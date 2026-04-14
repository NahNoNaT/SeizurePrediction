from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

DEFAULT_START_SEC = 0.0
DEFAULT_DURATION_SEC = 10.0
DEFAULT_TARGET_SAMPLING_RATE = 256.0
DEFAULT_BANDPASS_LOW_HZ = 0.5
DEFAULT_BANDPASS_HIGH_HZ = 40.0
MODEL_IMAGE_SIZE = 32
FEATURE_VECTOR_SIZE = 178
EEG_NAME_PATTERN = re.compile(
    r"(EEG|FP1|FP2|F3|F4|F7|F8|FZ|CZ|PZ|C3|C4|P3|P4|O1|O2|T3|T4|T5|T6|A1|A2)",
    re.IGNORECASE,
)
NON_EEG_MARKERS = ("ECG", "EKG", "EOG", "EMG", "RESP", "PHOTIC", "IBI", "BURSTS", "DC")


class EEGPreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedEEGSegment:
    raw_signal: np.ndarray
    spectrogram_tensor: torch.Tensor
    feature_vector: np.ndarray
    selected_channel: str
    original_sampling_rate: float
    sampling_rate: float
    duration_used_sec: float
    start_sec: float
    total_duration_sec: float
    warnings: list[str] = field(default_factory=list)


def prepare_segment(
    edf_path: str | Path,
    *,
    requested_channel: str | None = None,
    start_sec: float = DEFAULT_START_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
    target_sampling_rate: float = DEFAULT_TARGET_SAMPLING_RATE,
) -> PreparedEEGSegment:
    path = Path(edf_path)
    if path.suffix.lower() != ".edf":
        raise EEGPreprocessingError("Only EDF files are supported.")
    if start_sec < 0:
        raise EEGPreprocessingError("start_sec must be greater than or equal to 0.")
    if duration_sec <= 0:
        raise EEGPreprocessingError("duration_sec must be greater than 0.")

    try:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    except Exception as exc:
        raise EEGPreprocessingError(f"Unable to read EDF file: {exc}") from exc

    if not raw.ch_names:
        raise EEGPreprocessingError("The EDF file does not contain any channels.")

    selected_channel, warnings = _select_channel(raw, requested_channel)
    sampling_rate = float(raw.info["sfreq"])
    total_duration_sec = float(raw.n_times / sampling_rate) if sampling_rate > 0 else 0.0
    if total_duration_sec <= 0:
        raise EEGPreprocessingError("The EDF file does not contain a readable signal duration.")
    if start_sec >= total_duration_sec:
        raise EEGPreprocessingError(
            f"start_sec={start_sec} is outside the recording duration of {round(total_duration_sec, 3)} seconds."
        )

    effective_duration_sec = min(duration_sec, total_duration_sec - start_sec)
    start_sample = int(round(start_sec * sampling_rate))
    stop_sample = int(round((start_sec + effective_duration_sec) * sampling_rate))
    if stop_sample <= start_sample:
        raise EEGPreprocessingError("Requested EDF window produced no samples.")

    try:
        signal = raw.get_data(picks=[selected_channel], start=start_sample, stop=stop_sample)[0]
    except Exception as exc:
        raise EEGPreprocessingError(f"Unable to extract channel '{selected_channel}' from the EDF: {exc}") from exc

    if signal.size < 32:
        raise EEGPreprocessingError("The selected EDF window is too short for benchmarking.")

    processed = _preprocess_signal(signal.astype(np.float32, copy=False), sampling_rate, target_sampling_rate)
    spectrogram_tensor = _signal_to_spectrogram_tensor(processed, target_sampling_rate)
    feature_vector = _build_feature_vector(processed, target_sampling_rate)

    logger.info(
        "Prepared EDF segment for benchmarking",
        extra={"event": "benchmark_preprocessing_complete", "status": "ok"},
    )
    return PreparedEEGSegment(
        raw_signal=processed,
        spectrogram_tensor=spectrogram_tensor,
        feature_vector=feature_vector,
        selected_channel=selected_channel,
        original_sampling_rate=sampling_rate,
        sampling_rate=target_sampling_rate,
        duration_used_sec=round(processed.shape[0] / target_sampling_rate, 3),
        start_sec=round(start_sec, 3),
        total_duration_sec=round(total_duration_sec, 3),
        warnings=warnings,
    )


def _select_channel(raw: mne.io.BaseRaw, requested_channel: str | None) -> tuple[str, list[str]]:
    warnings: list[str] = []
    available_names = list(raw.ch_names)
    if requested_channel:
        requested_normalized = requested_channel.strip().casefold()
        for name in available_names:
            if name.casefold() == requested_normalized:
                return name, warnings
        warnings.append(
            f"Requested channel '{requested_channel}' was not found. Falling back to an EEG-like channel."
        )

    eeg_candidates = [name for name in available_names if _is_eeg_like_channel(raw, name)]
    if eeg_candidates:
        return eeg_candidates[0], warnings

    warnings.append("No EEG-like channel name was found. Falling back to the first channel in the EDF.")
    return available_names[0], warnings


def _is_eeg_like_channel(raw: mne.io.BaseRaw, name: str) -> bool:
    upper_name = name.upper()
    if any(marker in upper_name for marker in NON_EEG_MARKERS):
        return False
    try:
        index = raw.ch_names.index(name)
        if raw.get_channel_types(picks=[index])[0] == "eeg":
            return True
    except Exception:
        pass
    return EEG_NAME_PATTERN.search(name) is not None


def _preprocess_signal(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    centered = signal - np.mean(signal)
    if source_rate > 0 and abs(source_rate - target_rate) > 1e-6:
        centered = _resample(centered, source_rate, target_rate)
    filtered = _bandpass(centered, target_rate)
    scaled = filtered / max(float(np.std(filtered)), 1e-6)
    return np.nan_to_num(scaled.astype(np.float32, copy=False))


def _resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    source_rounded = max(int(round(source_rate)), 1)
    target_rounded = max(int(round(target_rate)), 1)
    gcd = np.gcd(source_rounded, target_rounded) or 1
    return scipy_signal.resample_poly(
        signal,
        up=target_rounded // gcd,
        down=source_rounded // gcd,
    ).astype(np.float32)


def _bandpass(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    nyquist = max(sampling_rate / 2.0, 1.0)
    low = max(DEFAULT_BANDPASS_LOW_HZ / nyquist, 1e-4)
    high = min(DEFAULT_BANDPASS_HIGH_HZ / nyquist, 0.999)
    if low >= high:
        return signal.astype(np.float32, copy=False)
    sos = scipy_signal.butter(4, [low, high], btype="bandpass", output="sos")
    if signal.shape[0] <= 32:
        return scipy_signal.sosfilt(sos, signal).astype(np.float32)
    return scipy_signal.sosfiltfilt(sos, signal).astype(np.float32)


def _signal_to_spectrogram_tensor(signal: np.ndarray, sampling_rate: float) -> torch.Tensor:
    nperseg = min(256, signal.shape[0])
    noverlap = min(nperseg // 2, max(nperseg - 1, 0))
    _, _, spectrogram = scipy_signal.spectrogram(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    if spectrogram.size == 0:
        raise EEGPreprocessingError("Unable to generate a spectrogram from the selected EEG window.")

    image = np.log1p(spectrogram.astype(np.float32, copy=False))
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum - minimum < 1e-6:
        normalized = np.zeros_like(image, dtype=np.float32)
    else:
        normalized = (image - minimum) / (maximum - minimum)

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), mode="bilinear", align_corners=False)
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(dtype=torch.float32)


def _build_feature_vector(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    frequencies, power = scipy_signal.welch(signal, fs=sampling_rate, nperseg=min(256, signal.shape[0]))
    power = np.maximum(power.astype(np.float32, copy=False), 1e-8)
    autocorrelation = scipy_signal.correlate(signal, signal, mode="full")
    autocorrelation = autocorrelation[autocorrelation.shape[0] // 2 :]
    autocorrelation = autocorrelation / max(float(autocorrelation[0]), 1e-6)

    scalar_features = np.array(
        [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.min(signal)),
            float(np.max(signal)),
            float(np.median(signal)),
            float(np.sqrt(np.mean(np.square(signal)))),
            float(skew(signal)),
            float(kurtosis(signal)),
            float(np.mean(np.abs(np.diff(signal)))),
            float(np.sum(np.abs(np.diff(np.signbit(signal).astype(np.int8))))),
            float(_band_power(frequencies, power, 0.5, 4.0)),
            float(_band_power(frequencies, power, 4.0, 8.0)),
            float(_band_power(frequencies, power, 8.0, 13.0)),
            float(_band_power(frequencies, power, 13.0, 30.0)),
        ],
        dtype=np.float32,
    )

    waveform_features = scipy_signal.resample(signal, 64).astype(np.float32)
    gradient_features = scipy_signal.resample(np.gradient(signal).astype(np.float32), 32).astype(np.float32)
    psd_features = scipy_signal.resample(np.log1p(power), 40).astype(np.float32)
    autocorr_features = scipy_signal.resample(autocorrelation[: min(autocorrelation.shape[0], 128)], 28).astype(np.float32)

    frequencies, _, spectrogram = scipy_signal.spectrogram(
        signal,
        fs=sampling_rate,
        nperseg=min(128, signal.shape[0]),
        noverlap=min(64, max(signal.shape[0] // 4, 1)),
        mode="magnitude",
    )
    spectral_summary = np.mean(spectrogram.astype(np.float32), axis=1) if spectrogram.size else np.zeros(32, dtype=np.float32)
    spectral_summary = scipy_signal.resample(spectral_summary, 32).astype(np.float32)

    combined = np.concatenate(
        [
            scalar_features,
            waveform_features,
            gradient_features,
            psd_features,
            autocorr_features,
            spectral_summary,
        ]
    ).astype(np.float32, copy=False)
    return np.nan_to_num(_fit_feature_vector(combined, FEATURE_VECTOR_SIZE))


def _band_power(frequencies: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(power[mask], frequencies[mask]))
    return float(np.trapz(power[mask], frequencies[mask]))


def _fit_feature_vector(features: np.ndarray, target_size: int) -> np.ndarray:
    if features.shape[0] == target_size:
        return features.astype(np.float32, copy=False)
    if features.shape[0] > target_size:
        return features[:target_size].astype(np.float32, copy=False)
    padded = np.zeros(target_size, dtype=np.float32)
    padded[: features.shape[0]] = features
    return padded
