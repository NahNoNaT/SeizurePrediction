from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np

from app.config import RuntimeConfig, runtime_config
from app.services.eeg_intake import EEGIntakeResult, EEGIntakeService
from app.services.errors import EEGValidationError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from scipy import signal as scipy_signal
except ImportError:  # pragma: no cover - optional dependency
    scipy_signal = None


@dataclass
class PreprocessedRecording:
    mapped_signal: np.ndarray
    segments: np.ndarray
    segment_times: list[tuple[float, float]]
    sampling_rate: float
    mapped_channels: list[str]
    duration_sec: float
    notes: list[str] = field(default_factory=list)


class SignalPreprocessingService:
    def __init__(
        self,
        config: RuntimeConfig = runtime_config,
        intake_service: EEGIntakeService | None = None,
    ):
        self.config = config
        self.intake_service = intake_service or EEGIntakeService(config)
        self.required_channel_order = list(config.required_channel_order)
        self.target_rate = float(config.target_sampling_rate_hz)

    def load_recording(self, file_path: Path, *, clinician_mode: bool = False) -> EEGIntakeResult:
        return self.intake_service.inspect(Path(file_path), clinician_mode=clinician_mode)

    def load_edf(self, file_path: Path) -> EEGIntakeResult:
        return self.load_recording(file_path, clinician_mode=True)

    def prepare(
        self,
        intake: EEGIntakeResult | Path,
        *,
        clinician_mode: bool = False,
    ) -> PreprocessedRecording:
        resolved_intake = intake if isinstance(intake, EEGIntakeResult) else self.load_recording(
            Path(intake),
            clinician_mode=clinician_mode,
        )
        expected_samples = max(
            int(
                round(
                    max(
                        resolved_intake.duration_sec,
                        self.config.default_window_length_seconds,
                    )
                    * self.target_rate
                )
            ),
            self.config.default_window_size,
        )
        mapped_signal, notes = self.map_channels(intake=resolved_intake, expected_samples=expected_samples)
        filtered_signal = self.bandpass_filter(mapped_signal)
        normalized_signal = self.normalize(filtered_signal)
        segments, segment_times = self.segment_windows(normalized_signal)
        return PreprocessedRecording(
            mapped_signal=normalized_signal,
            segments=segments,
            segment_times=segment_times,
            sampling_rate=self.target_rate,
            mapped_channels=self.required_channel_order.copy(),
            duration_sec=round(normalized_signal.shape[1] / self.target_rate, 2),
            notes=notes,
        )

    def map_channels(self, *, intake: EEGIntakeResult, expected_samples: int) -> tuple[np.ndarray, list[str]]:
        notes: list[str] = list(intake.conversion_messages)
        channels: list[np.ndarray] = []
        missing_channels: list[str] = []

        if intake.missing_channels and not intake.zero_fill_allowed:
            raise EEGValidationError(
                "validation_failed",
                "Zero-filled channels were requested for a recording that did not pass validation.",
            )

        for channel in self.required_channel_order:
            trace = intake.traces_by_channel.get(channel)
            if trace is None:
                missing_channels.append(channel)
                channels.append(np.zeros(expected_samples, dtype=np.float32))
                continue
            resampled = self._resample(trace.signal, trace.sampling_rate, self.target_rate)
            channels.append(self._fit_length(resampled, expected_samples))

        if missing_channels:
            notes.append(
                f"Zero-filled {len(missing_channels)} missing mapped channels to preserve the model input order."
            )
            logger.info(
                "Zero-filled validated missing channels",
                extra={"event": "preprocessing_zero_fill", "status": "VALIDATED"},
            )

        if intake.input_montage_type == "bipolar":
            notes.append(
                "The uploaded EEG used a bipolar montage and was converted heuristically into the fixed model input channel order."
            )
        if intake.derived_channels:
            notes.append(
                f"Derived channels from bipolar chains: {', '.join(intake.derived_channels[:10])}."
            )
        if intake.approximated_channels:
            notes.append(
                f"Approximated channels from neighboring bipolar signals: {', '.join(intake.approximated_channels[:10])}."
            )

        notes.append(
            f"Reordered EEG channels into the fixed analysis montage ({len(self.required_channel_order)} channels)."
        )
        return np.stack(channels, axis=0).astype(np.float32, copy=False), notes

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        if scipy_signal is None:
            centered = signal - np.mean(signal, axis=-1, keepdims=True)
            return np.nan_to_num(centered.astype(np.float32, copy=False))

        nyquist = max(self.target_rate / 2.0, 1.0)
        low = max(self.config.bandpass_low_hz / nyquist, 1e-4)
        high = min(self.config.bandpass_high_hz / nyquist, 0.999)
        if low >= high:
            return signal.astype(np.float32, copy=False)
        sos = scipy_signal.butter(4, [low, high], btype="bandpass", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, signal, axis=-1)
        return np.nan_to_num(filtered.astype(np.float32, copy=False))

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True)
        normalized = (signal - mean) / np.maximum(std, 1e-6)
        return np.nan_to_num(normalized.astype(np.float32, copy=False))

    def segment_windows(self, signal: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
        window_size = self.config.default_window_size
        hop_size = int(
            round(
                (self.config.default_window_length_seconds - self.config.default_overlap_seconds) * self.target_rate
            )
        )
        if hop_size <= 0:
            raise ValueError("Invalid analysis configuration: overlap must be smaller than the EEG window length.")

        if signal.shape[1] < window_size:
            padded = np.zeros((signal.shape[0], window_size), dtype=np.float32)
            padded[:, : signal.shape[1]] = signal
            signal = padded

        windows: list[np.ndarray] = []
        times: list[tuple[float, float]] = []
        for start in range(0, signal.shape[1] - window_size + 1, hop_size):
            end = start + window_size
            windows.append(signal[:, start:end])
            times.append((start / self.target_rate, end / self.target_rate))

        if not windows:
            windows.append(signal[:, :window_size])
            times.append((0.0, window_size / self.target_rate))

        return np.stack(windows, axis=0).astype(np.float32, copy=False), times

    def _build_mapped_signal(self, *, intake: EEGIntakeResult, expected_samples: int) -> tuple[np.ndarray, list[str]]:
        return self.map_channels(intake=intake, expected_samples=expected_samples)

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        return self.bandpass_filter(signal)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        return self.normalize(signal)

    def _segment(self, signal: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
        return self.segment_windows(signal)

    def _resample(self, signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
        if signal.size == 0:
            return np.zeros(self.config.default_window_size, dtype=np.float32)
        if source_rate <= 0 or abs(source_rate - target_rate) < 1e-6:
            return signal.astype(np.float32, copy=False)

        if scipy_signal is not None:
            gcd = np.gcd(int(round(source_rate)), int(round(target_rate))) or 1
            up = int(round(target_rate / gcd))
            down = int(round(source_rate / gcd))
            if up > 0 and down > 0:
                return scipy_signal.resample_poly(signal, up=up, down=down).astype(np.float32)

        duration_sec = signal.shape[0] / source_rate
        target_length = max(int(round(duration_sec * target_rate)), 1)
        source_times = np.linspace(0.0, duration_sec, signal.shape[0], endpoint=False, dtype=np.float32)
        target_times = np.linspace(0.0, duration_sec, target_length, endpoint=False, dtype=np.float32)
        return np.interp(target_times, source_times, signal).astype(np.float32)

    def _fit_length(self, signal: np.ndarray, expected_samples: int) -> np.ndarray:
        if signal.shape[0] == expected_samples:
            return signal.astype(np.float32, copy=False)
        if signal.shape[0] > expected_samples:
            return signal[:expected_samples].astype(np.float32, copy=False)
        padded = np.zeros(expected_samples, dtype=np.float32)
        padded[: signal.shape[0]] = signal
        return padded
