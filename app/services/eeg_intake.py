from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import RuntimeConfig, runtime_config
from app.schemas import RecordingValidationStatus
from app.services.errors import EEGValidationError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None

try:  # pragma: no cover - optional dependency
    import pyedflib
except ImportError:  # pragma: no cover - optional dependency
    pyedflib = None


COMMON_CHANNEL_ALIASES = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "F7": "F7",
    "F3": "F3",
    "FZ": "Fz",
    "F4": "F4",
    "F8": "F8",
    "T3": "T3",
    "T4": "T4",
    "T5": "T5",
    "T6": "T6",
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "C3": "C3",
    "CZ": "Cz",
    "C4": "C4",
    "P3": "P3",
    "PZ": "Pz",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "A1": "A1",
    "A2": "A2",
}

ANATOMICAL_COVERAGE = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "central": ("T3", "C3", "Cz", "C4", "T4"),
    "posterior": ("T5", "P3", "Pz", "P4", "T6", "O1"),
}


@dataclass
class ChannelTrace:
    source_name: str
    canonical_name: str
    sampling_rate: float
    signal: np.ndarray


@dataclass
class EEGIntakeResult:
    file_type: str
    duration_sec: float
    channel_count: int
    channel_names: list[str]
    traces_by_channel: dict[str, ChannelTrace]
    mapped_channels: list[str]
    missing_channels: list[str]
    validation_status: RecordingValidationStatus
    zero_fill_allowed: bool
    validation_messages: list[str] = field(default_factory=list)


@dataclass
class EEGPreviewResult:
    sampling_rate: float
    start_sec: float
    duration_sec: float
    total_duration_sec: float
    channels: list[str]
    available_channels: list[str]
    missing_channels: list[str]
    times: list[float]
    signals: list[list[float]]


class EEGIntakeService:
    default_preview_duration_sec = 30.0
    max_preview_duration_sec = 60.0
    default_preview_channel_count = 6
    max_preview_channel_count = 8

    def __init__(self, config: RuntimeConfig = runtime_config):
        self.config = config
        self.required_channel_order = list(config.required_channel_order)

    def inspect(self, file_path: Path, *, clinician_mode: bool = True) -> EEGIntakeResult:
        extension = file_path.suffix.lower()
        if clinician_mode and extension not in self.config.clinician_upload_extensions:
            raise EEGValidationError(
                "validation_failed",
                "Clinical workflow accepts EEG recording files in EDF format only.",
            )

        if extension == ".edf":
            result = self._read_edf(file_path)
        elif not clinician_mode and extension in self.config.internal_demo_extensions:
            result = self._read_internal_array(file_path)
        else:
            raise EEGValidationError("validation_failed", f"Unsupported EEG recording type '{extension}'.")

        self._validate_for_analysis(result)
        logger.info(
            "Recording validated",
            extra={"event": "recording_validated", "status": result.validation_status},
        )
        return result

    def _validate_for_analysis(self, result: EEGIntakeResult) -> None:
        mapped_count = len(result.mapped_channels)
        missing_count = len(result.missing_channels)
        missing_regions = [
            region
            for region, channels in ANATOMICAL_COVERAGE.items()
            if not any(channel in result.mapped_channels for channel in channels)
        ]

        validation_messages = list(result.validation_messages)
        if result.duration_sec < self.config.minimum_recording_duration_seconds:
            validation_messages.append(
                "Recording duration is too short for clinical analysis."
            )

        if mapped_count < self.config.minimum_mapped_channels:
            validation_messages.append(
                f"At least {self.config.minimum_mapped_channels} required channels must be mapped before analysis can run."
            )

        if missing_regions:
            regions_label = ", ".join(missing_regions)
            validation_messages.append(
                f"Clinical coverage is incomplete across the following regions: {regions_label}."
            )

        zero_fill_allowed = (
            mapped_count >= self.config.minimum_mapped_channels
            and missing_count <= self.config.max_zero_fill_channels
            and not missing_regions
            and result.duration_sec >= self.config.minimum_recording_duration_seconds
        )
        result.zero_fill_allowed = zero_fill_allowed

        if result.missing_channels:
            preview = ", ".join(result.missing_channels[:6])
            if zero_fill_allowed:
                validation_messages.append(
                    f"Missing required channels will be zero-filled for analysis: {preview}."
                )
            else:
                validation_messages.append(
                    f"Analysis is blocked because too many required channels are missing: {preview}."
                )

        if not zero_fill_allowed and result.missing_channels:
            result.validation_status = "BLOCKED"
        elif mapped_count < self.config.minimum_mapped_channels or missing_regions or result.duration_sec < self.config.minimum_recording_duration_seconds:
            result.validation_status = "BLOCKED"
        else:
            result.validation_status = "VALIDATED"

        result.validation_messages = validation_messages
        if result.validation_status != "VALIDATED":
            logger.warning(
                "Recording validation blocked analysis",
                extra={"event": "recording_blocked", "status": result.validation_status},
            )
            failure_message = " ".join(validation_messages)
            raise EEGValidationError(
                "validation_failed",
                failure_message,
                public_detail=failure_message,
            )

    def _read_edf(self, file_path: Path) -> EEGIntakeResult:
        if mne is not None:
            return self._read_edf_with_mne(file_path)
        if pyedflib is not None:
            return self._read_edf_with_pyedflib(file_path)
        raise EEGValidationError(
            "validation_failed",
            "EDF reading support is unavailable. Install pyEDFlib or MNE to enable clinical EDF intake.",
        )

    def preview(
        self,
        file_path: Path,
        *,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        channels: Sequence[str] | None = None,
    ) -> EEGPreviewResult:
        if not file_path.exists():
            raise EEGValidationError(
                "validation_failed",
                f"EEG recording file was not found at '{file_path}'.",
                public_detail="The EEG recording file is no longer available for preview.",
            )

        if file_path.suffix.lower() != ".edf":
            raise EEGValidationError(
                "validation_failed",
                "Waveform preview is available for EDF recordings only.",
            )

        if mne is not None:
            return self._preview_edf_with_mne(file_path, start_sec=start_sec, duration_sec=duration_sec, channels=channels)
        if pyedflib is not None:
            return self._preview_edf_with_pyedflib(file_path, start_sec=start_sec, duration_sec=duration_sec, channels=channels)
        raise EEGValidationError(
            "validation_failed",
            "EDF preview support is unavailable. Install pyEDFlib or MNE to enable waveform review.",
        )

    def _read_edf_with_mne(self, file_path: Path) -> EEGIntakeResult:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")
        channel_names = [str(name).strip() or f"Signal {index + 1}" for index, name in enumerate(raw.ch_names)]
        signals = raw.get_data()
        sampling_rate = float(raw.info.get("sfreq", 0.0))
        duration_sec = float(signals.shape[1] / sampling_rate) if sampling_rate > 0 else 0.0
        traces = self._build_trace_map(
            channel_names=channel_names,
            sample_rates=[sampling_rate] * len(channel_names),
            signals=[signals[index] for index in range(signals.shape[0])],
        )
        return self._finalize_intake(".edf", duration_sec, channel_names, traces)

    def _read_edf_with_pyedflib(self, file_path: Path) -> EEGIntakeResult:
        reader = pyedflib.EdfReader(str(file_path))
        try:
            channel_names = [label.strip() or f"Signal {index + 1}" for index, label in enumerate(reader.getSignalLabels())]
            sample_rates = [float(rate) for rate in reader.getSampleFrequencies()]
            duration_sec = float(reader.getFileDuration() or 0.0)
            signals = [np.asarray(reader.readSignal(index), dtype=np.float32) for index in range(reader.signals_in_file)]
        finally:
            reader.close()

        traces = self._build_trace_map(channel_names=channel_names, sample_rates=sample_rates, signals=signals)
        return self._finalize_intake(".edf", duration_sec, channel_names, traces)

    def _preview_edf_with_mne(
        self,
        file_path: Path,
        *,
        start_sec: float,
        duration_sec: float | None,
        channels: Sequence[str] | None,
    ) -> EEGPreviewResult:
        raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose="ERROR")
        try:
            available_channels = [str(name).strip() or f"Signal {index + 1}" for index, name in enumerate(raw.ch_names)]
            sampling_rate = float(raw.info.get("sfreq", 0.0))
            if sampling_rate <= 0:
                raise EEGValidationError("validation_failed", "The EDF recording does not expose a valid sampling rate.")

            total_duration_sec = float(raw.n_times / sampling_rate) if raw.n_times else 0.0
            selected_channels, missing_channels = self._resolve_preview_channels(channels, available_channels)
            start_sample, stop_sample, clamped_start_sec = self._resolve_preview_samples(
                total_duration_sec=total_duration_sec,
                sampling_rate=sampling_rate,
                requested_start_sec=start_sec,
                requested_duration_sec=duration_sec,
            )
            picks = [available_channels.index(channel_name) for channel_name in selected_channels]
            data = raw.get_data(picks=picks, start=start_sample, stop=stop_sample)
            times = (np.arange(start_sample, stop_sample, dtype=np.float32) / sampling_rate).tolist()
            signals = [
                np.nan_to_num(trace.astype(np.float32, copy=False), copy=False).tolist()
                for trace in data
            ]
            actual_duration_sec = float(len(times) / sampling_rate) if times else 0.0
            return EEGPreviewResult(
                sampling_rate=sampling_rate,
                start_sec=clamped_start_sec,
                duration_sec=actual_duration_sec,
                total_duration_sec=total_duration_sec,
                channels=selected_channels,
                available_channels=available_channels,
                missing_channels=missing_channels,
                times=times,
                signals=signals,
            )
        finally:
            close_method = getattr(raw, "close", None)
            if callable(close_method):
                close_method()

    def _read_internal_array(self, file_path: Path) -> EEGIntakeResult:
        array = self._load_internal_array(file_path)
        if array.ndim == 3:
            if array.shape[1] == len(self.required_channel_order):
                signal = np.transpose(array, (1, 0, 2)).reshape(len(self.required_channel_order), -1)
            elif array.shape[2] == len(self.required_channel_order):
                signal = np.transpose(array, (2, 0, 1)).reshape(len(self.required_channel_order), -1)
            else:
                raise EEGValidationError(
                    "validation_failed",
                    "Internal EEG tensor does not match the configured channel order.",
                )
        elif array.ndim == 2:
            signal = array if array.shape[0] <= array.shape[1] else array.transpose(1, 0)
        else:
            raise EEGValidationError(
                "validation_failed",
                "Internal EEG array must be two-dimensional or three-dimensional.",
            )

        if signal.shape[0] < len(self.required_channel_order):
            raise EEGValidationError(
                "validation_failed",
                "Internal EEG array does not include the required number of channels.",
            )

        signal = signal[: len(self.required_channel_order)].astype(np.float32, copy=False)
        sampling_rate = float(self.config.target_sampling_rate_hz)
        duration_sec = float(signal.shape[1] / sampling_rate) if sampling_rate > 0 else 0.0
        channel_names = list(self.required_channel_order)
        traces = {
            channel: ChannelTrace(
                source_name=channel,
                canonical_name=channel,
                sampling_rate=sampling_rate,
                signal=signal[index],
            )
            for index, channel in enumerate(channel_names)
        }
        return self._finalize_intake(file_path.suffix.lower(), duration_sec, channel_names, traces)

    def _build_trace_map(
        self,
        *,
        channel_names: list[str],
        sample_rates: list[float],
        signals: list[np.ndarray],
    ) -> dict[str, ChannelTrace]:
        trace_map: dict[str, ChannelTrace] = {}
        for name, sample_rate, signal in zip(channel_names, sample_rates, signals, strict=False):
            canonical_name = self._canonical_channel_name(name)
            if canonical_name in trace_map:
                continue
            cleaned_signal = np.nan_to_num(np.asarray(signal, dtype=np.float32), copy=False)
            if cleaned_signal.size == 0:
                continue
            trace_map[canonical_name] = ChannelTrace(
                source_name=name,
                canonical_name=canonical_name,
                sampling_rate=float(sample_rate),
                signal=cleaned_signal,
            )
        return trace_map

    def _finalize_intake(
        self,
        file_type: str,
        duration_sec: float,
        channel_names: list[str],
        trace_map: dict[str, ChannelTrace],
    ) -> EEGIntakeResult:
        mapped_channels = [channel for channel in self.required_channel_order if channel in trace_map]
        missing_channels = [channel for channel in self.required_channel_order if channel not in trace_map]
        validation_messages = [
            f"Mapped {len(mapped_channels)} of {len(self.required_channel_order)} required scalp EEG channels.",
            f"Recording duration: {duration_sec / 60.0:.1f} minutes." if duration_sec else "Recording duration could not be determined precisely.",
        ]
        return EEGIntakeResult(
            file_type=file_type,
            duration_sec=round(duration_sec, 2),
            channel_count=len(channel_names),
            channel_names=channel_names,
            traces_by_channel=trace_map,
            mapped_channels=mapped_channels,
            missing_channels=missing_channels,
            validation_status="PENDING",
            zero_fill_allowed=False,
            validation_messages=validation_messages,
        )

    def _preview_edf_with_pyedflib(
        self,
        file_path: Path,
        *,
        start_sec: float,
        duration_sec: float | None,
        channels: Sequence[str] | None,
    ) -> EEGPreviewResult:
        reader = pyedflib.EdfReader(str(file_path))
        try:
            available_channels = [
                label.strip() or f"Signal {index + 1}"
                for index, label in enumerate(reader.getSignalLabels())
            ]
            selected_channels, missing_channels = self._resolve_preview_channels(channels, available_channels)
            channel_indices = [available_channels.index(channel_name) for channel_name in selected_channels]
            sample_rates = [float(reader.getSampleFrequency(index)) for index in channel_indices]
            reference_rate = sample_rates[0] if sample_rates else 0.0
            if reference_rate <= 0:
                raise EEGValidationError("validation_failed", "The EDF recording does not expose a valid sampling rate.")
            if any(abs(rate - reference_rate) > 1e-6 for rate in sample_rates[1:]):
                raise EEGValidationError(
                    "validation_failed",
                    "The selected channels do not share a common sampling rate for preview rendering.",
                    public_detail="The selected EEG channels could not be previewed together because their sample rates differ.",
                )

            channel_lengths = [int(reader.getNSamples()[index]) for index in channel_indices]
            total_duration_sec = min(length / reference_rate for length in channel_lengths) if channel_lengths else 0.0
            start_sample, stop_sample, clamped_start_sec = self._resolve_preview_samples(
                total_duration_sec=total_duration_sec,
                sampling_rate=reference_rate,
                requested_start_sec=start_sec,
                requested_duration_sec=duration_sec,
            )
            sample_count = max(stop_sample - start_sample, 0)
            signals = [
                np.nan_to_num(
                    np.asarray(reader.readSignal(channel_index, start_sample, sample_count), dtype=np.float32),
                    copy=False,
                ).tolist()
                for channel_index in channel_indices
            ]
            times = (np.arange(start_sample, stop_sample, dtype=np.float32) / reference_rate).tolist()
            actual_duration_sec = float(len(times) / reference_rate) if times else 0.0
            return EEGPreviewResult(
                sampling_rate=reference_rate,
                start_sec=clamped_start_sec,
                duration_sec=actual_duration_sec,
                total_duration_sec=total_duration_sec,
                channels=selected_channels,
                available_channels=available_channels,
                missing_channels=missing_channels,
                times=times,
                signals=signals,
            )
        finally:
            reader.close()

    def _canonical_channel_name(self, value: str) -> str:
        cleaned = (
            value.replace("EEG", "")
            .replace("-REF", "")
            .replace("REF", "")
            .replace("-LE", "")
            .replace("-A1", "")
            .replace("-A2", "")
            .replace(" ", "")
            .replace("-", "")
            .upper()
        )
        return COMMON_CHANNEL_ALIASES.get(cleaned, value.strip() or "Unknown")

    def _load_internal_array(self, file_path: Path) -> np.ndarray:
        extension = file_path.suffix.lower()
        if extension == ".npy":
            return np.load(file_path, allow_pickle=False)
        if extension == ".npz":
            with np.load(file_path, allow_pickle=False) as archive:
                key = archive.files[0] if archive.files else None
                if key is None:
                    raise EEGValidationError("validation_failed", "The internal EEG archive does not contain any arrays.")
                return archive[key]
        if extension == ".csv":
            return np.genfromtxt(file_path, delimiter=",", dtype=np.float32)
        raise EEGValidationError("validation_failed", f"Unsupported internal EEG file type '{extension}'.")

    def _resolve_preview_channels(
        self,
        requested_channels: Sequence[str] | None,
        available_channels: list[str],
    ) -> tuple[list[str], list[str]]:
        if not available_channels:
            raise EEGValidationError("validation_failed", "The EDF recording does not contain any channels for preview.")

        normalized_lookup: dict[str, str] = {}
        for channel_name in available_channels:
            for key in self._channel_lookup_keys(channel_name):
                normalized_lookup.setdefault(key, channel_name)

        if requested_channels:
            selected: list[str] = []
            missing: list[str] = []
            for channel_name in requested_channels:
                cleaned = channel_name.strip()
                if not cleaned:
                    continue
                resolved = None
                for key in self._channel_lookup_keys(cleaned):
                    resolved = normalized_lookup.get(key)
                    if resolved is not None:
                        break
                if resolved is None:
                    missing.append(cleaned)
                    continue
                if resolved not in selected:
                    selected.append(resolved)
                if len(selected) >= self.max_preview_channel_count:
                    break
            if not selected:
                requested_label = ", ".join(missing or requested_channels)
                raise EEGValidationError(
                    "validation_failed",
                    f"None of the requested preview channels were found: {requested_label}.",
                    public_detail="None of the requested EEG channels were available for waveform preview.",
                )
            return selected, missing

        return available_channels[: self.default_preview_channel_count], []

    def _resolve_preview_samples(
        self,
        *,
        total_duration_sec: float,
        sampling_rate: float,
        requested_start_sec: float,
        requested_duration_sec: float | None,
    ) -> tuple[int, int, float]:
        if total_duration_sec <= 0:
            raise EEGValidationError("validation_failed", "The EDF recording does not contain a measurable preview duration.")

        duration_sec = requested_duration_sec if requested_duration_sec and requested_duration_sec > 0 else self.default_preview_duration_sec
        duration_sec = min(duration_sec, self.max_preview_duration_sec)
        duration_sec = min(duration_sec, total_duration_sec)

        max_start_sec = max(total_duration_sec - duration_sec, 0.0)
        start_sec = min(max(float(requested_start_sec or 0.0), 0.0), max_start_sec)

        total_sample_count = max(int(np.floor(total_duration_sec * sampling_rate)), 1)
        start_sample = min(int(round(start_sec * sampling_rate)), max(total_sample_count - 1, 0))
        sample_count = max(int(round(duration_sec * sampling_rate)), 1)
        stop_sample = min(start_sample + sample_count, total_sample_count)
        if stop_sample <= start_sample:
            stop_sample = min(start_sample + 1, total_sample_count)
        return start_sample, stop_sample, start_sec

    def _channel_lookup_keys(self, value: str) -> set[str]:
        stripped = value.strip()
        compact = "".join(character for character in stripped.upper() if character.isalnum())
        canonical = self._canonical_channel_name(stripped).strip().upper()
        canonical_compact = "".join(character for character in canonical if character.isalnum())
        return {stripped.casefold(), compact.casefold(), canonical.casefold(), canonical_compact.casefold()}
