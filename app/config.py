from __future__ import annotations

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or default


STANDARD_MODEL_CHANNEL_ORDER = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        for key in ("event", "case_id", "recording_id", "analysis_id", "status", "code", "upload_filename"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


class RuntimeConfig(BaseModel):
    app_title: str = "Clinical EEG Seizure Prediction Platform"
    app_subtitle: str = "Doctor-facing support for EEG seizure-risk review"
    platform_name: str = "NeuroVue EEG Clinical Support"
    model_name: str = os.getenv("SEIZURE_MODEL_NAME", "RawWindow CNN-SeqBiGRU Preictal Risk Estimator")
    default_model_version: str = os.getenv("SEIZURE_MODEL_VERSION", "raw-window-cnn-seqbigru-v1")
    model_versions: tuple[str, ...] = _get_csv("SEIZURE_MODEL_VERSIONS", ("raw-window-cnn-seqbigru-v1",))
    input_format: str = "EDF scalp EEG recordings"
    required_channel_order: tuple[str, ...] = _get_csv("SEIZURE_REQUIRED_CHANNELS", STANDARD_MODEL_CHANNEL_ORDER)
    minimum_mapped_channels: int = _get_int("SEIZURE_MINIMUM_MAPPED_CHANNELS", 14)
    max_zero_fill_channels: int = _get_int("SEIZURE_MAX_ZERO_FILL_CHANNELS", 4)
    minimum_recording_duration_seconds: float = _get_float("SEIZURE_MIN_RECORDING_DURATION_SECONDS", 30.0)
    target_sampling_rate_hz: int = _get_int("SEIZURE_TARGET_SAMPLE_RATE_HZ", 256)
    bandpass_low_hz: float = _get_float("SEIZURE_BANDPASS_LOW_HZ", 0.5)
    bandpass_high_hz: float = _get_float("SEIZURE_BANDPASS_HIGH_HZ", 40.0)
    default_window_length_seconds: float = Field(default=_get_float("SEIZURE_WINDOW_LENGTH_SECONDS", 5.0), gt=0.0)
    default_overlap_seconds: float = Field(default=_get_float("SEIZURE_OVERLAP_SECONDS", 2.5), ge=0.0)
    default_window_size: int = _get_int("SEIZURE_EXPECTED_WINDOW_SIZE", 1280)
    default_sequence_length: int = _get_int("SEIZURE_SEQUENCE_LENGTH", 8)
    inference_batch_size: int = _get_int("SEIZURE_INFERENCE_BATCH_SIZE", 16)
    default_threshold: float = Field(default=_get_float("SEIZURE_DEFAULT_THRESHOLD", 0.45), ge=0.0, le=1.0)
    smoothing_method: str = os.getenv("SEIZURE_SMOOTHING_METHOD", "ema")
    smoothing_alpha: float = Field(default=_get_float("SEIZURE_SMOOTHING_ALPHA", 0.35), ge=0.0, le=1.0)
    smoothing_window: int = _get_int("SEIZURE_SMOOTHING_WINDOW", 3)
    consecutive_segments_required: int = _get_int("SEIZURE_CONSECUTIVE_SEGMENTS_REQUIRED", 3)
    model_device_preference: str = os.getenv("SEIZURE_MODEL_DEVICE", "cpu")
    strict_checkpoint_loading: bool = _get_bool("SEIZURE_STRICT_CHECKPOINT_LOADING", True)
    auto_discover_checkpoints: bool = _get_bool("SEIZURE_AUTO_DISCOVER_CHECKPOINTS", True)
    checkpoint_path: str | None = os.getenv("SEIZURE_MODEL_CHECKPOINT")
    checkpoint_paths: tuple[str, ...] = _get_csv("SEIZURE_MODEL_CHECKPOINTS", ())
    checkpoint_directory_name: str = os.getenv("SEIZURE_MODEL_CHECKPOINT_DIR", "models/checkpoints")
    checkpoint_extensions: tuple[str, ...] = tuple(
        extension.lower()
        for extension in _get_csv("SEIZURE_MODEL_CHECKPOINT_EXTENSIONS", (".pt", ".pth", ".ckpt"))
    )
    inference_status: str = "model_unavailable"
    supported_upload_extensions: tuple[str, ...] = (".edf",)
    clinician_upload_extensions: tuple[str, ...] = (".edf",)
    internal_demo_extensions: tuple[str, ...] = (".npy", ".npz", ".csv")
    max_upload_size_mb: int = _get_int("SEIZURE_MAX_UPLOAD_SIZE_MB", 2048)
    data_directory_name: str = "data"
    uploads_directory_name: str = "uploads"
    reports_directory_name: str = "reports"
    database_file_name: str = "clinical_cases.db"
    research_disclaimer: str = (
        "This platform supports clinician review and does not replace professional medical diagnosis."
    )
    log_level: str = os.getenv("APP_LOG_LEVEL", "INFO")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def zero_fill_allowed(self) -> bool:
        return self.max_zero_fill_channels > 0

    def uploads_directory(self, project_root: Path) -> Path:
        return project_root / self.uploads_directory_name

    def data_directory(self, project_root: Path) -> Path:
        return project_root / self.data_directory_name

    def reports_directory(self, project_root: Path) -> Path:
        return project_root / self.reports_directory_name

    def database_file(self, project_root: Path) -> Path:
        return self.data_directory(project_root) / self.database_file_name

    def checkpoint_directory(self, project_root: Path) -> Path:
        path = Path(self.checkpoint_directory_name)
        return path if path.is_absolute() else project_root / path

    def explicit_checkpoint_paths(self) -> tuple[str, ...]:
        if self.checkpoint_paths:
            return tuple(path for path in self.checkpoint_paths if path)
        return (self.checkpoint_path,) if self.checkpoint_path else ()

    def discovered_checkpoint_paths(self, project_root: Path) -> tuple[Path, ...]:
        if not self.auto_discover_checkpoints:
            return ()
        checkpoint_directory = self.checkpoint_directory(project_root)
        if not checkpoint_directory.exists() or not checkpoint_directory.is_dir():
            return ()
        allowed_extensions = {extension.lower() for extension in self.checkpoint_extensions}
        discovered = sorted(
            (
                path
                for path in checkpoint_directory.iterdir()
                if path.is_file() and path.suffix.lower() in allowed_extensions
            ),
            key=lambda path: path.name.lower(),
        )
        return tuple(discovered)

    def resolved_checkpoint_path(self, project_root: Path) -> Path | None:
        configured_paths = self.configured_checkpoint_paths(project_root)
        if not configured_paths:
            return None
        path = Path(configured_paths[0])
        return path if path.is_absolute() else project_root / path

    def configured_checkpoint_paths(self, project_root: Path | None = None) -> tuple[str, ...]:
        explicit_paths = self.explicit_checkpoint_paths()
        if explicit_paths:
            return explicit_paths
        if project_root is None:
            return ()
        configured_paths: list[str] = []
        for path in self.discovered_checkpoint_paths(project_root):
            if path.is_absolute():
                try:
                    configured_paths.append(path.relative_to(project_root).as_posix())
                except ValueError:
                    configured_paths.append(str(path))
            else:
                configured_paths.append(path.as_posix())
        return tuple(configured_paths)

    def resolved_checkpoint_paths(self, project_root: Path) -> tuple[Path, ...]:
        resolved: list[Path] = []
        for checkpoint_path in self.configured_checkpoint_paths(project_root):
            path = Path(checkpoint_path)
            resolved.append(path if path.is_absolute() else project_root / path)
        return tuple(resolved)


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "app.config.JsonLogFormatter",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "level": level.upper(),
                }
            },
            "root": {"handlers": ["default"], "level": level.upper()},
        }
    )


runtime_config = RuntimeConfig()
