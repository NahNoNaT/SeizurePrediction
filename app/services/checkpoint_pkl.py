from __future__ import annotations

import importlib
import importlib.util
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import mne
import numpy as np


TARGET_FS = 256.0
EXPECTED_WINDOW_FEATURES = 1214
EXPECTED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


class CheckpointUnavailableError(RuntimeError):
    pass


class CheckpointPredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointBundle:
    model: Any
    scaler: Any
    top_idx: np.ndarray
    preprocess_window: Callable[[np.ndarray, int], np.ndarray]
    extract_features: Callable[[np.ndarray, list[str]], np.ndarray]
    model_path: Path
    scaler_path: Path
    top_idx_path: Path
    extractor_path: Path


@dataclass(frozen=True)
class CheckpointScanResult:
    scores: list[float]
    segment_times: list[tuple[float, float]]
    total_duration_sec: float
    window_sec: float
    hop_sec: float
    processed_windows: int
    truncated: bool
    warnings: list[str]
    model_label: str
    model_path: Path
    scaler_path: Path
    top_idx_path: Path
    extractor_path: Path
    inference_time_seconds: float


class CheckpointPklPredictionService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.checkpoint_dir = self.project_root / os.getenv("SEIZURE_PKL_CHECKPOINT_DIR", "checkpoint")
        self.model_file_name = os.getenv("SEIZURE_PKL_MODEL_FILE", "global_model_smart.pkl")
        self.scaler_file_name = os.getenv("SEIZURE_PKL_SCALER_FILE", "global_scaler.pkl")
        self.top_idx_file_name = os.getenv("SEIZURE_PKL_TOP_IDX_FILE", "global_top_idx.pkl")
        self.feature_extractor_file_name = os.getenv("SEIZURE_PKL_FEATURE_EXTRACTOR_FILE", "feature_extractor.py")
        self.max_missing_channels = self._resolve_int("SEIZURE_PKL_MAX_MISSING_CHANNELS", 4, minimum=0)
        self._bundle: CheckpointBundle | None = None

    def warmup(self) -> None:
        self._load_bundle()

    def predict_scan(
        self,
        edf_path: str | Path,
        *,
        window_sec: float = 10.0,
        hop_sec: float = 10.0,
        max_windows: int = 2000,
    ) -> CheckpointScanResult:
        if window_sec <= 0:
            raise CheckpointPredictionError("window_sec must be greater than 0.")
        if hop_sec <= 0:
            raise CheckpointPredictionError("hop_sec must be greater than 0.")
        if max_windows <= 0:
            raise CheckpointPredictionError("max_windows must be greater than 0.")

        started = time.perf_counter()
        bundle = self._load_bundle()
        matrix_uv, total_duration_sec, warnings = self._load_edf_matrix(Path(edf_path))
        starts = self._build_window_starts(
            total_duration_sec=total_duration_sec,
            window_sec=float(window_sec),
            hop_sec=float(hop_sec),
        )
        truncated = False
        if len(starts) > max_windows:
            starts = starts[:max_windows]
            truncated = True
            warnings.append(
                f"Checkpoint scan truncated to {max_windows} windows. Increase max window limit to scan the full recording."
            )

        segment_times: list[tuple[float, float]] = []
        scores: list[float] = []
        for start_sec in starts:
            end_sec = min(start_sec + window_sec, total_duration_sec)
            window = self._slice_window(matrix_uv, start_sec=start_sec, window_sec=window_sec)
            try:
                clean = bundle.preprocess_window(window.copy(), int(TARGET_FS))
                features = bundle.extract_features(clean, list(EXPECTED_CHANNELS))
            except Exception as exc:
                raise CheckpointPredictionError(f"Checkpoint feature extraction failed at {start_sec:.2f}s: {exc}") from exc

            vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
            if vector.shape[1] != EXPECTED_WINDOW_FEATURES:
                raise CheckpointPredictionError(
                    f"Checkpoint feature extractor returned {vector.shape[1]} values, expected {EXPECTED_WINDOW_FEATURES}."
                )

            score = self._predict_probability(bundle, vector)
            segment_times.append((round(start_sec, 3), round(end_sec, 3)))
            scores.append(score)

        elapsed = round(time.perf_counter() - started, 3)
        return CheckpointScanResult(
            scores=scores,
            segment_times=segment_times,
            total_duration_sec=round(total_duration_sec, 3),
            window_sec=float(window_sec),
            hop_sec=float(hop_sec),
            processed_windows=len(scores),
            truncated=truncated,
            warnings=warnings,
            model_label=bundle.model_path.name,
            model_path=bundle.model_path,
            scaler_path=bundle.scaler_path,
            top_idx_path=bundle.top_idx_path,
            extractor_path=bundle.extractor_path,
            inference_time_seconds=elapsed,
        )

    def _predict_probability(self, bundle: CheckpointBundle, features_1214: np.ndarray) -> float:
        try:
            scaled = bundle.scaler.transform(features_1214)
            selected = scaled[:, bundle.top_idx]
            if hasattr(bundle.model, "predict_proba"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    proba = np.asarray(bundle.model.predict_proba(selected), dtype=np.float32)
                if proba.ndim != 2 or proba.shape[0] == 0:
                    raise CheckpointPredictionError("predict_proba returned an invalid shape.")
                positive_index = self._resolve_positive_class_index(getattr(bundle.model, "classes_", None), proba.shape[1])
                return float(np.clip(proba[0, positive_index], 0.0, 1.0))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                prediction = np.asarray(bundle.model.predict(selected)).reshape(-1)
            if prediction.size == 0:
                raise CheckpointPredictionError("Model prediction returned an empty value.")
            return float(np.clip(float(prediction[0]), 0.0, 1.0))
        except CheckpointPredictionError:
            raise
        except Exception as exc:
            raise CheckpointPredictionError(f"Checkpoint model inference failed: {exc}") from exc

    def _resolve_positive_class_index(self, classes: Any, class_count: int) -> int:
        if class_count <= 1:
            return 0
        if classes is None:
            return 1 if class_count > 1 else 0

        labels = [str(item) for item in np.asarray(classes).tolist()]
        lowered = [label.strip().lower() for label in labels]
        for token in ("seizure", "preictal", "ictal", "positive", "1"):
            for index, label in enumerate(lowered):
                if token == "1":
                    if label in {"1", "1.0", "true"}:
                        return min(index, class_count - 1)
                    continue
                if token in label:
                    return min(index, class_count - 1)
        return min(class_count - 1, 1)

    def _slice_window(self, matrix_uv: np.ndarray, *, start_sec: float, window_sec: float) -> np.ndarray:
        sample_start = int(round(start_sec * TARGET_FS))
        sample_len = max(int(round(window_sec * TARGET_FS)), 1)
        sample_end = min(sample_start + sample_len, matrix_uv.shape[1])
        window = matrix_uv[:, sample_start:sample_end]
        if window.shape[1] == 0:
            return np.zeros((matrix_uv.shape[0], sample_len), dtype=np.float32)
        if window.shape[1] < sample_len:
            pad = sample_len - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad)), mode="edge")
        return window.astype(np.float32, copy=False)

    def _build_window_starts(self, *, total_duration_sec: float, window_sec: float, hop_sec: float) -> list[float]:
        if total_duration_sec <= 0:
            return [0.0]
        if total_duration_sec <= window_sec:
            return [0.0]
        starts = np.arange(
            0.0,
            max(total_duration_sec - window_sec, 0.0) + 1e-6,
            hop_sec,
            dtype=np.float32,
        ).tolist()
        last_start = max(total_duration_sec - window_sec, 0.0)
        if not starts or abs(float(starts[-1]) - last_start) > 1e-3:
            starts.append(last_start)
        return [round(float(item), 3) for item in starts]

    def _load_edf_matrix(self, edf_path: Path) -> tuple[np.ndarray, float, list[str]]:
        warnings: list[str] = []
        if not edf_path.exists():
            raise CheckpointPredictionError(f"EDF file was not found: {edf_path}")

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        try:
            if abs(float(raw.info["sfreq"]) - TARGET_FS) > 1e-6:
                raw.resample(TARGET_FS)

            raw_data_volts = raw.get_data()
            channel_map = {self._normalize_channel(name): index for index, name in enumerate(raw.ch_names)}
            rows: list[np.ndarray] = []
            missing: list[str] = []
            for expected in EXPECTED_CHANNELS:
                normalized = self._normalize_channel(expected)
                index = channel_map.get(normalized)
                if index is None:
                    missing.append(expected)
                    rows.append(np.zeros(raw.n_times, dtype=np.float32))
                else:
                    rows.append(raw_data_volts[index].astype(np.float32, copy=False))

            if missing:
                warnings.append(f"Missing checkpoint channels were zero-filled: {', '.join(missing)}.")
            if len(missing) > self.max_missing_channels:
                raise CheckpointPredictionError(
                    f"Checkpoint prediction requires at most {self.max_missing_channels} missing channels, got {len(missing)}."
                )

            matrix_volts = np.stack(rows, axis=0).astype(np.float32, copy=False)
            matrix_uv = matrix_volts * 1e6
            total_duration_sec = float(matrix_uv.shape[1] / TARGET_FS) if matrix_uv.shape[1] > 0 else 0.0
            return matrix_uv, total_duration_sec, warnings
        finally:
            raw.close()

    def _load_bundle(self) -> CheckpointBundle:
        if self._bundle is not None:
            return self._bundle

        checkpoint_dir = self.checkpoint_dir
        if not checkpoint_dir.exists():
            raise CheckpointUnavailableError(f"Checkpoint directory was not found: {checkpoint_dir}")

        model_path = checkpoint_dir / self.model_file_name
        scaler_path = checkpoint_dir / self.scaler_file_name
        top_idx_path = checkpoint_dir / self.top_idx_file_name
        extractor_path = checkpoint_dir / self.feature_extractor_file_name
        for path in (model_path, scaler_path, top_idx_path, extractor_path):
            if not path.exists():
                raise CheckpointUnavailableError(f"Missing checkpoint artifact: {path}")

        try:
            joblib = importlib.import_module("joblib")
        except ModuleNotFoundError as exc:
            raise CheckpointUnavailableError("Missing dependency 'joblib'. Install joblib and lightgbm.") from exc

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        top_idx_raw = joblib.load(top_idx_path)
        top_idx = np.asarray(top_idx_raw, dtype=np.int32).reshape(-1)
        if top_idx.size == 0:
            raise CheckpointUnavailableError("global_top_idx.pkl is empty.")

        extractor_module = self._load_feature_extractor_module(extractor_path)
        preprocess_window = getattr(extractor_module, "preprocess_window", None)
        extract_features = getattr(extractor_module, "extract_features", None)
        if not callable(preprocess_window) or not callable(extract_features):
            raise CheckpointUnavailableError(
                "Feature extractor must define callable preprocess_window(data, fs) and extract_features(data, ch_names)."
            )

        self._bundle = CheckpointBundle(
            model=model,
            scaler=scaler,
            top_idx=top_idx,
            preprocess_window=preprocess_window,
            extract_features=extract_features,
            model_path=model_path,
            scaler_path=scaler_path,
            top_idx_path=top_idx_path,
            extractor_path=extractor_path,
        )
        return self._bundle

    def _load_feature_extractor_module(self, extractor_path: Path) -> ModuleType:
        module_name = f"checkpoint_feature_extractor_{abs(hash(str(extractor_path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(extractor_path))
        if spec is None or spec.loader is None:
            raise CheckpointUnavailableError(f"Unable to import feature extractor from {extractor_path}.")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise CheckpointUnavailableError(f"Feature extractor import failed: {exc}") from exc
        return module

    def _normalize_channel(self, value: str) -> str:
        normalized = value.strip().upper().replace(" ", "").replace("_", "-")
        normalized = re.sub(r"-\d+$", "", normalized)
        return normalized

    def _resolve_int(self, env_name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(value, minimum)
