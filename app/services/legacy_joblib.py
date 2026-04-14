from __future__ import annotations

import importlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from preprocessing import DEFAULT_DURATION_SEC, DEFAULT_START_SEC, PreparedEEGSegment, prepare_segment

logger = logging.getLogger(__name__)

LegacySource = Literal["universal_lopo", "loso_final"]
LegacyFeatureSet = Literal["LBP", "GLCM", "COMBINED"]

UNIVERSAL_DIR_NAME = "UNIVERSAL_LOPO_MODELS"
LOSO_DIR_NAME = "LOSO_Models_Final"
FEATURE_DIMENSIONS: dict[LegacyFeatureSet, int] = {"LBP": 220, "GLCM": 440, "COMBINED": 660}

_UNIVERSAL_MODEL_RE = re.compile(
    r"^model_UNIVERSAL_(?P<algorithm>[^_]+)_(?P<feature_set>LBP|GLCM|COMBINED)\.joblib$"
)
_UNIVERSAL_SCALER_RE = re.compile(r"^scaler_UNIVERSAL_(?P<feature_set>LBP|GLCM|COMBINED)\.joblib$")
_LOSO_MODEL_RE = re.compile(
    r"^model_(?P<subject_id>chb\d{2})_(?P<algorithm>[^_]+)_(?P<feature_set>LBP|GLCM|COMBINED)\.joblib$"
)
_LOSO_SCALER_RE = re.compile(r"^scaler_(?P<subject_id>chb\d{2})_(?P<feature_set>LBP|GLCM|COMBINED)\.joblib$")


class LegacyPredictionError(RuntimeError):
    pass


class LegacyDependencyError(LegacyPredictionError):
    pass


class LegacyFeatureExtractorUnavailableError(LegacyPredictionError):
    pass


class LegacyFeatureExtractor(Protocol):
    def extract(
        self,
        *,
        prepared_segment: PreparedEEGSegment,
        feature_set: LegacyFeatureSet,
        edf_path: str | Path | None = None,
    ) -> np.ndarray:
        ...


class MissingLegacyFeatureExtractor:
    def extract(
        self,
        *,
        prepared_segment: PreparedEEGSegment,
        feature_set: LegacyFeatureSet,
        edf_path: str | Path | None = None,
    ) -> np.ndarray:
        raise LegacyFeatureExtractorUnavailableError(
            "Legacy feature extractor is not configured yet. "
            "Provide step1_extract_features.py + chb_mit_preprocess.py with required dependencies, "
            "or set SEIZURE_LEGACY_FEATURE_EXTRACTOR to a custom extractor object exposing "
            "extract(prepared_segment=..., feature_set=..., edf_path=...) -> np.ndarray."
        )


@dataclass(frozen=True)
class LegacyModelSpec:
    source: LegacySource
    subject_id: str | None
    algorithm: str
    feature_set: LegacyFeatureSet
    model_path: Path
    scaler_path: Path | None

    @property
    def model_id(self) -> str:
        if self.source == "universal_lopo":
            return f"universal:{self.algorithm}:{self.feature_set}".lower()
        subject = (self.subject_id or "unknown").lower()
        return f"loso:{subject}:{self.algorithm}:{self.feature_set}".lower()

    @property
    def display_name(self) -> str:
        subject_tag = (self.subject_id or "UNIVERSAL").upper()
        return f"{subject_tag} | {self.algorithm} | {self.feature_set}"

    @property
    def model_name(self) -> str:
        subject_tag = self.subject_id or "UNIVERSAL"
        return f"{subject_tag}_{self.algorithm}_{self.feature_set}"


@dataclass(frozen=True)
class LegacyPredictionModelResult:
    model_name: str
    source: LegacySource
    subject_id: str | None
    algorithm: str
    feature_set: LegacyFeatureSet
    predicted_label: str
    positive_class_label: str
    seizure_probability: float | None
    raw_score: float | None
    inference_time_ms: float | None
    notes: str


@dataclass(frozen=True)
class LegacyPredictionSummary:
    majority_vote: str
    average_probability: float | None
    positive_votes: int
    total_models: int
    confidence_note: str


@dataclass(frozen=True)
class LegacyPredictionResponseData:
    selected_channel: str
    sampling_rate: float
    duration_used_sec: float
    source: LegacySource
    selected_model_id: str | None
    matched_model_count: int
    subject_id: str | None
    algorithm: str | None
    feature_set: LegacyFeatureSet | None
    models: list[LegacyPredictionModelResult]
    summary: LegacyPredictionSummary
    warnings: list[str]
    scan: LegacyScanResultData | None = None


@dataclass(frozen=True)
class LegacyScanTimelinePointData:
    window_index: int
    start_sec: float
    end_sec: float
    average_probability: float | None
    majority_vote: str
    positive_votes: int
    successful_models: int
    total_models: int
    top_probability: float | None
    top_model_name: str | None


@dataclass(frozen=True)
class LegacyScanPeakWindowData:
    window_index: int
    start_sec: float
    end_sec: float
    average_probability: float | None
    majority_vote: str
    positive_votes: int
    successful_models: int
    total_models: int
    top_probability: float | None
    top_model_name: str | None
    models: list[LegacyPredictionModelResult]
    summary: LegacyPredictionSummary


@dataclass(frozen=True)
class LegacyScanResultData:
    enabled: bool
    total_duration_sec: float
    window_sec: float
    hop_sec: float
    window_count: int
    truncated: bool
    timeline: list[LegacyScanTimelinePointData]
    peak_window: LegacyScanPeakWindowData | None


class LegacyJoblibPredictionService:
    def __init__(self, project_root: Path, *, feature_extractor: LegacyFeatureExtractor | None = None):
        self.project_root = Path(project_root)
        self.universal_dir = self.project_root / UNIVERSAL_DIR_NAME
        self.loso_dir = self.project_root / LOSO_DIR_NAME
        self.feature_extractor: LegacyFeatureExtractor = feature_extractor or self._resolve_feature_extractor()
        self._catalog_cache: list[LegacyModelSpec] | None = None
        self._artifact_cache: dict[Path, Any] = {}

    def warmup(self) -> None:
        discovered = len(self.catalog())
        logger.info(
            "Legacy joblib catalog warmup completed",
            extra={"event": "legacy_catalog_ready", "status": f"models:{discovered}"},
        )

    def catalog(self) -> list[LegacyModelSpec]:
        if self._catalog_cache is None:
            self._catalog_cache = self._discover_catalog()
        return list(self._catalog_cache)

    def catalog_summary(self) -> dict[str, Any]:
        models = self.catalog()
        subjects = sorted({item.subject_id for item in models if item.subject_id})
        algorithms = sorted({item.algorithm for item in models})
        feature_sets = sorted({item.feature_set for item in models})
        counts_by_source: dict[str, int] = {"universal_lopo": 0, "loso_final": 0}
        counts_by_subject: dict[str, int] = {}
        for item in models:
            counts_by_source[item.source] = counts_by_source.get(item.source, 0) + 1
            if item.subject_id:
                subject_key = item.subject_id.lower()
                counts_by_subject[subject_key] = counts_by_subject.get(subject_key, 0) + 1
        return {
            "model_count": len(models),
            "subjects": subjects,
            "algorithms": algorithms,
            "feature_sets": feature_sets,
            "counts_by_source": counts_by_source,
            "counts_by_subject": dict(sorted(counts_by_subject.items())),
            "models": models,
        }

    def predict(
        self,
        edf_path: str | Path,
        *,
        model_id: str | None = None,
        source: LegacySource = "universal_lopo",
        subject_id: str | None = None,
        algorithm: str | None = None,
        feature_set: LegacyFeatureSet | None = "COMBINED",
        channel: str | None = None,
        start_sec: float = DEFAULT_START_SEC,
        duration_sec: float = DEFAULT_DURATION_SEC,
        max_models: int = 24,
        scan_full_file: bool = False,
        scan_window_sec: float = 5.0,
        scan_hop_sec: float = 5.0,
        scan_start_sec: float = 0.0,
        scan_end_sec: float | None = None,
        scan_max_windows: int = 240,
    ) -> LegacyPredictionResponseData:
        if scan_full_file:
            return self._predict_scan_full_file(
                edf_path=edf_path,
                model_id=model_id,
                source=source,
                subject_id=subject_id,
                algorithm=algorithm,
                feature_set=feature_set,
                channel=channel,
                max_models=max_models,
                scan_window_sec=scan_window_sec,
                scan_hop_sec=scan_hop_sec,
                scan_start_sec=scan_start_sec,
                scan_end_sec=scan_end_sec,
                scan_max_windows=scan_max_windows,
            )

        resolved_edf_path = Path(edf_path)
        prepared = prepare_segment(
            resolved_edf_path,
            requested_channel=channel,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
        warnings = list(prepared.warnings)

        specs = self._select_specs(
            model_id=model_id,
            source=source,
            subject_id=subject_id,
            algorithm=algorithm,
            feature_set=feature_set,
        )
        if not specs:
            raise LegacyPredictionError("No matching legacy model was found for the requested filters.")

        if len(specs) > max_models:
            warnings.append(
                f"Selected {len(specs)} models, but only the first {max_models} were executed. "
                "Narrow source/subject/algorithm/feature_set to run fewer models."
            )
            specs = specs[:max_models]

        resolved_source = specs[0].source
        resolved_subject = specs[0].subject_id if model_id else subject_id
        resolved_algorithm = specs[0].algorithm if model_id else algorithm
        resolved_feature_set = specs[0].feature_set if model_id else feature_set

        feature_cache: dict[tuple[float, float, LegacyFeatureSet], np.ndarray] = {}
        model_results = self._predict_specs_for_window(
            specs=specs,
            prepared=prepared,
            edf_path=resolved_edf_path,
            warnings=warnings,
            feature_cache=feature_cache,
            warning_prefix="",
        )

        warnings.append(
            "Legacy probability is mapped to a positive class inferred from estimator metadata. "
            "Confirm class mapping after plugging in the original feature extractor."
        )
        summary = self._build_summary(model_results)
        return LegacyPredictionResponseData(
            selected_channel=prepared.selected_channel,
            sampling_rate=prepared.sampling_rate,
            duration_used_sec=prepared.duration_used_sec,
            source=resolved_source,
            selected_model_id=model_id.strip().lower() if model_id else None,
            matched_model_count=len(specs),
            subject_id=resolved_subject,
            algorithm=resolved_algorithm,
            feature_set=resolved_feature_set,
            models=model_results,
            summary=summary,
            warnings=warnings,
            scan=None,
        )

    def _predict_one(
        self,
        *,
        spec: LegacyModelSpec,
        prepared: PreparedEEGSegment,
        edf_path: Path,
        feature_cache: dict[tuple[float, float, LegacyFeatureSet], np.ndarray],
    ) -> LegacyPredictionModelResult:
        matrix = self._get_or_extract_feature_matrix(
            prepared=prepared,
            feature_set=spec.feature_set,
            edf_path=edf_path,
            feature_cache=feature_cache,
        )

        scaler = self._load_artifact(spec.scaler_path) if spec.scaler_path else None
        if scaler is not None and hasattr(scaler, "transform"):
            matrix = scaler.transform(matrix)

        model = self._load_artifact(spec.model_path)
        predicted_label, positive_class_label, probability, raw_score = self._run_estimator(model, matrix)
        return LegacyPredictionModelResult(
            model_name=spec.model_name,
            source=spec.source,
            subject_id=spec.subject_id,
            algorithm=spec.algorithm,
            feature_set=spec.feature_set,
            predicted_label=predicted_label,
            positive_class_label=positive_class_label,
            seizure_probability=probability,
            raw_score=raw_score,
            inference_time_ms=0.0,
            notes=(
                f"Legacy {spec.source} model using {spec.feature_set} features. "
                f"Positive class interpreted as '{positive_class_label}'."
            ),
        )

    def _get_or_extract_feature_matrix(
        self,
        *,
        prepared: PreparedEEGSegment,
        feature_set: LegacyFeatureSet,
        edf_path: Path,
        feature_cache: dict[tuple[float, float, LegacyFeatureSet], np.ndarray],
    ) -> np.ndarray:
        key = (round(float(prepared.start_sec), 3), round(float(prepared.duration_used_sec), 3), feature_set)
        cached = feature_cache.get(key)
        if cached is not None:
            return cached

        try:
            features = self.feature_extractor.extract(
                prepared_segment=prepared,
                feature_set=feature_set,
                edf_path=edf_path,
            )
        except TypeError:
            # Backward compatibility for custom extractors that still implement the old signature.
            features = self.feature_extractor.extract(
                prepared_segment=prepared,
                feature_set=feature_set,
            )
        matrix = self._normalize_features(features=features, feature_set=feature_set)
        feature_cache[key] = matrix
        return matrix

    def _predict_specs_for_window(
        self,
        *,
        specs: list[LegacyModelSpec],
        prepared: PreparedEEGSegment,
        edf_path: Path,
        warnings: list[str],
        feature_cache: dict[tuple[float, float, LegacyFeatureSet], np.ndarray],
        warning_prefix: str,
    ) -> list[LegacyPredictionModelResult]:
        model_results: list[LegacyPredictionModelResult] = []
        for spec in specs:
            started = time.perf_counter()
            try:
                result = self._predict_one(
                    spec=spec,
                    prepared=prepared,
                    edf_path=edf_path,
                    feature_cache=feature_cache,
                )
                elapsed = round((time.perf_counter() - started) * 1000.0, 3)
                model_results.append(
                    LegacyPredictionModelResult(
                        model_name=result.model_name,
                        source=result.source,
                        subject_id=result.subject_id,
                        algorithm=result.algorithm,
                        feature_set=result.feature_set,
                        predicted_label=result.predicted_label,
                        positive_class_label=result.positive_class_label,
                        seizure_probability=result.seizure_probability,
                        raw_score=result.raw_score,
                        inference_time_ms=elapsed,
                        notes=result.notes,
                    )
                )
            except (LegacyFeatureExtractorUnavailableError, LegacyDependencyError):
                raise
            except Exception as exc:
                logger.exception(
                    "Legacy model prediction failed",
                    extra={"event": "legacy_prediction_failed", "status": spec.model_name},
                )
                warnings.append(f"{warning_prefix}{spec.model_name} failed: {exc}")
                model_results.append(
                    LegacyPredictionModelResult(
                        model_name=spec.model_name,
                        source=spec.source,
                        subject_id=spec.subject_id,
                        algorithm=spec.algorithm,
                        feature_set=spec.feature_set,
                        predicted_label="error",
                        positive_class_label="unknown",
                        seizure_probability=None,
                        raw_score=None,
                        inference_time_ms=None,
                        notes=f"Load/inference failed: {exc}",
                    )
                )
        return model_results

    def _predict_scan_full_file(
        self,
        *,
        edf_path: str | Path,
        model_id: str | None,
        source: LegacySource,
        subject_id: str | None,
        algorithm: str | None,
        feature_set: LegacyFeatureSet | None,
        channel: str | None,
        max_models: int,
        scan_window_sec: float,
        scan_hop_sec: float,
        scan_start_sec: float,
        scan_end_sec: float | None,
        scan_max_windows: int,
    ) -> LegacyPredictionResponseData:
        if scan_window_sec <= 0:
            raise LegacyPredictionError("scan_window_sec must be greater than 0.")
        if scan_hop_sec <= 0:
            raise LegacyPredictionError("scan_hop_sec must be greater than 0.")
        if scan_max_windows <= 0:
            raise LegacyPredictionError("scan_max_windows must be greater than 0.")

        resolved_edf_path = Path(edf_path)
        scan_start = max(float(scan_start_sec), 0.0)
        initial = prepare_segment(
            resolved_edf_path,
            requested_channel=channel,
            start_sec=scan_start,
            duration_sec=scan_window_sec,
        )
        warnings = list(initial.warnings)

        specs = self._select_specs(
            model_id=model_id,
            source=source,
            subject_id=subject_id,
            algorithm=algorithm,
            feature_set=feature_set,
        )
        if not specs:
            raise LegacyPredictionError("No matching legacy model was found for the requested filters.")

        if len(specs) > max_models:
            warnings.append(
                f"Selected {len(specs)} models, but only the first {max_models} were executed. "
                "Narrow source/subject/algorithm/feature_set to run fewer models."
            )
            specs = specs[:max_models]

        total_duration = float(initial.total_duration_sec)
        window_starts = self._build_scan_window_starts(
            total_duration_sec=total_duration,
            window_sec=float(scan_window_sec),
            hop_sec=float(scan_hop_sec),
            scan_start_sec=scan_start,
            scan_end_sec=scan_end_sec,
        )
        if not window_starts:
            raise LegacyPredictionError("No scan window could be created for the provided scan parameters.")

        truncated = False
        if len(window_starts) > scan_max_windows:
            window_starts = window_starts[:scan_max_windows]
            truncated = True
            warnings.append(
                f"Scan was truncated to {scan_max_windows} windows. Increase scan_max_windows to scan more."
            )

        feature_cache: dict[tuple[float, float, LegacyFeatureSet], np.ndarray] = {}
        timeline: list[LegacyScanTimelinePointData] = []
        peak_score = -1.0
        peak_window: LegacyScanPeakWindowData | None = None

        for window_index, window_start in enumerate(window_starts):
            if window_index == 0 and abs(window_start - initial.start_sec) <= 1e-3:
                prepared = initial
            else:
                prepared = prepare_segment(
                    resolved_edf_path,
                    requested_channel=channel,
                    start_sec=window_start,
                    duration_sec=scan_window_sec,
                )

            model_results = self._predict_specs_for_window(
                specs=specs,
                prepared=prepared,
                edf_path=resolved_edf_path,
                warnings=warnings,
                feature_cache=feature_cache,
                warning_prefix=f"[window {window_index}] ",
            )
            summary = self._build_summary(model_results)
            successful = [item for item in model_results if isinstance(item.seizure_probability, (float, int))]
            top_item = max(successful, key=lambda item: float(item.seizure_probability)) if successful else None
            point = LegacyScanTimelinePointData(
                window_index=window_index,
                start_sec=round(float(prepared.start_sec), 3),
                end_sec=round(float(prepared.start_sec + prepared.duration_used_sec), 3),
                average_probability=summary.average_probability,
                majority_vote=summary.majority_vote,
                positive_votes=summary.positive_votes,
                successful_models=len(successful),
                total_models=summary.total_models,
                top_probability=float(top_item.seizure_probability) if top_item else None,
                top_model_name=top_item.model_name if top_item else None,
            )
            timeline.append(point)

            point_score = float(point.average_probability) if point.average_probability is not None else -1.0
            if point_score > peak_score:
                peak_score = point_score
                peak_window = LegacyScanPeakWindowData(
                    window_index=point.window_index,
                    start_sec=point.start_sec,
                    end_sec=point.end_sec,
                    average_probability=point.average_probability,
                    majority_vote=point.majority_vote,
                    positive_votes=point.positive_votes,
                    successful_models=point.successful_models,
                    total_models=point.total_models,
                    top_probability=point.top_probability,
                    top_model_name=point.top_model_name,
                    models=model_results,
                    summary=summary,
                )

        if peak_window is None:
            raise LegacyPredictionError("Scan completed without any usable prediction window.")

        resolved_source = specs[0].source
        resolved_subject = specs[0].subject_id if model_id else subject_id
        resolved_algorithm = specs[0].algorithm if model_id else algorithm
        resolved_feature_set = specs[0].feature_set if model_id else feature_set

        warnings.append(
            "Scan mode reports one timeline value per window using average probability across successful models."
        )
        warnings.append(
            "Legacy probability is mapped to a positive class inferred from estimator metadata. "
            "Confirm class mapping after plugging in the original feature extractor."
        )

        return LegacyPredictionResponseData(
            selected_channel=initial.selected_channel,
            sampling_rate=initial.sampling_rate,
            duration_used_sec=initial.duration_used_sec,
            source=resolved_source,
            selected_model_id=model_id.strip().lower() if model_id else None,
            matched_model_count=len(specs),
            subject_id=resolved_subject,
            algorithm=resolved_algorithm,
            feature_set=resolved_feature_set,
            models=peak_window.models,
            summary=peak_window.summary,
            warnings=warnings,
            scan=LegacyScanResultData(
                enabled=True,
                total_duration_sec=round(total_duration, 3),
                window_sec=float(scan_window_sec),
                hop_sec=float(scan_hop_sec),
                window_count=len(timeline),
                truncated=truncated,
                timeline=timeline,
                peak_window=peak_window,
            ),
        )

    def _build_scan_window_starts(
        self,
        *,
        total_duration_sec: float,
        window_sec: float,
        hop_sec: float,
        scan_start_sec: float,
        scan_end_sec: float | None,
    ) -> list[float]:
        start = max(scan_start_sec, 0.0)
        effective_end = min(scan_end_sec, total_duration_sec) if scan_end_sec is not None else total_duration_sec
        if start >= effective_end:
            raise LegacyPredictionError("scan_start_sec must be smaller than scan_end_sec/recording duration.")

        if (effective_end - start) <= window_sec:
            return [round(start, 3)]

        starts = np.arange(start, max(effective_end - window_sec, start) + 1e-6, hop_sec, dtype=np.float32).tolist()
        last_start = max(effective_end - window_sec, start)
        if not starts or abs(float(starts[-1]) - last_start) > 1e-3:
            starts.append(last_start)
        return [round(float(item), 3) for item in starts]

    def _normalize_features(self, *, features: np.ndarray, feature_set: LegacyFeatureSet) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim > 2:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim == 2 and matrix.shape[0] != 1:
            matrix = matrix.reshape(1, -1)

        expected_features = FEATURE_DIMENSIONS[feature_set]
        if matrix.shape[1] != expected_features:
            raise LegacyPredictionError(
                f"Feature extractor produced {matrix.shape[1]} values for {feature_set}, "
                f"expected {expected_features}."
            )
        return matrix

    def _run_estimator(self, model: Any, matrix: np.ndarray) -> tuple[str, str, float | None, float | None]:
        predicted_label = "unknown"
        if hasattr(model, "predict"):
            prediction = model.predict(matrix)
            predicted_label = str(np.asarray(prediction).reshape(-1)[0])

        classes = np.asarray(getattr(model, "classes_", []))
        positive_index, positive_label = self._resolve_positive_class(classes)

        probability: float | None = None
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(matrix))
            if proba.ndim == 2 and proba.shape[0] > 0:
                row = proba[0]
                if 0 <= positive_index < row.shape[0]:
                    probability = float(row[positive_index])
                elif row.shape[0] > 0:
                    probability = float(row[-1])

        raw_score: float | None = None
        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(matrix)).reshape(-1)
            if decision.size > 0:
                raw_score = float(decision[0])

        return predicted_label, positive_label, probability, raw_score

    def _resolve_positive_class(self, classes: np.ndarray) -> tuple[int, str]:
        if classes.size == 0:
            return 1, "positive"

        candidates = [str(item) for item in classes.tolist()]
        lowered = [item.lower() for item in candidates]
        priority_tokens = ("seizure", "preictal", "ictal", "positive")
        for token in priority_tokens:
            for index, label in enumerate(lowered):
                if token in label:
                    return index, candidates[index]

        if np.issubdtype(classes.dtype, np.number):
            max_value = np.max(classes)
            for index, value in enumerate(classes.tolist()):
                if value == max_value:
                    return index, str(value)

        return len(candidates) - 1, candidates[-1]

    def _select_specs(
        self,
        *,
        model_id: str | None,
        source: LegacySource,
        subject_id: str | None,
        algorithm: str | None,
        feature_set: LegacyFeatureSet | None,
    ) -> list[LegacyModelSpec]:
        normalized_model_id = model_id.strip().lower() if model_id else None
        normalized_subject = subject_id.strip().lower() if subject_id else None
        normalized_algorithm = algorithm.strip().lower() if algorithm else None
        normalized_feature = feature_set.upper() if feature_set else None

        if normalized_model_id:
            match = next((item for item in self.catalog() if item.model_id == normalized_model_id), None)
            if match is None:
                raise LegacyPredictionError(
                    f"model_id '{normalized_model_id}' was not found. Use /legacy/health to list valid model_id values."
                )
            return [match]

        if source == "loso_final" and not normalized_subject:
            raise LegacyPredictionError("subject_id is required when source='loso_final' (example: chb01).")

        filtered: list[LegacyModelSpec] = []
        for spec in self.catalog():
            if spec.source != source:
                continue
            if normalized_subject and (spec.subject_id or "").lower() != normalized_subject:
                continue
            if normalized_algorithm and spec.algorithm.lower() != normalized_algorithm:
                continue
            if normalized_feature and spec.feature_set != normalized_feature:
                continue
            filtered.append(spec)

        return sorted(
            filtered,
            key=lambda item: (
                item.source,
                item.subject_id or "",
                item.algorithm.lower(),
                item.feature_set,
            ),
        )

    def _discover_catalog(self) -> list[LegacyModelSpec]:
        universal_scalers = self._collect_scalers(self.universal_dir, _UNIVERSAL_SCALER_RE, source="universal_lopo")
        loso_scalers = self._collect_scalers(self.loso_dir, _LOSO_SCALER_RE, source="loso_final")

        discovered: list[LegacyModelSpec] = []
        discovered.extend(
            self._collect_models(
                self.universal_dir,
                _UNIVERSAL_MODEL_RE,
                source="universal_lopo",
                scaler_map=universal_scalers,
            )
        )
        discovered.extend(
            self._collect_models(
                self.loso_dir,
                _LOSO_MODEL_RE,
                source="loso_final",
                scaler_map=loso_scalers,
            )
        )
        return sorted(
            discovered,
            key=lambda item: (
                item.source,
                item.subject_id or "",
                item.algorithm.lower(),
                item.feature_set,
            ),
        )

    def _collect_scalers(
        self,
        directory: Path,
        pattern: re.Pattern[str],
        *,
        source: LegacySource,
    ) -> dict[tuple[str | None, LegacyFeatureSet], Path]:
        scalers: dict[tuple[str | None, LegacyFeatureSet], Path] = {}
        if not directory.exists():
            return scalers
        for path in directory.rglob("*.joblib"):
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            feature_set = match.group("feature_set").upper()
            subject_id = match.groupdict().get("subject_id")
            subject_key = subject_id.lower() if subject_id else None
            scalers[(subject_key, feature_set)] = path
        logger.info(
            "Legacy scaler discovery completed",
            extra={"event": "legacy_scaler_discovered", "status": f"{source}:{len(scalers)}"},
        )
        return scalers

    def _collect_models(
        self,
        directory: Path,
        pattern: re.Pattern[str],
        *,
        source: LegacySource,
        scaler_map: dict[tuple[str | None, LegacyFeatureSet], Path],
    ) -> list[LegacyModelSpec]:
        models: list[LegacyModelSpec] = []
        if not directory.exists():
            return models

        for path in directory.rglob("*.joblib"):
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            groups = match.groupdict()
            subject_id = groups.get("subject_id")
            subject_key = subject_id.lower() if subject_id else None
            feature_set = groups["feature_set"].upper()
            algorithm = groups["algorithm"]
            models.append(
                LegacyModelSpec(
                    source=source,
                    subject_id=subject_id,
                    algorithm=algorithm,
                    feature_set=feature_set,  # type: ignore[arg-type]
                    model_path=path,
                    scaler_path=scaler_map.get((subject_key, feature_set)),
                )
            )
        logger.info(
            "Legacy model discovery completed",
            extra={"event": "legacy_model_discovered", "status": f"{source}:{len(models)}"},
        )
        return models

    def _load_artifact(self, path: Path | None) -> Any:
        if path is None:
            return None
        if path in self._artifact_cache:
            return self._artifact_cache[path]

        joblib = self._import_joblib()
        artifact = joblib.load(path)
        self._configure_estimator_runtime(artifact)
        self._artifact_cache[path] = artifact
        return artifact

    def _configure_estimator_runtime(self, estimator: Any) -> None:
        visited: set[int] = set()

        def walk(node: Any) -> None:
            node_id = id(node)
            if node is None or node_id in visited:
                return
            visited.add(node_id)

            try:
                if hasattr(node, "set_params") and callable(node.set_params):
                    params = {}
                    if hasattr(node, "get_params") and callable(node.get_params):
                        try:
                            known = node.get_params(deep=False)
                            if "n_jobs" in known:
                                params["n_jobs"] = 1
                            if "nthread" in known:
                                params["nthread"] = 1
                        except Exception:
                            pass
                    if params:
                        node.set_params(**params)
                else:
                    if hasattr(node, "n_jobs"):
                        setattr(node, "n_jobs", 1)
                    if hasattr(node, "nthread"):
                        setattr(node, "nthread", 1)
            except Exception:
                pass

            for collection_attr in ("estimators", "estimators_"):
                collection = getattr(node, collection_attr, None)
                if collection is None:
                    continue
                for child in collection:
                    # sklearn Voting/Stacking can store (name, estimator) tuples.
                    if isinstance(child, tuple) and len(child) >= 2:
                        walk(child[1])
                    else:
                        walk(child)

            for single_attr in ("final_estimator", "final_estimator_", "base_estimator", "base_estimator_"):
                walk(getattr(node, single_attr, None))

        walk(estimator)

    def _import_joblib(self):
        try:
            return importlib.import_module("joblib")
        except ModuleNotFoundError as exc:
            raise LegacyDependencyError(
                "Missing dependency 'joblib'. Install legacy dependencies: "
                "joblib, scikit-learn, xgboost, lightgbm."
            ) from exc

    def _resolve_feature_extractor(self) -> LegacyFeatureExtractor:
        env_reference = os.getenv("SEIZURE_LEGACY_FEATURE_EXTRACTOR", "").strip()
        if env_reference:
            resolved = self._resolve_feature_extractor_from_env(env_reference)
            if resolved is not None:
                return resolved

        try:
            module = importlib.import_module("app.services.legacy_feature_extractor")
            extractor_cls = getattr(module, "ChbMitLegacyFeatureExtractor")
            return extractor_cls()
        except Exception:
            logger.exception(
                "Failed to initialize default CHB-MIT legacy feature extractor",
                extra={"event": "legacy_feature_extractor_failed", "status": "default_auto"},
            )
            return MissingLegacyFeatureExtractor()

    def _resolve_feature_extractor_from_env(self, reference: str) -> LegacyFeatureExtractor | None:
        module_name: str
        attr_name: str
        if ":" in reference:
            module_name, attr_name = reference.split(":", maxsplit=1)
        else:
            try:
                module_name, attr_name = reference.rsplit(".", maxsplit=1)
            except ValueError:
                logger.warning(
                    "Invalid legacy feature extractor reference",
                    extra={"event": "legacy_feature_extractor_invalid", "status": reference},
                )
                return None

        try:
            module = importlib.import_module(module_name)
            target = getattr(module, attr_name)
            candidate = target() if isinstance(target, type) else target
            if callable(candidate) and not hasattr(candidate, "extract"):
                candidate = candidate()
            if hasattr(candidate, "extract"):
                return candidate
        except Exception:
            logger.exception(
                "Failed to import legacy feature extractor",
                extra={"event": "legacy_feature_extractor_failed", "status": reference},
            )
            return None

        logger.warning(
            "Legacy feature extractor does not expose extract()",
            extra={"event": "legacy_feature_extractor_invalid", "status": reference},
        )
        return None

    def _build_summary(self, model_results: list[LegacyPredictionModelResult]) -> LegacyPredictionSummary:
        successful = [
            item
            for item in model_results
            if item.predicted_label != "error" and isinstance(item.seizure_probability, (float, int))
        ]
        if not successful:
            return LegacyPredictionSummary(
                majority_vote="unavailable",
                average_probability=None,
                positive_votes=0,
                total_models=len(model_results),
                confidence_note="No legacy model produced a usable prediction.",
            )

        positive_votes = sum(
            1
            for item in successful
            if item.predicted_label.strip().lower() == item.positive_class_label.strip().lower()
        )
        average_probability = float(sum(float(item.seizure_probability) for item in successful) / len(successful))
        majority_vote = "positive-class" if positive_votes >= (len(successful) / 2.0) else "negative-class"
        vote_margin = abs((2 * positive_votes) - len(successful)) / max(len(successful), 1)
        if vote_margin >= 0.75:
            confidence_note = "High agreement among successful legacy models."
        elif vote_margin >= 0.4:
            confidence_note = "Moderate agreement among successful legacy models."
        else:
            confidence_note = "Low agreement among successful legacy models."
        return LegacyPredictionSummary(
            majority_vote=majority_vote,
            average_probability=round(average_probability, 6),
            positive_votes=positive_votes,
            total_models=len(model_results),
            confidence_note=confidence_note,
        )
