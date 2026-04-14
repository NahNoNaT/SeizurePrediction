from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from preprocessing import DEFAULT_DURATION_SEC, DEFAULT_START_SEC, EEGPreprocessingError

from app.schemas import (
    BenchmarkEnsembleSummary,
    BenchmarkModelResult,
    DemoHealthResponse,
    DemoPredictionResponse,
    LegacyCatalogEntry,
    LegacyHealthResponse,
    LegacyPredictionModelResult,
    LegacyPredictionResponse,
    LegacyScanPeakWindow,
    LegacyScanResult,
    LegacyScanTimelinePoint,
    LegacyPredictionSummary,
)
from app.services.legacy_joblib import LegacyDependencyError, LegacyFeatureExtractorUnavailableError, LegacyPredictionError
from app.web import app_http_exception, page_context, save_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    legacy_service = request.app.state.legacy_joblib_service
    catalog = legacy_service.catalog_summary()["models"]
    preferred = [item for item in catalog if item.source == "universal_lopo" and item.feature_set == "COMBINED"]
    display_models = preferred if preferred else catalog[:24]
    return templates.TemplateResponse(
        request=request,
        name="demo.html",
        context=page_context(
            request,
            page_title="EDF Benchmark Demo",
            page_subtitle="Upload one EDF segment and run legacy LOSO/UNIVERSAL models from local joblib artifacts.",
            active_page="demo",
            extra={
                "default_start_sec": DEFAULT_START_SEC,
                "default_duration_sec": DEFAULT_DURATION_SEC,
                "demo_models": [
                    {
                        "model_name": item.display_name,
                        "notes": f"model_id={item.model_id}",
                    }
                    for item in display_models
                ],
            },
        ),
    )


@router.get("/health", response_model=DemoHealthResponse)
async def health(request: Request) -> DemoHealthResponse:
    service = request.app.state.legacy_joblib_service
    summary = service.catalog_summary()
    return DemoHealthResponse(
        status="ok",
        task_type="legacy_joblib_prediction",
        model_count=summary["model_count"],
        models=[
            {
                "model_name": item.display_name,
                "notes": f"model_id={item.model_id}",
            }
            for item in summary["models"][:50]
        ],
    )


@router.post("/predict", response_model=DemoPredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    channel: str | None = Form(default=None),
    start_sec: float = Form(default=DEFAULT_START_SEC),
    duration_sec: float = Form(default=DEFAULT_DURATION_SEC),
    source: Literal["universal_lopo", "loso_final"] = Form(default="universal_lopo"),
    subject_id: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    feature_set: Literal["LBP", "GLCM", "COMBINED"] | None = Form(default="COMBINED"),
    max_models: int = Form(default=24),
) -> DemoPredictionResponse:
    upload_path: Path | None = None
    try:
        _, upload_path = await save_upload(
            file,
            uploads_dir=request.app.state.uploads_dir,
            config=request.app.state.runtime_config,
            clinician_mode=True,
        )
        legacy_service = request.app.state.legacy_joblib_service
        result = legacy_service.predict(
            upload_path,
            model_id=model_id.strip() if model_id and model_id.strip() else None,
            source=source,
            subject_id=subject_id.strip() if subject_id and subject_id.strip() else None,
            algorithm=algorithm.strip() if algorithm and algorithm.strip() else None,
            feature_set=feature_set,
            channel=channel.strip() if channel and channel.strip() else None,
            start_sec=start_sec,
            duration_sec=duration_sec,
            max_models=max(max_models, 1),
        )
        return DemoPredictionResponse(
            selected_channel=result.selected_channel,
            sampling_rate=result.sampling_rate,
            duration_used_sec=result.duration_used_sec,
            models=[
                BenchmarkModelResult(
                    model_name=item.model_name,
                    predicted_label=item.predicted_label,
                    seizure_probability=item.seizure_probability,
                    raw_score=item.raw_score,
                    inference_time_ms=item.inference_time_ms,
                    notes=f"{item.notes} source={item.source}",
                )
                for item in result.models
            ],
            ensemble_summary=BenchmarkEnsembleSummary(
                majority_vote=result.summary.majority_vote,
                average_probability=result.summary.average_probability,
                num_models_predicting_seizure=result.summary.positive_votes,
                num_models_total=result.summary.total_models,
                confidence_note=result.summary.confidence_note,
            ),
            warnings=result.warnings,
        )
    except EEGPreprocessingError as exc:
        raise app_http_exception(400, "invalid_edf", str(exc)) from exc
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("Benchmark model file was not found")
        raise app_http_exception(500, "model_unavailable", str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected benchmark prediction failure")
        raise app_http_exception(500, "prediction_failed", f"Prediction failed: {exc}") from exc
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)


@router.get("/legacy/health", response_model=LegacyHealthResponse)
async def legacy_health(request: Request) -> LegacyHealthResponse:
    service = request.app.state.legacy_joblib_service
    summary = service.catalog_summary()
    return LegacyHealthResponse(
        status="ok",
        task_type="legacy_joblib_prediction",
        model_count=summary["model_count"],
        subjects=summary["subjects"],
        algorithms=summary["algorithms"],
        feature_sets=summary["feature_sets"],
        counts_by_source=summary["counts_by_source"],
        counts_by_subject=summary["counts_by_subject"],
        models=[
            LegacyCatalogEntry(
                model_id=item.model_id,
                display_name=item.display_name,
                source=item.source,
                subject_id=item.subject_id,
                algorithm=item.algorithm,
                feature_set=item.feature_set,
                model_path=str(item.model_path),
                scaler_path=str(item.scaler_path) if item.scaler_path else None,
            )
            for item in summary["models"]
        ],
    )


@router.post("/legacy/predict", response_model=LegacyPredictionResponse)
async def legacy_predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    source: Literal["universal_lopo", "loso_final"] = Form(default="universal_lopo"),
    subject_id: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    feature_set: Literal["LBP", "GLCM", "COMBINED"] | None = Form(default="COMBINED"),
    channel: str | None = Form(default=None),
    start_sec: float = Form(default=DEFAULT_START_SEC),
    duration_sec: float = Form(default=DEFAULT_DURATION_SEC),
    max_models: int = Form(default=24),
    scan_full_file: bool = Form(default=False),
    scan_window_sec: float = Form(default=5.0),
    scan_hop_sec: float = Form(default=5.0),
    scan_start_sec: float = Form(default=0.0),
    scan_end_sec: float | None = Form(default=None),
    scan_max_windows: int = Form(default=240),
) -> LegacyPredictionResponse:
    upload_path: Path | None = None
    try:
        _, upload_path = await save_upload(
            file,
            uploads_dir=request.app.state.uploads_dir,
            config=request.app.state.runtime_config,
            clinician_mode=True,
        )
        service = request.app.state.legacy_joblib_service
        result = service.predict(
            upload_path,
            model_id=model_id.strip() if model_id and model_id.strip() else None,
            source=source,
            subject_id=subject_id.strip() if subject_id and subject_id.strip() else None,
            algorithm=algorithm.strip() if algorithm and algorithm.strip() else None,
            feature_set=feature_set,
            channel=channel.strip() if channel and channel.strip() else None,
            start_sec=start_sec,
            duration_sec=duration_sec,
            max_models=max(max_models, 1),
            scan_full_file=scan_full_file,
            scan_window_sec=scan_window_sec,
            scan_hop_sec=scan_hop_sec,
            scan_start_sec=scan_start_sec,
            scan_end_sec=scan_end_sec,
            scan_max_windows=max(scan_max_windows, 1),
        )
        return LegacyPredictionResponse(
            selected_channel=result.selected_channel,
            sampling_rate=result.sampling_rate,
            duration_used_sec=result.duration_used_sec,
            source=result.source,
            selected_model_id=result.selected_model_id,
            matched_model_count=result.matched_model_count,
            subject_id=result.subject_id,
            algorithm=result.algorithm,
            feature_set=result.feature_set,
            models=[
                LegacyPredictionModelResult(
                    model_name=item.model_name,
                    source=item.source,
                    subject_id=item.subject_id,
                    algorithm=item.algorithm,
                    feature_set=item.feature_set,
                    predicted_label=item.predicted_label,
                    positive_class_label=item.positive_class_label,
                    seizure_probability=item.seizure_probability,
                    raw_score=item.raw_score,
                    inference_time_ms=item.inference_time_ms,
                    notes=item.notes,
                )
                for item in result.models
            ],
            summary=LegacyPredictionSummary(
                majority_vote=result.summary.majority_vote,
                average_probability=result.summary.average_probability,
                positive_votes=result.summary.positive_votes,
                total_models=result.summary.total_models,
                confidence_note=result.summary.confidence_note,
            ),
            warnings=result.warnings,
            scan=(
                LegacyScanResult(
                    enabled=result.scan.enabled,
                    total_duration_sec=result.scan.total_duration_sec,
                    window_sec=result.scan.window_sec,
                    hop_sec=result.scan.hop_sec,
                    window_count=result.scan.window_count,
                    truncated=result.scan.truncated,
                    timeline=[
                        LegacyScanTimelinePoint(
                            window_index=item.window_index,
                            start_sec=item.start_sec,
                            end_sec=item.end_sec,
                            average_probability=item.average_probability,
                            majority_vote=item.majority_vote,
                            positive_votes=item.positive_votes,
                            successful_models=item.successful_models,
                            total_models=item.total_models,
                            top_probability=item.top_probability,
                            top_model_name=item.top_model_name,
                        )
                        for item in result.scan.timeline
                    ],
                    peak_window=(
                        LegacyScanPeakWindow(
                            window_index=result.scan.peak_window.window_index,
                            start_sec=result.scan.peak_window.start_sec,
                            end_sec=result.scan.peak_window.end_sec,
                            average_probability=result.scan.peak_window.average_probability,
                            majority_vote=result.scan.peak_window.majority_vote,
                            positive_votes=result.scan.peak_window.positive_votes,
                            successful_models=result.scan.peak_window.successful_models,
                            total_models=result.scan.peak_window.total_models,
                            top_probability=result.scan.peak_window.top_probability,
                            top_model_name=result.scan.peak_window.top_model_name,
                            models=[
                                LegacyPredictionModelResult(
                                    model_name=item.model_name,
                                    source=item.source,
                                    subject_id=item.subject_id,
                                    algorithm=item.algorithm,
                                    feature_set=item.feature_set,
                                    predicted_label=item.predicted_label,
                                    positive_class_label=item.positive_class_label,
                                    seizure_probability=item.seizure_probability,
                                    raw_score=item.raw_score,
                                    inference_time_ms=item.inference_time_ms,
                                    notes=item.notes,
                                )
                                for item in result.scan.peak_window.models
                            ],
                            summary=LegacyPredictionSummary(
                                majority_vote=result.scan.peak_window.summary.majority_vote,
                                average_probability=result.scan.peak_window.summary.average_probability,
                                positive_votes=result.scan.peak_window.summary.positive_votes,
                                total_models=result.scan.peak_window.summary.total_models,
                                confidence_note=result.scan.peak_window.summary.confidence_note,
                            ),
                        )
                        if result.scan.peak_window is not None
                        else None
                    ),
                )
                if result.scan is not None
                else None
            ),
        )
    except EEGPreprocessingError as exc:
        raise app_http_exception(400, "invalid_edf", str(exc)) from exc
    except LegacyFeatureExtractorUnavailableError as exc:
        raise app_http_exception(503, "feature_extractor_unavailable", str(exc)) from exc
    except LegacyDependencyError as exc:
        raise app_http_exception(500, "legacy_backend_unavailable", str(exc)) from exc
    except LegacyPredictionError as exc:
        raise app_http_exception(400, "legacy_prediction_invalid", str(exc)) from exc
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("Legacy model artifact was not found")
        raise app_http_exception(500, "legacy_model_unavailable", str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected legacy prediction failure")
        raise app_http_exception(500, "legacy_prediction_failed", f"Legacy prediction failed: {exc}") from exc
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
