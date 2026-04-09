from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.config import RuntimeConfig, runtime_config
from app.schemas import (
    AnalysisOverview,
    AnalysisSegmentEntry,
    AnalysisStateResponse,
    AnalysisTimelineEntry,
    CaseDetail,
    ClinicalSummaryState,
    ModelComparison,
    RecordingPreviewResponse,
    RecordingOverview,
    ReportSummary,
)
from app.services.clinical import ClinicalAssessmentResult, ClinicalDecisionService
from app.services.eeg_intake import EEGIntakeResult, EEGIntakeService
from app.services.errors import AnalysisExecutionError, CheckpointInvalidError, ModelUnavailableError
from app.services.inference import InferenceOutput, SeizureInferenceService
from app.services.pipeline import SeizurePredictionPipeline
from app.services.preprocessing import PreprocessedRecording, SignalPreprocessingService
from app.services.reporting import ClinicalReportService
from app.services.temporal import TemporalAnalysisResult, TemporalAnalysisService

logger = logging.getLogger(__name__)


@dataclass
class WorkflowRunResult:
    recording: RecordingOverview
    intake: EEGIntakeResult
    preprocessed: PreprocessedRecording | None
    inference: InferenceOutput | None
    temporal: TemporalAnalysisResult | None
    assessment: ClinicalAssessmentResult
    model_comparisons: list[ModelComparison]
    trace_json: dict


class ClinicalAnalysisService:
    def __init__(
        self,
        project_root: Path,
        inference_service: SeizureInferenceService,
        intake_service: EEGIntakeService | None = None,
        preprocessing_service: SignalPreprocessingService | None = None,
        temporal_service: TemporalAnalysisService | None = None,
        clinical_service: ClinicalDecisionService | None = None,
        reporting_service: ClinicalReportService | None = None,
        config: RuntimeConfig = runtime_config,
    ):
        self.project_root = project_root
        self.config = config
        self.inference_service = inference_service
        self.intake_service = intake_service or EEGIntakeService(config)
        self.preprocessing_service = preprocessing_service or SignalPreprocessingService(config)
        self.temporal_service = temporal_service or TemporalAnalysisService(config)
        self.clinical_service = clinical_service or ClinicalDecisionService(config)
        self.reporting_service = reporting_service or ClinicalReportService(project_root, config)
        self.pipeline = SeizurePredictionPipeline(
            project_root=project_root,
            inference_service=inference_service,
            preprocessing_service=self.preprocessing_service,
            temporal_service=self.temporal_service,
            clinical_service=self.clinical_service,
            config=config,
        )

    def inspect_recording(
        self,
        file_path: Path,
        *,
        clinician_mode: bool = True,
        enforce_validation: bool = True,
    ) -> EEGIntakeResult:
        return self.intake_service.inspect(
            file_path,
            clinician_mode=clinician_mode,
            enforce_validation=enforce_validation,
        )

    def preview_recording(
        self,
        *,
        recording: RecordingOverview,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        channels: list[str] | None = None,
    ) -> RecordingPreviewResponse:
        preview = self.intake_service.preview(
            Path(recording.file_path),
            start_sec=start_sec,
            duration_sec=duration_sec,
            channels=channels,
        )
        return RecordingPreviewResponse(
            recording_id=recording.recording_id,
            sampling_rate=preview.sampling_rate,
            start_sec=preview.start_sec,
            duration_sec=preview.duration_sec,
            total_duration_sec=preview.total_duration_sec,
            channels=preview.channels,
            available_channels=preview.available_channels,
            missing_channels=preview.missing_channels,
            times=preview.times,
            signals=preview.signals,
        )

    def build_recording_overview(
        self,
        *,
        recording_id: str,
        case_id: str,
        file_name: str,
        file_path: str,
        intake: EEGIntakeResult,
        uploaded_at: datetime,
    ) -> RecordingOverview:
        return RecordingOverview(
            recording_id=recording_id,
            case_id=case_id,
            file_name=file_name,
            file_path=file_path,
            file_type=intake.file_type,
            duration_sec=float(intake.duration_sec),
            sampling_rate=float(self.config.target_sampling_rate_hz),
            channel_count=intake.channel_count,
            channel_names=intake.channel_names,
            input_montage_type=intake.input_montage_type,
            conversion_status=intake.conversion_status,
            conversion_messages=intake.conversion_messages,
            mapped_channels=intake.mapped_channels,
            derived_channels=intake.derived_channels,
            approximated_channels=intake.approximated_channels,
            missing_channels=intake.missing_channels,
            mapped_channel_count=len(intake.mapped_channels),
            validation_status=intake.validation_status,
            validation_messages=intake.validation_messages,
            uploaded_at=uploaded_at,
        )

    def run_recording_analysis(self, *, case_id: str, recording: RecordingOverview) -> WorkflowRunResult:
        intake = self.inspect_recording(Path(recording.file_path), clinician_mode=False, enforce_validation=True)
        try:
            pipeline_result = self.pipeline.run_detailed(
                case_id=case_id,
                recording=recording,
                intake=intake,
            )
        except (ModelUnavailableError, CheckpointInvalidError, AnalysisExecutionError) as exc:
            logger.exception(
                "Analysis execution failed",
                extra={"event": "analysis_failed", "case_id": case_id, "recording_id": recording.recording_id, "code": exc.code},
            )
            failure_assessment = self.clinical_service.build_failed_assessment(
                case_id=case_id,
                recording=recording,
                failure_code=exc.code,  # type: ignore[arg-type]
                failure_message=exc.public_detail,
                model_version=self.config.default_model_version,
            )
            return WorkflowRunResult(
                recording=recording,
                intake=intake,
                preprocessed=None,
                inference=None,
                temporal=None,
                assessment=failure_assessment,
                model_comparisons=[],
                trace_json={
                    "backend_status": self.config.inference_status,
                    "failure_code": exc.code,
                    "failure_detail": exc.detail,
                    "missing_channels": intake.missing_channels,
                    "required_channel_order": list(self.config.required_channel_order),
                },
            )

        logger.info(
            "Analysis completed",
            extra={"event": "analysis_completed", "case_id": case_id, "recording_id": recording.recording_id, "status": "COMPLETED"},
        )
        model_comparisons = self._build_model_comparisons(
            case_id=case_id,
            recording=recording,
            inference=pipeline_result.inference,
            aggregate_scores=pipeline_result.inference.risk_scores,
            segment_times=pipeline_result.preprocessed.segment_times,
        )
        return WorkflowRunResult(
            recording=recording,
            intake=pipeline_result.intake,
            preprocessed=pipeline_result.preprocessed,
            inference=pipeline_result.inference,
            temporal=pipeline_result.temporal,
            assessment=pipeline_result.assessment,
            model_comparisons=model_comparisons,
            trace_json=pipeline_result.trace_json,
        )

    def build_report(self, *, report_id: str, case_detail) -> ReportSummary:
        return self.reporting_service.generate(report_id=report_id, case_detail=case_detail)

    def build_case_analysis_state(self, case_detail: CaseDetail) -> AnalysisStateResponse:
        if case_detail.recording is not None and case_detail.recording.validation_status == "BLOCKED":
            return self.build_failed_state(
                error=" ".join(case_detail.recording.validation_messages) or "Recording validation blocked analysis.",
                model_comparisons=case_detail.model_comparisons,
            )
        if case_detail.recording is None or case_detail.analysis is None:
            return self.build_pending_state()
        if case_detail.analysis.status == "FAILED":
            return self.build_failed_state(
                error=case_detail.analysis.failure_message,
                model_comparisons=case_detail.model_comparisons,
            )
        return self.build_completed_state(case_detail)

    def build_pending_state(self) -> AnalysisStateResponse:
        return AnalysisStateResponse(
            status="pending",
            error=None,
            clinical_summary=ClinicalSummaryState(
                risk_score=0,
                risk_level="pending",
                priority="awaiting_review",
                flagged_segments=0,
                summary_text="Clinical findings will appear here once the analysis is complete.",
                recommendation="Awaiting analysis.",
            ),
            model_comparisons=[],
            timeline=[],
            segments=[],
        )

    def build_failed_state(
        self,
        *,
        error: str | None,
        model_comparisons: list[ModelComparison] | None = None,
    ) -> AnalysisStateResponse:
        resolved_error = error or "Analysis could not be completed."
        if any(token in resolved_error.lower() for token in ("montage", "channel", "bipolar", "conversion")):
            summary_text = "Analysis was blocked because the uploaded EEG montage could not be converted into a model-ready input with sufficient coverage."
            recommendation = "Review the montage/conversion notes and confirm the EDF channel layout before retrying analysis."
        else:
            summary_text = "Analysis could not be completed."
            recommendation = "Please review the technical issue before retrying."
        return AnalysisStateResponse(
            status="failed",
            error=resolved_error,
            clinical_summary=ClinicalSummaryState(
                risk_score=0,
                risk_level="failed",
                priority="analysis_failed",
                flagged_segments=0,
                summary_text=summary_text,
                recommendation=recommendation,
            ),
            model_comparisons=model_comparisons or [],
            timeline=[],
            segments=[],
        )

    def build_completed_state(self, case_detail: CaseDetail) -> AnalysisStateResponse:
        assert case_detail.analysis is not None

        timeline = [
            AnalysisTimelineEntry(
                segment_index=segment.segment_index,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                risk_score=segment.risk_score,
                risk_level=segment.risk_label.lower(),  # type: ignore[arg-type]
                is_flagged=segment.is_flagged,
            )
            for segment in case_detail.segment_results
        ]
        segments = [
            AnalysisSegmentEntry(
                segment_index=segment.segment_index,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                risk_score=segment.risk_score,
                risk_level=segment.risk_label.lower(),  # type: ignore[arg-type]
                is_flagged=segment.is_flagged,
            )
            for segment in case_detail.segment_results
            if segment.is_flagged
        ]

        summary = ClinicalSummaryState(
            risk_score=int(round((case_detail.analysis.estimated_seizure_risk or 0.0) * 100)),
            risk_level=(case_detail.analysis.overall_risk or "Low").lower(),  # type: ignore[arg-type]
            priority=(case_detail.analysis.review_priority or "Routine").lower(),  # type: ignore[arg-type]
            flagged_segments=case_detail.analysis.flagged_segments_count,
            summary_text=case_detail.analysis.clinical_summary or "Clinical findings are available for review.",
            recommendation=(
                case_detail.analysis.recommendation
                or case_detail.analysis.interpretation
                or "Review the elevated-risk windows alongside the clinical context."
            ),
        )
        return AnalysisStateResponse(
            status="completed",
            error=None,
            clinical_summary=summary,
            model_comparisons=case_detail.model_comparisons,
            timeline=timeline,
            segments=segments,
        )

    def _build_model_comparisons(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        inference: InferenceOutput,
        aggregate_scores: np.ndarray,
        segment_times: list[tuple[float, float]],
    ) -> list[ModelComparison]:
        comparisons: list[ModelComparison] = []
        for model_result in inference.model_results:
            if model_result.status != "COMPLETED" or model_result.risk_scores is None:
                comparisons.append(
                    ModelComparison(
                        model_run_id=str(uuid4()),
                        analysis_id="",
                        model_key=model_result.model_key,
                        model_label=model_result.model_label,
                        model_version=model_result.model_version,
                        checkpoint_path=model_result.checkpoint_path,
                        status="FAILED",
                        backend_status=model_result.backend_status,
                        failure_code=model_result.failure_code,  # type: ignore[arg-type]
                        failure_message=model_result.failure_message,
                    )
                )
                continue

            temporal = self.temporal_service.analyze(model_result.risk_scores, segment_times)
            assessment = self.clinical_service.build_assessment(
                case_id=case_id,
                recording=recording,
                model_version=model_result.model_version or self.config.default_model_version,
                temporal_result=temporal,
            )
            confidence_score = self._confidence_score(model_result.risk_scores, aggregate_scores, temporal)
            comparisons.append(
                ModelComparison(
                    model_run_id=str(uuid4()),
                    analysis_id=assessment.analysis.analysis_id,
                    model_key=model_result.model_key,
                    model_label=model_result.model_label,
                    model_version=model_result.model_version,
                    checkpoint_path=model_result.checkpoint_path,
                    status="COMPLETED",
                    backend_status=model_result.backend_status,
                    overall_risk=assessment.analysis.overall_risk,
                    review_priority=assessment.analysis.review_priority,
                    estimated_seizure_risk=assessment.analysis.estimated_seizure_risk,
                    max_risk_score=assessment.analysis.max_risk_score,
                    mean_risk_score=assessment.analysis.mean_risk_score,
                    flagged_segments_count=assessment.analysis.flagged_segments_count,
                    high_risk_intervals_count=assessment.analysis.high_risk_intervals_count,
                    confidence_score=confidence_score,
                    confidence_label=self._confidence_label(confidence_score),
                    agreement_score=self._agreement_score(model_result.risk_scores, aggregate_scores),
                    inference_time_seconds=model_result.inference_time_seconds,
                    is_primary=False,
                )
            )
        return comparisons

    def _confidence_score(
        self,
        model_scores: np.ndarray,
        aggregate_scores: np.ndarray,
        temporal: TemporalAnalysisResult,
    ) -> float:
        threshold = self.config.default_threshold
        margin_score = float(np.clip(np.mean(np.abs(model_scores - threshold)) / max(1.0 - threshold, threshold), 0.0, 1.0))
        agreement_score = self._agreement_score(model_scores, aggregate_scores)
        flagged_ratio = sum(1 for point in temporal.timeline if point.is_flagged) / max(len(temporal.timeline), 1)
        stability_score = max(flagged_ratio, 1.0 - flagged_ratio)
        return float(np.clip(0.45 * margin_score + 0.35 * agreement_score + 0.20 * stability_score, 0.05, 0.99))

    def _agreement_score(self, model_scores: np.ndarray, aggregate_scores: np.ndarray) -> float:
        if model_scores.shape != aggregate_scores.shape or model_scores.size == 0:
            return 0.0
        return float(np.clip(1.0 - np.mean(np.abs(model_scores - aggregate_scores)), 0.0, 1.0))

    def _confidence_label(self, confidence_score: float) -> str:
        if confidence_score >= 0.75:
            return "High"
        if confidence_score >= 0.5:
            return "Moderate"
        return "Low"
