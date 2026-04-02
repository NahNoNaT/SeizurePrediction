from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import RuntimeConfig, runtime_config
from app.schemas import (
    AnalysisOverview,
    AnalysisSegmentEntry,
    AnalysisStateResponse,
    AnalysisTimelineEntry,
    CaseDetail,
    ClinicalSummaryState,
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

    def inspect_recording(self, file_path: Path, *, clinician_mode: bool = True) -> EEGIntakeResult:
        return self.intake_service.inspect(file_path, clinician_mode=clinician_mode)

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
            mapped_channels=intake.mapped_channels,
            missing_channels=intake.missing_channels,
            mapped_channel_count=len(intake.mapped_channels),
            validation_status=intake.validation_status,
            validation_messages=intake.validation_messages,
            uploaded_at=uploaded_at,
        )

    def run_recording_analysis(self, *, case_id: str, recording: RecordingOverview) -> WorkflowRunResult:
        intake = self.inspect_recording(Path(recording.file_path), clinician_mode=False)
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
        return WorkflowRunResult(
            recording=recording,
            intake=pipeline_result.intake,
            preprocessed=pipeline_result.preprocessed,
            inference=pipeline_result.inference,
            temporal=pipeline_result.temporal,
            assessment=pipeline_result.assessment,
            trace_json=pipeline_result.trace_json,
        )

    def build_report(self, *, report_id: str, case_detail) -> ReportSummary:
        return self.reporting_service.generate(report_id=report_id, case_detail=case_detail)

    def build_case_analysis_state(self, case_detail: CaseDetail) -> AnalysisStateResponse:
        if case_detail.recording is None or case_detail.analysis is None:
            return self.build_pending_state()
        if case_detail.analysis.status == "FAILED":
            return self.build_failed_state(error=case_detail.analysis.failure_message)
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
            timeline=[],
            segments=[],
        )

    def build_failed_state(self, *, error: str | None) -> AnalysisStateResponse:
        return AnalysisStateResponse(
            status="failed",
            error=error or "Analysis could not be completed.",
            clinical_summary=ClinicalSummaryState(
                risk_score=0,
                risk_level="failed",
                priority="analysis_failed",
                flagged_segments=0,
                summary_text="Analysis could not be completed.",
                recommendation="Please review the technical issue before retrying.",
            ),
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
            timeline=timeline,
            segments=segments,
        )
