from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import RuntimeConfig, runtime_config
from app.schemas import RecordingOverview
from app.services.clinical import ClinicalAssessmentResult, ClinicalDecisionService
from app.services.eeg_intake import EEGIntakeResult
from app.services.inference import InferenceOutput, SeizureInferenceService
from app.services.preprocessing import PreprocessedRecording, SignalPreprocessingService
from app.services.temporal import TemporalAnalysisResult, TemporalAnalysisService


@dataclass
class PipelineRunResult:
    recording: RecordingOverview
    intake: EEGIntakeResult
    preprocessed: PreprocessedRecording
    inference: InferenceOutput
    temporal: TemporalAnalysisResult
    assessment: ClinicalAssessmentResult
    output: dict[str, Any]
    trace_json: dict[str, Any]


class SeizurePredictionPipeline:
    def __init__(
        self,
        project_root: Path,
        inference_service: SeizureInferenceService,
        preprocessing_service: SignalPreprocessingService | None = None,
        temporal_service: TemporalAnalysisService | None = None,
        clinical_service: ClinicalDecisionService | None = None,
        config: RuntimeConfig = runtime_config,
    ):
        self.project_root = Path(project_root)
        self.config = config
        self.inference_service = inference_service
        self.preprocessing_service = preprocessing_service or SignalPreprocessingService(config)
        self.temporal_service = temporal_service or TemporalAnalysisService(config)
        self.clinical_service = clinical_service or ClinicalDecisionService(config)

    def run(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        intake: EEGIntakeResult | None = None,
    ) -> dict[str, Any]:
        return self.run_detailed(case_id=case_id, recording=recording, intake=intake).output

    def run_detailed(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        intake: EEGIntakeResult | None = None,
    ) -> PipelineRunResult:
        resolved_intake = intake or self.preprocessing_service.load_recording(
            Path(recording.file_path),
            clinician_mode=False,
        )
        preprocessed = self.preprocessing_service.prepare(resolved_intake)
        inference = self.inference_service.run(preprocessed.segments)
        temporal = self.temporal_service.analyze(inference.risk_scores, preprocessed.segment_times)
        assessment = self.clinical_service.build_assessment(
            case_id=case_id,
            recording=recording,
            model_version=inference.model_version,
            temporal_result=temporal,
        )
        output = self._build_output(assessment=assessment, temporal=temporal)
        trace_json = self._build_trace_json(intake=resolved_intake, preprocessed=preprocessed, inference=inference)
        return PipelineRunResult(
            recording=recording,
            intake=resolved_intake,
            preprocessed=preprocessed,
            inference=inference,
            temporal=temporal,
            assessment=assessment,
            output=output,
            trace_json=trace_json,
        )

    def _build_output(
        self,
        *,
        assessment: ClinicalAssessmentResult,
        temporal: TemporalAnalysisResult,
    ) -> dict[str, Any]:
        return {
            "overall_risk": assessment.analysis.overall_risk,
            "review_priority": assessment.analysis.review_priority,
            "timeline_scores": [
                {
                    "segment_index": point.segment_index,
                    "start_sec": point.start_sec,
                    "end_sec": point.end_sec,
                    "risk_score": point.risk_score,
                    "risk_label": point.risk_label,
                    "is_flagged": point.is_flagged,
                }
                for point in temporal.timeline
            ],
            "intervals": [interval.model_dump() for interval in assessment.high_risk_intervals],
            "segment_results": [segment.model_dump() for segment in assessment.segment_results],
        }

    def _build_trace_json(
        self,
        *,
        intake: EEGIntakeResult,
        preprocessed: PreprocessedRecording,
        inference: InferenceOutput,
    ) -> dict[str, Any]:
        return {
            "backend_status": inference.backend_status,
            "inference_time_seconds": inference.inference_time_seconds,
            "configured_model_count": inference.configured_model_count,
            "successful_model_count": inference.successful_model_count,
            "model_results": [
                {
                    "model_key": item.model_key,
                    "model_label": item.model_label,
                    "model_version": item.model_version,
                    "checkpoint_path": item.checkpoint_path,
                    "status": item.status,
                    "backend_status": item.backend_status,
                    "inference_time_seconds": item.inference_time_seconds,
                    "failure_code": item.failure_code,
                }
                for item in inference.model_results
            ],
            "preprocessing_notes": preprocessed.notes,
            "input_montage_type": intake.input_montage_type,
            "conversion_status": intake.conversion_status,
            "conversion_messages": intake.conversion_messages,
            "derived_channels": intake.derived_channels,
            "approximated_channels": intake.approximated_channels,
            "mapped_channels_present": intake.mapped_channels,
            "missing_channels": intake.missing_channels,
            "required_channel_order": list(self.config.required_channel_order),
        }


SessionInferenceService = SeizureInferenceService
SessionAnalysisPipeline = SeizurePredictionPipeline

__all__ = [
    "InferenceOutput",
    "PipelineRunResult",
    "SeizureInferenceService",
    "SeizurePredictionPipeline",
    "SessionAnalysisPipeline",
    "SessionInferenceService",
]
