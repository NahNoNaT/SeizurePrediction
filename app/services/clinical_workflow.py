from __future__ import annotations

import logging
import os
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
from app.services.checkpoint_pkl import (
    CheckpointPklPredictionService,
    CheckpointPredictionError,
    CheckpointUnavailableError,
)
from app.services.legacy_joblib import LegacyJoblibPredictionService, LegacyPredictionResponseData
from app.services.pipeline import SeizurePredictionPipeline
from app.services.preprocessing import PreprocessedRecording, SignalPreprocessingService
from app.services.reporting import ClinicalReportService
from app.services.temporal import TemporalAnalysisResult, TemporalAnalysisService, TemporalTimelinePoint

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
        self.checkpoint_pkl_service = CheckpointPklPredictionService(project_root=project_root)
        self.legacy_service = LegacyJoblibPredictionService(project_root=project_root)
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
        if not self.config.resolved_checkpoint_paths(self.project_root):
            try:
                return self._run_checkpoint_pkl_fallback(case_id=case_id, recording=recording, intake=intake)
            except (CheckpointUnavailableError, CheckpointPredictionError):
                logger.info(
                    "Checkpoint PKL fallback unavailable, switching to benchmark fallback",
                    extra={
                        "event": "checkpoint_pkl_unavailable",
                        "case_id": case_id,
                        "recording_id": recording.recording_id,
                        "status": "FALLBACK",
                    },
                )
            try:
                return self._run_benchmark_fallback(case_id=case_id, recording=recording, intake=intake)
            except Exception as exc:
                logger.exception(
                    "Benchmark fallback failed before clinical pipeline",
                    extra={
                        "event": "benchmark_fallback_failed",
                        "case_id": case_id,
                        "recording_id": recording.recording_id,
                        "status": "FAILED",
                    },
                )
                failure_assessment = self.clinical_service.build_failed_assessment(
                    case_id=case_id,
                    recording=recording,
                    failure_code="analysis_not_executed",
                    failure_message=f"Benchmark fallback failed: {exc}",
                    model_version="benchmark-ensemble-v1",
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
                        "backend_status": "benchmark_fallback_failed",
                        "failure_code": "analysis_not_executed",
                        "failure_detail": str(exc),
                        "missing_channels": intake.missing_channels,
                        "required_channel_order": list(self.config.required_channel_order),
                    },
                )
        try:
            pipeline_result = self.pipeline.run_detailed(
                case_id=case_id,
                recording=recording,
                intake=intake,
            )
        except (ModelUnavailableError, CheckpointInvalidError, AnalysisExecutionError) as exc:
            if isinstance(exc, ModelUnavailableError):
                try:
                    return self._run_checkpoint_pkl_fallback(case_id=case_id, recording=recording, intake=intake)
                except (CheckpointUnavailableError, CheckpointPredictionError):
                    logger.info(
                        "Checkpoint PKL fallback unavailable, trying benchmark fallback",
                        extra={
                            "event": "checkpoint_pkl_unavailable",
                            "case_id": case_id,
                            "recording_id": recording.recording_id,
                            "status": "FALLBACK",
                        },
                    )
                try:
                    return self._run_benchmark_fallback(case_id=case_id, recording=recording, intake=intake)
                except Exception:
                    logger.exception(
                        "Benchmark fallback failed",
                        extra={
                            "event": "benchmark_fallback_failed",
                            "case_id": case_id,
                            "recording_id": recording.recording_id,
                            "code": exc.code,
                        },
                    )
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

    def _run_benchmark_fallback(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        intake: EEGIntakeResult,
    ) -> WorkflowRunResult:
        scan_window_sec = self._benchmark_scan_window_sec()
        scan_hop_sec = self._benchmark_scan_hop_sec(scan_window_sec)
        scan_max_windows = self._benchmark_scan_max_windows()
        benchmark_result = self.legacy_service.predict(
            recording.file_path,
            source="universal_lopo",
            feature_set="COMBINED",
            max_models=24,
            scan_full_file=True,
            scan_window_sec=scan_window_sec,
            scan_hop_sec=scan_hop_sec,
            scan_max_windows=scan_max_windows,
        )
        probabilities = [
            float(item.seizure_probability)
            for item in benchmark_result.models
            if isinstance(item.seizure_probability, (int, float))
        ]
        if not probabilities:
            model_errors = [
                f"{item.model_name}: {item.notes}"
                for item in benchmark_result.models
            ]
            warning_details = " | ".join(benchmark_result.warnings[:3])
            detail = "; ".join(model_errors) if model_errors else warning_details
            raise RuntimeError(
                "Benchmark fallback did not produce any usable model scores."
                + (f" Details: {detail}" if detail else "")
            )

        scan_scores, segment_times = self._build_benchmark_scan_segments(benchmark_result)
        temporal = self.temporal_service.analyze(np.asarray(scan_scores, dtype=np.float32), segment_times)
        temporal = self._retune_benchmark_temporal(temporal)
        timeline_mean_score = float(np.mean(scan_scores)) if scan_scores else 0.0
        comparison_anchor_score = float(np.mean(probabilities))
        assessment = self.clinical_service.build_assessment(
            case_id=case_id,
            recording=recording,
            model_version="benchmark-ensemble-v1",
            temporal_result=temporal,
        )
        peak_window = benchmark_result.scan.peak_window if benchmark_result.scan else None
        if peak_window is not None:
            peak_range = f"{peak_window.start_sec:.1f}s-{peak_window.end_sec:.1f}s"
            peak_probability = (
                f"{peak_window.average_probability:.0%}"
                if isinstance(peak_window.average_probability, (int, float))
                else "n/a"
            )
            peak_note = f" Peak window: {peak_range} (avg {peak_probability})."
        else:
            peak_note = ""
        assessment.analysis.clinical_summary = (
            "This result was generated by the benchmark fallback because the original clinical checkpoint is not configured. "
            f"Legacy LOSO/UNIVERSAL models were scanned across {len(segment_times)} window(s) "
            f"({scan_window_sec:.1f}s window, {scan_hop_sec:.1f}s hop) with mean timeline risk {timeline_mean_score:.0%}."
            f"{peak_note}"
        )
        assessment.analysis.recommendation = (
            f"{benchmark_result.summary.confidence_note} "
            "Treat this output as exploratory benchmarking rather than the original clinical prediction pipeline."
        )
        assessment.analysis.interpretation = (
            f"Benchmark fallback majority vote: {benchmark_result.summary.majority_vote}. "
            "This case used the EDF benchmarking path because no clinical checkpoint was configured."
        )
        assessment.analysis.model_version = "benchmark-ensemble-v1"

        model_comparisons = self._build_benchmark_model_comparisons(
            benchmark_models=benchmark_result.models,
            aggregate_score=comparison_anchor_score,
        )
        trace_json = {
            "backend_status": "benchmark_fallback",
            "model_version": "benchmark-ensemble-v1",
            "selected_channel": benchmark_result.selected_channel,
            "sampling_rate": benchmark_result.sampling_rate,
            "duration_used_sec": benchmark_result.duration_used_sec,
            "scan": {
                "enabled": benchmark_result.scan.enabled if benchmark_result.scan else False,
                "window_sec": benchmark_result.scan.window_sec if benchmark_result.scan else scan_window_sec,
                "hop_sec": benchmark_result.scan.hop_sec if benchmark_result.scan else scan_hop_sec,
                "window_count": benchmark_result.scan.window_count if benchmark_result.scan else len(segment_times),
                "truncated": benchmark_result.scan.truncated if benchmark_result.scan else False,
                "retuned_flag_threshold": self._benchmark_flag_threshold(),
                "retuned_min_consecutive_windows": self._benchmark_min_consecutive_windows(),
                "retuned_min_interval_sec": self._benchmark_min_interval_sec(),
                "peak_window": (
                    {
                        "window_index": benchmark_result.scan.peak_window.window_index,
                        "start_sec": benchmark_result.scan.peak_window.start_sec,
                        "end_sec": benchmark_result.scan.peak_window.end_sec,
                        "average_probability": benchmark_result.scan.peak_window.average_probability,
                        "majority_vote": benchmark_result.scan.peak_window.majority_vote,
                    }
                    if benchmark_result.scan and benchmark_result.scan.peak_window is not None
                    else None
                ),
            },
            "ensemble_summary": {
                "majority_vote": benchmark_result.summary.majority_vote,
                "average_probability": benchmark_result.summary.average_probability,
                "num_models_predicting_seizure": benchmark_result.summary.positive_votes,
                "num_models_total": benchmark_result.summary.total_models,
                "confidence_note": benchmark_result.summary.confidence_note,
            },
            "warnings": benchmark_result.warnings,
            "missing_channels": intake.missing_channels,
            "required_channel_order": list(self.config.required_channel_order),
        }
        logger.info(
            "Clinical workflow used benchmark fallback",
            extra={"event": "benchmark_fallback_used", "case_id": case_id, "recording_id": recording.recording_id, "status": "COMPLETED"},
        )
        return WorkflowRunResult(
            recording=recording,
            intake=intake,
            preprocessed=None,
            inference=None,
            temporal=temporal,
            assessment=assessment,
            model_comparisons=model_comparisons,
            trace_json=trace_json,
        )

    def _run_checkpoint_pkl_fallback(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        intake: EEGIntakeResult,
    ) -> WorkflowRunResult:
        scan_window_sec = self._benchmark_scan_window_sec()
        scan_hop_sec = self._benchmark_scan_hop_sec(scan_window_sec)
        scan_max_windows = self._benchmark_scan_max_windows()
        checkpoint_result = self.checkpoint_pkl_service.predict_scan(
            recording.file_path,
            window_sec=scan_window_sec,
            hop_sec=scan_hop_sec,
            max_windows=scan_max_windows,
        )
        if not checkpoint_result.scores or not checkpoint_result.segment_times:
            raise CheckpointPredictionError("Checkpoint scan produced no usable prediction windows.")

        temporal = self.temporal_service.analyze(
            np.asarray(checkpoint_result.scores, dtype=np.float32),
            checkpoint_result.segment_times,
        )
        temporal = self._retune_benchmark_temporal(temporal)
        assessment = self.clinical_service.build_assessment(
            case_id=case_id,
            recording=recording,
            model_version="checkpoint-global-smart-v1",
            temporal_result=temporal,
        )

        peak_index = int(np.argmax(np.asarray(checkpoint_result.scores, dtype=np.float32)))
        peak_start, peak_end = checkpoint_result.segment_times[peak_index]
        peak_score = float(checkpoint_result.scores[peak_index])
        mean_score = float(np.mean(np.asarray(checkpoint_result.scores, dtype=np.float32)))

        assessment.analysis.clinical_summary = (
            "This result was generated by checkpoint PKL fallback because torch clinical checkpoints are not configured. "
            f"Scanned {len(checkpoint_result.scores)} window(s) across {checkpoint_result.total_duration_sec:.1f}s "
            f"({checkpoint_result.window_sec:.1f}s window, {checkpoint_result.hop_sec:.1f}s hop). "
            f"Mean risk {mean_score:.0%}, peak risk {peak_score:.0%} at {peak_start:.1f}s-{peak_end:.1f}s."
        )
        assessment.analysis.recommendation = (
            "Checkpoint PKL model output is exploratory and should be interpreted alongside clinical context. "
            "Review highlighted intervals instead of relying on a single peak score."
        )
        assessment.analysis.interpretation = (
            f"Checkpoint model '{checkpoint_result.model_label}' produced a timeline with "
            f"{assessment.analysis.high_risk_intervals_count} grouped high-risk interval(s)."
        )
        assessment.analysis.model_version = "checkpoint-global-smart-v1"

        estimated = float(assessment.analysis.estimated_seizure_risk or 0.0)
        confidence_score = float(np.clip(abs(estimated - 0.5) * 2.0, 0.0, 1.0))
        model_comparisons = [
            ModelComparison(
                model_run_id=str(uuid4()),
                analysis_id="",
                model_key="checkpoint_global_smart",
                model_label=checkpoint_result.model_label,
                model_version="checkpoint-global-smart-v1",
                checkpoint_path=str(checkpoint_result.model_path),
                status="COMPLETED",
                backend_status="checkpoint_pkl_ready",
                overall_risk=assessment.analysis.overall_risk,
                review_priority=assessment.analysis.review_priority,
                estimated_seizure_risk=assessment.analysis.estimated_seizure_risk,
                max_risk_score=assessment.analysis.max_risk_score,
                mean_risk_score=assessment.analysis.mean_risk_score,
                flagged_segments_count=assessment.analysis.flagged_segments_count,
                high_risk_intervals_count=assessment.analysis.high_risk_intervals_count,
                confidence_score=confidence_score,
                confidence_label=self._confidence_label(confidence_score),
                agreement_score=1.0,
                inference_time_seconds=checkpoint_result.inference_time_seconds,
                is_primary=True,
            )
        ]

        trace_json = {
            "backend_status": "checkpoint_pkl_fallback",
            "model_version": "checkpoint-global-smart-v1",
            "checkpoint": {
                "model_path": str(checkpoint_result.model_path),
                "scaler_path": str(checkpoint_result.scaler_path),
                "top_idx_path": str(checkpoint_result.top_idx_path),
                "extractor_path": str(checkpoint_result.extractor_path),
            },
            "scan": {
                "enabled": True,
                "window_sec": checkpoint_result.window_sec,
                "hop_sec": checkpoint_result.hop_sec,
                "window_count": checkpoint_result.processed_windows,
                "truncated": checkpoint_result.truncated,
                "retuned_flag_threshold": self._benchmark_flag_threshold(),
                "retuned_min_consecutive_windows": self._benchmark_min_consecutive_windows(),
                "retuned_min_interval_sec": self._benchmark_min_interval_sec(),
                "peak_window": {
                    "window_index": peak_index,
                    "start_sec": peak_start,
                    "end_sec": peak_end,
                    "risk_score": peak_score,
                },
            },
            "warnings": list(checkpoint_result.warnings),
            "missing_channels": intake.missing_channels,
            "required_channel_order": list(self.config.required_channel_order),
        }
        logger.info(
            "Clinical workflow used checkpoint PKL fallback",
            extra={"event": "checkpoint_pkl_fallback_used", "case_id": case_id, "recording_id": recording.recording_id, "status": "COMPLETED"},
        )
        return WorkflowRunResult(
            recording=recording,
            intake=intake,
            preprocessed=None,
            inference=None,
            temporal=temporal,
            assessment=assessment,
            model_comparisons=model_comparisons,
            trace_json=trace_json,
        )

    def _benchmark_scan_max_windows(self) -> int:
        value = os.getenv("SEIZURE_BENCHMARK_SCAN_MAX_WINDOWS", "").strip()
        if not value:
            return 2000
        try:
            parsed = int(value)
        except ValueError:
            return 2000
        return max(parsed, 32)

    def _benchmark_scan_window_sec(self) -> float:
        value = os.getenv("SEIZURE_BENCHMARK_SCAN_WINDOW_SEC", "").strip()
        if not value:
            return 10.0
        try:
            parsed = float(value)
        except ValueError:
            return 10.0
        return float(np.clip(parsed, 2.5, 60.0))

    def _benchmark_scan_hop_sec(self, window_sec: float) -> float:
        value = os.getenv("SEIZURE_BENCHMARK_SCAN_HOP_SEC", "").strip()
        if not value:
            return window_sec
        try:
            parsed = float(value)
        except ValueError:
            return window_sec
        return float(np.clip(parsed, 1.0, window_sec))

    def _benchmark_flag_threshold(self) -> float:
        value = os.getenv("SEIZURE_BENCHMARK_FLAG_THRESHOLD", "").strip()
        if not value:
            return 0.60
        try:
            parsed = float(value)
        except ValueError:
            return 0.60
        return float(np.clip(parsed, 0.45, 0.95))

    def _benchmark_min_consecutive_windows(self) -> int:
        value = os.getenv("SEIZURE_BENCHMARK_MIN_CONSECUTIVE_WINDOWS", "").strip()
        if not value:
            return 2
        try:
            parsed = int(value)
        except ValueError:
            return 2
        return max(parsed, 1)

    def _benchmark_min_interval_sec(self) -> float:
        value = os.getenv("SEIZURE_BENCHMARK_MIN_INTERVAL_SEC", "").strip()
        if not value:
            return 20.0
        try:
            parsed = float(value)
        except ValueError:
            return 20.0
        return float(np.clip(parsed, 0.0, 300.0))

    def _build_benchmark_scan_segments(
        self,
        benchmark_result: LegacyPredictionResponseData,
    ) -> tuple[list[float], list[tuple[float, float]]]:
        scan = benchmark_result.scan
        if scan is None or not scan.timeline:
            average_probability = benchmark_result.summary.average_probability
            fallback_score = float(average_probability) if isinstance(average_probability, (int, float)) else 0.0
            return [fallback_score], [(0.0, benchmark_result.duration_used_sec)]

        segment_scores: list[float] = []
        segment_times: list[tuple[float, float]] = []
        for item in scan.timeline:
            if not isinstance(item.average_probability, (int, float)):
                continue
            segment_scores.append(float(item.average_probability))
            segment_times.append((float(item.start_sec), float(item.end_sec)))

        if not segment_scores:
            average_probability = benchmark_result.summary.average_probability
            fallback_score = float(average_probability) if isinstance(average_probability, (int, float)) else 0.0
            return [fallback_score], [(0.0, benchmark_result.duration_used_sec)]
        return segment_scores, segment_times

    def _retune_benchmark_temporal(
        self,
        temporal: TemporalAnalysisResult,
    ) -> TemporalAnalysisResult:
        if not temporal.timeline:
            return temporal

        threshold = self._benchmark_flag_threshold()
        min_consecutive = self._benchmark_min_consecutive_windows()
        min_interval_sec = self._benchmark_min_interval_sec()

        raw_flags = [point.risk_score >= threshold for point in temporal.timeline]
        tuned_flags = self._apply_min_consecutive(raw_flags, min_consecutive)

        retuned_timeline: list[TemporalTimelinePoint] = []
        for point, tuned_flag in zip(temporal.timeline, tuned_flags, strict=False):
            retuned_timeline.append(
                TemporalTimelinePoint(
                    segment_index=point.segment_index,
                    start_sec=point.start_sec,
                    end_sec=point.end_sec,
                    risk_score=point.risk_score,
                    risk_label=point.risk_label,
                    is_flagged=bool(tuned_flag),
                )
            )

        retuned_intervals = self.temporal_service.merge_intervals(retuned_timeline)
        if min_interval_sec > 0.0 and retuned_intervals:
            kept_intervals = [item for item in retuned_intervals if (item.end_sec - item.start_sec) >= min_interval_sec]
            if len(kept_intervals) != len(retuned_intervals):
                retuned_timeline = self._restrict_timeline_to_intervals(retuned_timeline, kept_intervals)
                retuned_intervals = self.temporal_service.merge_intervals(retuned_timeline)

        return TemporalAnalysisResult(
            timeline=retuned_timeline,
            intervals=retuned_intervals,
            smoothed_scores=temporal.smoothed_scores,
        )

    def _apply_min_consecutive(self, flags: list[bool], minimum_run: int) -> list[bool]:
        if minimum_run <= 1:
            return [bool(item) for item in flags]

        derived = [False] * len(flags)
        run_start: int | None = None
        for index, is_high in enumerate(flags + [False]):
            if is_high and run_start is None:
                run_start = index
                continue
            if is_high:
                continue
            if run_start is not None and index - run_start >= minimum_run:
                for flagged_index in range(run_start, index):
                    derived[flagged_index] = True
            run_start = None
        return derived

    def _restrict_timeline_to_intervals(
        self,
        timeline: list[TemporalTimelinePoint],
        intervals,
    ) -> list[TemporalTimelinePoint]:
        if not intervals:
            return [
                TemporalTimelinePoint(
                    segment_index=point.segment_index,
                    start_sec=point.start_sec,
                    end_sec=point.end_sec,
                    risk_score=point.risk_score,
                    risk_label=point.risk_label,
                    is_flagged=False,
                )
                for point in timeline
            ]

        ranges = [(float(item.start_sec), float(item.end_sec)) for item in intervals]
        restricted: list[TemporalTimelinePoint] = []
        for point in timeline:
            keep = any((point.start_sec >= start - 1e-6 and point.end_sec <= end + 1e-6) for start, end in ranges)
            restricted.append(
                TemporalTimelinePoint(
                    segment_index=point.segment_index,
                    start_sec=point.start_sec,
                    end_sec=point.end_sec,
                    risk_score=point.risk_score,
                    risk_label=point.risk_label,
                    is_flagged=keep and point.is_flagged,
                )
            )
        return restricted

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

    def _build_benchmark_model_comparisons(
        self,
        *,
        benchmark_models: list,
        aggregate_score: float,
    ) -> list[ModelComparison]:
        comparisons: list[ModelComparison] = []
        for index, model in enumerate(benchmark_models, start=1):
            probability = getattr(model, "seizure_probability", None)
            if not isinstance(probability, (int, float)):
                comparisons.append(
                    ModelComparison(
                        model_run_id=str(uuid4()),
                        analysis_id="",
                        model_key=f"benchmark_model_{index}",
                        model_label=str(getattr(model, "model_name", f"Benchmark Model {index}")),
                        model_version="benchmark-fallback",
                        checkpoint_path=None,
                        status="FAILED",
                        backend_status="benchmark_failed",
                        failure_message=str(getattr(model, "notes", "Benchmark model failed.")),
                    )
                )
                continue

            probability_value = float(probability)
            overall_risk = self.clinical_service.map_risk_score(probability_value)
            review_priority = self.clinical_service.assign_review_priority(overall_risk)
            confidence_score = float(np.clip(abs(probability_value - 0.5) * 2.0, 0.0, 1.0))
            agreement_score = float(np.clip(1.0 - abs(probability_value - aggregate_score), 0.0, 1.0))
            comparisons.append(
                ModelComparison(
                    model_run_id=str(uuid4()),
                    analysis_id="",
                    model_key=f"benchmark_model_{index}",
                    model_label=str(getattr(model, "model_name", f"Benchmark Model {index}")),
                    model_version="benchmark-fallback",
                    checkpoint_path=None,
                    status="COMPLETED",
                    backend_status="benchmark_ready",
                    overall_risk=overall_risk,
                    review_priority=review_priority,
                    estimated_seizure_risk=probability_value,
                    max_risk_score=probability_value,
                    mean_risk_score=probability_value,
                    flagged_segments_count=1 if probability_value >= self.config.default_threshold else 0,
                    high_risk_intervals_count=1 if probability_value >= 0.75 else 0,
                    confidence_score=confidence_score,
                    confidence_label=self._confidence_label(confidence_score),
                    agreement_score=agreement_score,
                    inference_time_seconds=(
                        round(float(getattr(model, "inference_time_ms")) / 1000.0, 3)
                        if isinstance(getattr(model, "inference_time_ms", None), (int, float))
                        else None
                    ),
                    failure_message=None,
                    is_primary=index == 1,
                )
            )
        return comparisons
