from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import runtime_config
from app.main import create_app
from app.schemas import (
    AnalysisSegmentEntry,
    AnalysisStateResponse,
    AnalysisTimelineEntry,
    CaseDetail,
    ClinicalSummaryState,
    RecordingOverview,
    RecordingPreviewResponse,
    ReportSummary,
)
from app.services.clinical import ClinicalDecisionService
from app.services.clinical_workflow import WorkflowRunResult
from app.services.eeg_intake import ChannelTrace, EEGIntakeResult
from app.services.inference import InferenceOutput
from app.services.store import ClinicalCaseStore
from app.services.temporal import TemporalAnalysisService


class FakeInferenceService:
    def __init__(self, status: str = "model_ready"):
        self.status = status

    def warmup(self) -> None:
        runtime_config.inference_status = self.status

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        runtime_config.checkpoint_path = checkpoint_path
        runtime_config.inference_status = self.status


class FakeWorkflowService:
    def __init__(self, project_root: Path, *, fail_analysis: bool = False, failure_code: str = "model_unavailable"):
        self.project_root = project_root
        self.fail_analysis = fail_analysis
        self.failure_code = failure_code
        self.temporal_service = TemporalAnalysisService(runtime_config)
        self.clinical_service = ClinicalDecisionService(runtime_config)
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def inspect_recording(self, file_path: Path, *, clinician_mode: bool = True) -> EEGIntakeResult:
        traces = {
            channel: ChannelTrace(
                source_name=channel,
                canonical_name=channel,
                sampling_rate=float(runtime_config.target_sampling_rate_hz),
                signal=np.ones(runtime_config.default_window_size, dtype=np.float32),
            )
            for channel in runtime_config.required_channel_order
        }
        return EEGIntakeResult(
            file_type=".edf",
            duration_sec=60.0,
            channel_count=len(runtime_config.required_channel_order),
            channel_names=list(runtime_config.required_channel_order),
            traces_by_channel=traces,
            mapped_channels=list(runtime_config.required_channel_order),
            missing_channels=[],
            validation_status="VALIDATED",
            zero_fill_allowed=True,
            validation_messages=["Recording validated for analysis."],
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
            duration_sec=intake.duration_sec,
            sampling_rate=float(runtime_config.target_sampling_rate_hz),
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
        if self.fail_analysis:
            assessment = self.clinical_service.build_failed_assessment(
                case_id=case_id,
                recording=recording,
                failure_code=self.failure_code,  # type: ignore[arg-type]
                failure_message="Analysis could not be completed because the model is unavailable.",
                model_version=runtime_config.default_model_version,
            )
            return WorkflowRunResult(
                recording=recording,
                intake=intake,
                preprocessed=None,
                inference=None,
                temporal=None,
                assessment=assessment,
                trace_json={"backend_status": self.failure_code, "failure_code": self.failure_code},
            )

        segment_times = [(0.0, 5.0), (2.5, 7.5), (5.0, 10.0)]
        risk_scores = np.array([0.18, 0.82, 0.84], dtype=np.float32)
        temporal = self.temporal_service.analyze(risk_scores, segment_times)
        assessment = self.clinical_service.build_assessment(
            case_id=case_id,
            recording=recording,
            model_version=runtime_config.default_model_version,
            temporal_result=temporal,
        )
        return WorkflowRunResult(
            recording=recording,
            intake=intake,
            preprocessed=None,
            inference=InferenceOutput(
                model_version=runtime_config.default_model_version,
                risk_scores=risk_scores,
                inference_time_seconds=0.02,
                backend_status="model_ready",
            ),
            temporal=temporal,
            assessment=assessment,
            trace_json={"backend_status": "model_ready"},
        )

    def build_report(self, *, report_id: str, case_detail) -> ReportSummary:
        report_path = self.reports_dir / f"{report_id}.html"
        report_path.write_text("<html><body>report</body></html>", encoding="utf-8")
        return ReportSummary(
            report_id=report_id,
            analysis_id=case_detail.analysis.analysis_id,
            case_id=case_detail.case_id,
            patient_id=case_detail.patient_id,
            clinician_name=case_detail.clinician_name,
            report_title=f"Clinical EEG Analysis Report - {case_detail.patient_id}",
            report_path=str(report_path),
            report_url=f"/reports/{report_id}",
            report_status="Generated",
            generated_at=datetime.now(timezone.utc),
        )

    def build_case_analysis_state(self, case_detail: CaseDetail) -> AnalysisStateResponse:
        if case_detail.recording is None or case_detail.analysis is None:
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
        if case_detail.analysis.status == "FAILED":
            return AnalysisStateResponse(
                status="failed",
                error=case_detail.analysis.failure_message,
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
        return AnalysisStateResponse(
            status="completed",
            error=None,
            clinical_summary=ClinicalSummaryState(
                risk_score=int(round((case_detail.analysis.estimated_seizure_risk or 0.0) * 100)),
                risk_level=(case_detail.analysis.overall_risk or "Low").lower(),  # type: ignore[arg-type]
                priority=(case_detail.analysis.review_priority or "Routine").lower(),  # type: ignore[arg-type]
                flagged_segments=case_detail.analysis.flagged_segments_count,
                summary_text=case_detail.analysis.clinical_summary or "Clinical findings are available for review.",
                recommendation=case_detail.analysis.recommendation or "Review the elevated-risk windows alongside the clinical context.",
            ),
            timeline=timeline,
            segments=segments,
        )

    def preview_recording(
        self,
        *,
        recording: RecordingOverview,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        channels: list[str] | None = None,
    ) -> RecordingPreviewResponse:
        sampling_rate = float(runtime_config.target_sampling_rate_hz)
        total_duration_sec = recording.duration_sec
        resolved_duration = duration_sec if duration_sec and duration_sec > 0 else 30.0
        resolved_duration = min(resolved_duration, total_duration_sec)
        resolved_start = min(max(start_sec, 0.0), max(total_duration_sec - resolved_duration, 0.0))
        sample_count = max(int(round(resolved_duration * sampling_rate)), 1)
        times = [resolved_start + (index / sampling_rate) for index in range(sample_count)]
        available_channels = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4"]
        if channels:
            selected_channels = [channel for channel in channels if channel in available_channels][:8]
            missing_channels = [channel for channel in channels if channel not in available_channels]
        else:
            selected_channels = available_channels[:4]
            missing_channels = []
        if not selected_channels:
            selected_channels = available_channels[:4]
        signals = [
            [round(20.0 * np.sin((index / sampling_rate) * 6.0 + channel_index), 4) for index in range(sample_count)]
            for channel_index, _ in enumerate(selected_channels)
        ]
        return RecordingPreviewResponse(
            recording_id=recording.recording_id,
            sampling_rate=sampling_rate,
            start_sec=resolved_start,
            duration_sec=resolved_duration,
            total_duration_sec=total_duration_sec,
            channels=selected_channels,
            available_channels=available_channels,
            missing_channels=missing_channels,
            times=times,
            signals=signals,
        )


def build_test_app(tmp_path: Path, *, fail_analysis: bool = False, failure_code: str = "model_unavailable"):
    case_store = ClinicalCaseStore(tmp_path / "clinical_cases.db")
    inference_service = FakeInferenceService(status="model_unavailable" if fail_analysis else "model_ready")
    workflow_service = FakeWorkflowService(tmp_path, fail_analysis=fail_analysis, failure_code=failure_code)
    app = create_app(
        case_store=case_store,
        inference_service=inference_service,  # type: ignore[arg-type]
        workflow_service=workflow_service,  # type: ignore[arg-type]
    )
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.state.uploads_dir = uploads_dir
    return app


@pytest.fixture
def client(tmp_path: Path):
    app = build_test_app(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failing_client(tmp_path: Path):
    app = build_test_app(tmp_path, fail_analysis=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def case_payload() -> dict[str, object]:
    return {
        "patient_id": f"PT-{uuid4().hex[:6]}",
        "clinician_name": "Dr Example",
        "recording_date": date(2026, 4, 2).isoformat(),
        "notes": "Initial review request",
    }
