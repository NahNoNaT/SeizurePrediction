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
    ModelComparison,
    ModelSlotStatus,
    RecordingOverview,
    RecordingPreviewResponse,
    ReplaySessionStateResponse,
    ReplayTimelinePoint,
    ReplayUploadResponse,
    ReportSummary,
)
from app.services.clinical import ClinicalDecisionService
from app.services.clinical_workflow import WorkflowRunResult
from app.services.eeg_intake import ChannelTrace, EEGIntakeResult
from app.services.inference import InferenceOutput
from app.services.store import ClinicalCaseStore
from app.services.temporal import TemporalAnalysisService


class FakeInferenceService:
    def __init__(self, project_root: Path, status: str = "model_ready"):
        self.project_root = project_root
        self.status = status

    def warmup(self) -> None:
        runtime_config.inference_status = self.status

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        runtime_config.checkpoint_path = checkpoint_path
        runtime_config.checkpoint_paths = (checkpoint_path,) if checkpoint_path else ()
        runtime_config.inference_status = self.status

    def load_models(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is not None:
            runtime_config.checkpoint_path = checkpoint_path
            runtime_config.checkpoint_paths = (checkpoint_path,)
        runtime_config.inference_status = self.status

    def model_slot_statuses(self) -> list[ModelSlotStatus]:
        paths = list(runtime_config.configured_checkpoint_paths(self.project_root))
        return [
            ModelSlotStatus(
                model_key=f"model_{index}",
                model_label=f"Model {index}",
                checkpoint_path=path,
                status="READY",
                backend_status=self.status,
                model_version=runtime_config.default_model_version,
                detail=None,
            )
            for index, path in enumerate(paths or ["models/checkpoints/model-1.pt"], start=1)
        ]


class FakeWorkflowService:
    def __init__(
        self,
        project_root: Path,
        *,
        fail_analysis: bool = False,
        failure_code: str = "model_unavailable",
        recording_mode: str = "referential",
        segment_count: int = 3,
    ):
        self.project_root = project_root
        self.fail_analysis = fail_analysis
        self.failure_code = failure_code
        self.recording_mode = recording_mode
        self.segment_count = max(segment_count, 1)
        self.temporal_service = TemporalAnalysisService(runtime_config)
        self.clinical_service = ClinicalDecisionService(runtime_config)
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def inspect_recording(
        self,
        file_path: Path,
        *,
        clinician_mode: bool = True,
        enforce_validation: bool = True,
    ) -> EEGIntakeResult:
        if self.recording_mode == "blocked_bipolar":
            bipolar_channels = ["Fp1-F7", "F7-T3", "Fp2-F8", "F8-T4"]
            traces = {
                channel: ChannelTrace(
                    source_name=channel,
                    canonical_name=channel,
                    sampling_rate=float(runtime_config.target_sampling_rate_hz),
                    signal=np.ones(runtime_config.default_window_size, dtype=np.float32),
                )
                for channel in bipolar_channels
            }
            return EEGIntakeResult(
                file_type=".edf",
                duration_sec=60.0,
                channel_count=len(bipolar_channels),
                channel_names=bipolar_channels,
                input_montage_type="bipolar",
                conversion_status="blocked",
                conversion_messages=["Bipolar channel pairs were recognized, but posterior coverage was insufficient for montage conversion."],
                traces_by_channel={
                    "Fp1": traces["Fp1-F7"],
                    "F7": traces["F7-T3"],
                    "Fp2": traces["Fp2-F8"],
                    "F8": traces["F8-T4"],
                },
                mapped_channels=["Fp1", "F7", "Fp2", "F8"],
                derived_channels=["Fp1", "F7", "Fp2", "F8"],
                approximated_channels=[],
                missing_channels=[channel for channel in runtime_config.required_channel_order if channel not in {"Fp1", "F7", "Fp2", "F8"}],
                validation_status="BLOCKED",
                zero_fill_allowed=False,
                validation_messages=["Montage conversion was not sufficient to build a model-ready EEG input."],
            )
        if self.recording_mode == "converted_bipolar":
            bipolar_channels = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
            traces = {
                channel: ChannelTrace(
                    source_name=channel,
                    canonical_name=channel,
                    sampling_rate=float(runtime_config.target_sampling_rate_hz),
                    signal=np.ones(runtime_config.default_window_size, dtype=np.float32),
                )
                for channel in bipolar_channels
            }
            return EEGIntakeResult(
                file_type=".edf",
                duration_sec=60.0,
                channel_count=len(bipolar_channels),
                channel_names=bipolar_channels,
                input_montage_type="bipolar",
                conversion_status="converted",
                conversion_messages=["The EDF used bipolar channel labels and was converted heuristically into the fixed model input montage."],
                traces_by_channel={
                    channel: ChannelTrace(
                        source_name=f"derived:{channel}",
                        canonical_name=channel,
                        sampling_rate=float(runtime_config.target_sampling_rate_hz),
                        signal=np.ones(runtime_config.default_window_size, dtype=np.float32),
                    )
                    for channel in runtime_config.required_channel_order
                },
                mapped_channels=list(runtime_config.required_channel_order),
                derived_channels=["Fp1", "F7", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6"],
                approximated_channels=["F3", "C3", "P3", "F4", "C4", "P4", "Fz", "Cz", "Pz"],
                missing_channels=[],
                validation_status="VALIDATED",
                zero_fill_allowed=True,
                validation_messages=["Recording validated for analysis after bipolar montage conversion."],
            )
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
            input_montage_type="referential",
            conversion_status="direct",
            conversion_messages=["Referential scalp EEG channels were mapped directly to the model input montage."],
            traces_by_channel=traces,
            mapped_channels=list(runtime_config.required_channel_order),
            derived_channels=[],
            approximated_channels=[],
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
                model_comparisons=[],
                trace_json={"backend_status": self.failure_code, "failure_code": self.failure_code},
            )

        if self.segment_count == 3:
            segment_times = [(0.0, 5.0), (2.5, 7.5), (5.0, 10.0)]
            risk_scores = np.array([0.18, 0.82, 0.84], dtype=np.float32)
        else:
            hop_seconds = runtime_config.default_window_length_seconds - runtime_config.default_overlap_seconds
            segment_times = [
                (index * hop_seconds, index * hop_seconds + runtime_config.default_window_length_seconds)
                for index in range(self.segment_count)
            ]
            risk_scores = np.full(self.segment_count, 0.82, dtype=np.float32)
        temporal = self.temporal_service.analyze(risk_scores, segment_times)
        assessment = self.clinical_service.build_assessment(
            case_id=case_id,
            recording=recording,
            model_version=runtime_config.default_model_version,
            temporal_result=temporal,
        )
        model_comparisons = [
            ModelComparison(
                model_run_id=str(uuid4()),
                analysis_id="",
                model_key=f"model_{index}",
                model_label=f"Model {index}",
                model_version="raw-window-cnn-seqbigru-v1",
                checkpoint_path=f"models/checkpoints/model-{index}.pt",
                status="COMPLETED",
                backend_status="model_ready",
                overall_risk="Moderate" if index < 4 else "High",
                review_priority="Recommended" if index < 4 else "Urgent",
                estimated_seizure_risk=0.56 + (index * 0.05),
                max_risk_score=0.7 + (index * 0.04),
                mean_risk_score=0.48 + (index * 0.03),
                flagged_segments_count=index,
                high_risk_intervals_count=1,
                confidence_score=0.62 + (index * 0.07),
                confidence_label="High" if index >= 3 else "Moderate",
                agreement_score=0.74 + (index * 0.04),
                inference_time_seconds=0.01 * index,
            )
            for index in range(1, 5)
        ]
        return WorkflowRunResult(
            recording=recording,
            intake=intake,
            preprocessed=None,
            inference=InferenceOutput(
                model_version=runtime_config.default_model_version,
                risk_scores=risk_scores,
                inference_time_seconds=0.02,
                backend_status="model_ready",
                model_results=[],
                successful_model_count=4,
                configured_model_count=4,
            ),
            temporal=temporal,
            assessment=assessment,
            model_comparisons=model_comparisons,
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
        if case_detail.recording is not None and case_detail.recording.validation_status == "BLOCKED":
            return AnalysisStateResponse(
                status="failed",
                error=" ".join(case_detail.recording.validation_messages),
                clinical_summary=ClinicalSummaryState(
                    risk_score=0,
                    risk_level="failed",
                    priority="analysis_failed",
                    flagged_segments=0,
                    summary_text="Analysis was blocked because the uploaded EEG montage could not be converted into a model-ready input with sufficient coverage.",
                    recommendation="Review the montage/conversion notes and confirm the EDF channel layout before retrying analysis.",
                ),
                model_comparisons=case_detail.model_comparisons,
                timeline=[],
                segments=[],
            )
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
                model_comparisons=case_detail.model_comparisons,
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
            model_comparisons=case_detail.model_comparisons,
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


class FakeReplayService:
    def __init__(self):
        self._sessions: dict[str, ReplaySessionStateResponse] = {}

    def create_session(self, *, file_path: Path, file_name: str) -> ReplayUploadResponse:
        session_id = str(uuid4())
        state = ReplaySessionStateResponse(
            session_id=session_id,
            file_name=file_name,
            status="uploaded",
            total_duration_sec=60.0,
            sampling_rate=256.0,
            window_sec=10.0,
            hop_sec=2.5,
            replay_speed=5.0,
            available_channels=["Fp1-F7", "F7-T3", "T3-T5"],
            processed_windows=0,
            total_windows=0,
            replay_position_sec=0.0,
            latest_risk_score=None,
            latest_top_channel=None,
            error=None,
            timeline=[],
        )
        self._sessions[session_id] = state
        return ReplayUploadResponse(
            session_id=session_id,
            file_name=file_name,
            status="uploaded",
            total_duration_sec=60.0,
            sampling_rate=256.0,
            available_channels=state.available_channels,
            message="EDF replay session created. Start replay to stream sliding-window risk scores.",
        )

    def start_session(self, session_id: str, *, window_sec: float, hop_sec: float, replay_speed: float) -> ReplaySessionStateResponse:
        state = self._sessions[session_id]
        updated = state.model_copy(
            update={
                "status": "running",
                "window_sec": window_sec,
                "hop_sec": hop_sec,
                "replay_speed": replay_speed,
                "processed_windows": 1,
                "total_windows": 24,
                "replay_position_sec": window_sec,
                "latest_risk_score": 0.83,
                "latest_top_channel": "Fp1-F7",
                "timeline": [
                    ReplayTimelinePoint(
                        window_index=0,
                        start_sec=0.0,
                        end_sec=window_sec,
                        risk_score=0.83,
                        risk_label="High",
                        is_flagged=True,
                        top_channel="Fp1-F7",
                    )
                ],
            }
        )
        self._sessions[session_id] = updated
        return updated

    def stop_session(self, session_id: str) -> ReplaySessionStateResponse:
        state = self._sessions[session_id].model_copy(update={"status": "stopped"})
        self._sessions[session_id] = state
        return state

    def get_session_state(self, session_id: str) -> ReplaySessionStateResponse:
        return self._sessions[session_id]


def build_test_app(
    tmp_path: Path,
    *,
    fail_analysis: bool = False,
    failure_code: str = "model_unavailable",
    recording_mode: str = "referential",
    segment_count: int = 3,
):
    runtime_config.auth_enabled = False
    case_store = ClinicalCaseStore(tmp_path / "clinical_cases.db")
    inference_service = FakeInferenceService(
        tmp_path,
        status="model_unavailable" if fail_analysis else "model_ready",
    )
    workflow_service = FakeWorkflowService(
        tmp_path,
        fail_analysis=fail_analysis,
        failure_code=failure_code,
        recording_mode=recording_mode,
        segment_count=segment_count,
    )
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
def bipolar_client(tmp_path: Path):
    app = build_test_app(tmp_path, recording_mode="converted_bipolar")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def blocked_bipolar_client(tmp_path: Path):
    app = build_test_app(tmp_path, recording_mode="blocked_bipolar")
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
