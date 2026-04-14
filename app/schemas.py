from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


ClinicalRiskCategory = Literal["Low", "Moderate", "High"]
ReviewPriority = Literal["Routine", "Recommended", "Urgent"]
LifecycleStatus = Literal["NEW", "UPLOADED", "VALIDATED", "ANALYZING", "COMPLETED", "FAILED", "REPORT_READY"]
RecordingValidationStatus = Literal["PENDING", "VALIDATED", "BLOCKED"]
InputMontageType = Literal["referential", "bipolar", "unsupported"]
ConversionStatus = Literal["direct", "converted", "blocked"]
AnalysisFailureCode = Literal["model_unavailable", "checkpoint_invalid", "analysis_not_executed", "validation_failed"]
AnalysisStateStatus = Literal["pending", "completed", "failed"]
AnalysisStateRiskLevel = Literal["pending", "failed", "low", "moderate", "high"]
AnalysisStatePriority = Literal["awaiting_review", "analysis_failed", "routine", "recommended", "urgent"]
ConfidenceLabel = Literal["Low", "Moderate", "High"]
ModelRunStatus = Literal["COMPLETED", "FAILED"]


class ErrorResponse(BaseModel):
    code: str
    detail: str


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    stored_filename: str
    extension: str
    size_bytes: int
    upload_path: str
    message: str
    created_at: datetime


class CaseCreateRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=120)
    clinician_name: str = Field(min_length=1, max_length=160)
    recording_date: date
    notes: str | None = Field(default=None, max_length=4000)


class DashboardStats(BaseModel):
    total_analyses: int = 0
    high_risk_cases: int = 0
    pending_review_cases: int = 0
    recent_reports: int = 0


class TimelinePoint(BaseModel):
    segment_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: ClinicalRiskCategory
    is_flagged: bool


class Recording(BaseModel):
    recording_id: str
    case_id: str
    file_name: str
    file_path: str
    file_type: str
    duration_sec: float = Field(ge=0.0)
    sampling_rate: float = Field(ge=0.0)
    channel_count: int = Field(ge=0)
    channel_names: list[str]
    input_montage_type: InputMontageType = "unsupported"
    conversion_status: ConversionStatus = "blocked"
    conversion_messages: list[str] = Field(default_factory=list)
    mapped_channels: list[str]
    derived_channels: list[str] = Field(default_factory=list)
    approximated_channels: list[str] = Field(default_factory=list)
    missing_channels: list[str] = Field(default_factory=list)
    mapped_channel_count: int = Field(ge=0)
    validation_status: RecordingValidationStatus = "PENDING"
    validation_messages: list[str] = Field(default_factory=list)
    uploaded_at: datetime


class RecordingPreviewResponse(BaseModel):
    recording_id: str
    sampling_rate: float = Field(ge=0.0)
    start_sec: float = Field(ge=0.0)
    duration_sec: float = Field(ge=0.0)
    total_duration_sec: float = Field(ge=0.0)
    channels: list[str]
    available_channels: list[str] = Field(default_factory=list)
    missing_channels: list[str] = Field(default_factory=list)
    times: list[float] = Field(default_factory=list)
    signals: list[list[float]] = Field(default_factory=list)


class ReplayUploadResponse(BaseModel):
    session_id: str
    file_name: str
    status: str
    total_duration_sec: float = Field(ge=0.0)
    sampling_rate: float = Field(ge=0.0)
    available_channels: list[str] = Field(default_factory=list)
    message: str


class ReplayStartRequest(BaseModel):
    window_sec: float = Field(default=10.0, gt=0.0, le=60.0)
    hop_sec: float = Field(default=2.5, gt=0.0, le=30.0)
    replay_speed: float = Field(default=5.0, gt=0.0, le=60.0)


class ReplayTimelinePoint(BaseModel):
    window_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: ClinicalRiskCategory
    is_flagged: bool
    top_channel: str


class ReplaySessionStateResponse(BaseModel):
    session_id: str
    file_name: str
    status: str
    total_duration_sec: float = Field(ge=0.0)
    sampling_rate: float = Field(ge=0.0)
    window_sec: float = Field(ge=0.0)
    hop_sec: float = Field(ge=0.0)
    replay_speed: float = Field(ge=0.0)
    available_channels: list[str] = Field(default_factory=list)
    processed_windows: int = Field(ge=0)
    total_windows: int = Field(ge=0)
    replay_position_sec: float = Field(ge=0.0)
    latest_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    latest_top_channel: str | None = None
    error: str | None = None
    timeline: list[ReplayTimelinePoint] = Field(default_factory=list)


class SegmentResult(BaseModel):
    id: str
    analysis_id: str
    segment_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: ClinicalRiskCategory
    is_flagged: bool


class HighRiskInterval(BaseModel):
    id: str
    analysis_id: str
    interval_index: int = Field(ge=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    mean_risk: float = Field(ge=0.0, le=1.0)
    max_risk: float = Field(ge=0.0, le=1.0)
    flagged_segments_count: int = Field(ge=0)
    review_status: str


class Analysis(BaseModel):
    analysis_id: str
    case_id: str
    recording_id: str
    status: LifecycleStatus
    model_version: str | None = None
    overall_risk: ClinicalRiskCategory | None = None
    review_priority: ReviewPriority | None = None
    max_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_seizure_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    flagged_segments_count: int = Field(default=0, ge=0)
    total_segments: int = Field(default=0, ge=0)
    high_risk_intervals_count: int = Field(default=0, ge=0)
    recording_duration_sec: float = Field(default=0.0, ge=0.0)
    clinical_summary: str | None = None
    recommendation: str | None = None
    interpretation: str | None = None
    failure_code: AnalysisFailureCode | None = None
    failure_message: str | None = None
    created_at: datetime
    report_generated: bool = False


class ModelComparison(BaseModel):
    model_run_id: str
    analysis_id: str
    model_key: str
    model_label: str
    model_version: str | None = None
    checkpoint_path: str | None = None
    status: ModelRunStatus
    backend_status: str
    overall_risk: ClinicalRiskCategory | None = None
    review_priority: ReviewPriority | None = None
    estimated_seizure_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    max_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    flagged_segments_count: int = Field(default=0, ge=0)
    high_risk_intervals_count: int = Field(default=0, ge=0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel | None = None
    agreement_score: float | None = Field(default=None, ge=0.0, le=1.0)
    inference_time_seconds: float | None = Field(default=None, ge=0.0)
    failure_code: AnalysisFailureCode | None = None
    failure_message: str | None = None
    is_primary: bool = False


class ModelSlotStatus(BaseModel):
    model_key: str
    model_label: str
    checkpoint_path: str
    status: str
    backend_status: str
    model_version: str | None = None
    detail: str | None = None


class ReportSummary(BaseModel):
    report_id: str
    analysis_id: str
    case_id: str
    patient_id: str
    clinician_name: str
    report_title: str
    report_path: str
    report_url: str
    report_status: str
    generated_at: datetime


class CaseSummary(BaseModel):
    case_id: str
    patient_id: str
    clinician_name: str
    recording_date: date
    status: LifecycleStatus
    created_at: datetime
    updated_at: datetime
    overall_risk: ClinicalRiskCategory | None = None
    review_priority: ReviewPriority | None = None
    recording_id: str | None = None
    latest_analysis_id: str | None = None
    recording_file_name: str | None = None
    report_available: bool = False


class Case(BaseModel):
    case_id: str
    patient_id: str
    clinician_name: str
    recording_date: date
    notes: str | None = None
    status: LifecycleStatus
    created_at: datetime
    updated_at: datetime
    recording: Recording | None = None
    recordings: list[Recording] = Field(default_factory=list)
    analysis: Analysis | None = None
    analyses: list[Analysis] = Field(default_factory=list)
    model_comparisons: list[ModelComparison] = Field(default_factory=list)
    high_risk_intervals: list[HighRiskInterval] = Field(default_factory=list)
    segment_results: list[SegmentResult] = Field(default_factory=list)
    report: ReportSummary | None = None


RecordingOverview = Recording
AnalysisOverview = Analysis
SegmentResultRecord = SegmentResult
CaseDetail = Case
CaseStatus = LifecycleStatus


class AdminSettingsSummary(BaseModel):
    checkpoint_path: str | None = None
    checkpoint_paths: list[str] = Field(default_factory=list)
    checkpoint_directory: str
    checkpoint_extensions: list[str] = Field(default_factory=list)
    auto_discovery_enabled: bool = True
    using_auto_discovery: bool = False
    model_version: str
    backend_status: str
    model_device: str
    configured_model_count: int = 0
    model_slots: list[ModelSlotStatus] = Field(default_factory=list)
    target_sample_rate_hz: int
    supported_input_modes: list[str] = Field(default_factory=list)
    required_channel_order: list[str]
    minimum_mapped_channels: int
    max_zero_fill_channels: int
    zero_fill_allowed: bool


class AppMetadataResponse(BaseModel):
    app_title: str
    app_subtitle: str
    supported_formats: list[str]
    max_upload_size_mb: int
    backend_status: str
    model_version: str
    configured_model_count: int = 0
    research_disclaimer: str


class BenchmarkModelResult(BaseModel):
    model_name: str
    predicted_label: str
    seizure_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = None
    inference_time_ms: float | None = Field(default=None, ge=0.0)
    notes: str


class BenchmarkEnsembleSummary(BaseModel):
    majority_vote: str
    average_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    num_models_predicting_seizure: int = Field(ge=0)
    num_models_total: int = Field(ge=0)
    confidence_note: str


class DemoPredictionResponse(BaseModel):
    selected_channel: str
    sampling_rate: float = Field(ge=0.0)
    duration_used_sec: float = Field(ge=0.0)
    models: list[BenchmarkModelResult] = Field(default_factory=list)
    ensemble_summary: BenchmarkEnsembleSummary
    warnings: list[str] = Field(default_factory=list)


class DemoHealthResponse(BaseModel):
    status: str
    task_type: str
    model_count: int = Field(ge=0)
    models: list[dict[str, str]] = Field(default_factory=list)


class LegacyCatalogEntry(BaseModel):
    model_id: str
    display_name: str
    source: Literal["universal_lopo", "loso_final"]
    subject_id: str | None = None
    algorithm: str
    feature_set: Literal["LBP", "GLCM", "COMBINED"]
    model_path: str
    scaler_path: str | None = None


class LegacyPredictionSummary(BaseModel):
    majority_vote: str
    average_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_votes: int = Field(ge=0)
    total_models: int = Field(ge=0)
    confidence_note: str


class LegacyPredictionModelResult(BaseModel):
    model_name: str
    source: Literal["universal_lopo", "loso_final"]
    subject_id: str | None = None
    algorithm: str
    feature_set: Literal["LBP", "GLCM", "COMBINED"]
    predicted_label: str
    positive_class_label: str
    seizure_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = None
    inference_time_ms: float | None = Field(default=None, ge=0.0)
    notes: str


class LegacyScanTimelinePoint(BaseModel):
    window_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    average_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    majority_vote: str
    positive_votes: int = Field(ge=0)
    successful_models: int = Field(ge=0)
    total_models: int = Field(ge=0)
    top_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    top_model_name: str | None = None


class LegacyScanPeakWindow(BaseModel):
    window_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    average_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    majority_vote: str
    positive_votes: int = Field(ge=0)
    successful_models: int = Field(ge=0)
    total_models: int = Field(ge=0)
    top_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    top_model_name: str | None = None
    models: list[LegacyPredictionModelResult] = Field(default_factory=list)
    summary: LegacyPredictionSummary


class LegacyScanResult(BaseModel):
    enabled: bool = False
    total_duration_sec: float = Field(default=0.0, ge=0.0)
    window_sec: float = Field(default=0.0, ge=0.0)
    hop_sec: float = Field(default=0.0, ge=0.0)
    window_count: int = Field(default=0, ge=0)
    truncated: bool = False
    timeline: list[LegacyScanTimelinePoint] = Field(default_factory=list)
    peak_window: LegacyScanPeakWindow | None = None


class LegacyPredictionResponse(BaseModel):
    selected_channel: str
    sampling_rate: float = Field(ge=0.0)
    duration_used_sec: float = Field(ge=0.0)
    source: Literal["universal_lopo", "loso_final"]
    selected_model_id: str | None = None
    matched_model_count: int = Field(ge=0)
    subject_id: str | None = None
    algorithm: str | None = None
    feature_set: Literal["LBP", "GLCM", "COMBINED"] | None = None
    models: list[LegacyPredictionModelResult] = Field(default_factory=list)
    summary: LegacyPredictionSummary
    warnings: list[str] = Field(default_factory=list)
    scan: LegacyScanResult | None = None


class LegacyHealthResponse(BaseModel):
    status: str
    task_type: str
    model_count: int = Field(ge=0)
    subjects: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    feature_sets: list[str] = Field(default_factory=list)
    counts_by_source: dict[str, int] = Field(default_factory=dict)
    counts_by_subject: dict[str, int] = Field(default_factory=dict)
    models: list[LegacyCatalogEntry] = Field(default_factory=list)


class CreateCaseResponse(BaseModel):
    case: CaseSummary
    message: str


class CreateRecordingResponse(BaseModel):
    recording: Recording
    message: str


class DeleteCaseResponse(BaseModel):
    case_id: str
    message: str


class AnalysisResponse(BaseModel):
    case: CaseSummary
    recording: Recording
    analysis: Analysis
    high_risk_intervals: list[HighRiskInterval]
    segment_results: list[SegmentResult]
    timeline: list[TimelinePoint]
    report: ReportSummary | None = None
    message: str


class ClinicalSummaryState(BaseModel):
    risk_score: int = Field(default=0, ge=0, le=100)
    risk_level: AnalysisStateRiskLevel
    priority: AnalysisStatePriority
    flagged_segments: int = Field(default=0, ge=0)
    summary_text: str
    recommendation: str


class AnalysisTimelineEntry(BaseModel):
    segment_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: AnalysisStateRiskLevel
    is_flagged: bool


class AnalysisSegmentEntry(BaseModel):
    segment_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: AnalysisStateRiskLevel
    is_flagged: bool


class AnalysisStateResponse(BaseModel):
    status: AnalysisStateStatus
    error: str | None = None
    clinical_summary: ClinicalSummaryState
    model_comparisons: list[ModelComparison] = Field(default_factory=list)
    timeline: list[AnalysisTimelineEntry] = Field(default_factory=list)
    segments: list[AnalysisSegmentEntry] = Field(default_factory=list)


class CaseAnalysesResponse(BaseModel):
    case_id: str
    analyses: list[Analysis]


class GenerateReportResponse(BaseModel):
    report: ReportSummary
    message: str


class ReportDetailResponse(BaseModel):
    report: ReportSummary
    case: CaseSummary
    recording: Recording
    analysis: Analysis
    model_comparisons: list[ModelComparison] = Field(default_factory=list)
    high_risk_intervals: list[HighRiskInterval]
