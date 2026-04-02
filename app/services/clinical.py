from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from app.config import RuntimeConfig, runtime_config
from app.schemas import (
    Analysis,
    AnalysisFailureCode,
    CaseStatus,
    ClinicalRiskCategory,
    HighRiskInterval,
    RecordingOverview,
    ReviewPriority,
    SegmentResultRecord,
    TimelinePoint,
)
from app.services.temporal import TemporalAnalysisResult


@dataclass
class ClinicalAssessmentResult:
    analysis: Analysis
    segment_results: list[SegmentResultRecord]
    high_risk_intervals: list[HighRiskInterval]
    timeline: list[TimelinePoint]
    case_status: CaseStatus
    estimated_seizure_risk: float | None


class ClinicalDecisionService:
    def __init__(self, config: RuntimeConfig = runtime_config):
        self.config = config

    def map_risk_score(self, risk_score: float) -> ClinicalRiskCategory:
        if risk_score >= 0.75:
            return "High"
        if risk_score >= 0.45:
            return "Moderate"
        return "Low"

    def assign_review_priority(self, overall_risk: ClinicalRiskCategory) -> ReviewPriority:
        if overall_risk == "High":
            return "Urgent"
        if overall_risk == "Moderate":
            return "Recommended"
        return "Routine"

    def generate_interpretation_text(
        self,
        *,
        overall_risk: ClinicalRiskCategory,
        review_priority: ReviewPriority,
        intervals: list[HighRiskInterval],
        duration_sec: float,
    ) -> str:
        duration_min = max(duration_sec / 60.0, 0.1)
        if intervals:
            first = intervals[0]
            return (
                f"Elevated seizure-risk activity begins near {first.start_sec:.0f} seconds, with "
                f"{len(intervals)} grouped period(s) of concern across this {duration_min:.1f}-minute recording. "
                f"Overall seizure risk is {overall_risk.lower()} and review priority is {review_priority.lower()}."
            )
        return (
            f"No sustained clusters of elevated seizure-risk activity were identified in this "
            f"{duration_min:.1f}-minute recording. Overall seizure risk is {overall_risk.lower()} "
            f"with {review_priority.lower()} review priority."
        )

    def build_assessment(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        model_version: str,
        temporal_result: TemporalAnalysisResult,
    ) -> ClinicalAssessmentResult:
        scores = [point.risk_score for point in temporal_result.timeline]
        max_risk_score = float(np.max(scores)) if scores else 0.0
        mean_risk_score = float(np.mean(scores)) if scores else 0.0
        flagged_segments_count = sum(1 for point in temporal_result.timeline if point.is_flagged)
        interval_burden = self._interval_burden(temporal_result, recording.duration_sec)
        flagged_ratio = flagged_segments_count / max(len(temporal_result.timeline), 1)
        estimated_seizure_risk = float(
            np.clip(
                0.50 * max_risk_score
                + 0.20 * mean_risk_score
                + 0.15 * flagged_ratio
                + 0.15 * interval_burden,
                0.0,
                0.99,
            )
        )
        overall_risk = self._overall_risk(
            max_risk_score=max_risk_score,
            estimated_seizure_risk=estimated_seizure_risk,
            flagged_segments_count=flagged_segments_count,
            total_segments=len(temporal_result.timeline),
            interval_count=len(temporal_result.intervals),
        )
        review_priority = self.assign_review_priority(overall_risk)
        case_status = "COMPLETED"

        created_at = datetime.now(timezone.utc)
        analysis_id = str(uuid4())
        timeline = [
            TimelinePoint(
                segment_index=point.segment_index,
                start_sec=point.start_sec,
                end_sec=point.end_sec,
                risk_score=point.risk_score,
                risk_label=point.risk_label,  # type: ignore[arg-type]
                is_flagged=point.is_flagged,
            )
            for point in temporal_result.timeline
        ]
        segment_results = [
            SegmentResultRecord(
                id=str(uuid4()),
                analysis_id=analysis_id,
                segment_index=point.segment_index,
                start_sec=point.start_sec,
                end_sec=point.end_sec,
                risk_score=point.risk_score,
                risk_label=point.risk_label,  # type: ignore[arg-type]
                is_flagged=point.is_flagged,
            )
            for point in temporal_result.timeline
        ]
        intervals = [
            HighRiskInterval(
                id=str(uuid4()),
                analysis_id=analysis_id,
                interval_index=item.interval_index,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                mean_risk=item.mean_risk,
                max_risk=item.max_risk,
                flagged_segments_count=item.flagged_segments_count,
                review_status=item.review_status,
            )
            for item in temporal_result.intervals
        ]
        analysis = Analysis(
            analysis_id=analysis_id,
            case_id=case_id,
            recording_id=recording.recording_id,
            status="COMPLETED",
            model_version=model_version,
            overall_risk=overall_risk,
            review_priority=review_priority,
            max_risk_score=max_risk_score,
            mean_risk_score=mean_risk_score,
            estimated_seizure_risk=estimated_seizure_risk,
            flagged_segments_count=flagged_segments_count,
            total_segments=len(segment_results),
            high_risk_intervals_count=len(intervals),
            recording_duration_sec=recording.duration_sec,
            clinical_summary=self._clinical_summary(
                overall_risk=overall_risk,
                review_priority=review_priority,
                flagged_segments_count=flagged_segments_count,
                duration_sec=recording.duration_sec,
                interval_count=len(intervals),
            ),
            recommendation=self._recommendation_text(
                overall_risk=overall_risk,
                review_priority=review_priority,
                interval_count=len(intervals),
            ),
            interpretation=self.generate_interpretation_text(
                overall_risk=overall_risk,
                review_priority=review_priority,
                intervals=intervals,
                duration_sec=recording.duration_sec,
            ),
            created_at=created_at,
            report_generated=False,
        )
        return ClinicalAssessmentResult(
            analysis=analysis,
            segment_results=segment_results,
            high_risk_intervals=intervals,
            timeline=timeline,
            case_status=case_status,
            estimated_seizure_risk=estimated_seizure_risk,
        )

    def build_failed_assessment(
        self,
        *,
        case_id: str,
        recording: RecordingOverview,
        failure_code: AnalysisFailureCode,
        failure_message: str,
        model_version: str | None,
    ) -> ClinicalAssessmentResult:
        analysis = Analysis(
            analysis_id=str(uuid4()),
            case_id=case_id,
            recording_id=recording.recording_id,
            status="FAILED",
            model_version=model_version,
            recording_duration_sec=recording.duration_sec,
            clinical_summary="Analysis could not be completed.",
            recommendation="Please review the technical issue before retrying.",
            interpretation=(
                "The system could not complete the clinical review for this recording. "
                "No seizure-risk result has been generated."
            ),
            failure_code=failure_code,
            failure_message=failure_message,
            created_at=datetime.now(timezone.utc),
            report_generated=False,
        )
        return ClinicalAssessmentResult(
            analysis=analysis,
            segment_results=[],
            high_risk_intervals=[],
            timeline=[],
            case_status="FAILED",
            estimated_seizure_risk=None,
        )

    def _overall_risk(
        self,
        *,
        max_risk_score: float,
        estimated_seizure_risk: float,
        flagged_segments_count: int,
        total_segments: int,
        interval_count: int,
    ) -> ClinicalRiskCategory:
        flagged_ratio = flagged_segments_count / max(total_segments, 1)
        if interval_count >= 1 and (max_risk_score >= 0.78 or estimated_seizure_risk >= 0.72 or flagged_ratio >= 0.12):
            return "High"
        if interval_count >= 1 or estimated_seizure_risk >= 0.42 or flagged_ratio >= 0.05:
            return "Moderate"
        return "Low"

    def _clinical_summary(
        self,
        *,
        overall_risk: ClinicalRiskCategory,
        review_priority: ReviewPriority,
        flagged_segments_count: int,
        duration_sec: float,
        interval_count: int,
    ) -> str:
        duration_min = max(duration_sec / 60.0, 0.1)
        if overall_risk == "High":
            return (
                f"This {duration_min:.1f}-minute EEG recording contains sustained high-risk activity across "
                f"{interval_count} grouped interval(s), involving {flagged_segments_count} flagged segment(s). "
                f"Review priority is {review_priority.lower()}."
            )
        if overall_risk == "Moderate":
            return (
                f"This {duration_min:.1f}-minute EEG recording shows elevated seizure-risk activity with "
                f"{flagged_segments_count} flagged segment(s) for closer review."
            )
        return (
            f"This {duration_min:.1f}-minute EEG recording does not show sustained high-risk activity. "
            "Routine clinical review is appropriate."
        )

    def _recommendation_text(
        self,
        *,
        overall_risk: ClinicalRiskCategory,
        review_priority: ReviewPriority,
        interval_count: int,
    ) -> str:
        if overall_risk == "High":
            return (
                f"Urgent neurologist review is recommended. Examine the {interval_count} flagged high-risk interval(s) "
                "and correlate with the clinical context before making treatment decisions."
            )
        if overall_risk == "Moderate":
            return (
                f"{review_priority} clinician review is advised, with attention to the elevated-risk windows "
                "identified in the recording."
            )
        return "Routine review is appropriate. Continue to interpret these findings alongside the full clinical picture."

    def _interval_burden(self, temporal_result: TemporalAnalysisResult, duration_sec: float) -> float:
        if duration_sec <= 0 or not temporal_result.intervals:
            return 0.0
        total_interval_duration = sum(interval.end_sec - interval.start_sec for interval in temporal_result.intervals)
        return float(np.clip(total_interval_duration / duration_sec, 0.0, 1.0))
