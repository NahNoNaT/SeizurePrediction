from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.schemas import CaseDetail, RecordingOverview, ReportSummary
from app.services.clinical_workflow import ClinicalAnalysisService
from app.services.store import ClinicalCaseStore


def persist_analysis_bundle(
    *,
    case_store: ClinicalCaseStore,
    workflow_service: ClinicalAnalysisService,
    case_id: str,
    recording: RecordingOverview,
) -> tuple[CaseDetail, ReportSummary | None]:
    case_store.update_case_status(case_id, "ANALYZING", datetime.now(timezone.utc))
    run_result = workflow_service.run_recording_analysis(case_id=case_id, recording=recording)
    case_store.save_analysis(
        run_result.assessment.analysis,
        run_result.assessment.segment_results,
        run_result.assessment.high_risk_intervals,
        trace_json=run_result.trace_json,
        case_status=run_result.assessment.case_status,
    )

    detail = case_store.get_analysis_detail(run_result.assessment.analysis.analysis_id)
    if detail is None:
        raise RuntimeError("Unable to reload the analysis after saving.")

    report: ReportSummary | None = None
    if detail.analysis is not None and detail.analysis.status == "COMPLETED":
        report = workflow_service.build_report(report_id=str(uuid4()), case_detail=detail)
        case_store.save_report(report)
        refreshed = case_store.get_analysis_detail(detail.analysis.analysis_id)
        if refreshed is None:
            raise RuntimeError("Unable to reload the analysis after report generation.")
        detail = refreshed

    return detail, report
