from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.dependencies import get_case_store, get_workflow_service
from app.schemas import GenerateReportResponse, ReportDetailResponse
from app.web import app_http_exception, build_case_summary, format_duration_label, page_context

router = APIRouter()


@router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    case_store = get_case_store(request)
    return templates.TemplateResponse(
        request=request,
        name="reports.html",
        context=page_context(
            request,
            page_title="Reports",
            page_subtitle="Open clinician-facing reports that have already been saved for case review.",
            active_page="reports",
            extra={"reports": case_store.list_reports()},
        ),
    )


@router.get("/reports/{report_id}", response_class=HTMLResponse)
async def report_view_page(request: Request, report_id: str) -> HTMLResponse:
    templates = request.app.state.templates
    case_store = get_case_store(request)
    report = case_store.get_report(report_id)
    if report is None:
        raise app_http_exception(404, "report_not_found", "Report not found.")
    case_detail = case_store.get_analysis_detail(report.analysis_id)
    if case_detail is None or case_detail.analysis is None or case_detail.recording is None:
        raise app_http_exception(404, "analysis_not_found", "Linked analysis not found.")
    return templates.TemplateResponse(
        request=request,
        name="report_print.html",
        context=page_context(
            request,
            page_title="Clinical Report",
            page_subtitle="Printable clinician-facing report for the selected case review.",
            active_page="reports",
            extra={
                "report": report,
                "case_detail": case_detail,
                "recording_duration_label": format_duration_label(case_detail.recording.duration_sec),
            },
        ),
    )


@router.get("/reports/{report_id}/download")
async def report_download(request: Request, report_id: str) -> FileResponse:
    case_store = get_case_store(request)
    report = case_store.get_report(report_id)
    if report is None:
        raise app_http_exception(404, "report_not_found", "Report not found.")
    report_path = Path(report.report_path)
    if not report_path.exists():
        raise app_http_exception(404, "report_file_missing", "Saved report document is not available on disk.")
    return FileResponse(path=report_path, filename=report_path.name, media_type="text/html")


@router.post("/api/analyses/{analysis_id}/report", response_model=GenerateReportResponse)
async def api_generate_report(request: Request, analysis_id: str) -> GenerateReportResponse:
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    detail = case_store.get_analysis_detail(analysis_id)
    if detail is None or detail.analysis is None:
        raise app_http_exception(404, "analysis_not_found", "Analysis not found.")
    if detail.analysis.status not in {"COMPLETED", "REPORT_READY"}:
        raise app_http_exception(409, "analysis_not_ready", "A completed analysis is required before generating a report.")
    report = workflow_service.build_report(report_id=str(uuid4()), case_detail=detail)
    case_store.save_report(report)
    return GenerateReportResponse(report=report, message="Analysis report generated successfully.")


@router.get("/api/reports/{report_id}", response_model=ReportDetailResponse)
async def api_report_detail(request: Request, report_id: str) -> ReportDetailResponse:
    case_store = get_case_store(request)
    report = case_store.get_report(report_id)
    if report is None:
        raise app_http_exception(404, "report_not_found", "Report not found.")
    detail = case_store.get_analysis_detail(report.analysis_id)
    if detail is None or detail.recording is None or detail.analysis is None:
        raise app_http_exception(404, "analysis_not_found", "Linked analysis not found.")
    return ReportDetailResponse(
        report=report,
        case=build_case_summary(detail),
        recording=detail.recording,
        analysis=detail.analysis,
        high_risk_intervals=detail.high_risk_intervals,
    )
