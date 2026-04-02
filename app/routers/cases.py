from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import runtime_config
from app.dependencies import get_case_store, get_workflow_service
from app.schemas import CaseCreateRequest, CaseDetail, CaseSummary, CreateCaseResponse, DeleteCaseResponse
from app.services.errors import EEGValidationError
from app.web import (
    app_http_exception,
    build_case_summary,
    build_redirect,
    chart_points_from_timeline,
    format_duration_label,
    page_context,
    save_upload,
)
from app.workflow_helpers import persist_analysis_bundle

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/cases", response_class=HTMLResponse)
async def cases_page(
    request: Request,
    search: str = Query(default=""),
    risk: str = Query(default=""),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
) -> HTMLResponse:
    templates = request.app.state.templates
    case_store = get_case_store(request)
    cases = case_store.list_cases(search=search, risk=risk, date_from=date_from, date_to=date_to)
    return templates.TemplateResponse(
        request=request,
        name="cases.html",
        context=page_context(
            request,
            page_title="Cases",
            page_subtitle="Search patient cases, open saved findings, and review case progress.",
            active_page="cases",
            extra={"cases": cases, "search": search, "risk": risk, "date_from": date_from, "date_to": date_to},
        ),
    )


@router.get("/cases/new", response_class=HTMLResponse)
async def new_analysis_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="new_analysis.html",
        context=page_context(
            request,
            page_title="New Analysis",
            page_subtitle="Create a case, upload an EEG recording, and generate a clinician-facing review.",
            active_page="new_analysis",
            extra={"today": date.today().isoformat()},
        ),
    )


@router.post("/cases/new")
async def create_case_and_run_analysis(
    request: Request,
    patient_id: str = Form(...),
    clinician_name: str = Form(...),
    recording_date: date = Form(...),
    notes: str = Form(default=""),
    eeg_file: UploadFile = File(...),
) -> RedirectResponse:
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    case_id = str(uuid4())
    upload_path: Path | None = None

    try:
        payload = CaseCreateRequest(
            patient_id=patient_id.strip(),
            clinician_name=clinician_name.strip(),
            recording_date=recording_date,
            notes=notes.strip() or None,
        )
        created_at = datetime.now(timezone.utc)
        case_store.create_case(
            case_id=case_id,
            patient_id=payload.patient_id,
            clinician_name=payload.clinician_name,
            recording_date=payload.recording_date,
            notes=payload.notes,
            status="NEW",
            created_at=created_at,
        )

        upload, upload_path = await save_upload(
            eeg_file,
            uploads_dir=request.app.state.uploads_dir,
            config=runtime_config,
            clinician_mode=True,
        )
        logger.info(
            "Recording uploaded",
            extra={"event": "recording_uploaded", "case_id": case_id, "upload_filename": upload["filename"], "status": "UPLOADED"},
        )
        case_store.update_case_status(case_id, "UPLOADED", upload["created_at"])

        intake = workflow_service.inspect_recording(upload_path, clinician_mode=True)
        recording = workflow_service.build_recording_overview(
            recording_id=str(uuid4()),
            case_id=case_id,
            file_name=upload["filename"],
            file_path=str(upload_path),
            intake=intake,
            uploaded_at=upload["created_at"],
        )
        case_store.create_recording(recording)
        case_store.update_case_status(case_id, "VALIDATED", upload["created_at"])

        detail, _ = persist_analysis_bundle(
            case_store=case_store,
            workflow_service=workflow_service,
            case_id=case_id,
            recording=recording,
        )
        if detail.analysis and detail.analysis.status == "FAILED":
            return build_redirect(
                f"/cases/{case_id}",
                detail.analysis.failure_message or "Analysis could not be completed.",
                "error",
            )
        return build_redirect(
            f"/cases/{case_id}",
            "Analysis completed and the clinical report is ready for review.",
            "success",
        )
    except HTTPException as exc:
        logger.exception("Case creation failed", extra={"event": "case_create_failed", "case_id": case_id, "status": "FAILED"})
        if case_store.get_case_detail(case_id) is not None:
            case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail)}
        return build_redirect("/cases/new", detail.get("detail", "Unable to process the uploaded recording."), "error")
    except EEGValidationError as exc:
        logger.warning("Recording validation failed", extra={"event": "recording_validation_failed", "case_id": case_id, "status": "FAILED", "code": exc.code})
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        return build_redirect(f"/cases/{case_id}", exc.public_detail, "error")
    except Exception as exc:
        logger.exception("Case workflow failed", extra={"event": "case_workflow_failed", "case_id": case_id, "status": "FAILED"})
        if case_store.get_case_detail(case_id) is not None:
            case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        return build_redirect(f"/cases/{case_id}", str(exc), "error")


@router.get("/cases/{case_id}", response_class=HTMLResponse)
async def case_detail_page(request: Request, case_id: str) -> HTMLResponse:
    templates = request.app.state.templates
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    case_detail = case_store.get_case_detail(case_id)
    if case_detail is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")

    analysis_state = workflow_service.build_case_analysis_state(case_detail)
    return templates.TemplateResponse(
        request=request,
        name="case_detail.html",
        context=page_context(
            request,
            page_title=f"Case {case_detail.patient_id}",
            page_subtitle="Clinical summary, risk timeline, flagged intervals, and interpretation for this case.",
            active_page="cases",
            extra={
                "case_detail": case_detail,
                "analysis_state": analysis_state,
                "recording_duration_label": format_duration_label(
                    case_detail.recording.duration_sec if case_detail.recording else 0.0
                ),
                "chart_points": chart_points_from_timeline(analysis_state.timeline),
            },
        ),
    )


@router.post("/cases/{case_id}/delete")
async def delete_case_page(request: Request, case_id: str) -> RedirectResponse:
    case_store = get_case_store(request)
    deleted = case_store.delete_case(case_id)
    if not deleted:
        return build_redirect("/cases", "Case could not be found.", "error")
    return build_redirect("/cases", "Case deleted successfully.", "success")


@router.post("/api/cases", response_model=CreateCaseResponse, status_code=201)
async def api_create_case(request: Request, payload: CaseCreateRequest) -> CreateCaseResponse:
    case_store = get_case_store(request)
    created_at = datetime.now(timezone.utc)
    case_id = str(uuid4())
    case_store.create_case(
        case_id=case_id,
        patient_id=payload.patient_id,
        clinician_name=payload.clinician_name,
        recording_date=payload.recording_date,
        notes=payload.notes,
        status="NEW",
        created_at=created_at,
    )
    detail = case_store.get_case_detail(case_id)
    if detail is None:
        raise app_http_exception(500, "case_persistence_error", "Case could not be reloaded after creation.")
    return CreateCaseResponse(case=build_case_summary(detail), message="Case created successfully.")


@router.get("/api/cases", response_model=list[CaseSummary])
async def api_list_cases(
    request: Request,
    search: str = Query(default=""),
    risk: str = Query(default=""),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
) -> list[CaseSummary]:
    case_store = get_case_store(request)
    return case_store.list_cases(search=search, risk=risk, date_from=date_from, date_to=date_to)


@router.get("/api/cases/{case_id}", response_model=CaseDetail)
async def api_case_detail(request: Request, case_id: str) -> CaseDetail:
    case_store = get_case_store(request)
    detail = case_store.get_case_detail(case_id)
    if detail is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")
    return detail


@router.delete("/api/cases/{case_id}", response_model=DeleteCaseResponse)
async def api_delete_case(request: Request, case_id: str) -> DeleteCaseResponse:
    case_store = get_case_store(request)
    if not case_store.delete_case(case_id):
        raise app_http_exception(404, "case_not_found", "Case not found.")
    return DeleteCaseResponse(case_id=case_id, message="Case deleted successfully.")
