from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.dependencies import get_case_store, get_workflow_service
from app.schemas import AnalysisStateResponse, CaseAnalysesResponse
from app.web import app_http_exception
from app.workflow_helpers import persist_analysis_bundle

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/recordings/{recording_id}/analyze", response_model=AnalysisStateResponse)
async def api_analyze_recording(request: Request, recording_id: str) -> AnalysisStateResponse:
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    recording = case_store.get_recording(recording_id)
    if recording is None:
        raise app_http_exception(404, "recording_not_found", "Recording not found.")
    if case_store.get_case_detail(recording.case_id) is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")

    try:
        detail, _ = persist_analysis_bundle(
            case_store=case_store,
            workflow_service=workflow_service,
            case_id=recording.case_id,
            recording=recording,
        )
        return workflow_service.build_case_analysis_state(detail)
    except Exception as exc:
        logger.exception(
            "Analysis request failed",
            extra={"event": "analysis_request_failed", "case_id": recording.case_id, "recording_id": recording_id, "status": "FAILED"},
        )
        raise app_http_exception(400, "analysis_error", str(exc)) from exc


@router.get("/api/analyses/{analysis_id}", response_model=AnalysisStateResponse)
async def api_analysis_detail(request: Request, analysis_id: str) -> AnalysisStateResponse:
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    detail = case_store.get_analysis_detail(analysis_id)
    if detail is None:
        raise app_http_exception(404, "analysis_not_found", "Analysis not found.")
    return workflow_service.build_case_analysis_state(detail)


@router.get("/api/cases/{case_id}/analysis", response_model=AnalysisStateResponse)
async def api_case_analysis_state(request: Request, case_id: str) -> AnalysisStateResponse:
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    detail = case_store.get_case_detail(case_id)
    if detail is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")
    if detail.recording is None:
        raise app_http_exception(404, "recording_not_found", "No EEG recording is linked to this case.")
    return workflow_service.build_case_analysis_state(detail)


@router.get("/api/cases/{case_id}/analyses", response_model=CaseAnalysesResponse)
async def api_case_analyses(request: Request, case_id: str) -> CaseAnalysesResponse:
    case_store = get_case_store(request)
    detail = case_store.get_case_detail(case_id)
    if detail is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")
    return CaseAnalysesResponse(case_id=case_id, analyses=case_store.list_analyses(case_id))
