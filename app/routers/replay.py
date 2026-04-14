from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.config import runtime_config
from app.dependencies import get_replay_service
from app.schemas import ReplaySessionStateResponse, ReplayStartRequest, ReplayUploadResponse
from app.web import app_http_exception, page_context, save_upload

router = APIRouter()


@router.get("/replay", response_class=HTMLResponse)
async def replay_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="replay.html",
        context=page_context(
            request,
            page_title="EDF Replay Mode",
            page_subtitle="Upload an EDF, stream sliding windows through the configured legacy model, and review the live risk timeline.",
            active_page="replay",
        ),
    )


@router.post("/api/replay/upload", response_model=ReplayUploadResponse, status_code=201)
async def upload_replay_edf(request: Request, eeg_file: UploadFile = File(...)) -> ReplayUploadResponse:
    replay_service = get_replay_service(request)
    upload_path: Path | None = None
    try:
        upload, upload_path = await save_upload(
            eeg_file,
            uploads_dir=request.app.state.uploads_dir,
            config=runtime_config,
            clinician_mode=True,
        )
        return replay_service.create_session(
            file_path=upload_path,
            file_name=upload["filename"],
        )
    except HTTPException:
        if upload_path is not None and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if upload_path is not None and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise app_http_exception(400, "replay_upload_error", str(exc)) from exc


@router.post("/api/replay/{session_id}/start", response_model=ReplaySessionStateResponse)
async def start_replay_session(
    request: Request,
    session_id: str,
    payload: ReplayStartRequest,
) -> ReplaySessionStateResponse:
    replay_service = get_replay_service(request)
    try:
        return replay_service.start_session(
            session_id,
            window_sec=payload.window_sec,
            hop_sec=payload.hop_sec,
            replay_speed=payload.replay_speed,
        )
    except KeyError as exc:
        raise app_http_exception(404, "replay_not_found", "Replay session not found.") from exc
    except Exception as exc:
        raise app_http_exception(400, "replay_start_error", str(exc)) from exc


@router.post("/api/replay/{session_id}/stop", response_model=ReplaySessionStateResponse)
async def stop_replay_session(request: Request, session_id: str) -> ReplaySessionStateResponse:
    replay_service = get_replay_service(request)
    try:
        return replay_service.stop_session(session_id)
    except KeyError as exc:
        raise app_http_exception(404, "replay_not_found", "Replay session not found.") from exc


@router.get("/api/replay/{session_id}", response_model=ReplaySessionStateResponse)
async def replay_session_state(request: Request, session_id: str) -> ReplaySessionStateResponse:
    replay_service = get_replay_service(request)
    try:
        return replay_service.get_session_state(session_id)
    except KeyError as exc:
        raise app_http_exception(404, "replay_not_found", "Replay session not found.") from exc
