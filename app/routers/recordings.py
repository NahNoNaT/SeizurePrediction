from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from app.config import runtime_config
from app.dependencies import get_case_store, get_workflow_service, require_api_role
from app.schemas import CreateRecordingResponse, RecordingOverview, RecordingPreviewResponse
from app.services.errors import EEGValidationError
from app.web import app_http_exception, save_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/cases/{case_id}/recordings", response_model=CreateRecordingResponse, status_code=201)
async def api_upload_recording(request: Request, case_id: str, eeg_file: UploadFile = File(...)) -> CreateRecordingResponse:
    require_api_role(request, "clinician", "admin")
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    if case_store.get_case_detail(case_id) is None:
        raise app_http_exception(404, "case_not_found", "Case not found.")

    upload_path: Path | None = None
    try:
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

        intake = workflow_service.inspect_recording(upload_path, clinician_mode=True, enforce_validation=False)
        recording = workflow_service.build_recording_overview(
            recording_id=str(uuid4()),
            case_id=case_id,
            file_name=upload["filename"],
            file_path=str(upload_path),
            intake=intake,
            uploaded_at=upload["created_at"],
        )
        case_store.create_recording(recording)
        if recording.validation_status != "VALIDATED":
            case_store.update_case_status(case_id, "FAILED", upload["created_at"])
            return CreateRecordingResponse(
                recording=recording,
                message="EEG recording uploaded, but validation blocked analysis for this file.",
            )
        case_store.update_case_status(case_id, "VALIDATED", upload["created_at"])
        return CreateRecordingResponse(recording=recording, message="EEG recording uploaded and validated.")
    except HTTPException as exc:
        case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        raise exc
    except EEGValidationError as exc:
        case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        raise app_http_exception(400, exc.code, exc.public_detail) from exc
    except Exception as exc:
        logger.exception("Recording upload failed", extra={"event": "recording_upload_failed", "case_id": case_id, "status": "FAILED"})
        case_store.update_case_status(case_id, "FAILED", datetime.now(timezone.utc))
        raise app_http_exception(400, "recording_validation_error", str(exc)) from exc


@router.get("/api/recordings/{recording_id}", response_model=RecordingOverview)
async def api_recording_detail(request: Request, recording_id: str) -> RecordingOverview:
    require_api_role(request, "viewer", "clinician", "admin")
    case_store = get_case_store(request)
    recording = case_store.get_recording(recording_id)
    if recording is None:
        raise app_http_exception(404, "recording_not_found", "Recording not found.")
    return recording


@router.get("/api/recordings/{recording_id}/preview", response_model=RecordingPreviewResponse)
async def api_recording_preview(
    request: Request,
    recording_id: str,
    start_sec: float = Query(default=0.0),
    duration_sec: float = Query(default=30.0),
    channels: str = Query(default=""),
) -> RecordingPreviewResponse:
    require_api_role(request, "viewer", "clinician", "admin")
    case_store = get_case_store(request)
    workflow_service = get_workflow_service(request)
    recording = case_store.get_recording(recording_id)
    if recording is None:
        raise app_http_exception(404, "recording_not_found", "Recording not found.")

    selected_channels = [value.strip() for value in channels.split(",") if value.strip()] if channels else None
    try:
        return workflow_service.preview_recording(
            recording=recording,
            start_sec=start_sec,
            duration_sec=duration_sec,
            channels=selected_channels,
        )
    except EEGValidationError as exc:
        raise app_http_exception(400, exc.code, exc.public_detail) from exc
    except Exception as exc:
        logger.exception(
            "Recording preview failed",
            extra={"event": "recording_preview_failed", "recording_id": recording_id, "status": "FAILED"},
        )
        raise app_http_exception(400, "recording_preview_error", str(exc)) from exc
