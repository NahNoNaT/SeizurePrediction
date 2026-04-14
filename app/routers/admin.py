from __future__ import annotations

import re

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.config import runtime_config
from app.dependencies import get_inference_service
from app.schemas import AdminSettingsSummary, AppMetadataResponse
from app.web import build_redirect, page_context

router = APIRouter()


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    project_root = request.app.state.project_root
    configured_paths = list(runtime_config.configured_checkpoint_paths(project_root))
    using_auto_discovery = not runtime_config.explicit_checkpoint_paths() and bool(configured_paths)
    admin_settings = AdminSettingsSummary(
        checkpoint_path=runtime_config.checkpoint_path or (configured_paths[0] if configured_paths else None),
        checkpoint_paths=configured_paths,
        checkpoint_directory=str(runtime_config.checkpoint_directory(project_root)),
        checkpoint_extensions=list(runtime_config.checkpoint_extensions),
        auto_discovery_enabled=runtime_config.auto_discover_checkpoints,
        using_auto_discovery=using_auto_discovery,
        model_version=runtime_config.default_model_version,
        backend_status=runtime_config.inference_status,
        model_device=runtime_config.model_device_preference,
        configured_model_count=len(configured_paths),
        model_slots=get_inference_service(request).model_slot_statuses(),
        target_sample_rate_hz=runtime_config.target_sampling_rate_hz,
        supported_input_modes=[
            "Referential scalp EEG channels are accepted directly.",
            "Recognized bipolar EEG chains are converted heuristically before inference.",
        ],
        required_channel_order=list(runtime_config.required_channel_order),
        minimum_mapped_channels=runtime_config.minimum_mapped_channels,
        max_zero_fill_channels=runtime_config.max_zero_fill_channels,
        zero_fill_allowed=runtime_config.zero_fill_allowed,
    )
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context=page_context(
            request,
            page_title="Admin",
            page_subtitle="Technical runtime status and model package settings for system administration.",
            active_page="admin",
            extra={"admin_settings": admin_settings},
        ),
    )


@router.post("/admin/checkpoint")
async def admin_update_checkpoint(request: Request, checkpoint_paths: str = Form(...)):
    inference_service = get_inference_service(request)
    try:
        parsed_paths = [value.strip() for value in re.split(r"[\r\n,;]+", checkpoint_paths) if value.strip()]
        runtime_config.checkpoint_paths = tuple(parsed_paths)
        runtime_config.checkpoint_path = parsed_paths[0] if parsed_paths else None
        inference_service.load_models()
    except Exception as exc:
        return build_redirect("/admin", str(exc), "error")
    if parsed_paths:
        return build_redirect("/admin", "Model checkpoint updated successfully.", "success")
    return build_redirect("/admin", "Manual checkpoint override cleared. Auto-discovery is active again.", "success")


@router.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "edf_intake": "enabled",
        "backend_status": runtime_config.inference_status,
    }


@router.get("/api/model/info", response_model=AppMetadataResponse)
async def model_info(request: Request) -> AppMetadataResponse:
    return AppMetadataResponse(
        app_title=runtime_config.app_title,
        app_subtitle=runtime_config.app_subtitle,
        supported_formats=list(runtime_config.supported_upload_extensions),
        max_upload_size_mb=runtime_config.max_upload_size_mb,
        backend_status=runtime_config.inference_status,
        model_version=runtime_config.default_model_version,
        configured_model_count=len(runtime_config.configured_checkpoint_paths(request.app.state.project_root)),
        research_disclaimer=runtime_config.research_disclaimer,
    )
