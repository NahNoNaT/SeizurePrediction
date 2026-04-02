from __future__ import annotations

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
    admin_settings = AdminSettingsSummary(
        checkpoint_path=runtime_config.checkpoint_path,
        model_version=runtime_config.default_model_version,
        backend_status=runtime_config.inference_status,
        model_device=runtime_config.model_device_preference,
        target_sample_rate_hz=runtime_config.target_sampling_rate_hz,
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
async def admin_update_checkpoint(request: Request, checkpoint_path: str = Form(...)):
    inference_service = get_inference_service(request)
    try:
        inference_service.load_checkpoint(checkpoint_path.strip())
    except Exception as exc:
        return build_redirect("/admin", str(exc), "error")
    return build_redirect("/admin", "Model checkpoint updated successfully.", "success")


@router.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "edf_intake": "enabled",
        "backend_status": runtime_config.inference_status,
    }


@router.get("/api/model/info", response_model=AppMetadataResponse)
async def model_info() -> AppMetadataResponse:
    return AppMetadataResponse(
        app_title=runtime_config.app_title,
        app_subtitle=runtime_config.app_subtitle,
        supported_formats=list(runtime_config.supported_upload_extensions),
        max_upload_size_mb=runtime_config.max_upload_size_mb,
        backend_status=runtime_config.inference_status,
        model_version=runtime_config.default_model_version,
        research_disclaimer=runtime_config.research_disclaimer,
    )
