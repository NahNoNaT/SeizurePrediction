from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.dependencies import get_case_store
from app.web import page_context

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    case_store = get_case_store(request)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=page_context(
            request,
            page_title="Dashboard",
            page_subtitle="Review active clinical cases, high-risk findings, and recently generated reports.",
            active_page="dashboard",
            extra={
                "stats": case_store.dashboard_stats(),
                "recent_cases": case_store.list_cases(limit=6),
                "recent_reports": case_store.list_reports()[:5],
            },
        ),
    )
