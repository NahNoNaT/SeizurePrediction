from __future__ import annotations

from urllib.parse import unquote

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.dependencies import get_case_store, get_current_user
from app.services.auth_service import authenticate_user, bootstrap_first_role, register_user
from app.web import build_redirect

router = APIRouter()


def _sanitize_next(next_path: str | None) -> str:
    if not next_path:
        return "/dashboard"
    decoded = unquote(next_path).strip()
    if not decoded.startswith("/") or decoded.startswith("//") or decoded.startswith("/auth/"):
        return "/dashboard"
    return decoded


@router.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = Query(default="/dashboard")) -> HTMLResponse:
    if not request.app.state.runtime_config.auth_enabled:
        return RedirectResponse(url="/dashboard", status_code=303)
    user = get_current_user(request)
    if user is not None:
        return RedirectResponse(url=_sanitize_next(next), status_code=303)
    templates = request.app.state.templates
    case_store = get_case_store(request)
    user_count = case_store.count_users()
    allow_public_registration = request.app.state.runtime_config.allow_public_registration
    can_self_register = user_count == 0 or allow_public_registration
    return templates.TemplateResponse(
        request=request,
        name="auth_login.html",
        context={
            "request": request,
            "page_title": "Sign In",
            "next_path": _sanitize_next(next),
            "can_self_register": can_self_register,
            "notice": request.query_params.get("notice"),
            "tone": request.query_params.get("tone", "info"),
        },
    )


@router.post("/auth/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_path: str = Form(default="/dashboard"),
) -> RedirectResponse:
    if not request.app.state.runtime_config.auth_enabled:
        return RedirectResponse(url="/dashboard", status_code=303)
    case_store = get_case_store(request)
    user = authenticate_user(case_store, username=username, password=password)
    if user is None:
        return build_redirect("/auth/login", "Invalid username or password.", "error")

    request.session["user"] = user.to_session()
    return RedirectResponse(url=_sanitize_next(next_path), status_code=303)


@router.get("/auth/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    if not request.app.state.runtime_config.auth_enabled:
        return RedirectResponse(url="/dashboard", status_code=303)
    case_store = get_case_store(request)
    current_user = get_current_user(request)
    templates = request.app.state.templates
    user_count = case_store.count_users()
    allow_public_registration = request.app.state.runtime_config.allow_public_registration
    allow_open_register = user_count == 0 or allow_public_registration
    is_admin_creator = current_user is not None and current_user.role == "admin"
    if not allow_open_register and not is_admin_creator:
        return build_redirect("/auth/login", "Only admins can create new accounts.", "error")
    return templates.TemplateResponse(
        request=request,
        name="auth_register.html",
        context={
            "request": request,
            "page_title": "Register",
            "allow_role_select": is_admin_creator,
            "default_role": bootstrap_first_role(user_count) if user_count == 0 else "viewer",
            "notice": request.query_params.get("notice"),
            "tone": request.query_params.get("tone", "info"),
        },
    )


@router.post("/auth/register")
async def register_submit(
    request: Request,
    username: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    role: str = Form(default="viewer"),
) -> RedirectResponse:
    if not request.app.state.runtime_config.auth_enabled:
        return RedirectResponse(url="/dashboard", status_code=303)
    case_store = get_case_store(request)
    current_user = get_current_user(request)
    user_count = case_store.count_users()
    allow_public_registration = request.app.state.runtime_config.allow_public_registration
    allow_open_register = user_count == 0 or allow_public_registration
    is_admin_creator = current_user is not None and current_user.role == "admin"
    if not allow_open_register and not is_admin_creator:
        return build_redirect("/auth/login", "Only admins can create new accounts.", "error")

    resolved_role = role.strip().lower()
    if user_count == 0:
        resolved_role = "admin"
    elif not is_admin_creator:
        resolved_role = "viewer"

    try:
        user = register_user(
            case_store,
            username=username,
            full_name=full_name,
            password=password,
            role=resolved_role,
        )
    except ValueError as exc:
        return build_redirect("/auth/register", str(exc), "error")

    if user_count == 0:
        request.session["user"] = user.to_session()
        return build_redirect("/dashboard", "Admin account created.", "success")
    return build_redirect("/auth/register", "User account created successfully.", "success")


@router.post("/auth/logout")
async def logout_submit(request: Request) -> RedirectResponse:
    if not request.app.state.runtime_config.auth_enabled:
        return RedirectResponse(url="/dashboard", status_code=303)
    request.session.clear()
    return build_redirect("/auth/login", "Signed out successfully.", "success")
