from __future__ import annotations

from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse

from app.services.auth_service import SessionUser
from app.services.clinical_workflow import ClinicalAnalysisService
from app.services.inference import SeizureInferenceService
from app.services.legacy_joblib import LegacyJoblibPredictionService
from app.services.store import ClinicalCaseStore
from app.web import app_http_exception


def get_case_store(request: Request) -> ClinicalCaseStore:
    return request.app.state.case_store


def get_workflow_service(request: Request) -> ClinicalAnalysisService:
    return request.app.state.workflow_service


def get_inference_service(request: Request) -> SeizureInferenceService:
    return request.app.state.inference_service


def get_legacy_joblib_service(request: Request) -> LegacyJoblibPredictionService:
    return request.app.state.legacy_joblib_service


def get_templates(request: Request):
    return request.app.state.templates


def get_current_user(request: Request) -> SessionUser | None:
    if not request.app.state.runtime_config.auth_enabled:
        return SessionUser(user_id="local-dev", username="local-dev", full_name="Local Dev", role="admin")
    session_user = request.session.get("user")
    if not isinstance(session_user, dict):
        return None
    return SessionUser.from_dict(session_user)


def require_api_user(request: Request) -> SessionUser:
    user = get_current_user(request)
    if user is None:
        raise app_http_exception(401, "auth_required", "Authentication required.")
    return user


def require_api_role(request: Request, *allowed_roles: str) -> SessionUser:
    user = require_api_user(request)
    if allowed_roles and user.role not in allowed_roles:
        raise app_http_exception(403, "forbidden", "You do not have permission to perform this action.")
    return user


def require_page_role(request: Request, *allowed_roles: str) -> tuple[SessionUser | None, RedirectResponse | None]:
    user = get_current_user(request)
    if user is None:
        next_path = quote(str(request.url.path))
        return None, RedirectResponse(url=f"/auth/login?next={next_path}", status_code=303)
    if allowed_roles and user.role not in allowed_roles:
        return user, RedirectResponse(url="/dashboard?notice=Access+denied.&tone=error", status_code=303)
    return user, None
