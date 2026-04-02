from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import configure_logging, runtime_config
from app.routers import admin, analyses, cases, pages, recordings, reports
from app.schemas import ErrorResponse
from app.services.clinical_workflow import ClinicalAnalysisService
from app.services.inference import SeizureInferenceService
from app.services.store import ClinicalCaseStore

configure_logging(runtime_config.log_level)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def create_app(
    *,
    config=runtime_config,
    case_store: ClinicalCaseStore | None = None,
    inference_service: SeizureInferenceService | None = None,
    workflow_service: ClinicalAnalysisService | None = None,
) -> FastAPI:
    uploads_dir = config.uploads_directory(PROJECT_ROOT)
    data_dir = config.data_directory(PROJECT_ROOT)
    reports_dir = config.reports_directory(PROJECT_ROOT)
    database_file = config.database_file(PROJECT_ROOT)

    uploads_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    resolved_inference_service = inference_service or SeizureInferenceService(project_root=PROJECT_ROOT, config=config)
    resolved_workflow_service = workflow_service or ClinicalAnalysisService(
        project_root=PROJECT_ROOT,
        inference_service=resolved_inference_service,
        config=config,
    )
    resolved_case_store = case_store or ClinicalCaseStore(database_file)
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Application startup", extra={"event": "startup", "status": "starting"})
        resolved_case_store.initialize()
        resolved_inference_service.warmup()
        logger.info("Application startup complete", extra={"event": "startup_complete", "status": config.inference_status})
        yield

    app = FastAPI(title=config.app_title, version="2.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    app.state.runtime_config = config
    app.state.project_root = PROJECT_ROOT
    app.state.uploads_dir = uploads_dir
    app.state.data_dir = data_dir
    app.state.reports_dir = reports_dir
    app.state.case_store = resolved_case_store
    app.state.inference_service = resolved_inference_service
    app.state.workflow_service = resolved_workflow_service
    app.state.templates = templates

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Invalid request."}
        payload = ErrorResponse(code="validation_error", detail=str(first_error.get("msg", "Invalid request.")))
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "request_error", "detail": str(exc.detail)}
        payload = ErrorResponse(code=detail.get("code", "request_error"), detail=detail.get("detail", "Request failed."))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request failure", extra={"event": "request_failure", "status": "FAILED"})
        payload = ErrorResponse(code="server_error", detail=f"Unexpected server error: {exc}")
        return JSONResponse(status_code=500, content=payload.model_dump())

    for router in (pages.router, cases.router, recordings.router, analyses.router, reports.router, admin.router):
        app.include_router(router)

    return app


app = create_app()
