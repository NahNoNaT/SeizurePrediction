from __future__ import annotations

from fastapi import Request

from app.services.clinical_workflow import ClinicalAnalysisService
from app.services.inference import SeizureInferenceService
from app.services.store import ClinicalCaseStore


def get_case_store(request: Request) -> ClinicalCaseStore:
    return request.app.state.case_store


def get_workflow_service(request: Request) -> ClinicalAnalysisService:
    return request.app.state.workflow_service


def get_inference_service(request: Request) -> SeizureInferenceService:
    return request.app.state.inference_service


def get_templates(request: Request):
    return request.app.state.templates
