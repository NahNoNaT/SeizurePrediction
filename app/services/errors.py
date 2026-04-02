from __future__ import annotations


class ClinicalWorkflowError(Exception):
    def __init__(self, code: str, detail: str, *, public_detail: str | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.public_detail = public_detail or detail


class EEGValidationError(ClinicalWorkflowError):
    pass


class ModelUnavailableError(ClinicalWorkflowError):
    pass


class CheckpointInvalidError(ClinicalWorkflowError):
    pass


class AnalysisExecutionError(ClinicalWorkflowError):
    pass
