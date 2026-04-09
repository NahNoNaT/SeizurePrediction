from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import RuntimeConfig, runtime_config
from app.schemas import ModelSlotStatus
from app.services.errors import AnalysisExecutionError, CheckpointInvalidError, ModelUnavailableError
from app.services.torch_model import build_model, extract_state_dict, strip_module_prefix

logger = logging.getLogger(__name__)


@dataclass
class ModelInferenceResult:
    model_key: str
    model_label: str
    checkpoint_path: str
    model_version: str | None
    status: str
    backend_status: str
    risk_scores: np.ndarray | None
    inference_time_seconds: float | None
    failure_code: str | None = None
    failure_message: str | None = None


@dataclass
class InferenceOutput:
    model_version: str
    risk_scores: np.ndarray
    inference_time_seconds: float
    backend_status: str
    model_results: list[ModelInferenceResult]
    successful_model_count: int
    configured_model_count: int


@dataclass
class TorchModelState:
    model_key: str
    model_label: str
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_metadata: dict[str, Any]
    model_version: str
    expected_channel_count: int
    sequence_length: int


class SeizureInferenceService:
    def __init__(self, project_root: Path, config: RuntimeConfig = runtime_config):
        self.project_root = project_root
        self.config = config
        self._states: dict[str, TorchModelState] = {}
        self._slot_statuses: list[ModelSlotStatus] = []

    def warmup(self) -> None:
        try:
            self.load_models()
        except (ModelUnavailableError, CheckpointInvalidError) as exc:
            logger.warning(
                "Model warmup did not complete",
                extra={"event": "model_warmup_failed", "status": exc.code},
            )

    def load_model(self, checkpoint_path: str | None = None) -> None:
        self.load_models(checkpoint_path)

    def load_models(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is not None:
            self.config.checkpoint_path = checkpoint_path
            self.config.checkpoint_paths = (checkpoint_path,)
            self._states = {}
            self._slot_statuses = []

        resolved_paths = self.config.resolved_checkpoint_paths(self.project_root)
        if not resolved_paths:
            self.config.inference_status = "model_unavailable"
            raise ModelUnavailableError("model_unavailable", "Model checkpoint path is not configured.")

        loaded_states: dict[str, TorchModelState] = {}
        slot_statuses: list[ModelSlotStatus] = []
        failures: list[Exception] = []
        for index, resolved_path in enumerate(resolved_paths, start=1):
            model_key = f"model_{index}"
            model_label = f"Model {index}"
            try:
                state = self._load_state(
                    resolved_path=resolved_path,
                    model_key=model_key,
                    model_label=model_label,
                )
                loaded_states[model_key] = state
                slot_statuses.append(
                    ModelSlotStatus(
                        model_key=model_key,
                        model_label=model_label,
                        checkpoint_path=str(resolved_path),
                        status="READY",
                        backend_status="model_ready",
                        model_version=state.model_version,
                        detail=None,
                    )
                )
            except (ModelUnavailableError, CheckpointInvalidError) as exc:
                failures.append(exc)
                slot_statuses.append(
                    ModelSlotStatus(
                        model_key=model_key,
                        model_label=model_label,
                        checkpoint_path=str(resolved_path),
                        status="FAILED",
                        backend_status=exc.code,
                        model_version=None,
                        detail=exc.public_detail,
                    )
                )
                logger.warning(
                    "Model slot failed to load",
                    extra={"event": "model_slot_failed", "status": exc.code, "code": exc.code},
                )

        self._slot_statuses = slot_statuses
        if not loaded_states:
            self.config.inference_status = "model_unavailable"
            if failures and isinstance(failures[-1], (ModelUnavailableError, CheckpointInvalidError)):
                raise failures[-1]
            raise ModelUnavailableError("model_unavailable", "No model checkpoints could be loaded.")

        self._states = loaded_states
        self.config.default_model_version = "ensemble-mean"
        self.config.inference_status = "model_ready"
        logger.info(
            "Model checkpoints loaded",
            extra={"event": "model_loaded", "status": "model_ready"},
        )

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        self.load_models(checkpoint_path)

    def run(self, segments: np.ndarray) -> InferenceOutput:
        if not self._states:
            self.load_models()
        if not self._states:
            raise ModelUnavailableError("model_unavailable", "No model checkpoints are available for inference.")

        model_results: list[ModelInferenceResult] = []
        successful_scores: list[np.ndarray] = []

        for slot in self._slot_statuses:
            state = self._states.get(slot.model_key)
            if state is None:
                model_results.append(
                    ModelInferenceResult(
                        model_key=slot.model_key,
                        model_label=slot.model_label,
                        checkpoint_path=slot.checkpoint_path,
                        model_version=slot.model_version,
                        status="FAILED",
                        backend_status=slot.backend_status,
                        risk_scores=None,
                        inference_time_seconds=None,
                        failure_code=slot.backend_status,
                        failure_message=slot.detail,
                    )
                )
                continue
            started = time.perf_counter()
            try:
                scores = self._predict_with_torch(state, segments)
                elapsed = round(time.perf_counter() - started, 3)
                successful_scores.append(scores)
                model_results.append(
                    ModelInferenceResult(
                        model_key=state.model_key,
                        model_label=state.model_label,
                        checkpoint_path=str(state.checkpoint_path),
                        model_version=state.model_version,
                        status="COMPLETED",
                        backend_status="model_ready",
                        risk_scores=np.clip(scores.astype(np.float32, copy=False), 0.0, 1.0),
                        inference_time_seconds=elapsed,
                    )
                )
            except (AnalysisExecutionError, ModelUnavailableError, CheckpointInvalidError) as exc:
                logger.exception(
                    "Inference execution failed",
                    extra={"event": "inference_failed", "status": exc.code},
                )
                model_results.append(
                    ModelInferenceResult(
                        model_key=state.model_key,
                        model_label=state.model_label,
                        checkpoint_path=str(state.checkpoint_path),
                        model_version=state.model_version,
                        status="FAILED",
                        backend_status=exc.code,
                        risk_scores=None,
                        inference_time_seconds=None,
                        failure_code=exc.code,
                        failure_message=exc.public_detail,
                    )
                )

        if not successful_scores:
            raise AnalysisExecutionError("analysis_not_executed", "None of the configured models produced a result.")

        aggregate_scores = np.mean(np.stack(successful_scores, axis=0), axis=0).astype(np.float32, copy=False)
        total_time = round(
            sum(result.inference_time_seconds or 0.0 for result in model_results if result.status == "COMPLETED"),
            3,
        )
        logger.info(
            "Inference completed",
            extra={"event": "inference_completed", "status": "model_ready"},
        )
        return InferenceOutput(
            model_version="ensemble-mean",
            risk_scores=np.clip(aggregate_scores, 0.0, 1.0),
            inference_time_seconds=total_time,
            backend_status="model_ready",
            model_results=model_results,
            successful_model_count=len(successful_scores),
            configured_model_count=len(self._slot_statuses),
        )

    def infer_segments(self, segments: np.ndarray) -> list[float]:
        return self.run(segments).risk_scores.astype(np.float32, copy=False).tolist()

    def model_slot_statuses(self) -> list[ModelSlotStatus]:
        return list(self._slot_statuses)

    def _load_state(self, *, resolved_path: Path, model_key: str, model_label: str) -> TorchModelState:
        if not resolved_path.exists():
            self.config.inference_status = "model_unavailable"
            raise ModelUnavailableError(
                "model_unavailable",
                f"Model checkpoint was not found at '{resolved_path}'.",
            )

        device = self._resolve_device()
        try:
            checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
            state_dict, metadata = extract_state_dict(checkpoint)
            state_dict = strip_module_prefix(state_dict)

            resolved_model_version = str(metadata.get("model_version", "raw-window-cnn-seqbigru-v1"))
            resolved_channels = int(
                metadata.get("expected_channel_count", metadata.get("num_channels", len(self.config.required_channel_order)))
            )
            resolved_seq_len = int(
                metadata.get("sequence_length", metadata.get("seq_len", self.config.default_sequence_length))
            )

            model = build_model(
                model_version=resolved_model_version,
                num_channels=resolved_channels,
                seq_len=resolved_seq_len,
            )
            incompatible = model.load_state_dict(state_dict, strict=self.config.strict_checkpoint_loading)
            if not self.config.strict_checkpoint_loading:
                missing = list(getattr(incompatible, "missing_keys", []))
                unexpected = list(getattr(incompatible, "unexpected_keys", []))
                if missing or unexpected:
                    raise ValueError(
                        "Checkpoint did not fully match the configured model architecture. "
                        f"Missing keys: {missing}. Unexpected keys: {unexpected}."
                    )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            self.config.inference_status = "checkpoint_invalid"
            logger.exception(
                "Checkpoint loading failed",
                extra={"event": "model_load_failed", "status": "checkpoint_invalid"},
            )
            raise CheckpointInvalidError("checkpoint_invalid", f"Unable to load the model checkpoint: {exc}") from exc

        model.to(device)
        model.eval()
        return TorchModelState(
            model_key=model_key,
            model_label=model_label,
            model=model,
            device=device,
            checkpoint_path=resolved_path,
            checkpoint_metadata=metadata,
            model_version=resolved_model_version,
            expected_channel_count=resolved_channels,
            sequence_length=resolved_seq_len,
        )

    def _predict_with_torch(self, state: TorchModelState, segments: np.ndarray) -> np.ndarray:
        if segments.ndim != 3:
            raise AnalysisExecutionError(
                "analysis_not_executed",
                "Expected EEG segments with shape (segments, channels, samples).",
            )
        if segments.shape[1] != state.expected_channel_count:
            raise AnalysisExecutionError(
                "analysis_not_executed",
                f"The loaded model expects {state.expected_channel_count} channels, but received {segments.shape[1]}.",
            )

        try:
            sequences = self._build_context_sequences(segments, seq_len=state.sequence_length)
            outputs: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(sequences), self.config.inference_batch_size):
                    batch = sequences[start : start + self.config.inference_batch_size]
                    batch_tensor = torch.from_numpy(batch).to(state.device)
                    logits = state.model(batch_tensor).squeeze(-1)
                    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                    outputs.append(probabilities.astype(np.float32))
        except (RuntimeError, ValueError) as exc:
            raise AnalysisExecutionError("analysis_not_executed", f"Unable to execute model inference: {exc}") from exc

        if not outputs:
            raise AnalysisExecutionError("analysis_not_executed", "The model did not produce any seizure-risk scores.")
        concatenated = np.concatenate(outputs, axis=0)
        if concatenated.shape[0] != segments.shape[0]:
            raise AnalysisExecutionError(
                "analysis_not_executed",
                "Model output length did not match the number of EEG segments.",
            )
        return concatenated

    def _resolve_device(self) -> torch.device:
        preference = self.config.model_device_preference.strip().lower()
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise ModelUnavailableError(
                    "model_unavailable",
                    "CUDA was requested for inference, but no CUDA device is available.",
                )
            return torch.device("cuda")
        if preference == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _build_context_sequences(self, segments: np.ndarray, seq_len: int) -> np.ndarray:
        sequences = np.zeros((segments.shape[0], seq_len, segments.shape[1], segments.shape[2]), dtype=np.float32)
        for index in range(segments.shape[0]):
            start = max(0, index - seq_len + 1)
            context = segments[start : index + 1]
            sequences[index, -context.shape[0] :] = context
            if context.shape[0] < seq_len:
                sequences[index, : seq_len - context.shape[0]] = context[0]
        return sequences
