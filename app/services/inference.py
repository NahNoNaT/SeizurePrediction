from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import RuntimeConfig, runtime_config
from app.services.errors import AnalysisExecutionError, CheckpointInvalidError, ModelUnavailableError
from app.services.torch_model import build_model, extract_state_dict, strip_module_prefix

logger = logging.getLogger(__name__)


@dataclass
class InferenceOutput:
    model_version: str
    risk_scores: np.ndarray
    inference_time_seconds: float
    backend_status: str


@dataclass
class TorchModelState:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_metadata: dict[str, Any]


class SeizureInferenceService:
    def __init__(self, project_root: Path, config: RuntimeConfig = runtime_config):
        self.project_root = project_root
        self.config = config
        self._state: TorchModelState | None = None

    def warmup(self) -> None:
        try:
            self.load_model()
        except (ModelUnavailableError, CheckpointInvalidError) as exc:
            logger.warning(
                "Model warmup did not complete",
                extra={"event": "model_warmup_failed", "status": exc.code},
            )

    def load_model(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is not None:
            self.config.checkpoint_path = checkpoint_path
            self._state = None

        if self._state is not None:
            return

        resolved_path = self.config.resolved_checkpoint_path(self.project_root)
        if resolved_path is None:
            self.config.inference_status = "model_unavailable"
            raise ModelUnavailableError("model_unavailable", "Model checkpoint path is not configured.")
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

            resolved_model_version = str(metadata.get("model_version", self.config.default_model_version))
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
        self._state = TorchModelState(
            model=model,
            device=device,
            checkpoint_path=resolved_path,
            checkpoint_metadata=metadata,
        )
        self.config.default_model_version = resolved_model_version
        self.config.inference_status = "model_ready"
        logger.info(
            "Model checkpoint loaded",
            extra={"event": "model_loaded", "status": "model_ready"},
        )

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        self.load_model(checkpoint_path)

    def run(self, segments: np.ndarray) -> InferenceOutput:
        started = time.perf_counter()
        scores = np.asarray(self.infer_segments(segments), dtype=np.float32)
        inference_time_seconds = round(time.perf_counter() - started, 3)
        logger.info(
            "Inference completed",
            extra={"event": "inference_completed", "status": "model_ready"},
        )
        return InferenceOutput(
            model_version=self.config.default_model_version,
            risk_scores=np.clip(scores.astype(np.float32, copy=False), 0.0, 1.0),
            inference_time_seconds=inference_time_seconds,
            backend_status=self.config.inference_status,
        )

    def infer_segments(self, segments: np.ndarray) -> list[float]:
        return self._predict_with_torch(segments).astype(np.float32, copy=False).tolist()

    def _predict_with_torch(self, segments: np.ndarray) -> np.ndarray:
        if segments.ndim != 3:
            raise AnalysisExecutionError(
                "analysis_not_executed",
                "Expected EEG segments with shape (segments, channels, samples).",
            )
        if segments.shape[1] != len(self.config.required_channel_order):
            raise AnalysisExecutionError(
                "analysis_not_executed",
                f"The loaded model expects {len(self.config.required_channel_order)} channels, but received {segments.shape[1]}.",
            )

        if self._state is None:
            self.load_model()
        assert self._state is not None

        try:
            sequences = self._build_context_sequences(segments, seq_len=self.config.default_sequence_length)
            outputs: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(sequences), self.config.inference_batch_size):
                    batch = sequences[start : start + self.config.inference_batch_size]
                    batch_tensor = torch.from_numpy(batch).to(self._state.device)
                    logits = self._state.model(batch_tensor).squeeze(-1)
                    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                    outputs.append(probabilities.astype(np.float32))
        except (RuntimeError, ValueError) as exc:
            logger.exception(
                "Inference execution failed",
                extra={"event": "inference_failed", "status": "analysis_not_executed"},
            )
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
