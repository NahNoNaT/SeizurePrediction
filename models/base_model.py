from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    predicted_label: str
    seizure_probability: float
    raw_score: float
    inference_time_ms: float
    notes: str


class BaseBenchmarkModel(ABC):
    model_name: str
    notes: str

    def timed_predict(self, signal, sampling_rate: float, metadata: dict[str, Any] | None = None) -> ModelPrediction:
        started = perf_counter()
        prediction = self.predict(signal, sampling_rate, metadata=metadata)
        elapsed_ms = round((perf_counter() - started) * 1000.0, 3)
        return ModelPrediction(
            model_name=prediction.model_name,
            predicted_label=prediction.predicted_label,
            seizure_probability=prediction.seizure_probability,
            raw_score=prediction.raw_score,
            inference_time_ms=elapsed_ms,
            notes=prediction.notes,
        )

    @abstractmethod
    def load_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, signal, sampling_rate: float, metadata: dict[str, Any] | None = None) -> ModelPrediction:
        raise NotImplementedError
