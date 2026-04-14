from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from model_registry import get_registered_models
from preprocessing import DEFAULT_DURATION_SEC, DEFAULT_START_SEC, PreparedEEGSegment, prepare_segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleSummary:
    majority_vote: str
    average_probability: float | None
    num_models_predicting_seizure: int
    num_models_total: int
    confidence_note: str


@dataclass(frozen=True)
class ComparisonResult:
    selected_channel: str
    sampling_rate: float
    duration_used_sec: float
    models: list[dict]
    ensemble_summary: EnsembleSummary
    warnings: list[str]


def run_benchmark(
    edf_path: str | Path,
    *,
    channel: str | None = None,
    start_sec: float = DEFAULT_START_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> ComparisonResult:
    prepared = prepare_segment(
        edf_path,
        requested_channel=channel,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    warnings = list(prepared.warnings)
    model_results: list[dict] = []

    for model in get_registered_models():
        try:
            result = model.timed_predict(
                prepared.raw_signal,
                prepared.sampling_rate,
                metadata={"prepared_segment": prepared},
            )
            model_results.append(
                {
                    "model_name": result.model_name,
                    "predicted_label": result.predicted_label,
                    "seizure_probability": result.seizure_probability,
                    "raw_score": result.raw_score,
                    "inference_time_ms": result.inference_time_ms,
                    "notes": result.notes,
                }
            )
        except Exception as exc:
            logger.exception("Model benchmark failed", extra={"event": "benchmark_model_failed", "status": model.model_name})
            warnings.append(f"{model.model_name} failed: {exc}")
            model_results.append(
                {
                    "model_name": model.model_name,
                    "predicted_label": "error",
                    "seizure_probability": None,
                    "raw_score": None,
                    "inference_time_ms": None,
                    "notes": f"{model.notes} Load/inference failed: {exc}",
                }
            )

    warnings.append(
        "Probabilities are model-specific and not clinically calibrated against each other. Compare them as benchmark scores, not absolute risk."
    )
    ensemble = _build_ensemble(model_results)
    logger.info(
        "Completed EEG benchmark request",
        extra={"event": "benchmark_complete", "status": ensemble.majority_vote},
    )
    return ComparisonResult(
        selected_channel=prepared.selected_channel,
        sampling_rate=prepared.sampling_rate,
        duration_used_sec=prepared.duration_used_sec,
        models=model_results,
        ensemble_summary=ensemble,
        warnings=warnings,
    )


def _build_ensemble(model_results: list[dict]) -> EnsembleSummary:
    successful = [item for item in model_results if isinstance(item.get("seizure_probability"), (float, int))]
    if not successful:
        return EnsembleSummary(
            majority_vote="unavailable",
            average_probability=None,
            num_models_predicting_seizure=0,
            num_models_total=len(model_results),
            confidence_note="No model produced a usable prediction.",
        )

    seizure_votes = sum(1 for item in successful if item["predicted_label"] == "seizure")
    average_probability = float(sum(float(item["seizure_probability"]) for item in successful) / len(successful))
    majority_vote = "seizure" if seizure_votes >= (len(successful) / 2.0) else "non-seizure"
    confidence_note = _confidence_note(seizure_votes, len(successful), average_probability)
    return EnsembleSummary(
        majority_vote=majority_vote,
        average_probability=round(average_probability, 6),
        num_models_predicting_seizure=seizure_votes,
        num_models_total=len(model_results),
        confidence_note=confidence_note,
    )


def _confidence_note(seizure_votes: int, num_successful: int, average_probability: float) -> str:
    vote_margin = abs((2 * seizure_votes) - num_successful) / max(num_successful, 1)
    if vote_margin >= 0.75 and abs(average_probability - 0.5) >= 0.25:
        return "High agreement among the available models."
    if vote_margin >= 0.4:
        return "Moderate agreement among the available models."
    return "Low agreement; treat this benchmark as exploratory."
