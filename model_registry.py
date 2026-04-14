from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_registered_models():
    # Prediction now runs from local LOSO/UNIVERSAL joblib artifacts only.
    return []


def get_model_catalog() -> list[dict[str, str]]:
    catalog = []
    for model in get_registered_models():
        catalog.append({"model_name": model.model_name, "notes": model.notes})
    return catalog
