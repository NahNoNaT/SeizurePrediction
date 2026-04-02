from __future__ import annotations

from pathlib import Path


class PredictionHistoryStore:
    """Deprecated compatibility shim retained for older imports."""

    def __init__(self, history_file: Path):
        self.history_file = history_file

    def add(self, prediction) -> None:
        return None

    def list_items(self) -> list:
        return []

    def get(self, session_id: str):
        return None

    def delete(self, session_id: str) -> bool:
        return False
