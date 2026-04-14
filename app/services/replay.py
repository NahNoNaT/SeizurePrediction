from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import mne
import numpy as np

from app.config import RuntimeConfig, runtime_config
from app.schemas import ReplaySessionStateResponse, ReplayTimelinePoint, ReplayUploadResponse
from app.services.legacy_joblib import LegacyDependencyError, LegacyFeatureExtractorUnavailableError, LegacyJoblibPredictionService
from app.web import downsample_items
from preprocessing import _is_eeg_like_channel

MAX_REPLAY_TIMELINE_POINTS = 360


@dataclass
class ReplaySession:
    session_id: str
    file_name: str
    file_path: Path
    sampling_rate: float
    total_duration_sec: float
    available_channels: list[str]
    created_at: datetime
    status: str = "uploaded"
    window_sec: float = 10.0
    hop_sec: float = 2.5
    replay_speed: float = 5.0
    started_monotonic: float | None = None
    processed_windows: int = 0
    total_windows: int = 0
    replay_position_sec: float = 0.0
    latest_risk_score: float | None = None
    latest_top_channel: str | None = None
    error: str | None = None
    timeline: list[ReplayTimelinePoint] = field(default_factory=list)
    raw: mne.io.BaseRaw | None = None
    window_starts: list[float] = field(default_factory=list)


class ReplaySessionService:
    def __init__(self, project_root: Path, config: RuntimeConfig = runtime_config):
        self.config = config
        self.project_root = Path(project_root)
        self._legacy_service = LegacyJoblibPredictionService(project_root=self.project_root)
        self._replay_model_id = os.getenv("SEIZURE_REPLAY_LEGACY_MODEL_ID", "universal:dt:combined").strip().lower()
        self._sessions: dict[str, ReplaySession] = {}
        self._lock = Lock()

    def create_session(self, *, file_path: Path, file_name: str) -> ReplayUploadResponse:
        raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose="ERROR")
        try:
            sampling_rate = float(raw.info["sfreq"])
            total_duration_sec = float(raw.n_times / sampling_rate) if sampling_rate > 0 else 0.0
            channels = [name for name in raw.ch_names if _is_eeg_like_channel(raw, name)]
            if not channels:
                channels = list(raw.ch_names[:1])
        finally:
            raw.close()

        session = ReplaySession(
            session_id=str(uuid4()),
            file_name=file_name,
            file_path=file_path,
            sampling_rate=sampling_rate,
            total_duration_sec=round(total_duration_sec, 3),
            available_channels=channels,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return ReplayUploadResponse(
            session_id=session.session_id,
            file_name=session.file_name,
            status=session.status,
            total_duration_sec=session.total_duration_sec,
            sampling_rate=session.sampling_rate,
            available_channels=session.available_channels,
            message="EDF replay session created. Start replay to stream sliding-window risk scores.",
        )

    def start_session(
        self,
        session_id: str,
        *,
        window_sec: float,
        hop_sec: float,
        replay_speed: float,
    ) -> ReplaySessionStateResponse:
        with self._lock:
            session = self._require_session(session_id)
            session.window_sec = window_sec
            session.hop_sec = hop_sec
            session.replay_speed = replay_speed
            session.window_starts = self._build_window_starts(session.total_duration_sec, window_sec, hop_sec)
            session.total_windows = len(session.window_starts)
            session.started_monotonic = time.monotonic()
            session.processed_windows = 0
            session.replay_position_sec = 0.0
            session.latest_risk_score = None
            session.latest_top_channel = None
            session.error = None
            session.timeline = []
            session.status = "running"
            self._advance_session(session)
            return self._serialize_session(session)

    def stop_session(self, session_id: str) -> ReplaySessionStateResponse:
        with self._lock:
            session = self._require_session(session_id)
            if session.status == "running":
                session.status = "stopped"
            return self._serialize_session(session)

    def get_session_state(self, session_id: str) -> ReplaySessionStateResponse:
        with self._lock:
            session = self._require_session(session_id)
            self._advance_session(session)
            return self._serialize_session(session)

    def _require_session(self, session_id: str) -> ReplaySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def _build_window_starts(self, total_duration_sec: float, window_sec: float, hop_sec: float) -> list[float]:
        if total_duration_sec <= window_sec:
            return [0.0]
        starts = list(np.arange(0.0, max(total_duration_sec - window_sec, 0.0) + 1e-6, hop_sec, dtype=np.float32))
        last_start = max(total_duration_sec - window_sec, 0.0)
        if not starts or abs(starts[-1] - last_start) > 1e-3:
            starts.append(last_start)
        return [round(float(value), 3) for value in starts]

    def _advance_session(self, session: ReplaySession) -> None:
        if session.status != "running" or session.started_monotonic is None:
            return
        try:
            elapsed = max(time.monotonic() - session.started_monotonic, 0.0)
            target_windows = min(
                session.total_windows,
                int(math.floor((elapsed * session.replay_speed) / session.hop_sec)) + 1,
            )
            while session.processed_windows < target_windows:
                self._process_next_window(session)
            if session.processed_windows >= session.total_windows:
                session.status = "completed"
                self._close_raw(session)
        except Exception as exc:
            session.status = "failed"
            session.error = str(exc)
            self._close_raw(session)

    def _process_next_window(self, session: ReplaySession) -> None:
        window_index = session.processed_windows
        start_sec = session.window_starts[window_index]
        end_sec = min(start_sec + session.window_sec, session.total_duration_sec)
        try:
            prediction = self._legacy_service.predict(
                session.file_path,
                model_id=self._replay_model_id,
                start_sec=float(start_sec),
                duration_sec=float(session.window_sec),
                max_models=1,
            )
            risk_score = float(prediction.summary.average_probability or 0.0)
            top_channel = prediction.selected_channel
        except (LegacyFeatureExtractorUnavailableError, LegacyDependencyError, RuntimeError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Replay prediction failed: {exc}") from exc

        timeline_point = ReplayTimelinePoint(
            window_index=window_index,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            risk_score=round(risk_score, 6),
            risk_label=self._risk_label(risk_score),
            is_flagged=risk_score >= self.config.default_threshold,
            top_channel=top_channel,
        )
        session.timeline.append(timeline_point)
        session.latest_risk_score = risk_score
        session.latest_top_channel = top_channel or None
        session.processed_windows += 1
        session.replay_position_sec = round(end_sec, 3)

    def _risk_label(self, risk_score: float) -> str:
        if risk_score >= 0.75:
            return "High"
        if risk_score >= 0.45:
            return "Moderate"
        return "Low"

    def _serialize_session(self, session: ReplaySession) -> ReplaySessionStateResponse:
        timeline = downsample_items(session.timeline, MAX_REPLAY_TIMELINE_POINTS)
        return ReplaySessionStateResponse(
            session_id=session.session_id,
            file_name=session.file_name,
            status=session.status,
            total_duration_sec=session.total_duration_sec,
            sampling_rate=session.sampling_rate,
            window_sec=session.window_sec,
            hop_sec=session.hop_sec,
            replay_speed=session.replay_speed,
            available_channels=session.available_channels,
            processed_windows=session.processed_windows,
            total_windows=session.total_windows,
            replay_position_sec=session.replay_position_sec,
            latest_risk_score=session.latest_risk_score,
            latest_top_channel=session.latest_top_channel,
            error=session.error,
            timeline=timeline,
        )

    def _close_raw(self, session: ReplaySession) -> None:
        if session.raw is not None:
            session.raw.close()
            session.raw = None
