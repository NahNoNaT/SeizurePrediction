from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import RuntimeConfig, runtime_config


@dataclass
class TemporalTimelinePoint:
    segment_index: int
    start_sec: float
    end_sec: float
    risk_score: float
    risk_label: str
    is_flagged: bool


@dataclass
class TemporalInterval:
    interval_index: int
    start_sec: float
    end_sec: float
    mean_risk: float
    max_risk: float
    flagged_segments_count: int
    review_status: str


@dataclass
class TemporalAnalysisResult:
    timeline: list[TemporalTimelinePoint]
    intervals: list[TemporalInterval]
    smoothed_scores: list[float]


class TemporalAnalysisService:
    def __init__(self, config: RuntimeConfig = runtime_config):
        self.config = config

    def analyze(self, segment_scores: np.ndarray, segment_times: list[tuple[float, float]]) -> TemporalAnalysisResult:
        ordered_scores, ordered_times = self.sort_segments_by_time(segment_scores, segment_times)
        smoothed_scores = self.smooth_scores(ordered_scores)
        flags = self.apply_consecutive_rule(smoothed_scores, self.config.default_threshold)

        timeline: list[TemporalTimelinePoint] = []
        for index, (times, score, is_flagged) in enumerate(zip(ordered_times, smoothed_scores, flags, strict=False)):
            timeline.append(
                TemporalTimelinePoint(
                    segment_index=index,
                    start_sec=round(times[0], 2),
                    end_sec=round(times[1], 2),
                    risk_score=float(np.clip(score, 0.0, 1.0)),
                    risk_label=self._risk_label(score),
                    is_flagged=bool(is_flagged),
                )
            )

        intervals = self.merge_intervals(timeline)
        return TemporalAnalysisResult(
            timeline=timeline,
            intervals=intervals,
            smoothed_scores=smoothed_scores,
        )

    def sort_segments_by_time(
        self,
        segment_scores: np.ndarray | list[float],
        segment_times: list[tuple[float, float]],
    ) -> tuple[list[float], list[tuple[float, float]]]:
        ordered = sorted(
            zip(segment_times, np.asarray(segment_scores, dtype=np.float32).tolist(), strict=False),
            key=lambda item: (item[0][0], item[0][1]),
        )
        ordered_times = [times for times, _ in ordered]
        ordered_scores = [float(score) for _, score in ordered]
        return ordered_scores, ordered_times

    def smooth_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        if self.config.smoothing_method.lower() == "moving_average":
            return self._moving_average(scores, radius=max(self.config.smoothing_window, 1))
        return self._ema(scores, alpha=self.config.smoothing_alpha)

    def _ema(self, scores: list[float], *, alpha: float) -> list[float]:
        smoothed = [scores[0]]
        for score in scores[1:]:
            smoothed.append(alpha * score + (1.0 - alpha) * smoothed[-1])
        return [float(np.clip(value, 0.0, 1.0)) for value in smoothed]

    def _moving_average(self, scores: list[float], *, radius: int) -> list[float]:
        smoothed: list[float] = []
        for index in range(len(scores)):
            start = max(index - radius, 0)
            end = min(index + radius + 1, len(scores))
            smoothed.append(float(np.mean(scores[start:end])))
        return [float(np.clip(value, 0.0, 1.0)) for value in smoothed]

    def apply_consecutive_rule(self, scores: list[float], threshold: float) -> list[bool]:
        flags = [score >= threshold for score in scores]
        minimum_run = max(self.config.consecutive_segments_required, 1)
        if minimum_run == 1:
            return flags

        derived = [False] * len(flags)
        run_start: int | None = None
        for index, is_high in enumerate(flags + [False]):
            if is_high and run_start is None:
                run_start = index
                continue
            if is_high:
                continue
            if run_start is not None and index - run_start >= minimum_run:
                for flagged_index in range(run_start, index):
                    derived[flagged_index] = True
            run_start = None
        return derived

    def merge_intervals(self, timeline: list[TemporalTimelinePoint]) -> list[TemporalInterval]:
        flagged = [point for point in timeline if point.is_flagged]
        if not flagged:
            return []

        groups: list[list[TemporalTimelinePoint]] = [[flagged[0]]]
        for point in flagged[1:]:
            previous = groups[-1][-1]
            if point.segment_index == previous.segment_index + 1:
                groups[-1].append(point)
            else:
                groups.append([point])

        intervals: list[TemporalInterval] = []
        for index, group in enumerate(groups, start=1):
            mean_risk = float(np.mean([point.risk_score for point in group]))
            max_risk = float(np.max([point.risk_score for point in group]))
            review_status = "Urgent review" if max_risk >= 0.80 else "Recommended review"
            intervals.append(
                TemporalInterval(
                    interval_index=index,
                    start_sec=group[0].start_sec,
                    end_sec=group[-1].end_sec,
                    mean_risk=mean_risk,
                    max_risk=max_risk,
                    flagged_segments_count=len(group),
                    review_status=review_status,
                )
            )
        return intervals

    def _smooth_scores(self, scores: list[float]) -> list[float]:
        return self.smooth_scores(scores)

    def _apply_consecutive_rule(self, scores: list[float], threshold: float) -> list[bool]:
        return self.apply_consecutive_rule(scores, threshold)

    def _group_intervals(self, timeline: list[TemporalTimelinePoint]) -> list[TemporalInterval]:
        return self.merge_intervals(timeline)

    def _risk_label(self, score: float) -> str:
        if score >= 0.75:
            return "High"
        if score >= 0.45:
            return "Moderate"
        return "Low"
