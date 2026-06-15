from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class PrimaryBlinkPeakConfig:
    min_score: float = 0.04
    min_ratio: float = 0.8
    max_score: float = 0.25
    refractory_s: float = 1.2
    startup_ignore_s: float = 2.0


@dataclass(frozen=True)
class PrimaryBlinkPeakResult:
    is_event: bool
    event_id: int
    time_s: float
    score: float
    threshold: float
    ratio: float
    method: str = "primary_blink_peak"


@dataclass(frozen=True)
class _PeakPoint:
    time_s: float
    score: float
    threshold: float


class PrimaryBlinkPeakGate:
    """Streaming local-peak gate for the visible single-tone blink score curve."""

    def __init__(self, config: PrimaryBlinkPeakConfig):
        self.config = config
        self.points: Deque[_PeakPoint] = deque(maxlen=3)
        self.event_count = 0
        self.last_event_time_s = -1e9

    def update(self, time_s: float, score: float, threshold: float) -> PrimaryBlinkPeakResult:
        point = _PeakPoint(time_s=float(time_s), score=float(score), threshold=float(threshold))
        self.points.append(point)
        if len(self.points) < 3:
            return self._no_event(point)

        previous, candidate, current = self.points
        threshold_floor = max(float(candidate.threshold), 1e-9)
        ratio = float(candidate.score) / threshold_floor
        is_local_peak = candidate.score >= previous.score and candidate.score > current.score
        outside_refractory = (candidate.time_s - self.last_event_time_s) >= float(self.config.refractory_s)
        base_candidate = (
            candidate.time_s >= float(self.config.startup_ignore_s)
            and is_local_peak
            and candidate.score >= float(self.config.min_score)
            and ratio >= float(self.config.min_ratio)
            and outside_refractory
        )
        score_above_max = (
            float(self.config.max_score) > 0.0
            and candidate.score > float(self.config.max_score)
        )
        if base_candidate and score_above_max:
            self.last_event_time_s = float(candidate.time_s)
            return self._no_event(point)
        accepted = base_candidate and not score_above_max
        if not accepted:
            return self._no_event(point)

        self.event_count += 1
        self.last_event_time_s = float(candidate.time_s)
        return PrimaryBlinkPeakResult(
            is_event=True,
            event_id=int(self.event_count),
            time_s=float(candidate.time_s),
            score=float(candidate.score),
            threshold=float(candidate.threshold),
            ratio=float(ratio),
        )

    def _no_event(self, point: _PeakPoint) -> PrimaryBlinkPeakResult:
        threshold = max(float(point.threshold), 1e-9)
        return PrimaryBlinkPeakResult(
            is_event=False,
            event_id=int(self.event_count),
            time_s=float(point.time_s),
            score=float(point.score),
            threshold=float(point.threshold),
            ratio=float(point.score) / threshold,
        )
