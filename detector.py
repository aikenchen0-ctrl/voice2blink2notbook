from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from hp_acoustic_wave.config import DetectorConfig


@dataclass
class DetectionResult:
    is_event: bool
    event_id: int
    threshold: float
    baseline: float
    mad: float


class AdaptiveWaveDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.active = False

    def _stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_energy
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_energy, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(self, time_s: float, motion_energy: float) -> DetectionResult:
        baseline, mad, threshold = self._stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= getattr(self.config, "startup_ignore_s", 0.0)
        outside_refractory = (time_s - self.last_event_time_s) >= self.config.refractory_s
        release_ratio = max(0.0, min(1.0, float(self.config.release_ratio)))
        release_level = float(baseline) + (float(threshold) - float(baseline)) * release_ratio
        if self.active and motion_energy <= release_level:
            self.active = False
        above_threshold = bool(enough_history and outside_startup and motion_energy > threshold)
        is_event = bool(above_threshold and outside_refractory and not self.active)

        if is_event:
            self.event_count += 1
            self.last_event_time_s = time_s
            self.active = True

        in_baseline_freeze = (time_s - self.last_event_time_s) < self.config.baseline_freeze_s
        warmup_outlier = bool(
            getattr(self.config, "warmup_outlier_protection", True)
            and not enough_history
            and len(self.history) >= max(3, self.config.min_history // 3)
            and outside_startup
            and motion_energy > threshold
        )
        should_update_baseline = (not enough_history) or (
            not above_threshold and not in_baseline_freeze and not self.active
        )
        if should_update_baseline and not warmup_outlier:
            self.history.append(float(motion_energy))
        return DetectionResult(
            is_event=is_event,
            event_id=self.event_count,
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
        )
