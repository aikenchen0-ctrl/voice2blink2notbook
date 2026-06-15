from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Optional

import numpy as np

from hp_acoustic_wave.dsp import ChunkFeature, unwrap_delta


@dataclass
class BlinkDetectionConfig:
    method: str = "blinklistener"
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 2.5
    min_score: float = 0.006
    refractory_s: float = 1.05
    baseline_freeze_s: float = 0.60
    short_window: int = 9
    phase_pair_lag: int = 3
    phase_step_floor: float = 0.015
    absolute_score_floor: float = 0.0
    startup_ignore_s: float = 0.0
    release_ratio: float = 0.4
    twinkle_peak_gate_enabled: bool = True
    twinkle_peak_min_ratio: float = 1.0
    twinkle_min_peak_score: float = 0.035
    twinkle_max_peak_score: float = 1.2
    twinkle_min_motion_energy: float = 0.0
    twinkle_max_motion_energy: float = 0.22
    twinkle_min_sign_changes: int = 0
    twinkle_max_sign_changes: int = 2
    twinkle_large_motion_score: float = 1.5
    twinkle_large_motion_energy: float = 0.18
    twinkle_large_motion_suppress_s: float = 0.75


@dataclass
class BlinkDetectionResult:
    is_event: bool
    event_id: int
    method: str
    score: float
    threshold: float
    baseline: float
    mad: float
    metrics: Dict[str, float] = field(default_factory=dict)


class _RobustEventGate:
    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.active = False

    def stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_score
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(self, time_s: float, score: float, force_event: bool = False):
        baseline, mad, threshold = self.stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= self.config.startup_ignore_s
        event_level = threshold
        if force_event and self.config.absolute_score_floor > 0.0:
            event_level = min(event_level, self.config.absolute_score_floor)
        release_level = event_level * self.config.release_ratio
        if self.active and score <= release_level:
            self.active = False
        above_threshold = bool(enough_history and outside_startup and (score > threshold or force_event))
        outside_refractory = (time_s - self.last_event_time_s) >= self.config.refractory_s
        is_event = bool(above_threshold and outside_refractory and not self.active)

        if is_event:
            self.event_count += 1
            self.last_event_time_s = time_s
            self.active = True

        in_baseline_freeze = (time_s - self.last_event_time_s) < self.config.baseline_freeze_s
        warmup_outlier = bool(
            not enough_history
            and len(self.history) >= max(3, self.config.min_history // 3)
            and outside_startup
            and score > threshold
        )
        should_update_baseline = (
            (not enough_history and not warmup_outlier)
            or (enough_history and not above_threshold and not in_baseline_freeze and not self.active)
        )
        if should_update_baseline:
            self.history.append(float(score))

        return is_event, self.event_count, baseline, mad, threshold


@dataclass
class _TwinkleGatePoint:
    time_s: float
    score: float
    threshold: float
    above_threshold: bool
    motion_energy: float
    sign_changes: int


class _TwinklePeakEventGate:
    """Local peak/segment gate for Twinkle-style phase trajectories."""

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.points: Deque[_TwinkleGatePoint] = deque(maxlen=3)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.suppress_until_s = -1e9
        self.active = False
        self.last_peak_point: Optional[_TwinkleGatePoint] = None
        self.last_gate_metrics: Dict[str, float] = {}

    def stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_score
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(
        self,
        time_s: float,
        score: float,
        motion_energy: float,
        sign_changes: int,
    ):
        baseline, mad, threshold = self.stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= self.config.startup_ignore_s
        above_threshold = bool(enough_history and outside_startup and score > threshold)
        suppressing_large_motion = bool(
            score >= self.config.twinkle_large_motion_score
            and motion_energy >= self.config.twinkle_large_motion_energy
        )
        if suppressing_large_motion:
            self.suppress_until_s = max(
                self.suppress_until_s,
                time_s + self.config.twinkle_large_motion_suppress_s,
            )
        release_ratio = max(0.0, min(1.0, float(self.config.release_ratio)))
        release_level = float(baseline) + (float(threshold) - float(baseline)) * release_ratio
        if self.active and score <= release_level:
            self.active = False

        point = _TwinkleGatePoint(
            time_s=float(time_s),
            score=float(score),
            threshold=float(threshold),
            above_threshold=above_threshold,
            motion_energy=float(motion_energy),
            sign_changes=int(sign_changes),
        )
        self.points.append(point)

        is_event = False
        candidate = None
        candidate_is_local_peak = False
        candidate_is_rising_edge = False
        if len(self.points) == 3:
            previous, middle, current = self.points
            candidate_is_local_peak = middle.score >= previous.score and middle.score > current.score
            candidate_is_rising_edge = (not previous.above_threshold) and middle.above_threshold
            if candidate_is_local_peak or candidate_is_rising_edge:
                candidate = middle

        if candidate is not None:
            event_level = max(
                candidate.threshold * self.config.twinkle_peak_min_ratio,
                self.config.twinkle_min_peak_score,
            )
            score_floor_ok = candidate.score >= event_level
            score_ceiling_ok = (
                self.config.twinkle_max_peak_score <= 0.0
                or candidate.score <= self.config.twinkle_max_peak_score
            )
            motion_floor_ok = candidate.motion_energy >= self.config.twinkle_min_motion_energy
            motion_ceiling_ok = candidate.motion_energy <= self.config.twinkle_max_motion_energy
            sign_changes_floor_ok = candidate.sign_changes >= self.config.twinkle_min_sign_changes
            sign_changes_ceiling_ok = candidate.sign_changes <= self.config.twinkle_max_sign_changes
            suppress_ok = candidate.time_s >= self.suppress_until_s
            refractory_ok = (candidate.time_s - self.last_event_time_s) >= self.config.refractory_s
            candidate_ok = bool(
                candidate.above_threshold
                and score_floor_ok
                and score_ceiling_ok
                and motion_floor_ok
                and motion_ceiling_ok
                and sign_changes_floor_ok
                and sign_changes_ceiling_ok
                and suppress_ok
                and refractory_ok
                and not self.active
            )
            if candidate_ok:
                self.event_count += 1
                self.last_event_time_s = candidate.time_s
                self.last_peak_point = candidate
                self.active = True
                is_event = True
        else:
            event_level = max(
                threshold * self.config.twinkle_peak_min_ratio,
                self.config.twinkle_min_peak_score,
            )
            score_floor_ok = False
            score_ceiling_ok = True
            motion_floor_ok = motion_energy >= self.config.twinkle_min_motion_energy
            motion_ceiling_ok = motion_energy <= self.config.twinkle_max_motion_energy
            sign_changes_floor_ok = sign_changes >= self.config.twinkle_min_sign_changes
            sign_changes_ceiling_ok = sign_changes <= self.config.twinkle_max_sign_changes
            suppress_ok = time_s >= self.suppress_until_s
            refractory_ok = (time_s - self.last_event_time_s) >= self.config.refractory_s

        warmup_outlier = bool(
            not enough_history
            and len(self.history) >= max(3, self.config.min_history // 3)
            and outside_startup
            and score > threshold
        )
        should_update_baseline = (
            (not enough_history and not warmup_outlier)
            or (enough_history and not above_threshold)
        )
        if should_update_baseline:
            self.history.append(float(score))

        self.last_gate_metrics = {
            "twinkle_candidate_peak": float(candidate is not None),
            "twinkle_candidate_accepted": float(is_event),
            "twinkle_candidate_local_peak": float(candidate_is_local_peak),
            "twinkle_candidate_rising_edge": float(candidate_is_rising_edge),
            "twinkle_large_motion_suppressed": float(suppressing_large_motion),
            "twinkle_suppress_until_s": float(self.suppress_until_s),
            "twinkle_candidate_event_level": float(event_level),
            "twinkle_candidate_active": float(self.active),
            "twinkle_release_level": float(release_level),
            "twinkle_reject_low_score": float(candidate is not None and not score_floor_ok),
            "twinkle_reject_high_score": float(candidate is not None and not score_ceiling_ok),
            "twinkle_reject_low_motion": float(candidate is not None and not motion_floor_ok),
            "twinkle_reject_large_motion": float(candidate is not None and not motion_ceiling_ok),
            "twinkle_reject_few_reversals": float(candidate is not None and not sign_changes_floor_ok),
            "twinkle_reject_many_reversals": float(candidate is not None and not sign_changes_ceiling_ok),
            "twinkle_reject_suppressed": float(candidate is not None and not suppress_ok),
            "twinkle_reject_refractory": float(candidate is not None and not refractory_ok),
            "twinkle_reject_active": float(
                candidate is not None
                and not is_event
                and self.active
            ),
            "twinkle_peak_time_s": float(candidate.time_s) if candidate is not None else float(time_s),
            "twinkle_peak_score": float(candidate.score) if candidate is not None else float(score),
            "twinkle_peak_threshold": float(candidate.threshold) if candidate is not None else float(threshold),
            "twinkle_peak_motion_energy": (
                float(candidate.motion_energy) if candidate is not None else float(motion_energy)
            ),
            "twinkle_peak_sign_changes": (
                float(candidate.sign_changes) if candidate is not None else float(sign_changes)
            ),
        }

        return is_event, self.event_count, baseline, mad, threshold


class BlinkListenerBlinkDetector:
    """BlinkListener-inspired I/Q viewing-position bump detector.

    This is a practical first pass for laptop single-tone data. It keeps the
    BlinkListener idea that the useful bump may be clearer from a local I/Q
    viewing position than from the origin, then runs a robust LEVD-like bump
    gate on that viewing-position amplitude.
    """

    method = "blinklistener"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.gate = _RobustEventGate(config)
        self.baseline_i: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_q: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_amplitude: Deque[float] = deque(maxlen=config.history_size)
        self.amplitude_window: Deque[float] = deque(maxlen=config.short_window)
        self.projection_window: Deque[float] = deque(maxlen=config.short_window)

    def _center(self, feature: ChunkFeature):
        if not self.baseline_i:
            return feature.i_value, feature.q_value
        return float(np.median(self.baseline_i)), float(np.median(self.baseline_q))

    def _baseline_amplitude(self, feature: ChunkFeature) -> float:
        if not self.baseline_amplitude:
            return float(feature.amplitude)
        return float(np.median(self.baseline_amplitude))

    def _best_viewing_projection(self, feature: ChunkFeature, center_i: float, center_q: float) -> float:
        if len(self.baseline_i) < 3:
            return 0.0

        points_i = np.asarray(self.baseline_i, dtype=np.float64) - center_i
        points_q = np.asarray(self.baseline_q, dtype=np.float64) - center_q
        delta_i = float(feature.i_value - center_i)
        delta_q = float(feature.q_value - center_q)

        best = 0.0
        for angle in np.linspace(0.0, math.pi, num=12, endpoint=False):
            direction_i = math.cos(float(angle))
            direction_q = math.sin(float(angle))
            baseline_projection = points_i * direction_i + points_q * direction_q
            current_projection = delta_i * direction_i + delta_q * direction_q
            best = max(best, abs(float(current_projection - np.median(baseline_projection))))
        return float(best)

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        center_i, center_q = self._center(feature)
        baseline_amplitude = self._baseline_amplitude(feature)
        amplitude_bump = float(abs(feature.amplitude - baseline_amplitude))
        best_projection = self._best_viewing_projection(feature, center_i, center_q)
        gated_projection = min(best_projection, amplitude_bump * 2.0)
        phase_stable_projection = best_projection if abs(feature.phase_delta) <= 0.12 else gated_projection

        self.amplitude_window.append(amplitude_bump)
        self.projection_window.append(phase_stable_projection)

        if len(self.amplitude_window) >= 3:
            amplitude_range = float(max(self.amplitude_window) - min(self.amplitude_window))
            projection_range = float(max(self.projection_window) - min(self.projection_window))
        else:
            amplitude_range = amplitude_bump
            projection_range = phase_stable_projection
        raw_score = max(amplitude_bump, amplitude_range, phase_stable_projection, 0.75 * projection_range)
        score_scale = max(baseline_amplitude, 1e-4)
        score = raw_score / score_scale

        is_event, event_id, baseline, mad, threshold = self.gate.update(feature.time_s, score)

        in_baseline_freeze = (feature.time_s - self.gate.last_event_time_s) < self.config.baseline_freeze_s
        should_update_center = (not is_event) and (not in_baseline_freeze or len(self.baseline_i) < self.config.min_history)
        if should_update_center:
            self.baseline_i.append(float(feature.i_value))
            self.baseline_q.append(float(feature.q_value))
            self.baseline_amplitude.append(float(feature.amplitude))

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            metrics={
                "amplitude_bump": amplitude_bump,
                "amplitude_range": amplitude_range,
                "viewing_projection": best_projection,
                "gated_projection": gated_projection,
                "phase_stable_projection": phase_stable_projection,
                "viewing_amplitude": phase_stable_projection,
                "viewing_range": projection_range,
                "raw_viewing_score": raw_score,
                "score_scale": score_scale,
                "relative_viewing_score": score,
                "center_i": float(center_i),
                "center_q": float(center_q),
                "baseline_amplitude": baseline_amplitude,
                "origin_amplitude": float(feature.amplitude),
            },
        )


class TwinkleTwinkleBlinkDetector:
    """TwinkleTwinkle-inspired phase-pair trajectory detector.

    The paper's full method uses FMCW chirps and phase differences between
    chirp endpoints. Our HP prototype currently transmits one tone, so this
    detector uses the same phase-pair trajectory idea on the unwrapped
    baseband phase stream. The output is a real-time proxy that preserves the
    algorithm boundary for a future FMCW upgrade.
    """

    method = "twinkle"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        if config.twinkle_peak_gate_enabled:
            self.gate = _TwinklePeakEventGate(config)
        else:
            self.gate = _RobustEventGate(config)
        self.unwrapped_phases: Deque[float] = deque(maxlen=config.history_size)
        self.phase_window: Deque[float] = deque(maxlen=config.short_window)
        self.phase_steps: Deque[float] = deque(maxlen=config.short_window)
        self.previous_phase = None
        self.current_unwrapped_phase = 0.0

    def _append_unwrapped_phase(self, feature: ChunkFeature):
        if self.previous_phase is None:
            self.current_unwrapped_phase = float(feature.phase)
            phase_step = 0.0
        else:
            phase_step = unwrap_delta(feature.phase, self.previous_phase)
            self.current_unwrapped_phase += phase_step
        self.previous_phase = float(feature.phase)
        self.unwrapped_phases.append(float(self.current_unwrapped_phase))
        self.phase_window.append(float(self.current_unwrapped_phase))
        self.phase_steps.append(float(phase_step))
        return float(self.current_unwrapped_phase), float(phase_step)

    def _trajectory_score(self):
        if len(self.phase_window) < max(4, min(self.config.short_window, 4)):
            return 0.0, 0.0, 0.0, 0

        phases = np.asarray(self.phase_window, dtype=np.float64)
        steps = np.asarray(self.phase_steps, dtype=np.float64)
        trajectory_span = float(np.max(phases) - np.min(phases))
        if steps.size >= 2:
            acceleration = np.diff(steps)
            acceleration_rms = float(np.sqrt(np.mean(np.square(acceleration))))
        else:
            acceleration_rms = 0.0

        step_floor = max(0.0, self.config.phase_step_floor)
        signs: List[int] = []
        for step in steps:
            if abs(float(step)) < step_floor:
                continue
            signs.append(1 if step > 0.0 else -1)

        sign_changes = sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])
        if sign_changes == 0:
            if acceleration_rms >= step_floor:
                return float(max(trajectory_span, 2.0 * acceleration_rms)), trajectory_span, acceleration_rms, 0
            return 0.0, trajectory_span, acceleration_rms, 0

        reversal_score = max(trajectory_span, 2.0 * acceleration_rms)
        return float(reversal_score), trajectory_span, acceleration_rms, sign_changes

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        current_phase, phase_step = self._append_unwrapped_phase(feature)
        lag = max(1, int(self.config.phase_pair_lag))
        if len(self.unwrapped_phases) > lag:
            past_phase = list(self.unwrapped_phases)[-1 - lag]
            phase_pair_delta = float(abs(current_phase - past_phase))
        else:
            phase_pair_delta = 0.0

        score, trajectory_span, acceleration_rms, sign_changes = self._trajectory_score()

        force_event = bool(
            self.config.absolute_score_floor > 0.0
            and score >= self.config.absolute_score_floor
        )
        if self.config.twinkle_peak_gate_enabled:
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                feature.motion_energy,
                sign_changes,
            )
            gate_metrics = self.gate.last_gate_metrics
        else:
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                force_event=force_event,
            )
            gate_metrics = {}
        metrics = {
            "phase_step": phase_step,
            "phase_pair_delta": phase_pair_delta,
            "trajectory_span": trajectory_span,
            "acceleration_rms": acceleration_rms,
            "sign_changes": float(sign_changes),
            "unwrapped_phase": current_phase,
        }
        metrics.update(gate_metrics)
        result_score = float(metrics.get("twinkle_peak_score", score)) if is_event else float(score)
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=result_score,
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            metrics=metrics,
        )


class CompositeBlinkDetector:
    method = "both"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        child_config = BlinkDetectionConfig(**{**config.__dict__, "method": "blinklistener"})
        twinkle_config = BlinkDetectionConfig(**{**config.__dict__, "method": "twinkle"})
        self.detectors = [
            BlinkListenerBlinkDetector(child_config),
            TwinkleTwinkleBlinkDetector(twinkle_config),
        ]
        self.event_count = 0
        self.last_event_time_s = -1e9

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        results: List[BlinkDetectionResult] = [detector.update(feature) for detector in self.detectors]
        event_results = [result for result in results if result.is_event]
        if event_results:
            selected = max(event_results, key=lambda result: result.score / max(result.threshold, 1e-9))
            outside_refractory = (feature.time_s - self.last_event_time_s) >= self.config.refractory_s
            if outside_refractory:
                self.event_count += 1
                self.last_event_time_s = feature.time_s
                selected.event_id = self.event_count
            else:
                selected.is_event = False
                selected.event_id = self.event_count
        else:
            selected = max(results, key=lambda result: result.score / max(result.threshold, 1e-9))
            selected.event_id = self.event_count
        selected.metrics = dict(selected.metrics)
        selected.metrics["selected_method"] = selected.method
        selected.metrics["composite_event_count"] = float(self.event_count)
        return selected


def build_blink_detector(config: BlinkDetectionConfig):
    method = config.method.lower()
    if method == "blinklistener":
        return BlinkListenerBlinkDetector(config)
    if method in ("twinkle", "twinkletwinkle"):
        return TwinkleTwinkleBlinkDetector(config)
    if method == "both":
        return CompositeBlinkDetector(config)
    raise ValueError("blink method must be one of: blinklistener, twinkle, both")
