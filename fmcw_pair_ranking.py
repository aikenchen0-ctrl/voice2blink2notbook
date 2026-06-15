from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


PHASE_PAIR_SCORE_METRICS = ("delta_peak", "shape")


@dataclass(frozen=True)
class PhasePairPeaks:
    reference_index: int
    target_index: int
    blink_peaks: tuple[float, ...]
    background_peaks: tuple[float, ...]
    negative_peaks: tuple[float, ...] = ()


@dataclass(frozen=True)
class PhasePairRank:
    reference_index: int
    target_index: int
    blink_peak_median: float
    background_peak_p95: float
    negative_peak_p95: float
    decision_threshold: float
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    negative_trigger_rate: float
    blink_peak_count: int
    background_window_count: int
    negative_window_count: int


@dataclass(frozen=True)
class AggregatePhasePairRank:
    reference_index: int
    target_index: int
    session_count: int
    blink_peak_median: float
    background_peak_p95: float
    negative_peak_p95: float
    decision_threshold: float
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    negative_trigger_rate: float
    blink_peak_count: int
    background_window_count: int
    negative_window_count: int


@dataclass(frozen=True)
class PhasePairLineEvidence:
    reference_index: int
    target_index: int
    blink_correlations: tuple[float, ...]
    background_abs_correlations: tuple[float, ...]
    negative_abs_correlations: tuple[float, ...] = ()


@dataclass(frozen=True)
class PhasePairLineCorrelationRank:
    reference_index: int
    target_index: int
    blink_corr_median: float
    background_abs_corr_p95: float
    negative_abs_corr_p95: float
    decision_threshold: float
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    negative_trigger_rate: float
    blink_window_count: int
    background_window_count: int
    negative_window_count: int


@dataclass(frozen=True)
class AggregatePhasePairLineCorrelationRank:
    reference_index: int
    target_index: int
    session_count: int
    blink_corr_median: float
    background_abs_corr_p95: float
    negative_abs_corr_p95: float
    decision_threshold: float
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    negative_trigger_rate: float
    blink_window_count: int
    background_window_count: int
    negative_window_count: int


@dataclass(frozen=True)
class PhasePairTemplatePoint:
    group: str
    reference_index: int
    target_index: int
    center_offset_s: float
    point_index: int
    relative_time: float
    mean: float
    std: float
    count: int


def collect_phase_pair_peaks(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    blink_markers: Sequence[float],
    *,
    negative_markers: Sequence[float] = (),
    valid_start: int,
    valid_stop: int,
    marker_window_s: float,
    background_window_s: float,
    metric: str = "delta_peak",
    center_offset_s: float = 0.0,
) -> tuple[PhasePairPeaks, ...]:
    times = np.asarray(times, dtype=np.float64)
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if times.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != times.size:
        raise ValueError("times must be 1D and phase_matrix must have the same row count")
    if times.size < 2 or not blink_markers:
        return tuple()
    metric = _validate_metric(metric)

    matrix = np.unwrap(matrix, axis=1)
    valid_stop = min(int(valid_stop), int(matrix.shape[1]))
    valid_start = max(1, min(int(valid_start), valid_stop - 1))
    reference_start = max(0, valid_start - 8)
    shifted_blink_markers = tuple(float(value) + float(center_offset_s) for value in blink_markers)
    shifted_negative_markers = tuple(float(value) + float(center_offset_s) for value in negative_markers)
    background_windows = _background_windows(
        float(times[0]),
        float(times[-1]),
        shifted_blink_markers + shifted_negative_markers,
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
    )
    blink_windows = [(float(marker) - marker_window_s, float(marker) + marker_window_s) for marker in shifted_blink_markers]
    negative_windows = [
        (float(marker) - marker_window_s, float(marker) + marker_window_s)
        for marker in shifted_negative_markers
    ]

    peaks: list[PhasePairPeaks] = []
    for reference in range(reference_start, valid_stop - 1):
        # target 必须在论文/实时链路使用的有效 chirp 区间内；reference 可以稍早，用作相位差基准。
        for target in range(max(reference + 1, valid_start), valid_stop):
            trajectory = _pair_trajectory(matrix, reference, target)
            blink_peaks = tuple(_window_scores(times, trajectory, blink_windows, metric=metric))
            background_peaks = tuple(_window_scores(times, trajectory, background_windows, metric=metric))
            if not blink_peaks or not background_peaks:
                continue
            peaks.append(
                PhasePairPeaks(
                    reference_index=int(reference),
                    target_index=int(target),
                    blink_peaks=blink_peaks,
                    background_peaks=background_peaks,
                    negative_peaks=tuple(_window_scores(times, trajectory, negative_windows, metric=metric)),
                )
            )
    return tuple(peaks)


def rank_phase_pair_peaks(peaks: Iterable[PhasePairPeaks]) -> tuple[PhasePairRank, ...]:
    ranks = [_rank_from_peaks(item) for item in peaks]
    return tuple(sorted(ranks, key=_rank_sort_key))


def rank_phase_pairs(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    blink_markers: Sequence[float],
    *,
    negative_markers: Sequence[float] = (),
    valid_start: int,
    valid_stop: int,
    marker_window_s: float,
    background_window_s: float,
    metric: str = "delta_peak",
    center_offset_s: float = 0.0,
) -> tuple[PhasePairRank, ...]:
    return rank_phase_pair_peaks(
        collect_phase_pair_peaks(
            times,
            phase_matrix,
            blink_markers,
            negative_markers=negative_markers,
            valid_start=valid_start,
            valid_stop=valid_stop,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            metric=metric,
            center_offset_s=center_offset_s,
        )
    )


def aggregate_phase_pair_peaks(
    session_peak_sets: Iterable[Iterable[PhasePairPeaks]],
) -> tuple[AggregatePhasePairRank, ...]:
    by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for session_peaks in session_peak_sets:
        seen_in_session: set[tuple[int, int]] = set()
        for peaks in session_peaks:
            key = (int(peaks.reference_index), int(peaks.target_index))
            entry = by_pair.setdefault(
                key,
                {
                    "blink": [],
                    "background": [],
                    "negative": [],
                    "sessions": 0,
                },
            )
            entry["blink"].extend(float(value) for value in peaks.blink_peaks)  # type: ignore[union-attr]
            entry["background"].extend(float(value) for value in peaks.background_peaks)  # type: ignore[union-attr]
            entry["negative"].extend(float(value) for value in peaks.negative_peaks)  # type: ignore[union-attr]
            seen_in_session.add(key)
        for key in seen_in_session:
            by_pair[key]["sessions"] = int(by_pair[key]["sessions"]) + 1

    ranks: list[AggregatePhasePairRank] = []
    for (reference, target), entry in by_pair.items():
        rank = _rank_arrays(
            reference,
            target,
            np.asarray(entry["blink"], dtype=np.float64),
            np.asarray(entry["background"], dtype=np.float64),
            np.asarray(entry["negative"], dtype=np.float64),
        )
        ranks.append(
            AggregatePhasePairRank(
                reference_index=rank.reference_index,
                target_index=rank.target_index,
                session_count=int(entry["sessions"]),
                blink_peak_median=rank.blink_peak_median,
                background_peak_p95=rank.background_peak_p95,
                negative_peak_p95=rank.negative_peak_p95,
                decision_threshold=rank.decision_threshold,
                separation=rank.separation,
                blink_hit_rate=rank.blink_hit_rate,
                background_trigger_rate=rank.background_trigger_rate,
                negative_trigger_rate=rank.negative_trigger_rate,
                blink_peak_count=rank.blink_peak_count,
                background_window_count=rank.background_window_count,
                negative_window_count=rank.negative_window_count,
            )
        )
    return tuple(sorted(ranks, key=_aggregate_rank_sort_key))


def collect_phase_pair_line_evidence(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    blink_markers: Sequence[float],
    *,
    negative_markers: Sequence[float] = (),
    valid_start: int,
    valid_stop: int,
    marker_window_s: float,
    background_window_s: float,
    center_offset_s: float = 0.0,
    template_points: int = 81,
) -> tuple[PhasePairLineEvidence, ...]:
    times = np.asarray(times, dtype=np.float64)
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if times.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != times.size:
        raise ValueError("times must be 1D and phase_matrix must have the same row count")
    if times.size < 2 or not blink_markers:
        return tuple()
    if template_points < 4:
        raise ValueError("template_points must be >= 4")

    matrix = np.unwrap(matrix, axis=1)
    valid_stop = min(int(valid_stop), int(matrix.shape[1]))
    valid_start = max(1, min(int(valid_start), valid_stop - 1))
    reference_start = max(0, valid_start - 8)
    shifted_blink_markers = tuple(float(value) + float(center_offset_s) for value in blink_markers)
    shifted_negative_markers = tuple(float(value) + float(center_offset_s) for value in negative_markers)
    blink_windows = [(marker - marker_window_s, marker + marker_window_s) for marker in shifted_blink_markers]
    negative_windows = [(marker - marker_window_s, marker + marker_window_s) for marker in shifted_negative_markers]
    background_windows = _background_windows(
        float(times[0]),
        float(times[-1]),
        shifted_blink_markers + shifted_negative_markers,
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
    )

    evidence: list[PhasePairLineEvidence] = []
    for reference in range(reference_start, valid_stop - 1):
        for target in range(max(reference + 1, valid_start), valid_stop):
            trajectory = _pair_trajectory(matrix, reference, target)
            blink_rows = _resampled_centered_windows(times, trajectory, blink_windows, template_points=template_points)
            background_rows = _resampled_centered_windows(
                times,
                trajectory,
                background_windows,
                template_points=template_points,
            )
            if blink_rows.size == 0 or background_rows.size == 0:
                continue
            negative_rows = _resampled_centered_windows(
                times,
                trajectory,
                negative_windows,
                template_points=template_points,
            )
            template = np.mean(blink_rows, axis=0)
            blink_correlations = _template_correlations(template, blink_rows, absolute=False)
            background_abs_correlations = _template_correlations(template, background_rows, absolute=True)
            negative_abs_correlations = _template_correlations(template, negative_rows, absolute=True)
            if not blink_correlations or not background_abs_correlations:
                continue
            evidence.append(
                PhasePairLineEvidence(
                    reference_index=int(reference),
                    target_index=int(target),
                    blink_correlations=tuple(blink_correlations),
                    background_abs_correlations=tuple(background_abs_correlations),
                    negative_abs_correlations=tuple(negative_abs_correlations),
                )
            )
    return tuple(evidence)


def rank_phase_pair_line_evidence(
    evidence: Iterable[PhasePairLineEvidence],
) -> tuple[PhasePairLineCorrelationRank, ...]:
    ranks = [_rank_from_line_evidence(item) for item in evidence]
    return tuple(sorted(ranks, key=_line_rank_sort_key))


def rank_phase_pair_line_correlations(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    blink_markers: Sequence[float],
    *,
    negative_markers: Sequence[float] = (),
    valid_start: int,
    valid_stop: int,
    marker_window_s: float,
    background_window_s: float,
    center_offset_s: float = 0.0,
    template_points: int = 81,
) -> tuple[PhasePairLineCorrelationRank, ...]:
    return rank_phase_pair_line_evidence(
        collect_phase_pair_line_evidence(
            times,
            phase_matrix,
            blink_markers,
            negative_markers=negative_markers,
            valid_start=valid_start,
            valid_stop=valid_stop,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            center_offset_s=center_offset_s,
            template_points=template_points,
        )
    )


def aggregate_phase_pair_line_evidence(
    session_evidence_sets: Iterable[Iterable[PhasePairLineEvidence]],
) -> tuple[AggregatePhasePairLineCorrelationRank, ...]:
    by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for session_evidence in session_evidence_sets:
        seen_in_session: set[tuple[int, int]] = set()
        for evidence in session_evidence:
            key = (int(evidence.reference_index), int(evidence.target_index))
            entry = by_pair.setdefault(
                key,
                {
                    "blink": [],
                    "background": [],
                    "negative": [],
                    "sessions": 0,
                },
            )
            entry["blink"].extend(float(value) for value in evidence.blink_correlations)  # type: ignore[union-attr]
            entry["background"].extend(  # type: ignore[union-attr]
                float(value) for value in evidence.background_abs_correlations
            )
            entry["negative"].extend(float(value) for value in evidence.negative_abs_correlations)  # type: ignore[union-attr]
            seen_in_session.add(key)
        for key in seen_in_session:
            by_pair[key]["sessions"] = int(by_pair[key]["sessions"]) + 1

    ranks: list[AggregatePhasePairLineCorrelationRank] = []
    for (reference, target), entry in by_pair.items():
        rank = _rank_line_arrays(
            reference,
            target,
            np.asarray(entry["blink"], dtype=np.float64),
            np.asarray(entry["background"], dtype=np.float64),
            np.asarray(entry["negative"], dtype=np.float64),
        )
        ranks.append(
            AggregatePhasePairLineCorrelationRank(
                reference_index=rank.reference_index,
                target_index=rank.target_index,
                session_count=int(entry["sessions"]),
                blink_corr_median=rank.blink_corr_median,
                background_abs_corr_p95=rank.background_abs_corr_p95,
                negative_abs_corr_p95=rank.negative_abs_corr_p95,
                decision_threshold=rank.decision_threshold,
                separation=rank.separation,
                blink_hit_rate=rank.blink_hit_rate,
                background_trigger_rate=rank.background_trigger_rate,
                negative_trigger_rate=rank.negative_trigger_rate,
                blink_window_count=rank.blink_window_count,
                background_window_count=rank.background_window_count,
                negative_window_count=rank.negative_window_count,
            )
        )
    return tuple(sorted(ranks, key=_aggregate_line_rank_sort_key))


def phase_pair_template_points(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    reference_index: int,
    target_index: int,
    blink_markers: Sequence[float],
    *,
    negative_markers: Sequence[float] = (),
    marker_window_s: float,
    background_window_s: float,
    center_offset_s: float = 0.0,
    template_points: int = 81,
) -> tuple[PhasePairTemplatePoint, ...]:
    times = np.asarray(times, dtype=np.float64)
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if times.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != times.size:
        raise ValueError("times must be 1D and phase_matrix must have the same row count")
    if template_points < 2:
        raise ValueError("template_points must be >= 2")

    matrix = np.unwrap(matrix, axis=1)
    trajectory = _pair_trajectory(matrix, int(reference_index), int(target_index))
    shifted_blink_markers = tuple(float(value) + float(center_offset_s) for value in blink_markers)
    shifted_negative_markers = tuple(float(value) + float(center_offset_s) for value in negative_markers)
    blink_windows = [(marker - marker_window_s, marker + marker_window_s) for marker in shifted_blink_markers]
    negative_windows = [(marker - marker_window_s, marker + marker_window_s) for marker in shifted_negative_markers]
    background_windows = _background_windows(
        float(times[0]),
        float(times[-1]),
        shifted_blink_markers + shifted_negative_markers,
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
    )
    rows: list[PhasePairTemplatePoint] = []
    for group, windows in (
        ("blink", blink_windows),
        ("background", background_windows),
        ("negative", negative_windows),
    ):
        resampled = _resampled_centered_windows(times, trajectory, windows, template_points=template_points)
        if resampled.size == 0:
            continue
        means = np.mean(resampled, axis=0)
        stds = np.std(resampled, axis=0)
        for point_index, (mean, std) in enumerate(zip(means, stds)):
            relative_time = -1.0 + 2.0 * point_index / float(template_points - 1)
            rows.append(
                PhasePairTemplatePoint(
                    group=group,
                    reference_index=int(reference_index),
                    target_index=int(target_index),
                    center_offset_s=float(center_offset_s),
                    point_index=int(point_index),
                    relative_time=float(relative_time),
                    mean=float(mean),
                    std=float(std),
                    count=int(resampled.shape[0]),
                )
            )
    return tuple(rows)


def _rank_from_peaks(peaks: PhasePairPeaks) -> PhasePairRank:
    return _rank_arrays(
        int(peaks.reference_index),
        int(peaks.target_index),
        np.asarray(peaks.blink_peaks, dtype=np.float64),
        np.asarray(peaks.background_peaks, dtype=np.float64),
        np.asarray(peaks.negative_peaks, dtype=np.float64),
    )


def _rank_from_line_evidence(evidence: PhasePairLineEvidence) -> PhasePairLineCorrelationRank:
    return _rank_line_arrays(
        int(evidence.reference_index),
        int(evidence.target_index),
        np.asarray(evidence.blink_correlations, dtype=np.float64),
        np.asarray(evidence.background_abs_correlations, dtype=np.float64),
        np.asarray(evidence.negative_abs_correlations, dtype=np.float64),
    )


def _rank_arrays(
    reference: int,
    target: int,
    blink_peaks: np.ndarray,
    background_peaks: np.ndarray,
    negative_peaks: np.ndarray,
) -> PhasePairRank:
    blink_median = _median_or_zero(blink_peaks)
    background_p95 = _percentile_or_zero(background_peaks, 95.0)
    negative_p95 = _percentile_or_zero(negative_peaks, 95.0)
    threshold = max(background_p95, negative_p95, 1e-9)
    blink_hit_rate = _rate_at_or_above(blink_peaks, threshold)
    background_trigger_rate = _rate_at_or_above(background_peaks, threshold)
    negative_trigger_rate = _rate_at_or_above(negative_peaks, threshold)
    separation = (
        (blink_median / threshold)
        * (0.25 + blink_hit_rate)
        / (0.25 + background_trigger_rate + negative_trigger_rate)
    )
    return PhasePairRank(
        reference_index=int(reference),
        target_index=int(target),
        blink_peak_median=float(blink_median),
        background_peak_p95=float(background_p95),
        negative_peak_p95=float(negative_p95),
        decision_threshold=float(threshold),
        separation=float(separation),
        blink_hit_rate=float(blink_hit_rate),
        background_trigger_rate=float(background_trigger_rate),
        negative_trigger_rate=float(negative_trigger_rate),
        blink_peak_count=int(blink_peaks.size),
        background_window_count=int(background_peaks.size),
        negative_window_count=int(negative_peaks.size),
    )


def _rank_line_arrays(
    reference: int,
    target: int,
    blink_correlations: np.ndarray,
    background_abs_correlations: np.ndarray,
    negative_abs_correlations: np.ndarray,
) -> PhasePairLineCorrelationRank:
    blink_median = _median_or_zero(blink_correlations)
    background_p95 = _percentile_or_zero(background_abs_correlations, 95.0)
    negative_p95 = _percentile_or_zero(negative_abs_correlations, 95.0)
    threshold = max(background_p95, negative_p95, 0.35)
    blink_hit_rate = _rate_at_or_above(blink_correlations, threshold)
    background_trigger_rate = _rate_at_or_above(background_abs_correlations, threshold)
    negative_trigger_rate = _rate_at_or_above(negative_abs_correlations, threshold)
    separation = (
        (max(0.0, blink_median) / threshold)
        * (0.25 + blink_hit_rate)
        / (0.25 + background_trigger_rate + negative_trigger_rate)
    )
    return PhasePairLineCorrelationRank(
        reference_index=int(reference),
        target_index=int(target),
        blink_corr_median=float(blink_median),
        background_abs_corr_p95=float(background_p95),
        negative_abs_corr_p95=float(negative_p95),
        decision_threshold=float(threshold),
        separation=float(separation),
        blink_hit_rate=float(blink_hit_rate),
        background_trigger_rate=float(background_trigger_rate),
        negative_trigger_rate=float(negative_trigger_rate),
        blink_window_count=int(blink_correlations.size),
        background_window_count=int(background_abs_correlations.size),
        negative_window_count=int(negative_abs_correlations.size),
    )


def _pair_trajectory(matrix: np.ndarray, reference: int, target: int) -> np.ndarray:
    trajectory = np.unwrap(matrix[:, target] - matrix[:, reference])
    gap = max(1, abs(int(target) - int(reference)))
    return trajectory / float(gap)


def _window_peaks(times: np.ndarray, values: np.ndarray, windows: Iterable[tuple[float, float]]) -> list[float]:
    peaks: list[float] = []
    for start, end in windows:
        mask = (times >= float(start)) & (times <= float(end))
        if np.any(mask):
            peaks.append(float(np.max(values[mask])))
    return peaks


def _window_scores(
    times: np.ndarray,
    trajectory: np.ndarray,
    windows: Iterable[tuple[float, float]],
    *,
    metric: str,
) -> list[float]:
    metric = _validate_metric(metric)
    if metric == "delta_peak":
        delta = np.abs(np.diff(trajectory, prepend=trajectory[0]))
        return _window_peaks(times, delta, windows)

    scores: list[float] = []
    values = np.asarray(trajectory, dtype=np.float64)
    for start, end in windows:
        mask = (times >= float(start)) & (times <= float(end))
        if np.any(mask):
            scores.append(_trajectory_shape_score(values[mask]))
    return scores


def _resampled_centered_windows(
    times: np.ndarray,
    trajectory: np.ndarray,
    windows: Iterable[tuple[float, float]],
    *,
    template_points: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    template_x = np.linspace(0.0, 1.0, int(template_points), dtype=np.float64)
    values = np.asarray(trajectory, dtype=np.float64)
    for start, end in windows:
        mask = (times >= float(start)) & (times <= float(end))
        if np.count_nonzero(mask) < 2:
            continue
        segment = values[mask]
        segment = segment - float(np.median(segment))
        source_x = np.linspace(0.0, 1.0, int(segment.size), dtype=np.float64)
        rows.append(np.interp(template_x, source_x, segment))
    if not rows:
        return np.asarray([], dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _template_correlations(template: np.ndarray, windows: np.ndarray, *, absolute: bool) -> list[float]:
    if windows.size == 0:
        return []
    template_values = np.asarray(template, dtype=np.float64)
    template_values = template_values - float(np.mean(template_values))
    template_norm = float(np.linalg.norm(template_values))
    if template_norm <= 1e-12:
        return []

    correlations: list[float] = []
    for row in np.asarray(windows, dtype=np.float64):
        values = row - float(np.mean(row))
        norm = float(np.linalg.norm(values))
        if norm <= 1e-12:
            correlations.append(0.0)
            continue
        corr = float(np.dot(values, template_values) / (norm * template_norm))
        correlations.append(abs(corr) if absolute else corr)
    return correlations


def _trajectory_shape_score(values: np.ndarray) -> float:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size < 4:
        return 0.0
    smooth = _smooth_edge(samples, window=3)
    span = float(np.ptp(smooth))
    if span <= 1e-12:
        return 0.0

    # 眨眼在单条相位差轨迹里通常是一段“离开基线再回落”的开闭形态；
    # 单调漂移/大动作即使 span 大，也不应获得同等分数。
    endpoint_delta = abs(float(smooth[-1] - smooth[0]))
    return_score = max(0.0, min(1.0, 1.0 - endpoint_delta / span))
    directions = _compressed_directions(np.diff(smooth))
    sign_changes = sum(1 for left, right in zip(directions, directions[1:]) if left != right)
    if sign_changes == 0:
        reversal_score = 0.25
    elif sign_changes == 1:
        reversal_score = 1.0
    elif sign_changes == 2:
        reversal_score = 0.85
    elif sign_changes == 3:
        reversal_score = 0.60
    else:
        reversal_score = 0.35

    baseline = 0.5 * (float(smooth[0]) + float(smooth[-1]))
    extremum_index = int(np.argmax(np.abs(smooth - baseline)))
    edge_margin = max(1, int(round(smooth.size * 0.15)))
    center_score = 0.35 if extremum_index < edge_margin or extremum_index >= smooth.size - edge_margin else 1.0
    return float(span * (0.25 + 0.75 * return_score) * (0.35 + 0.65 * reversal_score) * center_score)


def _smooth_edge(values: np.ndarray, *, window: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if window <= 1 or samples.size < 2:
        return samples.copy()
    window = min(int(window), int(samples.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(samples, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _compressed_directions(deltas: np.ndarray) -> list[int]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        return []
    threshold = float(np.max(np.abs(values))) * 0.15
    if threshold <= 0.0:
        return []
    directions: list[int] = []
    for value in values:
        if abs(float(value)) <= threshold:
            continue
        direction = 1 if value > 0.0 else -1
        if not directions or directions[-1] != direction:
            directions.append(direction)
    return directions


def _background_windows(
    start_time: float,
    end_time: float,
    markers: Sequence[float],
    *,
    marker_window_s: float,
    background_window_s: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    step = max(float(background_window_s), 0.1)
    t = float(start_time)
    while t + background_window_s <= end_time:
        center = t + background_window_s / 2.0
        if all(abs(center - float(marker)) > marker_window_s * 1.5 for marker in markers):
            windows.append((t, t + background_window_s))
        t += step
    return windows


def _median_or_zero(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.median(values))


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return 0.0 if values.size == 0 else float(np.percentile(values, percentile))


def _rate_at_or_above(values: np.ndarray, threshold: float) -> float:
    return 0.0 if values.size == 0 else float(np.mean(values >= float(threshold)))


def _validate_metric(metric: str) -> str:
    metric = str(metric)
    if metric not in PHASE_PAIR_SCORE_METRICS:
        raise ValueError(f"unknown phase-pair score metric: {metric}")
    return metric


def _rank_sort_key(rank: PhasePairRank) -> tuple[float, float, float, float, float, int, int]:
    return (
        -rank.separation,
        -rank.blink_hit_rate,
        rank.negative_trigger_rate,
        rank.background_trigger_rate,
        -rank.blink_peak_median,
        rank.reference_index,
        rank.target_index,
    )


def _line_rank_sort_key(rank: PhasePairLineCorrelationRank) -> tuple[float, float, float, float, float, int, int]:
    return (
        -rank.separation,
        -rank.blink_hit_rate,
        rank.negative_trigger_rate,
        rank.background_trigger_rate,
        -rank.blink_corr_median,
        rank.reference_index,
        rank.target_index,
    )


def _aggregate_rank_sort_key(rank: AggregatePhasePairRank) -> tuple[float, float, float, float, float, int, int]:
    return (
        -rank.separation,
        -rank.blink_hit_rate,
        rank.negative_trigger_rate,
        rank.background_trigger_rate,
        -rank.blink_peak_median,
        rank.reference_index,
        rank.target_index,
    )


def _aggregate_line_rank_sort_key(
    rank: AggregatePhasePairLineCorrelationRank,
) -> tuple[float, float, float, float, float, int, int]:
    return (
        -rank.separation,
        -rank.blink_hit_rate,
        rank.negative_trigger_rate,
        rank.background_trigger_rate,
        -rank.blink_corr_median,
        rank.reference_index,
        rank.target_index,
    )
