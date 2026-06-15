from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from hp_acoustic_wave.config import DetectorConfig, FmcwConfig
from hp_acoustic_wave.detector import AdaptiveWaveDetector
from hp_acoustic_wave.dsp import select_fmcw_candidate_bundles, unwrap_delta
from hp_acoustic_wave.fmcw_blink import decode_phase_matrix_vote, decode_trajectory_pattern, segmentation_max_gap


@dataclass(frozen=True)
class CandidateDetectorParams:
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 3.0
    min_score: float = 0.05
    refractory_s: float = 1.05
    baseline_freeze_s: float = 0.75
    release_ratio: float = 0.4
    startup_ignore_s: float = 2.0
    score_source: str = "track0_delta"


@dataclass(frozen=True)
class CandidatePoint:
    time_s: float
    score: float
    baseline: float
    mad: float
    threshold: float
    is_event: bool
    event_id: int


@dataclass(frozen=True)
class MarkerWindowSummary:
    marker_index: int
    label: str
    key: str
    time_s: float
    window_s: float
    row_count: int
    max_delta_rms: float
    median_delta_rms: float
    max_vote_confidence: float
    pattern_rows: int
    dominant_pattern: str
    dominant_pattern_count: int
    current_marker_delta_rms: float
    current_marker_pattern: str
    pattern_triggered: bool
    candidate_triggered: bool
    candidate_event_count: int
    max_candidate_score: float
    min_candidate_threshold: float
    nearest_candidate_event_offset_s: float


@dataclass(frozen=True)
class ThresholdSweepRow:
    threshold: float
    blink_hits: int
    blink_total: int
    large_motion_hits: int
    large_motion_total: int
    blink_recall: float
    large_motion_trigger_rate: float


@dataclass(frozen=True)
class CandidateDetectorSummary:
    threshold_k: float
    min_score: float
    candidate_event_count: int
    blink_hits: int
    blink_total: int
    large_motion_hits: int
    large_motion_total: int
    blink_recall: float
    large_motion_trigger_rate: float


@dataclass(frozen=True)
class CandidateDetectorSweepRow:
    threshold_k: float
    min_score: float
    candidate_event_count: int
    blink_hits: int
    blink_total: int
    large_motion_hits: int
    large_motion_total: int
    blink_recall: float
    large_motion_trigger_rate: float


@dataclass(frozen=True)
class PaperVoteSummary:
    pattern: str
    confidence: float
    score: int
    candidate_count: int
    group_winners: tuple[str, ...]


@dataclass(frozen=True)
class FmcwConfirmFeature:
    marker_index: int
    label: str
    key: str
    time_s: float
    row_count: int
    candidate_triggered: bool
    candidate_event_count: int
    max_delta_rms: float
    median_delta_rms: float
    high_delta_duration_s: float
    max_confidence: float
    pattern_rows: int
    dominant_pattern: str
    pattern_stability: float
    paper_vote_pattern: str
    paper_vote_confidence: float
    paper_vote_score: int
    paper_vote_candidate_count: int
    paper_vote_group_winners: str
    track_peak_width_s: float
    track_sync_score: float
    large_motion_suppressed: bool
    fmcw_confirmed: bool


@dataclass(frozen=True)
class MarkerTrajectoryDetail:
    marker_index: int
    label: str
    key: str
    marker_time_s: float
    window_s: float
    row_count: int
    vote_group: int
    bundle_index: int
    criterion: str
    criterion_score: float
    reference_index: int
    target_index: int
    target_indices: str
    trajectory_index: int
    pattern: str
    trajectory_span: float
    trajectory_rms: float
    peak_index: int


@dataclass(frozen=True)
class SessionAnalysis:
    session_dir: str
    feature_rows: int
    marker_rows: int
    marker_counts: dict[str, int]
    window_s: float
    candidate_params: CandidateDetectorParams
    candidate_summary: CandidateDetectorSummary
    marker_windows: tuple[MarkerWindowSummary, ...]
    confirm_features: tuple[FmcwConfirmFeature, ...]
    marker_trajectory_details: tuple[MarkerTrajectoryDetail, ...]
    threshold_sweep: tuple[ThresholdSweepRow, ...]
    candidate_sweep: tuple[CandidateDetectorSweepRow, ...]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmcw_config_from_session(session_dir: Path) -> FmcwConfig:
    # 用采集时写入的 metadata 复盘历史 session，避免默认参数变更后把旧数据按新参数重解释。
    path = Path(session_dir) / "metadata.json"
    config = FmcwConfig()
    if not path.exists():
        return config
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return config
    config_payload = payload.get("config", {})
    fmcw_payload = payload.get("fmcw", {})
    if not isinstance(fmcw_payload, dict) and isinstance(config_payload, dict):
        fmcw_payload = config_payload.get("fmcw", {})
    if not fmcw_payload and isinstance(config_payload, dict):
        fmcw_payload = config_payload.get("fmcw", {})
    if not isinstance(fmcw_payload, dict):
        return config

    known_fields = {field.name for field in fields(FmcwConfig)}
    for key, value in fmcw_payload.items():
        if key in known_fields:
            setattr(config, key, value)
    return config


def analyze_session(
    session_dir: Path,
    *,
    window_s: float = 0.8,
    thresholds: Sequence[float] = (0.05, 0.07, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.00),
    candidate_params: CandidateDetectorParams = CandidateDetectorParams(),
    candidate_threshold_ks: Sequence[float] = (3.0, 4.0, 5.0, 6.0, 8.0),
    candidate_min_scores: Sequence[float] = (0.03, 0.05, 0.07, 0.10),
) -> SessionAnalysis:
    session_dir = Path(session_dir)
    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    fmcw_config = fmcw_config_from_session(session_dir)
    candidate_points = replay_candidate_detector(features, candidate_params)
    marker_windows = tuple(
        summarize_marker_window(index, marker, features, candidate_points, window_s=window_s)
        for index, marker in enumerate(markers)
    )
    confirm_features = tuple(
        compute_confirm_feature(index, marker, features, candidate_points, window_s=window_s, config=fmcw_config)
        for index, marker in enumerate(markers)
    )
    trajectory_details = tuple(
        detail
        for index, marker in enumerate(markers)
        for detail in marker_trajectory_details(index, marker, features, window_s=window_s, config=fmcw_config)
    )
    sweep = tuple(threshold_sweep(marker_windows, thresholds))
    candidate_summary = summarize_candidate_detector(
        marker_windows,
        params=candidate_params,
        candidate_event_count=sum(point.is_event for point in candidate_points),
    )
    candidate_sweep = tuple(
        candidate_detector_sweep(
            features,
            markers,
            window_s=window_s,
            base_params=candidate_params,
            threshold_ks=candidate_threshold_ks,
            min_scores=candidate_min_scores,
        )
    )
    return SessionAnalysis(
        session_dir=str(session_dir),
        feature_rows=len(features),
        marker_rows=len(markers),
        marker_counts=dict(Counter(marker.get("label", "") for marker in markers)),
        window_s=float(window_s),
        candidate_params=candidate_params,
        candidate_summary=candidate_summary,
        marker_windows=marker_windows,
        confirm_features=confirm_features,
        marker_trajectory_details=trajectory_details,
        threshold_sweep=sweep,
        candidate_sweep=candidate_sweep,
    )


def replay_candidate_detector(
    features: Sequence[dict[str, str]],
    params: CandidateDetectorParams = CandidateDetectorParams(),
) -> tuple[CandidatePoint, ...]:
    detector = AdaptiveWaveDetector(
        DetectorConfig(
            history_size=int(params.history_size),
            min_history=int(params.min_history),
            threshold_k=float(params.threshold_k),
            min_energy=float(params.min_score),
            refractory_s=float(params.refractory_s),
            baseline_freeze_s=float(params.baseline_freeze_s),
            release_ratio=float(params.release_ratio),
            startup_ignore_s=float(params.startup_ignore_s),
        )
    )
    points: list[CandidatePoint] = []
    previous_row = None
    for row in features:
        time_s = _float(row.get("time_s"))
        score = _candidate_score(row, params.score_source, previous_row)
        detection = detector.update(time_s, score)
        points.append(
            CandidatePoint(
                time_s=time_s,
                score=score,
                baseline=float(detection.baseline),
                mad=float(detection.mad),
                threshold=float(detection.threshold),
                is_event=bool(detection.is_event),
                event_id=int(detection.event_id),
            )
        )
        previous_row = row
    return tuple(points)


def summarize_marker_window(
    marker_index: int,
    marker: dict[str, str],
    features: Sequence[dict[str, str]],
    candidate_points: Sequence[CandidatePoint],
    *,
    window_s: float,
) -> MarkerWindowSummary:
    marker_time = _float(marker.get("time_s"))
    nearby = [
        row
        for row in features
        if abs(_float(row.get("time_s")) - marker_time) <= window_s
    ]
    delta_values = [_float(row.get("fmcw_track_delta_rms")) for row in nearby]
    confidence_values = [_float(row.get("fmcw_vote_confidence")) for row in nearby]
    patterns = [row.get("fmcw_pattern", "") for row in nearby if row.get("fmcw_pattern", "")]
    nearby_candidates = [
        point
        for point in candidate_points
        if abs(point.time_s - marker_time) <= window_s
    ]
    candidate_events = [point for point in nearby_candidates if point.is_event]
    pattern_counts = Counter(patterns)
    if pattern_counts:
        dominant_pattern, dominant_count = sorted(
            pattern_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )[0]
    else:
        dominant_pattern, dominant_count = "", 0
    max_delta = max(delta_values) if delta_values else 0.0
    candidate_thresholds = [point.threshold for point in nearby_candidates]
    if candidate_events:
        nearest_event = sorted(candidate_events, key=lambda point: abs(point.time_s - marker_time))[0]
        nearest_offset = nearest_event.time_s - marker_time
    else:
        nearest_offset = 0.0

    return MarkerWindowSummary(
        marker_index=int(marker_index),
        label=marker.get("label", ""),
        key=marker.get("key", ""),
        time_s=marker_time,
        window_s=float(window_s),
        row_count=len(nearby),
        max_delta_rms=max_delta,
        median_delta_rms=float(median(delta_values)) if delta_values else 0.0,
        max_vote_confidence=max(confidence_values) if confidence_values else 0.0,
        pattern_rows=len(patterns),
        dominant_pattern=dominant_pattern,
        dominant_pattern_count=int(dominant_count),
        current_marker_delta_rms=_float(marker.get("fmcw_track_delta_rms")),
        current_marker_pattern=marker.get("fmcw_pattern", ""),
        pattern_triggered=bool(patterns),
        candidate_triggered=bool(candidate_events),
        candidate_event_count=len(candidate_events),
        max_candidate_score=max((point.score for point in nearby_candidates), default=0.0),
        min_candidate_threshold=min(candidate_thresholds) if candidate_thresholds else 0.0,
        nearest_candidate_event_offset_s=float(nearest_offset),
    )


def threshold_sweep(
    marker_windows: Iterable[MarkerWindowSummary],
    thresholds: Sequence[float],
) -> list[ThresholdSweepRow]:
    windows = list(marker_windows)
    blink = [window for window in windows if window.label == "blink"]
    large = [window for window in windows if window.label == "large_motion"]
    rows: list[ThresholdSweepRow] = []
    for threshold in thresholds:
        blink_hits = sum(window.max_delta_rms >= threshold for window in blink)
        large_hits = sum(window.max_delta_rms >= threshold for window in large)
        rows.append(
            ThresholdSweepRow(
                threshold=float(threshold),
                blink_hits=int(blink_hits),
                blink_total=len(blink),
                large_motion_hits=int(large_hits),
                large_motion_total=len(large),
                blink_recall=_ratio(blink_hits, len(blink)),
                large_motion_trigger_rate=_ratio(large_hits, len(large)),
            )
        )
    return rows


def summarize_candidate_detector(
    marker_windows: Iterable[MarkerWindowSummary],
    *,
    params: CandidateDetectorParams,
    candidate_event_count: int,
) -> CandidateDetectorSummary:
    windows = list(marker_windows)
    blink = [window for window in windows if window.label == "blink"]
    large = [window for window in windows if window.label == "large_motion"]
    blink_hits = sum(window.candidate_triggered for window in blink)
    large_hits = sum(window.candidate_triggered for window in large)
    return CandidateDetectorSummary(
        threshold_k=float(params.threshold_k),
        min_score=float(params.min_score),
        candidate_event_count=int(candidate_event_count),
        blink_hits=int(blink_hits),
        blink_total=len(blink),
        large_motion_hits=int(large_hits),
        large_motion_total=len(large),
        blink_recall=_ratio(blink_hits, len(blink)),
        large_motion_trigger_rate=_ratio(large_hits, len(large)),
    )


def candidate_detector_sweep(
    features: Sequence[dict[str, str]],
    markers: Sequence[dict[str, str]],
    *,
    window_s: float,
    base_params: CandidateDetectorParams,
    threshold_ks: Sequence[float],
    min_scores: Sequence[float],
) -> list[CandidateDetectorSweepRow]:
    rows: list[CandidateDetectorSweepRow] = []
    for threshold_k in threshold_ks:
        for min_score in min_scores:
            params = CandidateDetectorParams(
                history_size=base_params.history_size,
                min_history=base_params.min_history,
                threshold_k=float(threshold_k),
                min_score=float(min_score),
                refractory_s=base_params.refractory_s,
                baseline_freeze_s=base_params.baseline_freeze_s,
                release_ratio=base_params.release_ratio,
                startup_ignore_s=base_params.startup_ignore_s,
                score_source=base_params.score_source,
            )
            candidate_points = replay_candidate_detector(features, params)
            marker_windows = tuple(
                summarize_marker_window(index, marker, features, candidate_points, window_s=window_s)
                for index, marker in enumerate(markers)
            )
            summary = summarize_candidate_detector(
                marker_windows,
                params=params,
                candidate_event_count=sum(point.is_event for point in candidate_points),
            )
            rows.append(
                CandidateDetectorSweepRow(
                    threshold_k=summary.threshold_k,
                    min_score=summary.min_score,
                    candidate_event_count=summary.candidate_event_count,
                    blink_hits=summary.blink_hits,
                    blink_total=summary.blink_total,
                    large_motion_hits=summary.large_motion_hits,
                    large_motion_total=summary.large_motion_total,
                    blink_recall=summary.blink_recall,
                    large_motion_trigger_rate=summary.large_motion_trigger_rate,
                )
            )
    return rows


def compute_confirm_feature(
    marker_index: int,
    marker: dict[str, str],
    features: Sequence[dict[str, str]],
    candidate_points: Sequence[CandidatePoint],
    *,
    window_s: float,
    high_delta_threshold: float = 0.10,
    large_motion_delta_threshold: float = 0.20,
    large_motion_duration_threshold_s: float = 0.25,
    confirm_min_delta: float = 0.07,
    confirm_max_delta: float = 0.20,
    confirm_max_high_delta_duration_s: float = 0.45,
    config: FmcwConfig | None = None,
) -> FmcwConfirmFeature:
    marker_time = _float(marker.get("time_s"))
    nearby = [
        row
        for row in features
        if abs(_float(row.get("time_s")) - marker_time) <= window_s
    ]
    nearby_candidates = [
        point
        for point in candidate_points
        if abs(point.time_s - marker_time) <= window_s
    ]
    candidate_events = [point for point in nearby_candidates if point.is_event]
    delta_values = [_float(row.get("fmcw_track_delta_rms")) for row in nearby]
    times = [_float(row.get("time_s")) for row in nearby]
    confidences = [_float(row.get("fmcw_vote_confidence")) for row in nearby]
    patterns = [row.get("fmcw_pattern", "") for row in nearby if row.get("fmcw_pattern", "")]
    paper_vote = paper_style_vote_from_rows(nearby, config=config)
    pattern_counts = Counter(patterns)
    if pattern_counts:
        dominant_pattern, dominant_count = sorted(
            pattern_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )[0]
    else:
        dominant_pattern, dominant_count = "", 0

    max_delta = max(delta_values) if delta_values else 0.0
    high_delta_duration = _duration_above_threshold(times, delta_values, high_delta_threshold)
    pattern_stability = _ratio(dominant_count, len(patterns))
    track_widths = []
    track_peak_times = []
    for track_name in ("fmcw_track_0", "fmcw_track_1", "fmcw_track_2", "fmcw_track_3", "fmcw_track_4"):
        values = [_float(row.get(track_name)) for row in nearby]
        width, peak_time = _track_peak_shape(times, values)
        if width > 0.0:
            track_widths.append(width)
        if peak_time is not None:
            track_peak_times.append(peak_time)
    track_peak_width = float(median(track_widths)) if track_widths else 0.0
    track_sync_score = _track_sync_score(track_peak_times, window_s=window_s)
    fmcw_config = config or FmcwConfig()
    dominant_confidence = max(confidences) if confidences else 0.0
    paper_vote_support = (
        paper_vote.pattern in tuple(fmcw_config.confirm_single_blink_patterns)
        and paper_vote.confidence >= float(fmcw_config.confirm_vote_min_confidence)
    )
    continuous_vote_support = (
        dominant_pattern in tuple(fmcw_config.confirm_single_blink_patterns)
        and dominant_confidence >= float(fmcw_config.confirm_vote_min_confidence)
        and pattern_stability >= float(fmcw_config.confirm_vote_min_stability)
    )
    vote_support = paper_vote_support or continuous_vote_support

    large_motion_suppressed = bool(
        not vote_support
        and (
            max_delta >= large_motion_delta_threshold
            or high_delta_duration >= large_motion_duration_threshold_s
        )
    )
    fmcw_confirmed = bool(
        candidate_events
        and (
            vote_support
            or (
                not large_motion_suppressed
                and confirm_min_delta <= max_delta <= confirm_max_delta
                and high_delta_duration <= confirm_max_high_delta_duration_s
            )
        )
    )

    return FmcwConfirmFeature(
        marker_index=int(marker_index),
        label=marker.get("label", ""),
        key=marker.get("key", ""),
        time_s=marker_time,
        row_count=len(nearby),
        candidate_triggered=bool(candidate_events),
        candidate_event_count=len(candidate_events),
        max_delta_rms=max_delta,
        median_delta_rms=float(median(delta_values)) if delta_values else 0.0,
        high_delta_duration_s=high_delta_duration,
        max_confidence=dominant_confidence,
        pattern_rows=len(patterns),
        dominant_pattern=dominant_pattern,
        pattern_stability=pattern_stability,
        paper_vote_pattern=paper_vote.pattern,
        paper_vote_confidence=paper_vote.confidence,
        paper_vote_score=paper_vote.score,
        paper_vote_candidate_count=paper_vote.candidate_count,
        paper_vote_group_winners="|".join(paper_vote.group_winners),
        track_peak_width_s=track_peak_width,
        track_sync_score=track_sync_score,
        large_motion_suppressed=large_motion_suppressed,
        fmcw_confirmed=fmcw_confirmed,
    )


def paper_style_vote_from_rows(
    rows: Sequence[dict[str, str]],
    config: FmcwConfig | None = None,
) -> PaperVoteSummary:
    # 论文式投票要求同一窗口内完整 phase matrix：phase-pair 轨迹 -> 分割 -> 三组投票。
    # 历史 session 没保存 fmcw_phase_points 时返回空摘要，避免把 5 条展示轨迹误当论文输入。
    phase_matrix = _phase_matrix_from_rows(rows)
    if phase_matrix.shape[0] == 0:
        return PaperVoteSummary("", 0.0, 0, 0, tuple())

    fmcw_config = config or FmcwConfig()
    minimum_rows = max(
        int(fmcw_config.candidate_interval_length),
        int(fmcw_config.trajectory_detrend_window),
        16,
    )
    if phase_matrix.shape[0] < minimum_rows:
        return PaperVoteSummary("", 0.0, 0, 0, tuple())

    decision = decode_phase_matrix_vote(phase_matrix, fmcw_config, minimum_rows=minimum_rows)
    return PaperVoteSummary(
        pattern=decision.pattern,
        confidence=float(decision.confidence),
        score=int(decision.score),
        candidate_count=int(decision.candidate_count),
        group_winners=tuple(decision.group_winners),
    )


def marker_trajectory_details(
    marker_index: int,
    marker: dict[str, str],
    features: Sequence[dict[str, str]],
    *,
    window_s: float,
    config: FmcwConfig | None = None,
) -> tuple[MarkerTrajectoryDetail, ...]:
    # 调试用：展开论文 45 条候选轨迹，便于定位误报来自哪个 criterion / phase-pair。
    marker_time = _float(marker.get("time_s"))
    nearby = [
        row
        for row in features
        if abs(_float(row.get("time_s")) - marker_time) <= window_s
    ]
    phase_matrix = _phase_matrix_from_rows(nearby)
    if phase_matrix.shape[0] == 0:
        return tuple()

    fmcw_config = config or FmcwConfig()
    minimum_rows = max(
        int(fmcw_config.candidate_interval_length),
        int(fmcw_config.trajectory_detrend_window),
        16,
    )
    if phase_matrix.shape[0] < minimum_rows:
        return tuple()

    bundles = select_fmcw_candidate_bundles(phase_matrix, fmcw_config)
    max_gap = segmentation_max_gap(fmcw_config)
    details: list[MarkerTrajectoryDetail] = []
    trajectory_index = 0
    for bundle_index, bundle in enumerate(bundles):
        vote_group = int(bundle_index // max(1, int(fmcw_config.candidate_intervals_per_criterion)))
        for local_index, trajectory in enumerate(bundle.trajectories):
            values = np.asarray(trajectory, dtype=np.float64)
            target_index = bundle.target_indices[local_index] if local_index < len(bundle.target_indices) else -1
            details.append(
                MarkerTrajectoryDetail(
                    marker_index=int(marker_index),
                    label=marker.get("label", ""),
                    key=marker.get("key", ""),
                    marker_time_s=marker_time,
                    window_s=float(window_s),
                    row_count=int(phase_matrix.shape[0]),
                    vote_group=vote_group,
                    bundle_index=int(bundle_index),
                    criterion=bundle.criterion,
                    criterion_score=float(bundle.score),
                    reference_index=int(bundle.reference_index),
                    target_index=int(target_index),
                    target_indices=";".join(str(index) for index in bundle.target_indices),
                    trajectory_index=int(trajectory_index),
                    pattern=decode_trajectory_pattern(values, max_gap=max_gap),
                    trajectory_span=float(np.ptp(values)) if values.size else 0.0,
                    trajectory_rms=float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
                    peak_index=int(np.argmax(np.abs(values))) if values.size else -1,
                )
            )
            trajectory_index += 1
    return tuple(details)


def write_analysis_outputs(analysis: SessionAnalysis, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "marker_windows.csv", analysis.marker_windows)
    _write_dataclass_csv(output_dir / "confirm_features.csv", analysis.confirm_features)
    _write_dataclass_csv(
        output_dir / "marker_trajectory_details.csv",
        analysis.marker_trajectory_details,
        fieldnames=_dataclass_fieldnames(MarkerTrajectoryDetail),
    )
    _write_dataclass_csv(output_dir / "drms_threshold_sweep.csv", analysis.threshold_sweep)
    _write_dataclass_csv(output_dir / "candidate_detector_summary.csv", (analysis.candidate_summary,))
    _write_dataclass_csv(output_dir / "candidate_detector_sweep.csv", analysis.candidate_sweep)
    payload = asdict(analysis)
    payload["marker_windows"] = [asdict(row) for row in analysis.marker_windows]
    payload["confirm_features"] = [asdict(row) for row in analysis.confirm_features]
    payload["marker_trajectory_details"] = [asdict(row) for row in analysis.marker_trajectory_details]
    payload["threshold_sweep"] = [asdict(row) for row in analysis.threshold_sweep]
    payload["candidate_sweep"] = [asdict(row) for row in analysis.candidate_sweep]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows and fieldnames is None:
            return
        names = list(fieldnames or asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _dataclass_fieldnames(row_type: type) -> list[str]:
    return list(getattr(row_type, "__dataclass_fields__", {}).keys())


def _float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _phase_matrix_from_rows(rows: Sequence[dict[str, str]]) -> np.ndarray:
    parsed_rows = []
    expected_count = 0
    for row in rows:
        values = _parse_phase_points(row.get("fmcw_phase_points"))
        if not values:
            continue
        if expected_count == 0:
            expected_count = len(values)
        if len(values) != expected_count:
            continue
        parsed_rows.append(values)
    if not parsed_rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(parsed_rows, dtype=np.float64)


def _parse_phase_points(value: str | None) -> list[float]:
    if value is None or value == "":
        return []
    parts = value.split(";")
    parsed = []
    for part in parts:
        if part == "":
            continue
        parsed.append(float(part))
    return parsed


def _candidate_score(
    row: dict[str, str],
    source: str = "track_delta_rms",
    previous_row: dict[str, str] | None = None,
) -> float:
    if source.startswith("track") and source.endswith("_delta"):
        raw_index = source[len("track") : -len("_delta")]
        if raw_index.isdigit() and previous_row is None:
            return 0.0
        if raw_index.isdigit() and previous_row is not None:
            key = f"fmcw_track_{int(raw_index)}"
            current = row.get(key)
            previous = previous_row.get(key)
            if current not in (None, "") and previous not in (None, ""):
                return abs(unwrap_delta(_float(current), _float(previous)))

    if source in ("max_track_delta", "mean_track_delta") and previous_row is not None:
        deltas = []
        for index in range(5):
            key = f"fmcw_track_{index}"
            current = row.get(key)
            previous = previous_row.get(key)
            if current not in (None, "") and previous not in (None, ""):
                deltas.append(abs(unwrap_delta(_float(current), _float(previous))))
        if deltas and source == "max_track_delta":
            return max(deltas)
        if deltas and source == "mean_track_delta":
            return float(sum(deltas) / float(len(deltas)))
    if source in ("max_track_delta", "mean_track_delta") and previous_row is None:
        return 0.0

    if source == "track_delta_rms":
        score = row.get("fmcw_track_delta_rms")
        if score not in (None, ""):
            return _float(score)

    score = row.get("fmcw_candidate_score")
    if score not in (None, ""):
        return _float(score)
    score = row.get("fmcw_track_delta_rms")
    if score not in (None, ""):
        return _float(score)
    return _float(row.get("motion_energy"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _duration_above_threshold(times: Sequence[float], values: Sequence[float], threshold: float) -> float:
    if len(times) < 2 or len(values) < 2:
        return 0.0
    duration = 0.0
    for index in range(len(times) - 1):
        if values[index] >= threshold:
            duration += max(0.0, float(times[index + 1]) - float(times[index]))
    return float(duration)


def _track_peak_shape(times: Sequence[float], values: Sequence[float]) -> tuple[float, float | None]:
    if len(times) < 3 or len(values) < 3:
        return 0.0, None
    baseline = float(median(values))
    deviations = [abs(float(value) - baseline) for value in values]
    peak = max(deviations)
    if peak <= 0.0:
        return 0.0, None
    threshold = peak * 0.50
    active_indices = [index for index, value in enumerate(deviations) if value >= threshold]
    if not active_indices:
        return 0.0, None
    width = float(times[active_indices[-1]]) - float(times[active_indices[0]])
    peak_index = max(range(len(deviations)), key=lambda index: deviations[index])
    return max(0.0, width), float(times[peak_index])


def _track_sync_score(peak_times: Sequence[float], *, window_s: float) -> float:
    if len(peak_times) < 2:
        return 0.0
    spread = max(peak_times) - min(peak_times)
    return max(0.0, 1.0 - float(spread) / max(float(window_s), 1e-6))
