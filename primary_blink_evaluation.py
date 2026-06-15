from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

from hp_acoustic_wave.event_evaluation import EventEvaluation, evaluate_events
from hp_acoustic_wave.primary_blink_peak import PrimaryBlinkPeakConfig, PrimaryBlinkPeakGate


@dataclass(frozen=True)
class PrimaryBlinkEventRow:
    session: str
    event_id: int
    time_s: float
    score: float
    threshold: float
    ratio: float
    method: str


@dataclass(frozen=True)
class PrimaryBlinkSweepRow:
    session: str
    min_score: float
    min_ratio: float
    max_score: float
    refractory_s: float
    event_total: int
    marker_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class PrimaryBlinkWindowRow:
    session: str
    event_id: int
    time_s: float
    matched: bool
    nearest_marker_offset_s: float
    score: float
    threshold: float
    ratio: float
    window_s: float
    row_count: int
    max_delta_rms: float
    median_delta_rms: float
    high_delta_duration_s: float
    dominant_pattern: str
    max_vote_confidence: float
    max_vote_score: float
    max_candidate_count: float
    max_fmcw_candidate_score: float
    max_fmcw_candidate_threshold: float


@dataclass(frozen=True)
class PrimaryBlinkEventDiagnosticRow:
    session: str
    event_id: int
    time_s: float
    matched: bool
    classification: str
    nearest_blink_offset_s: float
    nearest_large_motion_offset_s: float
    previous_event_gap_s: float
    next_event_gap_s: float
    burst_event_count_2s: int
    score: float
    threshold: float
    ratio: float
    max_delta_rms: float
    median_delta_rms: float
    high_delta_duration_s: float
    dominant_pattern: str
    max_vote_confidence: float
    max_vote_score: float
    max_candidate_count: float
    max_fmcw_candidate_score: float
    max_fmcw_candidate_threshold: float
    peak_prominence: float
    peak_prominence_ratio: float
    half_width_s: float
    rise_time_s: float
    fall_time_s: float
    abs_baseline_slope: float
    pre_post_symmetry: float


@dataclass(frozen=True)
class PrimaryBlinkConfirmSweepRow:
    session: str
    max_delta_rms: float
    max_high_delta_duration_s: float
    max_score: float
    require_pattern: bool
    min_vote_confidence: float
    event_total: int
    marker_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class PrimaryBlinkShapeSweepRow:
    session: str
    min_prominence: float
    min_prominence_ratio: float
    max_half_width_s: float
    max_abs_baseline_slope: float
    min_pre_post_symmetry: float
    max_delta_rms: float
    event_total: int
    marker_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class PrimaryBlinkRawPeakRow:
    session: str
    time_s: float
    score: float
    threshold: float
    ratio: float
    passes_score: bool
    passes_max_score: bool
    passes_ratio: bool
    passes_startup: bool
    eligible_without_refractory: bool


@dataclass(frozen=True)
class PrimaryBlinkMarkerDiagnosticRow:
    session: str
    marker_index: int
    marker_time_s: float
    matched: bool
    matched_event_id: str
    matched_event_offset_s: float
    reason: str
    best_score_time_s: float
    best_score_offset_s: float
    best_score: float
    best_threshold: float
    best_ratio: float
    best_local_peak_time_s: float
    best_local_peak_offset_s: float
    best_local_peak_score: float
    best_local_peak_ratio: float
    eligible_raw_peak_time_s: float
    eligible_raw_peak_offset_s: float
    nearest_event_time_s: float
    nearest_event_offset_s: float
    previous_event_gap_s: float


@dataclass(frozen=True)
class PrimaryBlinkEvaluation:
    session: str
    tolerance_s: float
    events: tuple[PrimaryBlinkEventRow, ...]
    raw_peaks: tuple[PrimaryBlinkRawPeakRow, ...]
    marker_diagnostics: tuple[PrimaryBlinkMarkerDiagnosticRow, ...]
    windows: tuple[PrimaryBlinkWindowRow, ...]
    event_diagnostics: tuple[PrimaryBlinkEventDiagnosticRow, ...]
    event_evaluation: EventEvaluation
    sweep: tuple[PrimaryBlinkSweepRow, ...]
    confirm_sweep: tuple[PrimaryBlinkConfirmSweepRow, ...]
    shape_sweep: tuple[PrimaryBlinkShapeSweepRow, ...]


def evaluate_primary_blink_session(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    ignore_startup_s: float = 2.0,
    min_score: float = 0.04,
    min_ratio: float = 0.8,
    max_score: float = 0.25,
    refractory_s: float = 1.2,
    window_s: float = 0.8,
    sweep_min_scores: Sequence[float] = (0.04, 0.05, 0.06, 0.07, 0.08),
    sweep_min_ratios: Sequence[float] = (0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0),
    sweep_max_scores: Sequence[float] = (0.2, 0.25, 0.3, 0.5, 999.0),
    sweep_refractory_s: Sequence[float] = (0.55, 0.75, 0.9, 1.05),
) -> PrimaryBlinkEvaluation:
    session_dir = Path(session_dir)
    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    events = detect_primary_blink_peaks(
        session_name=session_dir.name,
        feature_rows=features,
        min_score=min_score,
        min_ratio=min_ratio,
        max_score=max_score,
        refractory_s=refractory_s,
        ignore_startup_s=ignore_startup_s,
    )
    event_evaluation = evaluate_events(
        session_name=session_dir.name,
        markers=markers,
        events=primary_events_as_dicts(events),
        tolerance_s=tolerance_s,
        event_labels=("primary_blink_peak",),
        ignore_startup_s=ignore_startup_s,
    )
    raw_peaks = detect_primary_blink_raw_peaks(
        session_name=session_dir.name,
        feature_rows=features,
        min_score=min_score,
        min_ratio=min_ratio,
        max_score=max_score,
        ignore_startup_s=ignore_startup_s,
    )
    marker_diagnostics = summarize_primary_blink_marker_diagnostics(
        session_name=session_dir.name,
        feature_rows=features,
        markers=markers,
        events=events,
        raw_peaks=raw_peaks,
        event_evaluation=event_evaluation,
        tolerance_s=tolerance_s,
        min_score=min_score,
        min_ratio=min_ratio,
        max_score=max_score,
        refractory_s=refractory_s,
        ignore_startup_s=ignore_startup_s,
    )
    windows = summarize_primary_blink_windows(
        session_name=session_dir.name,
        feature_rows=features,
        markers=markers,
        events=events,
        event_evaluation=event_evaluation,
        window_s=window_s,
    )
    event_diagnostics = summarize_primary_blink_event_diagnostics(
        session_name=session_dir.name,
        feature_rows=features,
        markers=markers,
        events=events,
        windows=windows,
        event_evaluation=event_evaluation,
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
    )
    sweep_rows = tuple(
        _sweep_rows(
            session_name=session_dir.name,
            features=features,
            markers=markers,
            tolerance_s=tolerance_s,
            ignore_startup_s=ignore_startup_s,
            min_scores=sweep_min_scores,
            min_ratios=sweep_min_ratios,
            max_scores=sweep_max_scores,
            refractory_values=sweep_refractory_s,
        )
    )
    confirm_rows = tuple(
        _confirm_sweep_rows(
            session_name=session_dir.name,
            markers=markers,
            events=events,
            windows=windows,
            tolerance_s=tolerance_s,
            ignore_startup_s=ignore_startup_s,
        )
    )
    shape_rows = tuple(
        _shape_sweep_rows(
            session_name=session_dir.name,
            markers=markers,
            events=events,
            event_diagnostics=event_diagnostics,
            tolerance_s=tolerance_s,
            ignore_startup_s=ignore_startup_s,
        )
    )
    return PrimaryBlinkEvaluation(
        session=session_dir.name,
        tolerance_s=float(tolerance_s),
        events=events,
        raw_peaks=raw_peaks,
        marker_diagnostics=marker_diagnostics,
        windows=windows,
        event_diagnostics=event_diagnostics,
        event_evaluation=event_evaluation,
        sweep=sweep_rows,
        confirm_sweep=confirm_rows,
        shape_sweep=shape_rows,
    )


def detect_primary_blink_peaks(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    min_score: float,
    min_ratio: float,
    refractory_s: float,
    max_score: float = 0.25,
    ignore_startup_s: float = 0.0,
) -> tuple[PrimaryBlinkEventRow, ...]:
    gate = PrimaryBlinkPeakGate(
        PrimaryBlinkPeakConfig(
            min_score=float(min_score),
            min_ratio=float(min_ratio),
            max_score=float(max_score),
            refractory_s=float(refractory_s),
            startup_ignore_s=float(ignore_startup_s),
        )
    )
    events: list[PrimaryBlinkEventRow] = []
    for row in feature_rows:
        result = gate.update(
            _float(row.get("time_s")),
            _float(row.get("blink_score")),
            _float(row.get("blink_threshold")),
        )
        if not result.is_event:
            continue
        events.append(
            PrimaryBlinkEventRow(
                session=session_name,
                event_id=int(result.event_id),
                time_s=float(result.time_s),
                score=float(result.score),
                threshold=float(result.threshold),
                ratio=float(result.ratio),
                method=str(row.get("blink_method", "") or result.method),
            )
        )
    return tuple(events)


def detect_primary_blink_raw_peaks(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    min_score: float,
    min_ratio: float,
    max_score: float = 0.25,
    ignore_startup_s: float = 0.0,
) -> tuple[PrimaryBlinkRawPeakRow, ...]:
    rows: list[PrimaryBlinkRawPeakRow] = []
    if len(feature_rows) < 3:
        return tuple()
    for index in range(1, len(feature_rows) - 1):
        previous = feature_rows[index - 1]
        row = feature_rows[index]
        current = feature_rows[index + 1]
        score = _float(row.get("blink_score"))
        if not (score >= _float(previous.get("blink_score")) and score > _float(current.get("blink_score"))):
            continue
        threshold = _float(row.get("blink_threshold"))
        threshold_floor = max(threshold, 1e-9)
        ratio = score / threshold_floor
        passes_score = score >= float(min_score)
        passes_max_score = float(max_score) <= 0.0 or score <= float(max_score)
        passes_ratio = ratio >= float(min_ratio)
        passes_startup = _float(row.get("time_s")) >= float(ignore_startup_s)
        rows.append(
            PrimaryBlinkRawPeakRow(
                session=session_name,
                time_s=_float(row.get("time_s")),
                score=score,
                threshold=threshold,
                ratio=ratio,
                passes_score=passes_score,
                passes_max_score=passes_max_score,
                passes_ratio=passes_ratio,
                passes_startup=passes_startup,
                eligible_without_refractory=passes_score and passes_max_score and passes_ratio and passes_startup,
            )
        )
    return tuple(rows)


def primary_events_as_dicts(events: Sequence[PrimaryBlinkEventRow]) -> list[dict[str, str]]:
    return [
        {
            "event_id": str(event.event_id),
            "time_s": f"{event.time_s:.6f}",
            "label": "primary_blink_peak",
            "method": event.method,
            "score": f"{event.score:.9f}",
        }
        for event in events
    ]


def summarize_primary_blink_marker_diagnostics(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    markers: Sequence[dict[str, str]],
    events: Sequence[PrimaryBlinkEventRow],
    raw_peaks: Sequence[PrimaryBlinkRawPeakRow],
    event_evaluation: EventEvaluation,
    tolerance_s: float,
    min_score: float,
    min_ratio: float,
    max_score: float,
    refractory_s: float,
    ignore_startup_s: float,
) -> tuple[PrimaryBlinkMarkerDiagnosticRow, ...]:
    match_by_marker = {int(row.marker_index): row for row in event_evaluation.marker_matches}
    event_times = sorted(float(event.time_s) for event in events)
    diagnostics = []
    for marker_index, marker in enumerate(markers):
        if marker.get("label") != "blink":
            continue
        marker_time = _float(marker.get("time_s"))
        if marker_time < float(ignore_startup_s):
            continue
        match = match_by_marker.get(marker_index)
        window_rows = [
            row
            for row in feature_rows
            if abs(_float(row.get("time_s")) - marker_time) <= float(tolerance_s)
        ]
        best_row = max(window_rows, key=lambda row: _float(row.get("blink_score")), default=None)
        best_time = _float(best_row.get("time_s")) if best_row else 0.0
        best_score = _float(best_row.get("blink_score")) if best_row else 0.0
        best_threshold = _float(best_row.get("blink_threshold")) if best_row else 0.0
        best_ratio = best_score / max(best_threshold, 1e-9)
        local_peaks = [
            peak
            for peak in raw_peaks
            if abs(float(peak.time_s) - marker_time) <= float(tolerance_s)
        ]
        best_local_peak = max(local_peaks, key=lambda peak: peak.score, default=None)
        eligible_peaks = [peak for peak in local_peaks if peak.eligible_without_refractory]
        eligible_peak = max(eligible_peaks, key=lambda peak: peak.score, default=None)
        nearest_event_time = _nearest_time(marker_time, event_times)
        previous_event_time = _previous_time(float(eligible_peak.time_s), event_times) if eligible_peak else 0.0
        previous_event_gap = float(eligible_peak.time_s) - previous_event_time if eligible_peak and previous_event_time else 0.0
        reason = _diagnostic_reason(
            matched=bool(match and match.matched),
            has_window=bool(window_rows),
            best_score=best_score,
            best_ratio=best_ratio,
            min_score=min_score,
            min_ratio=min_ratio,
            max_score=max_score,
            best_local_peak=best_local_peak,
            eligible_peak=eligible_peak,
            previous_event_gap=previous_event_gap,
            refractory_s=refractory_s,
        )
        diagnostics.append(
            PrimaryBlinkMarkerDiagnosticRow(
                session=session_name,
                marker_index=int(marker_index),
                marker_time_s=marker_time,
                matched=bool(match and match.matched),
                matched_event_id=match.event_id if match and match.matched else "",
                matched_event_offset_s=float(match.offset_s) if match and match.matched else 0.0,
                reason=reason,
                best_score_time_s=best_time,
                best_score_offset_s=best_time - marker_time if best_row else 0.0,
                best_score=best_score,
                best_threshold=best_threshold,
                best_ratio=best_ratio,
                best_local_peak_time_s=float(best_local_peak.time_s) if best_local_peak else 0.0,
                best_local_peak_offset_s=float(best_local_peak.time_s) - marker_time if best_local_peak else 0.0,
                best_local_peak_score=float(best_local_peak.score) if best_local_peak else 0.0,
                best_local_peak_ratio=float(best_local_peak.ratio) if best_local_peak else 0.0,
                eligible_raw_peak_time_s=float(eligible_peak.time_s) if eligible_peak else 0.0,
                eligible_raw_peak_offset_s=float(eligible_peak.time_s) - marker_time if eligible_peak else 0.0,
                nearest_event_time_s=nearest_event_time,
                nearest_event_offset_s=nearest_event_time - marker_time if nearest_event_time else 0.0,
                previous_event_gap_s=previous_event_gap,
            )
        )
    return tuple(diagnostics)


def summarize_primary_blink_windows(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    markers: Sequence[dict[str, str]],
    events: Sequence[PrimaryBlinkEventRow],
    event_evaluation: EventEvaluation,
    window_s: float = 0.8,
    high_delta_threshold: float = 0.10,
) -> tuple[PrimaryBlinkWindowRow, ...]:
    marker_times = [_float(marker.get("time_s")) for marker in markers if marker.get("label") == "blink"]
    matched_ids = {
        int(row.event_id)
        for row in event_evaluation.marker_matches
        if row.matched and str(row.event_id).strip()
    }
    rows = []
    for event in events:
        window_rows = [
            row
            for row in feature_rows
            if abs(_float(row.get("time_s")) - float(event.time_s)) <= float(window_s)
        ]
        deltas = [_float(row.get("fmcw_track_delta_rms")) for row in window_rows]
        high_count = sum(1 for value in deltas if value >= float(high_delta_threshold))
        rows.append(
            PrimaryBlinkWindowRow(
                session=session_name,
                event_id=int(event.event_id),
                time_s=float(event.time_s),
                matched=int(event.event_id) in matched_ids,
                nearest_marker_offset_s=_nearest_offset(float(event.time_s), marker_times),
                score=float(event.score),
                threshold=float(event.threshold),
                ratio=float(event.ratio),
                window_s=float(window_s),
                row_count=len(window_rows),
                max_delta_rms=max(deltas, default=0.0),
                median_delta_rms=median(deltas) if deltas else 0.0,
                high_delta_duration_s=high_count * _median_sample_period_s(window_rows),
                dominant_pattern=_mode(
                    row.get("fmcw_confirm_window_pattern", "") or row.get("fmcw_final_pattern", "")
                    for row in window_rows
                ),
                max_vote_confidence=max(
                    (_float(row.get("fmcw_confirm_window_confidence")) for row in window_rows),
                    default=0.0,
                ),
                max_vote_score=max(
                    (_float(row.get("fmcw_confirm_window_vote_score")) for row in window_rows),
                    default=0.0,
                ),
                max_candidate_count=max(
                    (_float(row.get("fmcw_confirm_window_candidate_count")) for row in window_rows),
                    default=0.0,
                ),
                max_fmcw_candidate_score=max(
                    (_float(row.get("fmcw_candidate_score")) for row in window_rows),
                    default=0.0,
                ),
                max_fmcw_candidate_threshold=max(
                    (_float(row.get("fmcw_candidate_threshold")) for row in window_rows),
                    default=0.0,
                ),
            )
        )
    return tuple(rows)


def summarize_primary_blink_event_diagnostics(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    markers: Sequence[dict[str, str]],
    events: Sequence[PrimaryBlinkEventRow],
    windows: Sequence[PrimaryBlinkWindowRow],
    event_evaluation: EventEvaluation,
    tolerance_s: float,
    ignore_startup_s: float,
) -> tuple[PrimaryBlinkEventDiagnosticRow, ...]:
    blink_times = [
        _float(marker.get("time_s"))
        for marker in markers
        if marker.get("label") == "blink" and _float(marker.get("time_s")) >= float(ignore_startup_s)
    ]
    large_motion_times = [
        _float(marker.get("time_s"))
        for marker in markers
        if marker.get("label") == "large_motion" and _float(marker.get("time_s")) >= float(ignore_startup_s)
    ]
    matched_ids = {
        int(row.event_id)
        for row in event_evaluation.marker_matches
        if row.matched and str(row.event_id).strip()
    }
    windows_by_event_id = {int(window.event_id): window for window in windows}
    event_times = [float(event.time_s) for event in events]
    rows: list[PrimaryBlinkEventDiagnosticRow] = []
    for index, event in enumerate(events):
        event_id = int(event.event_id)
        time_s = float(event.time_s)
        previous_gap = time_s - event_times[index - 1] if index > 0 else 0.0
        next_gap = event_times[index + 1] - time_s if index + 1 < len(event_times) else 0.0
        burst_count = sum(1 for other in event_times if other != time_s and abs(other - time_s) <= 1.0)
        nearest_blink_offset = _nearest_offset(time_s, blink_times)
        nearest_large_offset = _nearest_offset(time_s, large_motion_times)
        matched = event_id in matched_ids
        classification = _event_diagnostic_classification(
            matched=matched,
            nearest_large_motion_offset_s=nearest_large_offset,
            has_large_motion_marker=bool(large_motion_times),
            burst_event_count_2s=burst_count,
            tolerance_s=tolerance_s,
        )
        window = windows_by_event_id.get(event_id)
        shape = _blink_score_peak_shape(feature_rows, event)
        rows.append(
            PrimaryBlinkEventDiagnosticRow(
                session=session_name,
                event_id=event_id,
                time_s=time_s,
                matched=matched,
                classification=classification,
                nearest_blink_offset_s=nearest_blink_offset,
                nearest_large_motion_offset_s=nearest_large_offset,
                previous_event_gap_s=previous_gap,
                next_event_gap_s=next_gap,
                burst_event_count_2s=int(burst_count),
                score=float(event.score),
                threshold=float(event.threshold),
                ratio=float(event.ratio),
                max_delta_rms=float(window.max_delta_rms) if window else 0.0,
                median_delta_rms=float(window.median_delta_rms) if window else 0.0,
                high_delta_duration_s=float(window.high_delta_duration_s) if window else 0.0,
                dominant_pattern=window.dominant_pattern if window else "",
                max_vote_confidence=float(window.max_vote_confidence) if window else 0.0,
                max_vote_score=float(window.max_vote_score) if window else 0.0,
                max_candidate_count=float(window.max_candidate_count) if window else 0.0,
                max_fmcw_candidate_score=float(window.max_fmcw_candidate_score) if window else 0.0,
                max_fmcw_candidate_threshold=float(window.max_fmcw_candidate_threshold) if window else 0.0,
                peak_prominence=shape["peak_prominence"],
                peak_prominence_ratio=shape["peak_prominence_ratio"],
                half_width_s=shape["half_width_s"],
                rise_time_s=shape["rise_time_s"],
                fall_time_s=shape["fall_time_s"],
                abs_baseline_slope=shape["abs_baseline_slope"],
                pre_post_symmetry=shape["pre_post_symmetry"],
            )
        )
    return tuple(rows)


def write_primary_blink_outputs(evaluation: PrimaryBlinkEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "primary_blink_events.csv", evaluation.events)
    _write_dataclass_csv(output_dir / "primary_blink_raw_peaks.csv", evaluation.raw_peaks)
    _write_dataclass_csv(output_dir / "primary_blink_marker_diagnostics.csv", evaluation.marker_diagnostics)
    _write_dataclass_csv(output_dir / "primary_blink_windows.csv", evaluation.windows)
    _write_dataclass_csv(output_dir / "primary_blink_event_diagnostics.csv", evaluation.event_diagnostics)
    _write_dataclass_csv(output_dir / "primary_blink_sweep.csv", evaluation.sweep)
    _write_dataclass_csv(output_dir / "primary_blink_confirm_sweep.csv", evaluation.confirm_sweep)
    _write_dataclass_csv(output_dir / "primary_blink_shape_sweep.csv", evaluation.shape_sweep)
    _write_dataclass_csv(output_dir / "marker_matches.csv", evaluation.event_evaluation.marker_matches)
    _write_dataclass_csv(output_dir / "false_positives.csv", evaluation.event_evaluation.false_positives)
    _write_dataclass_csv(output_dir / "metrics.csv", evaluation.event_evaluation.metrics)
    payload = asdict(evaluation)
    payload["events"] = [asdict(row) for row in evaluation.events]
    payload["raw_peaks"] = [asdict(row) for row in evaluation.raw_peaks]
    payload["marker_diagnostics"] = [asdict(row) for row in evaluation.marker_diagnostics]
    payload["windows"] = [asdict(row) for row in evaluation.windows]
    payload["event_diagnostics"] = [asdict(row) for row in evaluation.event_diagnostics]
    payload["event_evaluation"] = asdict(evaluation.event_evaluation)
    payload["sweep"] = [asdict(row) for row in evaluation.sweep]
    payload["confirm_sweep"] = [asdict(row) for row in evaluation.confirm_sweep]
    payload["shape_sweep"] = [asdict(row) for row in evaluation.shape_sweep]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sweep_rows(
    *,
    session_name: str,
    features: Sequence[dict[str, str]],
    markers: Sequence[dict[str, str]],
    tolerance_s: float,
    ignore_startup_s: float,
    min_scores: Sequence[float],
    min_ratios: Sequence[float],
    max_scores: Sequence[float],
    refractory_values: Sequence[float],
):
    for refractory_s in refractory_values:
        for min_score in min_scores:
            for min_ratio in min_ratios:
                for max_score in max_scores:
                    events = detect_primary_blink_peaks(
                        session_name=session_name,
                        feature_rows=features,
                        min_score=float(min_score),
                        min_ratio=float(min_ratio),
                        max_score=float(max_score),
                        refractory_s=float(refractory_s),
                        ignore_startup_s=float(ignore_startup_s),
                    )
                    evaluation = evaluate_events(
                        session_name=session_name,
                        markers=markers,
                        events=primary_events_as_dicts(events),
                        tolerance_s=float(tolerance_s),
                        event_labels=("primary_blink_peak",),
                        ignore_startup_s=float(ignore_startup_s),
                    )
                    metric = evaluation.metrics[0]
                    yield PrimaryBlinkSweepRow(
                        session=session_name,
                        min_score=float(min_score),
                        min_ratio=float(min_ratio),
                        max_score=float(max_score),
                        refractory_s=float(refractory_s),
                        event_total=metric.event_total,
                        marker_total=metric.marker_total,
                        true_positive=metric.true_positive,
                        false_negative=metric.false_negative,
                        false_positive=metric.false_positive,
                        recall=metric.recall,
                        precision=metric.precision,
                        f1=metric.f1,
                    )


def _confirm_sweep_rows(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    events: Sequence[PrimaryBlinkEventRow],
    windows: Sequence[PrimaryBlinkWindowRow],
    tolerance_s: float,
    ignore_startup_s: float,
    max_delta_values: Sequence[float] = (0.04, 0.06, 0.08, 0.10, 0.20, 999.0),
    max_high_duration_values: Sequence[float] = (0.0, 0.03, 0.06, 999.0),
    max_score_values: Sequence[float] = (0.2, 0.25, 0.3, 0.5, 999.0),
    min_vote_confidences: Sequence[float] = (0.0, 0.6, 0.8),
):
    event_by_id = {int(event.event_id): event for event in events}
    accepted_patterns = {"1", "11", "12", "21"}
    for max_delta in max_delta_values:
        for max_high_duration in max_high_duration_values:
            for max_score in max_score_values:
                for require_pattern in (False, True):
                    for min_confidence in min_vote_confidences:
                        if not require_pattern and min_confidence > 0.0:
                            continue
                        kept_events = []
                        for window in windows:
                            event = event_by_id.get(int(window.event_id))
                            if event is None:
                                continue
                            pattern_ok = (
                                not require_pattern
                                or (
                                    window.dominant_pattern in accepted_patterns
                                    and window.max_vote_confidence >= float(min_confidence)
                                )
                            )
                            accepted = (
                                event.score <= float(max_score)
                                and window.max_delta_rms <= float(max_delta)
                                and window.high_delta_duration_s <= float(max_high_duration)
                                and pattern_ok
                            )
                            if accepted:
                                kept_events.append(event)
                        evaluation = evaluate_events(
                            session_name=session_name,
                            markers=markers,
                            events=primary_events_as_dicts(kept_events),
                            tolerance_s=float(tolerance_s),
                            event_labels=("primary_blink_peak",),
                            ignore_startup_s=float(ignore_startup_s),
                        )
                        metric = evaluation.metrics[0]
                        yield PrimaryBlinkConfirmSweepRow(
                            session=session_name,
                            max_delta_rms=float(max_delta),
                            max_high_delta_duration_s=float(max_high_duration),
                            max_score=float(max_score),
                            require_pattern=bool(require_pattern),
                            min_vote_confidence=float(min_confidence),
                            event_total=metric.event_total,
                            marker_total=metric.marker_total,
                            true_positive=metric.true_positive,
                            false_negative=metric.false_negative,
                            false_positive=metric.false_positive,
                            recall=metric.recall,
                            precision=metric.precision,
                            f1=metric.f1,
                        )


def _shape_sweep_rows(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    events: Sequence[PrimaryBlinkEventRow],
    event_diagnostics: Sequence[PrimaryBlinkEventDiagnosticRow],
    tolerance_s: float,
    ignore_startup_s: float,
    min_prominence_values: Sequence[float] = (0.0, 0.01, 0.02, 0.03),
    min_prominence_ratio_values: Sequence[float] = (0.0, 0.25, 0.50),
    max_half_width_values: Sequence[float] = (0.15, 0.25, 0.40, 999.0),
    max_baseline_slope_values: Sequence[float] = (0.10, 0.30, 999.0),
    min_symmetry_values: Sequence[float] = (0.0, 0.20),
    max_delta_values: Sequence[float] = (0.10, 999.0),
):
    event_by_id = {int(event.event_id): event for event in events}
    for min_prominence in min_prominence_values:
        for min_prominence_ratio in min_prominence_ratio_values:
            for max_half_width in max_half_width_values:
                for max_baseline_slope in max_baseline_slope_values:
                    for min_symmetry in min_symmetry_values:
                        for max_delta in max_delta_values:
                            kept_events = []
                            for diagnostic in event_diagnostics:
                                event = event_by_id.get(int(diagnostic.event_id))
                                if event is None:
                                    continue
                                accepted = (
                                    diagnostic.peak_prominence >= float(min_prominence)
                                    and diagnostic.peak_prominence_ratio >= float(min_prominence_ratio)
                                    and diagnostic.half_width_s <= float(max_half_width)
                                    and diagnostic.abs_baseline_slope <= float(max_baseline_slope)
                                    and diagnostic.pre_post_symmetry >= float(min_symmetry)
                                    and diagnostic.max_delta_rms <= float(max_delta)
                                )
                                if accepted:
                                    kept_events.append(event)
                            evaluation = evaluate_events(
                                session_name=session_name,
                                markers=markers,
                                events=primary_events_as_dicts(kept_events),
                                tolerance_s=float(tolerance_s),
                                event_labels=("primary_blink_peak",),
                                ignore_startup_s=float(ignore_startup_s),
                            )
                            metric = evaluation.metrics[0]
                            yield PrimaryBlinkShapeSweepRow(
                                session=session_name,
                                min_prominence=float(min_prominence),
                                min_prominence_ratio=float(min_prominence_ratio),
                                max_half_width_s=float(max_half_width),
                                max_abs_baseline_slope=float(max_baseline_slope),
                                min_pre_post_symmetry=float(min_symmetry),
                                max_delta_rms=float(max_delta),
                                event_total=metric.event_total,
                                marker_total=metric.marker_total,
                                true_positive=metric.true_positive,
                                false_negative=metric.false_negative,
                                false_positive=metric.false_positive,
                                recall=metric.recall,
                                precision=metric.precision,
                                f1=metric.f1,
                            )


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _nearest_offset(time_s: float, marker_times: Sequence[float]) -> float:
    if not marker_times:
        return 0.0
    nearest = min(marker_times, key=lambda marker_time: abs(float(time_s) - float(marker_time)))
    return float(nearest) - float(time_s)


def _nearest_time(time_s: float, times: Sequence[float]) -> float:
    if not times:
        return 0.0
    return float(min(times, key=lambda value: abs(float(value) - float(time_s))))


def _previous_time(time_s: float, times: Sequence[float]) -> float:
    previous = [float(value) for value in times if float(value) < float(time_s)]
    return max(previous) if previous else 0.0


def _blink_score_peak_shape(
    feature_rows: Sequence[dict[str, str]],
    event: PrimaryBlinkEventRow,
    *,
    window_s: float = 0.8,
) -> dict[str, float]:
    event_time = float(event.time_s)
    rows = sorted(
        (
            row
            for row in feature_rows
            if abs(_float(row.get("time_s")) - event_time) <= float(window_s)
        ),
        key=lambda row: _float(row.get("time_s")),
    )
    if len(rows) < 3:
        return {
            "peak_prominence": 0.0,
            "peak_prominence_ratio": 0.0,
            "half_width_s": 0.0,
            "rise_time_s": 0.0,
            "fall_time_s": 0.0,
            "abs_baseline_slope": 0.0,
            "pre_post_symmetry": 0.0,
        }

    times = [_float(row.get("time_s")) for row in rows]
    scores = [_float(row.get("blink_score")) for row in rows]
    peak_index = min(range(len(rows)), key=lambda index: abs(times[index] - event_time))
    peak_time = times[peak_index]
    peak_score = float(event.score)
    left_indices = list(range(0, peak_index + 1))
    right_indices = list(range(peak_index, len(rows)))
    left_min_index = min(left_indices, key=lambda index: scores[index])
    right_min_index = min(right_indices, key=lambda index: scores[index])
    left_min_score = scores[left_min_index]
    right_min_score = scores[right_min_index]
    local_base = max(float(left_min_score), float(right_min_score))
    prominence = max(0.0, peak_score - local_base)
    threshold = max(float(event.threshold), 1e-9)
    prominence_ratio = prominence / threshold
    half_level = local_base + prominence * 0.5
    half_width = _half_width_s(times, scores, peak_index, half_level) if prominence > 0.0 else 0.0
    rise_time = max(0.0, peak_time - times[left_min_index])
    fall_time = max(0.0, times[right_min_index] - peak_time)
    denominator = times[right_min_index] - times[left_min_index]
    baseline_slope = (
        abs(float(right_min_score) - float(left_min_score)) / denominator
        if denominator > 0.0
        else 0.0
    )
    symmetry = min(rise_time, fall_time) / max(rise_time, fall_time) if max(rise_time, fall_time) > 0.0 else 0.0
    return {
        "peak_prominence": float(prominence),
        "peak_prominence_ratio": float(prominence_ratio),
        "half_width_s": float(half_width),
        "rise_time_s": float(rise_time),
        "fall_time_s": float(fall_time),
        "abs_baseline_slope": float(baseline_slope),
        "pre_post_symmetry": float(symmetry),
    }


def _half_width_s(times: Sequence[float], scores: Sequence[float], peak_index: int, level: float) -> float:
    left_time = float(times[peak_index])
    for index in range(int(peak_index), 0, -1):
        if float(scores[index - 1]) < float(level) <= float(scores[index]):
            left_time = _interpolate_time_for_level(
                float(times[index - 1]),
                float(scores[index - 1]),
                float(times[index]),
                float(scores[index]),
                float(level),
            )
            break
        if float(scores[index - 1]) >= float(level):
            left_time = float(times[index - 1])
    else:
        left_time = float(times[0])

    right_time = float(times[peak_index])
    for index in range(int(peak_index), len(times) - 1):
        if float(scores[index + 1]) < float(level) <= float(scores[index]):
            right_time = _interpolate_time_for_level(
                float(times[index]),
                float(scores[index]),
                float(times[index + 1]),
                float(scores[index + 1]),
                float(level),
            )
            break
        if float(scores[index + 1]) >= float(level):
            right_time = float(times[index + 1])
    else:
        right_time = float(times[-1])

    return max(0.0, right_time - left_time)


def _interpolate_time_for_level(t0: float, y0: float, t1: float, y1: float, level: float) -> float:
    if abs(float(y1) - float(y0)) <= 1e-12:
        return float(t0)
    ratio = (float(level) - float(y0)) / (float(y1) - float(y0))
    ratio = max(0.0, min(1.0, ratio))
    return float(t0) + (float(t1) - float(t0)) * ratio


def _diagnostic_reason(
    *,
    matched: bool,
    has_window: bool,
    best_score: float,
    best_ratio: float,
    min_score: float,
    min_ratio: float,
    max_score: float,
    best_local_peak: PrimaryBlinkRawPeakRow | None,
    eligible_peak: PrimaryBlinkRawPeakRow | None,
    previous_event_gap: float,
    refractory_s: float,
) -> str:
    if matched:
        return "matched"
    if not has_window:
        return "no_feature_window"
    if best_score < float(min_score):
        return "score_below_min"
    if float(max_score) > 0.0 and best_score > float(max_score):
        return "score_above_max"
    if best_ratio < float(min_ratio):
        return "ratio_below_min"
    if best_local_peak is None:
        return "no_local_peak"
    if eligible_peak is None:
        return "local_peak_below_gate"
    if 0.0 < previous_event_gap < float(refractory_s):
        return "refractory_suppressed"
    return "eligible_peak_not_emitted"


def _event_diagnostic_classification(
    *,
    matched: bool,
    nearest_large_motion_offset_s: float,
    has_large_motion_marker: bool,
    burst_event_count_2s: int,
    tolerance_s: float,
) -> str:
    if matched:
        return "true_positive"
    if has_large_motion_marker and abs(float(nearest_large_motion_offset_s)) <= float(tolerance_s):
        return "false_positive_near_large_motion"
    if int(burst_event_count_2s) >= 2:
        return "false_positive_burst"
    return "false_positive_isolated"


def _median_sample_period_s(rows: Sequence[dict[str, str]]) -> float:
    if len(rows) < 2:
        return 0.0
    times = sorted(_float(row.get("time_s")) for row in rows)
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    return median(diffs) if diffs else 0.0


def _mode(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value or "")
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda key: (counts[key], key))
