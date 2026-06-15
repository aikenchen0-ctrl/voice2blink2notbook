from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hp_acoustic_wave.event_evaluation import EventEvaluation, evaluate_events, write_event_evaluation_outputs
from hp_acoustic_wave.logged_gate_evaluation import (
    LoggedGateEventRow,
    detect_logged_gate_events,
    logged_gate_events_as_dicts,
)
from hp_acoustic_wave.negative_marker_evaluation import evaluate_negative_markers
from hp_acoustic_wave.primary_blink_evaluation import (
    PrimaryBlinkEventRow,
    detect_primary_blink_peaks,
    primary_events_as_dicts,
    read_csv_rows,
)


@dataclass(frozen=True)
class FusedCandidateEventRow:
    session: str
    event_id: int
    time_s: float
    source: str
    source_event_id: int
    source_label: str
    method: str
    score: float
    threshold: float
    ratio: float


@dataclass(frozen=True)
class CandidateFusionSummaryRow:
    session: str
    primary_event_total: int
    fallback_event_total: int
    fused_event_total: int
    added_fallback_event_total: int
    rescued_marker_total: int
    added_fallback_false_positive_total: int
    negative_marker_total: int
    primary_negative_conflict_total: int
    fallback_negative_conflict_total: int
    fused_negative_conflict_total: int


@dataclass(frozen=True)
class FusedCandidateDiagnosticRow:
    session: str
    event_id: int
    time_s: float
    source: str
    source_event_id: int
    matched: bool
    classification: str
    matched_marker_index: int
    matched_marker_offset_s: float
    nearest_marker_offset_s: float
    score: float
    method: str
    window_s: float
    row_count: int
    max_fmcw_blink_vote_evidence: float
    nearest_fmcw_blink_vote_evidence: float
    max_abs_fmcw_blink_trajectory_value: float
    nearest_fmcw_blink_trajectory_value: float
    max_abs_fmcw_fixed_trajectory_distance_mm: float
    nearest_fmcw_fixed_trajectory_distance_mm: float
    max_abs_fmcw_fixed_trajectory_phase_rad: float
    nearest_fmcw_fixed_trajectory_phase_rad: float
    dominant_fmcw_blink_trajectory_pair: str
    fmcw_blink_trajectory_pair_stability: float
    dominant_fmcw_blink_trajectory_pattern: str
    dominant_fmcw_blink_trajectory_criterion: str
    max_fmcw_track_delta_rms: float
    max_fmcw_confirm_window_confidence: float


@dataclass(frozen=True)
class CandidateFusionFmcwSweepRow:
    session: str
    min_vote_evidence: float
    min_abs_trajectory_value: float
    min_pair_stability: float
    event_total: int
    marker_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float
    negative_marker_total: int
    negative_conflict_total: int
    negative_conflict_rate: float


@dataclass(frozen=True)
class CandidateFusionAggregateStrategyRow:
    session: str
    strategy: str
    session_count: int
    marker_total: int
    event_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float
    negative_marker_total: int
    negative_conflict_total: int
    negative_conflict_rate: float


@dataclass(frozen=True)
class CandidateFusionAggregateFmcwSweepRow:
    session: str
    min_vote_evidence: float
    min_abs_trajectory_value: float
    min_pair_stability: float
    session_count: int
    marker_total: int
    event_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float
    negative_marker_total: int
    negative_conflict_total: int
    negative_conflict_rate: float


@dataclass(frozen=True)
class CandidateFusionEvaluation:
    session: str
    tolerance_s: float
    primary_events: tuple[PrimaryBlinkEventRow, ...]
    fallback_events: tuple[LoggedGateEventRow, ...]
    fused_events: tuple[FusedCandidateEventRow, ...]
    primary_evaluation: EventEvaluation
    fallback_evaluation: EventEvaluation
    fused_evaluation: EventEvaluation
    summary: CandidateFusionSummaryRow
    diagnostics: tuple[FusedCandidateDiagnosticRow, ...]
    fmcw_sweep: tuple[CandidateFusionFmcwSweepRow, ...]


@dataclass(frozen=True)
class CandidateFusionAggregateEvaluation:
    session: str
    session_count: int
    strategy_metrics: tuple[CandidateFusionAggregateStrategyRow, ...]
    fmcw_sweep: tuple[CandidateFusionAggregateFmcwSweepRow, ...]


def evaluate_candidate_fusion_session(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    ignore_startup_s: float = 2.0,
    primary_min_score: float = 0.06,
    primary_min_ratio: float = 0.85,
    primary_max_score: float = 0.25,
    primary_refractory_s: float = 1.05,
    fallback_event_column: str = "twinkle_candidate_peak",
    fallback_score_column: str = "blink_score",
    fallback_threshold: float = 0.5,
    fallback_min_score: float | None = None,
    fallback_max_score: float | None = 0.20,
    fallback_refractory_s: float = 1.05,
    fallback_exclusion_s: float = 0.8,
    diagnostic_window_s: float = 0.8,
    sweep_min_vote_evidence: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    sweep_min_abs_trajectory_value: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    sweep_min_pair_stability: Sequence[float] = (0.0, 0.5, 0.8),
) -> CandidateFusionEvaluation:
    session_dir = Path(session_dir)
    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    session_name = session_dir.name

    primary_events = detect_primary_blink_peaks(
        session_name=session_name,
        feature_rows=features,
        min_score=primary_min_score,
        min_ratio=primary_min_ratio,
        max_score=primary_max_score,
        refractory_s=primary_refractory_s,
        ignore_startup_s=ignore_startup_s,
    )
    fallback_events = detect_logged_gate_events(
        session_name=session_name,
        feature_rows=features,
        event_column=fallback_event_column,
        score_column=fallback_score_column,
        threshold=fallback_threshold,
        min_score=fallback_min_score,
        max_score=fallback_max_score,
        refractory_s=fallback_refractory_s,
        ignore_startup_s=ignore_startup_s,
    )
    fused_events = fuse_primary_and_fallback_events(
        session_name=session_name,
        primary_events=primary_events,
        fallback_events=fallback_events,
        fallback_exclusion_s=fallback_exclusion_s,
    )

    primary_evaluation = evaluate_events(
        session_name=session_name,
        markers=markers,
        events=primary_events_as_dicts(primary_events),
        tolerance_s=tolerance_s,
        event_labels=("primary_blink_peak",),
        ignore_startup_s=ignore_startup_s,
    )
    fallback_evaluation = evaluate_events(
        session_name=session_name,
        markers=markers,
        events=logged_gate_events_as_dicts(fallback_events),
        tolerance_s=tolerance_s,
        event_labels=(fallback_event_column,),
        ignore_startup_s=ignore_startup_s,
    )
    fused_evaluation = evaluate_events(
        session_name=session_name,
        markers=markers,
        events=fused_events_as_dicts(fused_events),
        tolerance_s=tolerance_s,
        event_labels=("fused_blink_candidate",),
        ignore_startup_s=ignore_startup_s,
    )
    summary = summarize_candidate_fusion(
        session_name=session_name,
        markers=markers,
        primary_evaluation=primary_evaluation,
        fused_evaluation=fused_evaluation,
        primary_events=primary_events,
        fallback_events=fallback_events,
        fused_events=fused_events,
        fallback_event_column=fallback_event_column,
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
    )
    diagnostics = summarize_fused_candidate_diagnostics(
        session_name=session_name,
        feature_rows=features,
        fused_events=fused_events,
        fused_evaluation=fused_evaluation,
        window_s=diagnostic_window_s,
    )
    fmcw_sweep = sweep_fallback_fmcw_filters(
        session_name=session_name,
        markers=markers,
        fused_events=fused_events,
        diagnostics=diagnostics,
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
        min_vote_values=sweep_min_vote_evidence,
        min_abs_trajectory_values=sweep_min_abs_trajectory_value,
        min_pair_stability_values=sweep_min_pair_stability,
    )
    return CandidateFusionEvaluation(
        session=session_name,
        tolerance_s=float(tolerance_s),
        primary_events=primary_events,
        fallback_events=fallback_events,
        fused_events=fused_events,
        primary_evaluation=primary_evaluation,
        fallback_evaluation=fallback_evaluation,
        fused_evaluation=fused_evaluation,
        summary=summary,
        diagnostics=diagnostics,
        fmcw_sweep=fmcw_sweep,
    )


def fuse_primary_and_fallback_events(
    *,
    session_name: str,
    primary_events: Sequence[PrimaryBlinkEventRow],
    fallback_events: Sequence[LoggedGateEventRow],
    fallback_exclusion_s: float,
) -> tuple[FusedCandidateEventRow, ...]:
    fused: list[FusedCandidateEventRow] = []
    primary_times = [float(event.time_s) for event in primary_events]
    for event in primary_events:
        fused.append(
            FusedCandidateEventRow(
                session=session_name,
                event_id=0,
                time_s=float(event.time_s),
                source="primary",
                source_event_id=int(event.event_id),
                source_label="primary_blink_peak",
                method=str(event.method),
                score=float(event.score),
                threshold=float(event.threshold),
                ratio=float(event.ratio),
            )
        )
    for event in fallback_events:
        if any(abs(float(event.time_s) - primary_time) <= float(fallback_exclusion_s) for primary_time in primary_times):
            continue
        fused.append(
            FusedCandidateEventRow(
                session=session_name,
                event_id=0,
                time_s=float(event.time_s),
                source="fallback",
                source_event_id=int(event.event_id),
                source_label=str(event.label),
                method=str(event.method),
                score=float(event.score),
                threshold=0.0,
                ratio=0.0,
            )
        )
    return tuple(
        _replace_event_id(event, event_id=index + 1)
        for index, event in enumerate(sorted(fused, key=lambda row: (row.time_s, row.source)))
    )


def fused_events_as_dicts(events: Sequence[FusedCandidateEventRow]) -> list[dict[str, str]]:
    return [
        {
            "event_id": str(event.event_id),
            "time_s": f"{event.time_s:.6f}",
            "label": "fused_blink_candidate",
            "method": f"{event.source}:{event.method}",
            "score": f"{event.score:.9f}",
        }
        for event in events
    ]


def summarize_candidate_fusion(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    primary_evaluation: EventEvaluation,
    fused_evaluation: EventEvaluation,
    primary_events: Sequence[PrimaryBlinkEventRow],
    fallback_events: Sequence[LoggedGateEventRow],
    fused_events: Sequence[FusedCandidateEventRow],
    fallback_event_column: str,
    tolerance_s: float,
    ignore_startup_s: float,
) -> CandidateFusionSummaryRow:
    primary_matched_markers = {
        int(row.marker_index)
        for row in primary_evaluation.marker_matches
        if row.matched
    }
    fused_matched_markers = {
        int(row.marker_index)
        for row in fused_evaluation.marker_matches
        if row.matched
    }
    fused_false_positive_ids = {
        str(row.event_id)
        for row in fused_evaluation.false_positives
    }
    fallback_fused_events = [event for event in fused_events if event.source == "fallback"]
    added_fallback_fp = sum(1 for event in fallback_fused_events if str(event.event_id) in fused_false_positive_ids)
    primary_negative = _negative_conflict_metric(
        session_name=session_name,
        markers=markers,
        events=primary_events_as_dicts(primary_events),
        conflict_event_labels=("primary_blink_peak",),
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
    )
    fallback_negative = _negative_conflict_metric(
        session_name=session_name,
        markers=markers,
        events=logged_gate_events_as_dicts(fallback_events),
        conflict_event_labels=(fallback_event_column,),
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
    )
    fused_negative = _negative_conflict_metric(
        session_name=session_name,
        markers=markers,
        events=fused_events_as_dicts(fused_events),
        conflict_event_labels=("fused_blink_candidate",),
        tolerance_s=tolerance_s,
        ignore_startup_s=ignore_startup_s,
    )
    return CandidateFusionSummaryRow(
        session=session_name,
        primary_event_total=len(primary_evaluation.false_positives) + len(primary_matched_markers),
        fallback_event_total=len(fallback_events),
        fused_event_total=len(fused_events),
        added_fallback_event_total=len(fallback_fused_events),
        rescued_marker_total=len(fused_matched_markers - primary_matched_markers),
        added_fallback_false_positive_total=int(added_fallback_fp),
        negative_marker_total=int(fused_negative[0]),
        primary_negative_conflict_total=int(primary_negative[1]),
        fallback_negative_conflict_total=int(fallback_negative[1]),
        fused_negative_conflict_total=int(fused_negative[1]),
    )


def summarize_fused_candidate_diagnostics(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    fused_events: Sequence[FusedCandidateEventRow],
    fused_evaluation: EventEvaluation,
    window_s: float,
) -> tuple[FusedCandidateDiagnosticRow, ...]:
    matched_by_event_id = {
        str(row.event_id): row
        for row in fused_evaluation.marker_matches
        if row.matched and str(row.event_id).strip()
    }
    false_positive_by_event_id = {
        str(row.event_id): row
        for row in fused_evaluation.false_positives
    }
    rows: list[FusedCandidateDiagnosticRow] = []
    for event in fused_events:
        features = [
            row
            for row in feature_rows
            if abs(_float(row.get("time_s")) - float(event.time_s)) <= float(window_s)
        ]
        nearest_feature = min(
            features,
            key=lambda row: abs(_float(row.get("time_s")) - float(event.time_s)),
            default={},
        )
        pair_counts = _counts(
            row.get("fmcw_blink_trajectory_pair", "")
            for row in features
            if row.get("fmcw_blink_trajectory_pair", "")
        )
        dominant_pair, dominant_pair_count = _mode_with_count(pair_counts)
        pattern_counts = _counts(
            row.get("fmcw_blink_trajectory_pattern", "")
            for row in features
            if row.get("fmcw_blink_trajectory_pattern", "")
        )
        criterion_counts = _counts(
            row.get("fmcw_blink_trajectory_criterion", "")
            for row in features
            if row.get("fmcw_blink_trajectory_criterion", "")
        )
        event_id = str(event.event_id)
        match = matched_by_event_id.get(event_id)
        false_positive = false_positive_by_event_id.get(event_id)
        matched = match is not None
        rows.append(
            FusedCandidateDiagnosticRow(
                session=session_name,
                event_id=int(event.event_id),
                time_s=float(event.time_s),
                source=str(event.source),
                source_event_id=int(event.source_event_id),
                matched=matched,
                classification="matched_blink" if matched else "false_positive",
                matched_marker_index=int(match.marker_index) if match else -1,
                matched_marker_offset_s=float(match.offset_s) if match else 0.0,
                nearest_marker_offset_s=float(false_positive.nearest_marker_offset_s) if false_positive else 0.0,
                score=float(event.score),
                method=str(event.method),
                window_s=float(window_s),
                row_count=len(features),
                max_fmcw_blink_vote_evidence=max(
                    (_float(row.get("fmcw_blink_vote_evidence")) for row in features),
                    default=0.0,
                ),
                nearest_fmcw_blink_vote_evidence=_float(nearest_feature.get("fmcw_blink_vote_evidence")),
                max_abs_fmcw_blink_trajectory_value=max(
                    (abs(_float(row.get("fmcw_blink_trajectory_value"))) for row in features),
                    default=0.0,
                ),
                nearest_fmcw_blink_trajectory_value=_float(nearest_feature.get("fmcw_blink_trajectory_value")),
                max_abs_fmcw_fixed_trajectory_distance_mm=max(
                    (abs(_float(row.get("fmcw_fixed_trajectory_distance_mm"))) for row in features),
                    default=0.0,
                ),
                nearest_fmcw_fixed_trajectory_distance_mm=_float(
                    nearest_feature.get("fmcw_fixed_trajectory_distance_mm")
                ),
                max_abs_fmcw_fixed_trajectory_phase_rad=max(
                    (abs(_float(row.get("fmcw_fixed_trajectory_phase_rad"))) for row in features),
                    default=0.0,
                ),
                nearest_fmcw_fixed_trajectory_phase_rad=_float(
                    nearest_feature.get("fmcw_fixed_trajectory_phase_rad")
                ),
                dominant_fmcw_blink_trajectory_pair=dominant_pair,
                fmcw_blink_trajectory_pair_stability=(
                    float(dominant_pair_count) / float(len(features)) if features else 0.0
                ),
                dominant_fmcw_blink_trajectory_pattern=_mode_from_counts(pattern_counts),
                dominant_fmcw_blink_trajectory_criterion=_mode_from_counts(criterion_counts),
                max_fmcw_track_delta_rms=max(
                    (_float(row.get("fmcw_track_delta_rms")) for row in features),
                    default=0.0,
                ),
                max_fmcw_confirm_window_confidence=max(
                    (_float(row.get("fmcw_confirm_window_confidence")) for row in features),
                    default=0.0,
                ),
            )
        )
    return tuple(rows)


def sweep_fallback_fmcw_filters(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    fused_events: Sequence[FusedCandidateEventRow],
    diagnostics: Sequence[FusedCandidateDiagnosticRow],
    tolerance_s: float,
    ignore_startup_s: float,
    min_vote_values: Sequence[float],
    min_abs_trajectory_values: Sequence[float],
    min_pair_stability_values: Sequence[float],
) -> tuple[CandidateFusionFmcwSweepRow, ...]:
    diagnostics_by_event_id = {int(row.event_id): row for row in diagnostics}
    rows: list[CandidateFusionFmcwSweepRow] = []
    for min_vote in min_vote_values:
        for min_abs_trajectory in min_abs_trajectory_values:
            for min_pair_stability in min_pair_stability_values:
                kept = [
                    event
                    for event in fused_events
                    if event.source == "primary"
                    or _diagnostic_passes_fmcw_filter(
                        diagnostics_by_event_id.get(int(event.event_id)),
                        min_vote=float(min_vote),
                        min_abs_trajectory=float(min_abs_trajectory),
                        min_pair_stability=float(min_pair_stability),
                    )
                ]
                evaluation = evaluate_events(
                    session_name=session_name,
                    markers=markers,
                    events=fused_events_as_dicts(kept),
                    tolerance_s=tolerance_s,
                    event_labels=("fused_blink_candidate",),
                    ignore_startup_s=ignore_startup_s,
                )
                negative_total, negative_conflicts, negative_conflict_rate = _negative_conflict_metric(
                    session_name=session_name,
                    markers=markers,
                    events=fused_events_as_dicts(kept),
                    conflict_event_labels=("fused_blink_candidate",),
                    tolerance_s=tolerance_s,
                    ignore_startup_s=ignore_startup_s,
                )
                metric = evaluation.metrics[0]
                rows.append(
                    CandidateFusionFmcwSweepRow(
                        session=session_name,
                        min_vote_evidence=float(min_vote),
                        min_abs_trajectory_value=float(min_abs_trajectory),
                        min_pair_stability=float(min_pair_stability),
                        event_total=int(metric.event_total),
                        marker_total=int(metric.marker_total),
                        true_positive=int(metric.true_positive),
                        false_negative=int(metric.false_negative),
                        false_positive=int(metric.false_positive),
                        recall=float(metric.recall),
                        precision=float(metric.precision),
                        f1=float(metric.f1),
                        negative_marker_total=int(negative_total),
                        negative_conflict_total=int(negative_conflicts),
                        negative_conflict_rate=float(negative_conflict_rate),
                    )
                )
    return tuple(rows)


def aggregate_candidate_fusion_evaluations(
    evaluations: Sequence[CandidateFusionEvaluation],
    *,
    session_name: str = "ALL",
) -> CandidateFusionAggregateEvaluation:
    evaluations = tuple(evaluations)
    strategy_rows = tuple(
        _aggregate_strategy_row(
            session_name=session_name,
            strategy=strategy,
            evaluations=evaluations,
        )
        for strategy in ("primary", "fallback", "fused")
    )
    sweep_rows = _aggregate_fmcw_sweep_rows(
        session_name=session_name,
        evaluations=evaluations,
    )
    return CandidateFusionAggregateEvaluation(
        session=session_name,
        session_count=len(evaluations),
        strategy_metrics=strategy_rows,
        fmcw_sweep=sweep_rows,
    )


def write_candidate_fusion_outputs(evaluation: CandidateFusionEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "fused_candidate_events.csv", evaluation.fused_events)
    _write_dataclass_csv(output_dir / "fused_candidate_diagnostics.csv", evaluation.diagnostics)
    _write_dataclass_csv(output_dir / "candidate_fusion_fmcw_sweep.csv", evaluation.fmcw_sweep)
    _write_dataclass_csv(output_dir / "candidate_fusion_summary.csv", (evaluation.summary,))
    write_event_evaluation_outputs(evaluation.primary_evaluation, output_dir / "primary")
    write_event_evaluation_outputs(evaluation.fallback_evaluation, output_dir / "fallback")
    write_event_evaluation_outputs(evaluation.fused_evaluation, output_dir / "fused")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(evaluation), handle, indent=2, ensure_ascii=False)


def write_candidate_fusion_aggregate_outputs(
    aggregate: CandidateFusionAggregateEvaluation,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "candidate_fusion_aggregate_strategy_metrics.csv", aggregate.strategy_metrics)
    _write_dataclass_csv(output_dir / "candidate_fusion_aggregate_fmcw_sweep.csv", aggregate.fmcw_sweep)
    with (output_dir / "candidate_fusion_aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(aggregate), handle, indent=2, ensure_ascii=False)


def _replace_event_id(event: FusedCandidateEventRow, *, event_id: int) -> FusedCandidateEventRow:
    return FusedCandidateEventRow(
        session=event.session,
        event_id=int(event_id),
        time_s=event.time_s,
        source=event.source,
        source_event_id=event.source_event_id,
        source_label=event.source_label,
        method=event.method,
        score=event.score,
        threshold=event.threshold,
        ratio=event.ratio,
    )


def _aggregate_strategy_row(
    *,
    session_name: str,
    strategy: str,
    evaluations: Sequence[CandidateFusionEvaluation],
) -> CandidateFusionAggregateStrategyRow:
    metrics = []
    negative_conflicts = 0
    negative_markers = 0
    for evaluation in evaluations:
        if strategy == "primary":
            metrics.append(evaluation.primary_evaluation.metrics[0])
            negative_conflicts += int(evaluation.summary.primary_negative_conflict_total)
        elif strategy == "fallback":
            metrics.append(evaluation.fallback_evaluation.metrics[0])
            negative_conflicts += int(evaluation.summary.fallback_negative_conflict_total)
        elif strategy == "fused":
            metrics.append(evaluation.fused_evaluation.metrics[0])
            negative_conflicts += int(evaluation.summary.fused_negative_conflict_total)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        negative_markers += int(evaluation.summary.negative_marker_total)

    marker_total = sum(int(row.marker_total) for row in metrics)
    event_total = sum(int(row.event_total) for row in metrics)
    true_positive = sum(int(row.true_positive) for row in metrics)
    false_negative = sum(int(row.false_negative) for row in metrics)
    false_positive = sum(int(row.false_positive) for row in metrics)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return CandidateFusionAggregateStrategyRow(
        session=session_name,
        strategy=strategy,
        session_count=len(evaluations),
        marker_total=int(marker_total),
        event_total=int(event_total),
        true_positive=int(true_positive),
        false_negative=int(false_negative),
        false_positive=int(false_positive),
        recall=float(recall),
        precision=float(precision),
        f1=float(f1),
        negative_marker_total=int(negative_markers),
        negative_conflict_total=int(negative_conflicts),
        negative_conflict_rate=_ratio(negative_conflicts, negative_markers),
    )


def _aggregate_fmcw_sweep_rows(
    *,
    session_name: str,
    evaluations: Sequence[CandidateFusionEvaluation],
) -> tuple[CandidateFusionAggregateFmcwSweepRow, ...]:
    grouped: dict[tuple[float, float, float], list[CandidateFusionFmcwSweepRow]] = {}
    for evaluation in evaluations:
        for row in evaluation.fmcw_sweep:
            key = (
                float(row.min_vote_evidence),
                float(row.min_abs_trajectory_value),
                float(row.min_pair_stability),
            )
            grouped.setdefault(key, []).append(row)

    aggregate_rows: list[CandidateFusionAggregateFmcwSweepRow] = []
    for key, rows in sorted(grouped.items()):
        marker_total = sum(int(row.marker_total) for row in rows)
        event_total = sum(int(row.event_total) for row in rows)
        true_positive = sum(int(row.true_positive) for row in rows)
        false_negative = sum(int(row.false_negative) for row in rows)
        false_positive = sum(int(row.false_positive) for row in rows)
        negative_marker_total = sum(int(row.negative_marker_total) for row in rows)
        negative_conflict_total = sum(int(row.negative_conflict_total) for row in rows)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        aggregate_rows.append(
            CandidateFusionAggregateFmcwSweepRow(
                session=session_name,
                min_vote_evidence=float(key[0]),
                min_abs_trajectory_value=float(key[1]),
                min_pair_stability=float(key[2]),
                session_count=len(rows),
                marker_total=int(marker_total),
                event_total=int(event_total),
                true_positive=int(true_positive),
                false_negative=int(false_negative),
                false_positive=int(false_positive),
                recall=float(recall),
                precision=float(precision),
                f1=float(f1),
                negative_marker_total=int(negative_marker_total),
                negative_conflict_total=int(negative_conflict_total),
                negative_conflict_rate=_ratio(negative_conflict_total, negative_marker_total),
            )
        )
    return tuple(aggregate_rows)


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _diagnostic_passes_fmcw_filter(
    diagnostic: FusedCandidateDiagnosticRow | None,
    *,
    min_vote: float,
    min_abs_trajectory: float,
    min_pair_stability: float,
) -> bool:
    if diagnostic is None:
        return False
    return (
        float(diagnostic.max_fmcw_blink_vote_evidence) >= float(min_vote)
        and float(diagnostic.max_abs_fmcw_blink_trajectory_value) >= float(min_abs_trajectory)
        and float(diagnostic.fmcw_blink_trajectory_pair_stability) >= float(min_pair_stability)
    )


def _negative_conflict_metric(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    events: Sequence[dict[str, str]],
    conflict_event_labels: Sequence[str],
    tolerance_s: float,
    ignore_startup_s: float,
) -> tuple[int, int, float]:
    evaluation = evaluate_negative_markers(
        session_name=session_name,
        markers=markers,
        events=events,
        tolerance_s=tolerance_s,
        negative_labels=("large_motion", "wave", "w"),
        conflict_event_labels=conflict_event_labels,
        suppressed_event_labels=(),
        ignore_startup_s=ignore_startup_s,
    )
    metric = evaluation.metrics[0]
    return (
        int(metric.negative_total),
        int(metric.conflict_total),
        float(metric.conflict_rate),
    )


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mode_with_count(counts: dict[str, int]) -> tuple[str, int]:
    if not counts:
        return "", 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _mode_from_counts(counts: dict[str, int]) -> str:
    return _mode_with_count(counts)[0]


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
