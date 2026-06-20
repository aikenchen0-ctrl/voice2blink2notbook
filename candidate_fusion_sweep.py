from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hp_acoustic_wave.candidate_fusion_evaluation import (
    fuse_primary_and_fallback_events,
    fused_events_as_dicts,
)
from hp_acoustic_wave.event_evaluation import (
    evaluate_events,
    visual_face_time_ranges,
)
from hp_acoustic_wave.logged_gate_evaluation import (
    detect_logged_gate_events,
)
from hp_acoustic_wave.negative_marker_evaluation import evaluate_negative_markers
from hp_acoustic_wave.primary_blink_evaluation import (
    detect_primary_blink_peaks,
    primary_events_as_dicts,
    read_csv_rows,
)


@dataclass(frozen=True)
class CandidateFusionParameterRow:
    session: str
    primary_min_score: float
    primary_min_ratio: float
    primary_max_score: float
    primary_refractory_s: float
    fallback_enabled: bool
    fallback_min_score: float
    fallback_max_score: float
    fallback_refractory_s: float
    fallback_exclusion_s: float
    marker_total: int
    event_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float
    primary_event_total: int
    fallback_event_total: int
    added_fallback_event_total: int
    negative_marker_total: int
    negative_conflict_total: int
    negative_conflict_rate: float


@dataclass(frozen=True)
class CandidateFusionSweepResult:
    session: str
    rows: tuple[CandidateFusionParameterRow, ...]


def sweep_candidate_fusion_session(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    ignore_startup_s: float = 2.0,
    primary_min_scores: Sequence[float] = (0.04, 0.06, 0.08, 0.10, 0.12),
    primary_min_ratios: Sequence[float] = (0.8, 1.0, 1.5, 2.0),
    primary_max_scores: Sequence[float] = (0.25,),
    primary_refractory_values: Sequence[float] = (1.05, 1.2, 1.5, 2.0),
    fallback_enabled_values: Sequence[bool] = (False, True),
    fallback_event_column: str = "twinkle_candidate_peak",
    fallback_score_column: str = "blink_score",
    fallback_threshold: float = 0.5,
    fallback_min_scores: Sequence[float | None] = (None, 0.06, 0.08, 0.10, 0.12),
    fallback_max_scores: Sequence[float | None] = (0.20,),
    fallback_refractory_values: Sequence[float] = (1.05, 1.5, 2.0),
    fallback_exclusion_values: Sequence[float] = (0.8,),
    require_visual_face: bool = False,
) -> CandidateFusionSweepResult:
    session_dir = Path(session_dir)
    session_name = session_dir.name
    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    valid_time_ranges = (
        visual_face_time_ranges(session_dir / "visual_features.csv")
        if bool(require_visual_face)
        else None
    )

    primary_cache = {}
    for min_score in primary_min_scores:
        for min_ratio in primary_min_ratios:
            for max_score in primary_max_scores:
                for refractory_s in primary_refractory_values:
                    key = (
                        float(min_score),
                        float(min_ratio),
                        float(max_score),
                        float(refractory_s),
                    )
                    primary_cache[key] = detect_primary_blink_peaks(
                        session_name=session_name,
                        feature_rows=features,
                        min_score=float(min_score),
                        min_ratio=float(min_ratio),
                        max_score=float(max_score),
                        refractory_s=float(refractory_s),
                        ignore_startup_s=float(ignore_startup_s),
                    )

    fallback_cache = {}
    for min_score in fallback_min_scores:
        for max_score in fallback_max_scores:
            for refractory_s in fallback_refractory_values:
                key = (
                    _optional_float_key(min_score),
                    _optional_float_key(max_score),
                    float(refractory_s),
                )
                fallback_cache[key] = detect_logged_gate_events(
                    session_name=session_name,
                    feature_rows=features,
                    event_column=fallback_event_column,
                    score_column=fallback_score_column,
                    threshold=float(fallback_threshold),
                    min_score=None if min_score is None else float(min_score),
                    max_score=None if max_score is None else float(max_score),
                    refractory_s=float(refractory_s),
                    ignore_startup_s=float(ignore_startup_s),
                )

    rows: list[CandidateFusionParameterRow] = []
    for primary_key, primary_events in primary_cache.items():
        primary_min_score, primary_min_ratio, primary_max_score, primary_refractory_s = primary_key
        for fallback_enabled in fallback_enabled_values:
            fallback_min_iter: Sequence[float | None]
            fallback_max_iter: Sequence[float | None]
            fallback_refractory_iter: Sequence[float]
            if fallback_enabled:
                fallback_min_iter = tuple(fallback_min_scores)
                fallback_max_iter = tuple(fallback_max_scores)
                fallback_refractory_iter = tuple(float(v) for v in fallback_refractory_values)
            else:
                fallback_min_iter = (None,)
                fallback_max_iter = (None,)
                fallback_refractory_iter = (0.0,)

            for fallback_min_score in fallback_min_iter:
                for fallback_max_score in fallback_max_iter:
                    for fallback_refractory_s in fallback_refractory_iter:
                        fallback_key = (
                            _optional_float_key(fallback_min_score),
                            _optional_float_key(fallback_max_score),
                            float(fallback_refractory_s),
                        )
                        fallback_events = fallback_cache.get(fallback_key, ()) if fallback_enabled else ()
                        for fallback_exclusion_s in fallback_exclusion_values:
                            fused_events = fuse_primary_and_fallback_events(
                                session_name=session_name,
                                primary_events=primary_events,
                                fallback_events=fallback_events,
                                fallback_exclusion_s=float(fallback_exclusion_s),
                            )
                            evaluation = evaluate_events(
                                session_name=session_name,
                                markers=markers,
                                events=fused_events_as_dicts(fused_events),
                                tolerance_s=float(tolerance_s),
                                event_labels=("fused_blink_candidate",),
                                ignore_startup_s=float(ignore_startup_s),
                                valid_time_ranges=valid_time_ranges,
                            )
                            metric = evaluation.metrics[0]
                            negative_metric = evaluate_negative_markers(
                                session_name=session_name,
                                markers=markers,
                                events=fused_events_as_dicts(fused_events),
                                tolerance_s=float(tolerance_s),
                                negative_labels=("large_motion", "wave", "w"),
                                conflict_event_labels=("fused_blink_candidate",),
                                suppressed_event_labels=(),
                                ignore_startup_s=float(ignore_startup_s),
                            ).metrics[0]
                            rows.append(
                                CandidateFusionParameterRow(
                                    session=session_name,
                                    primary_min_score=float(primary_min_score),
                                    primary_min_ratio=float(primary_min_ratio),
                                    primary_max_score=float(primary_max_score),
                                    primary_refractory_s=float(primary_refractory_s),
                                    fallback_enabled=bool(fallback_enabled),
                                    fallback_min_score=0.0
                                    if fallback_min_score is None
                                    else float(fallback_min_score),
                                    fallback_max_score=0.0
                                    if fallback_max_score is None
                                    else float(fallback_max_score),
                                    fallback_refractory_s=float(fallback_refractory_s),
                                    fallback_exclusion_s=float(fallback_exclusion_s),
                                    marker_total=int(metric.marker_total),
                                    event_total=int(metric.event_total),
                                    true_positive=int(metric.true_positive),
                                    false_negative=int(metric.false_negative),
                                    false_positive=int(metric.false_positive),
                                    recall=float(metric.recall),
                                    precision=float(metric.precision),
                                    f1=float(metric.f1),
                                    primary_event_total=len(primary_events),
                                    fallback_event_total=len(fallback_events),
                                    added_fallback_event_total=max(0, len(fused_events) - len(primary_events)),
                                    negative_marker_total=int(negative_metric.negative_total),
                                    negative_conflict_total=int(negative_metric.conflict_total),
                                    negative_conflict_rate=float(negative_metric.conflict_rate),
                                )
                            )

    return CandidateFusionSweepResult(session=session_name, rows=tuple(rows))


def best_sweep_rows(
    rows: Sequence[CandidateFusionParameterRow],
    *,
    min_recall: float = 0.0,
    limit: int = 20,
) -> tuple[CandidateFusionParameterRow, ...]:
    eligible = [row for row in rows if float(row.recall) >= float(min_recall)]
    return tuple(
        sorted(
            eligible,
            key=lambda row: (
                row.negative_conflict_total,
                -row.f1,
                -row.precision,
                row.false_positive,
                row.event_total,
            ),
        )[: int(limit)]
    )


def write_candidate_fusion_sweep_outputs(result: CandidateFusionSweepResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "candidate_fusion_parameter_sweep.csv", result.rows)


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _optional_float_key(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)
