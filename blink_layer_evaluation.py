from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from hp_acoustic_wave.candidate_fusion_evaluation import evaluate_candidate_fusion_session
from hp_acoustic_wave.event_evaluation import evaluate_events, visual_face_time_ranges
from hp_acoustic_wave.logged_gate_evaluation import evaluate_logged_gate_session
from hp_acoustic_wave.primary_blink_evaluation import (
    detect_primary_blink_peaks,
    primary_events_as_dicts,
    read_csv_rows,
)


@dataclass(frozen=True)
class BlinkLayerMetricRow:
    session: str
    layer: str
    marker_total: int
    event_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class BlinkLayerEvaluation:
    session: str
    rows: tuple[BlinkLayerMetricRow, ...]


def evaluate_blink_layers_session(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    ignore_startup_s: float = 2.0,
    require_visual_face: bool = False,
) -> BlinkLayerEvaluation:
    session_dir = Path(session_dir)
    rows = []
    for layer, event_column, score_column in (
        ("detector_is_event", "blink_detector_is_event", "blink_score"),
        ("peak_is_event", "blink_peak_is_event", "blink_peak_score"),
        ("current_blink_is_event", "blink_is_event", "blink_score"),
        ("twinkle_accepted", "twinkle_candidate_accepted", "blink_score"),
    ):
        evaluation = evaluate_logged_gate_session(
            session_dir,
            event_column=event_column,
            score_column=score_column,
            threshold=0.5,
            tolerance_s=float(tolerance_s),
            ignore_startup_s=float(ignore_startup_s),
            require_visual_face=bool(require_visual_face),
        )
        rows.append(_metric_row(session_dir.name, layer, evaluation.event_evaluation.metrics[0]))

    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    valid_time_ranges = (
        visual_face_time_ranges(session_dir / "visual_features.csv")
        if bool(require_visual_face)
        else None
    )
    primary_events = detect_primary_blink_peaks(
        session_name=session_dir.name,
        feature_rows=features,
        min_score=0.04,
        min_ratio=0.8,
        max_score=0.25,
        refractory_s=1.2,
        ignore_startup_s=float(ignore_startup_s),
    )
    primary_eval = evaluate_events(
        session_name=session_dir.name,
        markers=markers,
        events=primary_events_as_dicts(primary_events),
        tolerance_s=float(tolerance_s),
        event_labels=("primary_blink_peak",),
        ignore_startup_s=float(ignore_startup_s),
        valid_time_ranges=valid_time_ranges,
    )
    rows.append(_metric_row(session_dir.name, "offline_primary_peak", primary_eval.metrics[0]))

    fused = evaluate_candidate_fusion_session(
        session_dir,
        tolerance_s=float(tolerance_s),
        ignore_startup_s=float(ignore_startup_s),
        require_visual_face=bool(require_visual_face),
    )
    rows.append(_metric_row(session_dir.name, "candidate_fused", fused.fused_evaluation.metrics[0]))
    return BlinkLayerEvaluation(session=session_dir.name, rows=tuple(rows))


def write_blink_layer_outputs(evaluation: BlinkLayerEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "blink_layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(evaluation.rows[0]).keys()))
        writer.writeheader()
        for row in evaluation.rows:
            writer.writerow(asdict(row))


def _metric_row(session: str, layer: str, metric) -> BlinkLayerMetricRow:
    return BlinkLayerMetricRow(
        session=session,
        layer=layer,
        marker_total=int(metric.marker_total),
        event_total=int(metric.event_total),
        true_positive=int(metric.true_positive),
        false_negative=int(metric.false_negative),
        false_positive=int(metric.false_positive),
        recall=float(metric.recall),
        precision=float(metric.precision),
        f1=float(metric.f1),
    )
