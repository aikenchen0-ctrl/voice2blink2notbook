#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.candidate_fusion_evaluation import evaluate_candidate_fusion_session  # noqa: E402
from hp_acoustic_wave.event_evaluation import evaluate_events, visual_face_time_ranges  # noqa: E402
from hp_acoustic_wave.logged_gate_evaluation import evaluate_logged_gate_session  # noqa: E402
from hp_acoustic_wave.primary_blink_evaluation import (  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare blink detector layers against the same labels.")
    parser.add_argument("session_dir", help="Session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--ignore-startup", type=float, default=2.0)
    parser.add_argument("--require-visual-face", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_blink_layer_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
            tolerance_s=float(args.tolerance),
            ignore_startup_s=float(args.ignore_startup),
            require_visual_face=bool(args.require_visual_face),
        )
        rows.append(_metric_row(session_dir.name, layer, evaluation.event_evaluation.metrics[0]))

    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    valid_time_ranges = (
        visual_face_time_ranges(session_dir / "visual_features.csv")
        if bool(args.require_visual_face)
        else None
    )
    primary_events = detect_primary_blink_peaks(
        session_name=session_dir.name,
        feature_rows=features,
        min_score=0.04,
        min_ratio=0.8,
        max_score=0.25,
        refractory_s=1.2,
        ignore_startup_s=float(args.ignore_startup),
    )
    primary_eval = evaluate_events(
        session_name=session_dir.name,
        markers=markers,
        events=primary_events_as_dicts(primary_events),
        tolerance_s=float(args.tolerance),
        event_labels=("primary_blink_peak",),
        ignore_startup_s=float(args.ignore_startup),
        valid_time_ranges=valid_time_ranges,
    )
    rows.append(_metric_row(session_dir.name, "offline_primary_peak", primary_eval.metrics[0]))

    fused = evaluate_candidate_fusion_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        require_visual_face=bool(args.require_visual_face),
    )
    rows.append(_metric_row(session_dir.name, "candidate_fused", fused.fused_evaluation.metrics[0]))

    output_file = output_dir / "blink_layer_metrics.csv"
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print("layer markers events tp fn fp recall precision f1")
    for row in rows:
        print(
            f"{row.layer} {row.marker_total} {row.event_total} "
            f"{row.true_positive} {row.false_negative} {row.false_positive} "
            f"{row.recall:.3f} {row.precision:.3f} {row.f1:.3f}"
        )
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
