#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.primary_blink_evaluation import (
    evaluate_primary_blink_session,
    write_primary_blink_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate primary blink score peaks against manual blink markers.")
    parser.add_argument("session_dir", help="Session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Maximum absolute event-marker offset in seconds.")
    parser.add_argument("--ignore-startup", type=float, default=2.0, help="Ignore markers/events before this time.")
    parser.add_argument("--min-score", type=float, default=0.04, help="Minimum blink_score peak.")
    parser.add_argument("--min-ratio", type=float, default=0.8, help="Minimum blink_score / blink_threshold ratio.")
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.25,
        help="Maximum blink_score peak. Use 0 to disable the upper gate.",
    )
    parser.add_argument("--refractory", type=float, default=1.2, help="Minimum seconds between peak events.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_primary_blink_eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_primary_blink_eval"
    )
    evaluation = evaluate_primary_blink_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        min_score=float(args.min_score),
        min_ratio=float(args.min_ratio),
        max_score=float(args.max_score),
        refractory_s=float(args.refractory),
    )
    write_primary_blink_outputs(evaluation, output_dir)
    metric = evaluation.event_evaluation.metrics[0]
    best = max(evaluation.sweep, key=lambda row: (row.f1, row.recall, row.precision), default=None)
    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print("markers events tp fn fp recall precision f1")
    print(
        f"{metric.marker_total} {metric.event_total} {metric.true_positive} "
        f"{metric.false_negative} {metric.false_positive} "
        f"{metric.recall:.3f} {metric.precision:.3f} {metric.f1:.3f}"
    )
    if best is not None:
        print(
            "best_sweep "
            f"min_score={best.min_score:.3f} min_ratio={best.min_ratio:.3f} "
            f"max_score={best.max_score:.3f} "
            f"refractory={best.refractory_s:.3f} recall={best.recall:.3f} "
            f"precision={best.precision:.3f} f1={best.f1:.3f}"
        )
    confirm_best = max(
        evaluation.confirm_sweep,
        key=lambda row: (
            row.recall >= metric.recall - 1e-9,
            row.f1,
            row.precision,
            -row.false_positive,
        ),
        default=None,
    )
    if confirm_best is not None:
        print(
            "best_confirm_sweep "
            f"max_delta={confirm_best.max_delta_rms:.3f} "
            f"max_high_duration={confirm_best.max_high_delta_duration_s:.3f} "
            f"max_score={confirm_best.max_score:.3f} "
            f"require_pattern={int(confirm_best.require_pattern)} "
            f"min_vote_confidence={confirm_best.min_vote_confidence:.3f} "
            f"recall={confirm_best.recall:.3f} precision={confirm_best.precision:.3f} "
            f"f1={confirm_best.f1:.3f}"
        )
    shape_best = max(
        evaluation.shape_sweep,
        key=lambda row: (
            row.recall >= metric.recall - 1e-9,
            row.f1,
            row.precision,
            -row.false_positive,
        ),
        default=None,
    )
    if shape_best is not None:
        print(
            "best_shape_sweep "
            f"min_prominence={shape_best.min_prominence:.3f} "
            f"min_prominence_ratio={shape_best.min_prominence_ratio:.3f} "
            f"max_half_width={shape_best.max_half_width_s:.3f} "
            f"max_baseline_slope={shape_best.max_abs_baseline_slope:.3f} "
            f"min_symmetry={shape_best.min_pre_post_symmetry:.3f} "
            f"max_delta={shape_best.max_delta_rms:.3f} "
            f"recall={shape_best.recall:.3f} precision={shape_best.precision:.3f} "
            f"f1={shape_best.f1:.3f}"
        )
    reasons = Counter(row.reason for row in evaluation.marker_diagnostics if not row.matched)
    if reasons:
        print("fn_reasons " + " ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())))
    classifications = Counter(
        row.classification for row in evaluation.event_diagnostics if row.classification != "true_positive"
    )
    if classifications:
        print(
            "fp_classes "
            + " ".join(f"{classification}={count}" for classification, count in sorted(classifications.items()))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
