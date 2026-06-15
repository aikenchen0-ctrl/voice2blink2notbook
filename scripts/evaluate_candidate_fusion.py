#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.candidate_fusion_evaluation import (
    evaluate_candidate_fusion_session,
    write_candidate_fusion_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate primary blink candidates plus a logged fallback gate.")
    parser.add_argument("session_dir", help="Session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Maximum event-marker offset in seconds.")
    parser.add_argument("--ignore-startup", type=float, default=2.0, help="Ignore markers/events before this time.")
    parser.add_argument("--primary-min-score", type=float, default=0.04)
    parser.add_argument("--primary-min-ratio", type=float, default=0.8)
    parser.add_argument("--primary-max-score", type=float, default=0.25)
    parser.add_argument("--primary-refractory", type=float, default=1.2)
    parser.add_argument("--fallback-event-column", default="twinkle_candidate_peak")
    parser.add_argument("--fallback-score-column", default="blink_score")
    parser.add_argument("--fallback-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-min-score", type=float, default=None)
    parser.add_argument("--fallback-max-score", type=float, default=0.20)
    parser.add_argument("--fallback-refractory", type=float, default=1.05)
    parser.add_argument(
        "--fallback-exclusion",
        type=float,
        default=0.8,
        help="Do not add fallback events this close to an existing primary event.",
    )
    parser.add_argument(
        "--diagnostic-window",
        type=float,
        default=0.8,
        help="Window around each event used to summarize FMCW vote/trajectory evidence.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_candidate_fusion_eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_candidate_fusion_eval"
    )
    evaluation = evaluate_candidate_fusion_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        primary_min_score=float(args.primary_min_score),
        primary_min_ratio=float(args.primary_min_ratio),
        primary_max_score=float(args.primary_max_score),
        primary_refractory_s=float(args.primary_refractory),
        fallback_event_column=str(args.fallback_event_column),
        fallback_score_column=str(args.fallback_score_column),
        fallback_threshold=float(args.fallback_threshold),
        fallback_min_score=args.fallback_min_score,
        fallback_max_score=args.fallback_max_score,
        fallback_refractory_s=float(args.fallback_refractory),
        fallback_exclusion_s=float(args.fallback_exclusion),
        diagnostic_window_s=float(args.diagnostic_window),
    )
    write_candidate_fusion_outputs(evaluation, output_dir)

    primary = evaluation.primary_evaluation.metrics[0]
    fallback = evaluation.fallback_evaluation.metrics[0]
    fused = evaluation.fused_evaluation.metrics[0]
    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print("strategy markers events tp fn fp recall precision f1")
    for name, metric in (("primary", primary), ("fallback", fallback), ("fused", fused)):
        print(
            f"{name} {metric.marker_total} {metric.event_total} "
            f"{metric.true_positive} {metric.false_negative} {metric.false_positive} "
            f"{metric.recall:.3f} {metric.precision:.3f} {metric.f1:.3f}"
        )
    summary = evaluation.summary
    print(
        "fallback_added "
        f"events={summary.added_fallback_event_total} "
        f"rescued_markers={summary.rescued_marker_total} "
        f"false_positives={summary.added_fallback_false_positive_total}"
    )
    print(
        "negative_conflicts "
        f"markers={summary.negative_marker_total} "
        f"primary={summary.primary_negative_conflict_total} "
        f"fallback={summary.fallback_negative_conflict_total} "
        f"fused={summary.fused_negative_conflict_total}"
    )
    eligible = [row for row in evaluation.fmcw_sweep if row.recall >= 0.95]
    if eligible:
        best = sorted(
            eligible,
            key=lambda row: (row.negative_conflict_total, -row.f1, -row.precision, row.false_positive),
        )[0]
        print(
            "best_fmcw_sweep "
            f"min_vote={best.min_vote_evidence:.3f} "
            f"min_abs_traj={best.min_abs_trajectory_value:.3f} "
            f"min_pair_stability={best.min_pair_stability:.3f} "
            f"recall={best.recall:.3f} precision={best.precision:.3f} f1={best.f1:.3f} "
            f"fp={best.false_positive} neg_conflicts={best.negative_conflict_total}"
        )
    else:
        print("best_fmcw_sweep none_recall_ge_0.95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
