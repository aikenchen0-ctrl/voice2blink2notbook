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
    aggregate_candidate_fusion_evaluations,
    evaluate_candidate_fusion_session,
    write_candidate_fusion_aggregate_outputs,
    write_candidate_fusion_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate candidate fusion evaluation across multiple sessions.")
    parser.add_argument("session_dirs", nargs="+", help="Session directories containing features.csv and markers.")
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
    parser.add_argument("--fallback-exclusion", type=float, default=0.8)
    parser.add_argument("--diagnostic-window", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        default="docs/experiments/candidate_fusion_aggregate",
        help="Directory for aggregate outputs and per-session subdirectories.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    sessions_dir = output_dir / "sessions"
    evaluations = []
    for raw_session_dir in args.session_dirs:
        session_dir = Path(raw_session_dir)
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
        write_candidate_fusion_outputs(evaluation, sessions_dir / session_dir.name)
        evaluations.append(evaluation)

    aggregate = aggregate_candidate_fusion_evaluations(evaluations)
    write_candidate_fusion_aggregate_outputs(aggregate, output_dir)

    print(f"output={output_dir}")
    print(f"sessions={aggregate.session_count}")
    print("strategy markers events tp fn fp recall precision f1 negative_markers negative_conflicts")
    for row in aggregate.strategy_metrics:
        print(
            f"{row.strategy} {row.marker_total} {row.event_total} "
            f"{row.true_positive} {row.false_negative} {row.false_positive} "
            f"{row.recall:.3f} {row.precision:.3f} {row.f1:.3f} "
            f"{row.negative_marker_total} {row.negative_conflict_total}"
        )
    eligible = [row for row in aggregate.fmcw_sweep if row.recall >= 0.95]
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
