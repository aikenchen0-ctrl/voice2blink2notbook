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

from hp_acoustic_wave.post_collection_evaluation import evaluate_post_collection_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-collection evaluation across multiple sessions.")
    parser.add_argument("session_dirs", nargs="+", help="Session directories containing collected CSV files.")
    parser.add_argument("--output-dir", default="docs/experiments/post_collection_dataset_eval")
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--ignore-startup", type=float, default=2.0)
    parser.add_argument("--no-require-visual-face", action="store_true")
    parser.add_argument("--min-visual-face-found-rate", type=float, default=0.50)
    parser.add_argument("--min-blink-markers", type=int, default=40)
    parser.add_argument("--min-negative-markers", type=int, default=20)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--sweep-min-recall", type=float, default=0.85)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    evaluation = evaluate_post_collection_dataset(
        [Path(raw) for raw in args.session_dirs],
        output_dir=output_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        require_visual_face=not bool(args.no_require_visual_face),
        min_visual_face_found_rate=float(args.min_visual_face_found_rate),
        min_blink_markers=int(args.min_blink_markers),
        min_negative_markers=int(args.min_negative_markers),
        target_recall=float(args.target_recall),
        min_precision=float(args.min_precision),
        sweep_min_recall=float(args.sweep_min_recall),
    )
    summary = evaluation.summary
    print(f"output={output_dir}")
    print(
        "sessions "
        f"count={summary.session_count} "
        f"visual_ready={summary.visual_face_ready_sessions} "
        f"visual_problem={summary.visual_face_problem_sessions}"
    )
    print(
        "markers "
        f"blink={summary.blink_markers}/{summary.min_blink_markers} "
        f"negative={summary.negative_markers}/{summary.min_negative_markers}"
    )
    print(
        "fused "
        f"markers={summary.fused_marker_total} events={summary.fused_event_total} "
        f"tp={summary.fused_true_positive} fn={summary.fused_false_negative} "
        f"fp={summary.fused_false_positive} recall={summary.fused_recall:.3f} "
        f"precision={summary.fused_precision:.3f} f1={summary.fused_f1:.3f} "
        f"negative_conflicts={summary.fused_negative_conflicts}"
    )
    print(
        "best_sweep "
        f"min_recall={summary.sweep_min_recall:.3f} "
        f"min_vote={summary.best_sweep_min_vote_evidence:.3f} "
        f"min_abs_traj={summary.best_sweep_min_abs_trajectory_value:.3f} "
        f"min_pair_stability={summary.best_sweep_min_pair_stability:.3f} "
        f"recall={summary.best_sweep_recall:.3f} "
        f"precision={summary.best_sweep_precision:.3f} "
        f"f1={summary.best_sweep_f1:.3f} "
        f"fp={summary.best_sweep_false_positive} "
        f"negative_conflicts={summary.best_sweep_negative_conflicts}"
    )
    print(
        "targets "
        f"recall={int(summary.reaches_target_recall)} "
        f"precision={int(summary.reaches_min_precision)}"
    )
    print(f"decision {summary.decision_status}: {summary.recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
