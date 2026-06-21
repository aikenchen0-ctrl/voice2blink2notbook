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

from hp_acoustic_wave.post_collection_evaluation import evaluate_post_collection_session  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fast post-collection blink evaluation bundle.")
    parser.add_argument("session_dir", help="Session directory containing collected CSV files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--ignore-startup", type=float, default=2.0)
    parser.add_argument("--no-require-visual-face", action="store_true")
    parser.add_argument("--min-negative-markers", type=int, default=20)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--sweep-min-recall", type=float, default=0.85)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_post_collection_eval"
    )
    evaluation = evaluate_post_collection_session(
        session_dir,
        output_dir=output_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        require_visual_face=not bool(args.no_require_visual_face),
        min_negative_markers=int(args.min_negative_markers),
        target_recall=float(args.target_recall),
        min_precision=float(args.min_precision),
        sweep_min_recall=float(args.sweep_min_recall),
    )
    summary = evaluation.summary
    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print(
        "markers "
        f"blink={summary.blink_markers} negative={summary.negative_markers} "
        f"needs_negative_labels={int(summary.needs_negative_labels)}"
    )
    print(
        "visual "
        f"events={summary.visual_events} valid={summary.valid_visual_events} "
        f"valid_rate={summary.visual_valid_event_rate:.3f}"
    )
    print(
        "layer_best "
        f"{summary.layer_best_name} recall={summary.layer_best_recall:.3f} "
        f"precision={summary.layer_best_precision:.3f} f1={summary.layer_best_f1:.3f}"
    )
    print(
        "fused "
        f"recall={summary.fused_recall:.3f} precision={summary.fused_precision:.3f} "
        f"f1={summary.fused_f1:.3f} fp={summary.fused_false_positive} "
        f"negative_conflicts={summary.fused_negative_conflicts}"
    )
    print(
        "sweep_best "
        f"min_recall={summary.sweep_min_recall:.3f} recall={summary.sweep_best_recall:.3f} "
        f"precision={summary.sweep_best_precision:.3f} f1={summary.sweep_best_f1:.3f} "
        f"fp={summary.sweep_best_false_positive}"
    )
    print(
        "targets "
        f"recall={int(summary.reaches_target_recall)} "
        f"precision={int(summary.reaches_min_precision)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
