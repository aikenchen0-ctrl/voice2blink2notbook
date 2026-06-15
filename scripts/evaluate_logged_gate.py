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

from hp_acoustic_wave.logged_gate_evaluation import evaluate_logged_gate_session, write_logged_gate_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate events logged in features.csv against manual blink markers.")
    parser.add_argument("session_dir", help="Session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--event-column", default="twinkle_candidate_accepted", help="Feature column treated as an event flag.")
    parser.add_argument("--score-column", default="blink_score", help="Feature column copied into event score.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Event flag must be greater than this value.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional minimum score-column value.")
    parser.add_argument("--max-score", type=float, default=None, help="Optional maximum score-column value.")
    parser.add_argument("--refractory", type=float, default=0.0, help="Minimum seconds between rising-edge events.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Maximum absolute event-marker offset in seconds.")
    parser.add_argument("--ignore-startup", type=float, default=2.0, help="Ignore markers/events before this time.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_<event_column>_eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_{args.event_column}_eval"
    )
    evaluation = evaluate_logged_gate_session(
        session_dir,
        event_column=str(args.event_column),
        score_column=str(args.score_column),
        threshold=float(args.threshold),
        min_score=args.min_score,
        max_score=args.max_score,
        refractory_s=float(args.refractory),
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
    )
    write_logged_gate_outputs(evaluation, output_dir)
    metric = evaluation.event_evaluation.metrics[0]
    print(f"session={session_dir}")
    print(f"event_column={evaluation.event_column} output={output_dir}")
    print("markers events tp fn fp recall precision f1 false_discovery_rate")
    print(
        f"{metric.marker_total} {metric.event_total} {metric.true_positive} "
        f"{metric.false_negative} {metric.false_positive} "
        f"{metric.recall:.3f} {metric.precision:.3f} {metric.f1:.3f} "
        f"{metric.false_discovery_rate:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
