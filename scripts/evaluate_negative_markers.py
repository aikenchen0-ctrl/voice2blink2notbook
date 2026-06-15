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

from hp_acoustic_wave.negative_marker_evaluation import (
    evaluate_negative_markers_session,
    write_negative_marker_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate blink conflicts around negative manual markers.")
    parser.add_argument("session_dir", help="Session directory containing events.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Maximum absolute event-marker offset in seconds.")
    parser.add_argument("--negative-labels", default="large_motion", help="Comma-separated negative marker labels.")
    parser.add_argument(
        "--conflict-event-labels",
        default="fmcw_confirmed_blink",
        help="Comma-separated event labels counted as conflicts near negative markers.",
    )
    parser.add_argument(
        "--suppressed-event-labels",
        default="fmcw_suppressed_motion",
        help="Comma-separated event labels counted as successful suppression near negative markers.",
    )
    parser.add_argument("--ignore-startup", type=float, default=2.0, help="Ignore markers/events before this time.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_negative_marker_eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_negative_marker_eval"
    )
    evaluation = evaluate_negative_markers_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        negative_labels=_parse_str_list(args.negative_labels) or ("large_motion",),
        conflict_event_labels=_parse_str_list(args.conflict_event_labels) or ("fmcw_confirmed_blink",),
        suppressed_event_labels=_parse_str_list(args.suppressed_event_labels) or ("fmcw_suppressed_motion",),
        ignore_startup_s=float(args.ignore_startup),
    )
    write_negative_marker_outputs(evaluation, output_dir)
    metric = evaluation.metrics[0]
    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print("negative conflict suppressed conflict_rate suppression_rate")
    print(
        f"{metric.negative_total} {metric.conflict_total} {metric.suppressed_total} "
        f"{metric.conflict_rate:.3f} {metric.suppression_rate:.3f}"
    )
    return 0


def _parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    return values or None


if __name__ == "__main__":
    raise SystemExit(main())
