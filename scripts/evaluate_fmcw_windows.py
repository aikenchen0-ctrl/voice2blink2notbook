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

from hp_acoustic_wave.fmcw_window_evaluation import (
    evaluate_fmcw_session_windows,
    write_window_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paper-style FMCW blink votes on marker and background windows.",
    )
    parser.add_argument("session_dirs", nargs="+", help="Session directories containing audio.wav.")
    parser.add_argument("--window", type=float, default=0.8, help="Half-window size around each center in seconds.")
    parser.add_argument(
        "--background-step",
        type=float,
        default=None,
        help="Distance between sampled background window centers. Defaults to 2*window.",
    )
    parser.add_argument(
        "--marker-exclusion",
        type=float,
        default=None,
        help="Exclude background centers this close to any manual marker. Defaults to 2*window.",
    )
    parser.add_argument(
        "--max-background-windows",
        type=int,
        default=40,
        help="Maximum sampled background windows per session. Use -1 for all.",
    )
    parser.add_argument(
        "--ignore-startup",
        type=float,
        default=2.0,
        help="Do not sample background centers before this many seconds from session start.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        default="0.5,0.6,0.7,0.8",
        help="Comma-separated vote confidence thresholds for metric sweep.",
    )
    parser.add_argument(
        "--accepted-patterns",
        default=None,
        help="Comma-separated patterns counted as a single blink. Defaults to FmcwConfig.confirm_single_blink_patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_window_eval for a single session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = _parse_float_list(args.confidence_thresholds)
    accepted_patterns = _parse_str_list(args.accepted_patterns)
    max_background_windows = None if int(args.max_background_windows) < 0 else int(args.max_background_windows)

    for raw_session in args.session_dirs:
        session_dir = Path(raw_session)
        if args.output_dir is None:
            output_dir = Path("docs") / "experiments" / f"{session_dir.name}_window_eval"
        else:
            base = Path(args.output_dir)
            output_dir = base if len(args.session_dirs) == 1 else base / session_dir.name

        evaluation = evaluate_fmcw_session_windows(
            session_dir,
            window_s=float(args.window),
            background_step_s=args.background_step,
            marker_exclusion_s=args.marker_exclusion,
            max_background_windows=max_background_windows,
            ignore_startup_s=float(args.ignore_startup),
            confidence_thresholds=thresholds,
            accepted_patterns=accepted_patterns,
        )
        write_window_evaluation_outputs(evaluation, output_dir)
        print(f"session={session_dir}")
        print(f"output={output_dir}")
        print("threshold recall precision false_positive_rate accuracy balanced_accuracy")
        for row in evaluation.metrics:
            print(
                f"{row.confidence_threshold:.2f} "
                f"{row.recall:.3f} "
                f"{row.precision:.3f} "
                f"{row.false_positive_rate:.3f} "
                f"{row.accuracy:.3f} "
                f"{row.balanced_accuracy:.3f}"
            )
    return 0


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return tuple(values)


def _parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    return values or None


if __name__ == "__main__":
    raise SystemExit(main())
