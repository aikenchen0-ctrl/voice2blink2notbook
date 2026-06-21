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

from hp_acoustic_wave.blink_layer_evaluation import (  # noqa: E402
    evaluate_blink_layers_session,
    write_blink_layer_outputs,
)


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
    evaluation = evaluate_blink_layers_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        require_visual_face=bool(args.require_visual_face),
    )
    write_blink_layer_outputs(evaluation, output_dir)

    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print("layer markers events tp fn fp recall precision f1")
    for row in evaluation.rows:
        print(
            f"{row.layer} {row.marker_total} {row.event_total} "
            f"{row.true_positive} {row.false_negative} {row.false_positive} "
            f"{row.recall:.3f} {row.precision:.3f} {row.f1:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
