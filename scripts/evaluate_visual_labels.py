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

from hp_acoustic_wave.visual_label_audit import audit_visual_labels, write_visual_label_audit_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MediaPipe visual blink labels and EAR traces.",
    )
    parser.add_argument("session_dirs", nargs="+", help="Session directories to audit.")
    parser.add_argument("--visual-key", default="v", help="manual_markers.csv key used by visual auto labels.")
    parser.add_argument("--ear-threshold", type=float, default=0.22, help="EAR threshold expected by visual detector.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_visual_label_audit for one session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for raw_session in args.session_dirs:
        session_dir = Path(raw_session)
        if args.output_dir is None:
            output_dir = Path("docs") / "experiments" / f"{session_dir.name}_visual_label_audit"
        else:
            base = Path(args.output_dir)
            output_dir = base if len(args.session_dirs) == 1 else base / session_dir.name
        audit = audit_visual_labels(
            session_dir,
            visual_key=str(args.visual_key),
            ear_threshold=float(args.ear_threshold),
        )
        write_visual_label_audit_outputs(audit, output_dir)
        summary = audit.summary
        print(f"session={session_dir}")
        print(f"source={summary.source} output={output_dir}")
        print("rows events valid invalid markers mismatch errors available_rate face_found_rate valid_event_rate")
        print(
            f"{summary.row_total} {summary.visual_event_total} "
            f"{summary.valid_visual_event_total} {summary.invalid_visual_event_total} "
            f"{summary.visual_marker_total} {summary.visual_event_marker_mismatch} "
            f"{summary.error_total} {summary.available_rate:.3f} "
            f"{summary.face_found_rate:.3f} {summary.valid_visual_event_rate:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
