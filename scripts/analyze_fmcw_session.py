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

from hp_acoustic_wave.fmcw_session_analysis import analyze_session, write_analysis_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a labeled FMCW session.")
    parser.add_argument("session_dir", help="Path to a session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--window", type=float, default=0.8, help="Marker-centered analysis window in seconds.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session-name>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    if args.output_dir is None:
        output_dir = Path("docs") / "experiments" / session_dir.name
    else:
        output_dir = Path(args.output_dir)

    analysis = analyze_session(session_dir, window_s=float(args.window))
    write_analysis_outputs(analysis, output_dir)
    print(f"Analyzed {analysis.marker_rows} markers from {session_dir}")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
