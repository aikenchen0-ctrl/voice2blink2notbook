#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.fmcw_template_diagnostics import summarize_template_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize exported FMCW phase-pair template curves.")
    parser.add_argument(
        "template_path",
        help="Path to fmcw_pair_templates.csv or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the input CSV directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template_path = _resolve_template_path(Path(args.template_path))
    output_dir = Path(args.output_dir) if args.output_dir is not None else template_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with template_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summaries = summarize_template_rows(rows)
    output_path = output_dir / "fmcw_pair_template_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summaries[0].__dataclass_fields__.keys()) if summaries else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for summary in summaries:
                writer.writerow(asdict(summary))

    print(f"output={output_path}")
    for summary in summaries:
        print(
            f"{summary.group}: span={summary.span:.4f} "
            f"peak_t={summary.peak_relative_time:+.2f} "
            f"endpoint={summary.endpoint_delta:.4f} "
            f"sign={summary.sign} blink_dist={summary.blink_distance:.4f}"
        )
    return 0


def _resolve_template_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "fmcw_pair_templates.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
