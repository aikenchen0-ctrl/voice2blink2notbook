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

from hp_acoustic_wave.candidate_fusion_sweep import (  # noqa: E402
    best_sweep_rows,
    sweep_candidate_fusion_session,
    write_candidate_fusion_sweep_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep primary/fallback blink fusion parameters.")
    parser.add_argument("session_dir", help="Session directory containing features.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--ignore-startup", type=float, default=2.0)
    parser.add_argument("--require-visual-face", action="store_true")
    parser.add_argument("--primary-min-scores", default="0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.14")
    parser.add_argument("--primary-min-ratios", default="0.8,1.0,1.5,2.0,3.0")
    parser.add_argument("--primary-max-scores", default="0.25")
    parser.add_argument("--primary-refractory-values", default="1.05,1.2,1.5,1.8,2.2")
    parser.add_argument("--fallback-min-scores", default="none,0.06,0.08,0.10,0.12,0.14")
    parser.add_argument("--fallback-max-scores", default="0.20")
    parser.add_argument("--fallback-refractory-values", default="1.05,1.5,2.0")
    parser.add_argument("--fallback-exclusion-values", default="0.8")
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("docs") / "experiments" / f"{session_dir.name}_candidate_fusion_parameter_sweep"
    )
    result = sweep_candidate_fusion_session(
        session_dir,
        tolerance_s=float(args.tolerance),
        ignore_startup_s=float(args.ignore_startup),
        primary_min_scores=_parse_float_list(args.primary_min_scores),
        primary_min_ratios=_parse_float_list(args.primary_min_ratios),
        primary_max_scores=_parse_float_list(args.primary_max_scores),
        primary_refractory_values=_parse_float_list(args.primary_refractory_values),
        fallback_min_scores=_parse_optional_float_list(args.fallback_min_scores),
        fallback_max_scores=_parse_optional_float_list(args.fallback_max_scores),
        fallback_refractory_values=_parse_float_list(args.fallback_refractory_values),
        fallback_exclusion_values=_parse_float_list(args.fallback_exclusion_values),
        require_visual_face=bool(args.require_visual_face),
    )
    write_candidate_fusion_sweep_outputs(result, output_dir)
    best = best_sweep_rows(result.rows, min_recall=float(args.min_recall), limit=int(args.limit))
    print(f"session={session_dir}")
    print(f"output={output_dir}")
    print(f"rows={len(result.rows)} min_recall={args.min_recall}")
    print(
        "rank recall precision f1 tp fn fp events primary_min_score primary_min_ratio "
        "primary_refractory fallback_enabled fallback_min_score fallback_refractory added_fallback"
    )
    for index, row in enumerate(best, start=1):
        print(
            f"{index} {row.recall:.3f} {row.precision:.3f} {row.f1:.3f} "
            f"{row.true_positive} {row.false_negative} {row.false_positive} {row.event_total} "
            f"{row.primary_min_score:.3f} {row.primary_min_ratio:.3f} {row.primary_refractory_s:.3f} "
            f"{int(row.fallback_enabled)} {row.fallback_min_score:.3f} "
            f"{row.fallback_refractory_s:.3f} {row.added_fallback_event_total}"
        )
    if not best:
        print("no_rows_at_min_recall")
    return 0


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_optional_float_list(value: str) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "off"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    return tuple(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
