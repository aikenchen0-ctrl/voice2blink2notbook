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

from hp_acoustic_wave.event_evaluation import (
    evaluate_session_events,
    summarize_event_evaluations,
    write_event_aggregate_outputs,
    write_event_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate final blink events against manual blink markers.",
    )
    parser.add_argument("session_dirs", nargs="+", help="Session directories containing events.csv and manual_markers.csv.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Maximum absolute event-marker offset in seconds.")
    parser.add_argument(
        "--event-labels",
        default=None,
        help="Comma-separated event labels to count. Defaults to fmcw_confirmed_blink when present, otherwise blink_candidate.",
    )
    parser.add_argument(
        "--positive-labels",
        default="blink",
        help="Comma-separated manual marker labels counted as ground-truth blink positives.",
    )
    parser.add_argument("--ignore-startup", type=float, default=2.0, help="Ignore markers/events before this time.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/experiments/<session>_event_eval for one session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    event_labels = _parse_str_list(args.event_labels)
    positive_labels = _parse_str_list(args.positive_labels) or ("blink",)
    evaluations = []
    for raw_session in args.session_dirs:
        session_dir = Path(raw_session)
        if args.output_dir is None:
            output_dir = Path("docs") / "experiments" / f"{session_dir.name}_event_eval"
        else:
            base = Path(args.output_dir)
            output_dir = base if len(args.session_dirs) == 1 else base / session_dir.name
        evaluation = evaluate_session_events(
            session_dir,
            tolerance_s=float(args.tolerance),
            positive_labels=positive_labels,
            event_labels=event_labels,
            ignore_startup_s=float(args.ignore_startup),
        )
        evaluations.append(evaluation)
        write_event_evaluation_outputs(evaluation, output_dir)
        metric = evaluation.metrics[0]
        print(f"session={session_dir}")
        print(f"event_labels={','.join(evaluation.event_labels)} output={output_dir}")
        print("markers events tp fn fp recall precision f1 false_discovery_rate")
        print(
            f"{metric.marker_total} {metric.event_total} {metric.true_positive} "
            f"{metric.false_negative} {metric.false_positive} "
            f"{metric.recall:.3f} {metric.precision:.3f} {metric.f1:.3f} "
            f"{metric.false_discovery_rate:.3f}"
        )
    if len(evaluations) > 1:
        base = Path(args.output_dir) if args.output_dir is not None else Path("docs") / "experiments"
        aggregate = summarize_event_evaluations(evaluations)
        write_event_aggregate_outputs(aggregate, base)
        metric = aggregate.metrics[0]
        print("session=ALL")
        print(f"output={base}")
        print("markers events tp fn fp recall precision f1 false_discovery_rate")
        print(
            f"{metric.marker_total} {metric.event_total} {metric.true_positive} "
            f"{metric.false_negative} {metric.false_positive} "
            f"{metric.recall:.3f} {metric.precision:.3f} {metric.f1:.3f} "
            f"{metric.false_discovery_rate:.3f}"
        )
    return 0


def _parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    return values or None


if __name__ == "__main__":
    raise SystemExit(main())
