#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.candidate_fusion_evaluation import (
    aggregate_candidate_fusion_evaluations,
    evaluate_candidate_fusion_session,
    write_candidate_fusion_aggregate_outputs,
    write_candidate_fusion_outputs,
)
from hp_acoustic_wave.fixed_posture_study import (
    decide_fixed_posture_study,
    summarize_session_markers,
)
from hp_acoustic_wave.fmcw_pair_ranking import (
    AggregatePhasePairLineCorrelationRank,
    AggregatePhasePairRank,
    PHASE_PAIR_SCORE_METRICS,
    PhasePairLineCorrelationRank,
    PhasePairRank,
)
from hp_acoustic_wave.fmcw_session_analysis import fmcw_config_from_session, read_csv_rows
from scripts.rank_fmcw_phase_pairs_many import (
    OffsetSweepRow,
    _SessionInput,
    _evaluate_offset,
    _evaluate_line_correlations,
    _offset_sort_key,
    _parse_label_set,
    _parse_offset_sweep,
    _per_session_template_summary_rows,
    _template_rows_for_pair,
    _write_dict_rows,
    _write_rows,
    load_phase_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-posture blink study gate: labels, FMCW pair ranking, templates, and fusion metrics.",
    )
    parser.add_argument("session_dirs", nargs="+", help="Fixed-posture session directories.")
    parser.add_argument("--output-dir", default="docs/experiments/fixed_posture_study_latest")
    parser.add_argument("--min-blink-markers", type=int, default=40)
    parser.add_argument("--min-negative-markers", type=int, default=20)
    parser.add_argument("--negative-labels", default="large_motion")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--max-negative-conflicts", type=int, default=0)

    parser.add_argument("--window", type=float, default=0.6)
    parser.add_argument("--background-window", type=float, default=0.6)
    parser.add_argument("--center-offset", type=float, default=0.0)
    parser.add_argument("--center-offset-sweep", default="-0.4:0.4:0.1")
    parser.add_argument("--template-points", type=int, default=81)
    parser.add_argument("--metric", choices=PHASE_PAIR_SCORE_METRICS, default="shape")
    parser.add_argument("--min-recommend-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-recommend-separation", type=float, default=1.0)
    parser.add_argument("--min-recommend-line-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-recommend-line-separation", type=float, default=1.0)

    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--ignore-startup", type=float, default=2.0)
    parser.add_argument("--primary-min-score", type=float, default=0.04)
    parser.add_argument("--primary-min-ratio", type=float, default=0.8)
    parser.add_argument("--primary-max-score", type=float, default=0.25)
    parser.add_argument("--primary-refractory", type=float, default=1.2)
    parser.add_argument("--fallback-event-column", default="twinkle_candidate_peak")
    parser.add_argument("--fallback-score-column", default="blink_score")
    parser.add_argument("--fallback-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-min-score", type=float, default=None)
    parser.add_argument("--fallback-max-score", type=float, default=0.20)
    parser.add_argument("--fallback-refractory", type=float, default=1.05)
    parser.add_argument("--fallback-exclusion", type=float, default=0.8)
    parser.add_argument("--diagnostic-window", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    negative_labels = _parse_label_set(args.negative_labels)

    marker_summaries = [
        summarize_session_markers(Path(session_dir), negative_labels=negative_labels)
        for session_dir in args.session_dirs
    ]
    _write_marker_summaries(output_dir / "fixed_posture_marker_summary.csv", marker_summaries)

    phase_pair_dir = output_dir / "phase_pair"
    phase_pair_summary = _run_phase_pair_gate(
        [Path(session_dir) for session_dir in args.session_dirs],
        output_dir=phase_pair_dir,
        negative_labels=negative_labels,
        center_offsets=_parse_offset_sweep(args.center_offset_sweep, float(args.center_offset)),
        metric=str(args.metric),
        marker_window_s=float(args.window),
        background_window_s=float(args.background_window),
        template_points=int(args.template_points),
        min_recommend_hit_rate=float(args.min_recommend_hit_rate),
        min_recommend_separation=float(args.min_recommend_separation),
        min_recommend_line_hit_rate=float(args.min_recommend_line_hit_rate),
        min_recommend_line_separation=float(args.min_recommend_line_separation),
    )

    fusion_dir = output_dir / "candidate_fusion"
    fusion_summary = _run_candidate_fusion_gate(args, fusion_dir)

    fused_metrics = fusion_summary.get("fused", {})
    decision = decide_fixed_posture_study(
        blink_marker_total=sum(summary.blink_markers for summary in marker_summaries),
        negative_marker_total=sum(summary.negative_markers for summary in marker_summaries),
        session_count=len(marker_summaries),
        phase_point_log_session_count=sum(1 for summary in marker_summaries if summary.has_phase_point_log),
        recommended_pair=phase_pair_summary.get("recommended_line_pair"),
        fused_recall=float(fused_metrics.get("recall", 0.0)),
        fused_precision=float(fused_metrics.get("precision", 0.0)),
        fused_false_positives=int(fused_metrics.get("false_positive", 0)),
        fused_negative_conflicts=int(fused_metrics.get("negative_conflict_total", 0)),
        min_blink_markers=int(args.min_blink_markers),
        min_negative_markers=int(args.min_negative_markers),
        target_recall=float(args.target_recall),
        min_precision=float(args.min_precision),
        max_false_positives=int(args.max_false_positives),
        max_negative_conflicts=int(args.max_negative_conflicts),
    )
    summary = {
        "sessions": [asdict(row) for row in marker_summaries],
        "marker_totals": {
            "blink": sum(row.blink_markers for row in marker_summaries),
            "negative": sum(row.negative_markers for row in marker_summaries),
            "phase_point_log_sessions": sum(1 for row in marker_summaries if row.has_phase_point_log),
        },
        "phase_pair": phase_pair_summary,
        "candidate_fusion": fusion_summary,
        "decision": asdict(decision),
    }
    (output_dir / "fixed_posture_study_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"output={output_dir}")
    print(
        f"markers blink={summary['marker_totals']['blink']} "
        f"negative={summary['marker_totals']['negative']} "
        f"phase_logs={summary['marker_totals']['phase_point_log_sessions']}/{len(marker_summaries)}"
    )
    print(
        "phase_pair "
        f"recommended={phase_pair_summary.get('recommended_pair')} "
        f"best={phase_pair_summary.get('best_pair')} "
        f"hit={phase_pair_summary.get('best_pair_blink_hit_rate', 0.0):.3f}"
    )
    print(
        "physical_line "
        f"recommended={phase_pair_summary.get('recommended_line_pair')} "
        f"best={phase_pair_summary.get('best_line_pair')} "
        f"hit={phase_pair_summary.get('best_line_blink_hit_rate', 0.0):.3f}"
    )
    print(
        "fusion "
        f"recall={float(fused_metrics.get('recall', 0.0)):.3f} "
        f"precision={float(fused_metrics.get('precision', 0.0)):.3f} "
        f"false_positive={int(fused_metrics.get('false_positive', 0))} "
        f"negative_conflicts={int(fused_metrics.get('negative_conflict_total', 0))}"
    )
    print(f"decision {decision.status}: {decision.recommendation}")
    return 0


def _run_phase_pair_gate(
    session_dirs: list[Path],
    *,
    output_dir: Path,
    negative_labels: tuple[str, ...],
    center_offsets: tuple[float, ...],
    metric: str,
    marker_window_s: float,
    background_window_s: float,
    template_points: int,
    min_recommend_hit_rate: float,
    min_recommend_separation: float,
    min_recommend_line_hit_rate: float,
    min_recommend_line_separation: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session_inputs: list[_SessionInput] = []
    for session_dir in session_dirs:
        times, phase_matrix, phase_source = load_phase_matrix(session_dir)
        markers = read_csv_rows(session_dir / "manual_markers.csv")
        blink_markers = _marker_times(markers, ("blink",))
        negative_markers = _marker_times(markers, negative_labels)
        config = fmcw_config_from_session(session_dir)
        if blink_markers:
            session_inputs.append(
                _SessionInput(
                    session_dir=session_dir,
                    times=times,
                    phase_matrix=phase_matrix,
                    phase_source=phase_source,
                    blink_markers=tuple(blink_markers),
                    negative_markers=tuple(negative_markers),
                    valid_start=int(config.valid_start),
                    valid_stop=int(config.valid_stop),
                )
            )
    offset_results = [
        _evaluate_offset(
            session_inputs,
            center_offset_s=float(offset),
            metric=metric,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            min_recommend_hit_rate=min_recommend_hit_rate,
            min_recommend_separation=min_recommend_separation,
        )
        for offset in center_offsets
    ]
    offset_results = [result for result in offset_results if result is not None]
    if not offset_results:
        return {"recommended_pair": None, "reason": "no_rankable_phase_pairs"}
    offset_sweep = [result[0] for result in offset_results]
    _write_rows(output_dir / "fmcw_pair_rank_offset_sweep.csv", offset_sweep, OffsetSweepRow)
    best_offset_row, aggregate, per_session_ranks, session_summaries = sorted(
        offset_results,
        key=lambda item: _offset_sort_key(item[0]),
    )[0]
    _write_rows(output_dir / "fmcw_pair_rank_aggregate.csv", aggregate, AggregatePhasePairRank)
    for session_input, ranks in per_session_ranks:
        _write_rows(output_dir / f"{session_input.session_dir.name}_pair_rank.csv", ranks, PhasePairRank)
    best = aggregate[0]
    recommended = (
        best.blink_hit_rate >= min_recommend_hit_rate and best.separation >= min_recommend_separation
    )
    line_aggregate, line_per_session_ranks = _evaluate_line_correlations(
        session_inputs,
        center_offset_s=float(best_offset_row.center_offset_s),
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
        template_points=template_points,
    )
    if line_aggregate:
        _write_rows(
            output_dir / "fmcw_pair_line_correlation_aggregate.csv",
            line_aggregate,
            AggregatePhasePairLineCorrelationRank,
        )
        for session_input, ranks in line_per_session_ranks:
            _write_rows(
                output_dir / f"{session_input.session_dir.name}_line_correlation.csv",
                ranks,
                PhasePairLineCorrelationRank,
            )
    best_line = line_aggregate[0] if line_aggregate else None
    line_recommended = bool(
        best_line is not None
        and best_line.blink_hit_rate >= float(min_recommend_line_hit_rate)
        and best_line.separation >= float(min_recommend_line_separation)
        and best_line.session_count == len(session_inputs)
    )
    recommended_line_pair = (
        f"{best_line.reference_index}:{best_line.target_index}" if best_line is not None and line_recommended else None
    )
    templates = _template_rows_for_pair(
        session_inputs,
        reference_index=int(best.reference_index),
        target_index=int(best.target_index),
        center_offset_s=float(best_offset_row.center_offset_s),
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
        template_points=template_points,
    )
    if templates:
        _write_dict_rows(output_dir / "fmcw_pair_templates.csv", templates)
    per_session_template_summary = _per_session_template_summary_rows(
        per_session_ranks,
        center_offset_s=float(best_offset_row.center_offset_s),
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
        template_points=template_points,
    )
    if per_session_template_summary:
        _write_dict_rows(output_dir / "fmcw_pair_template_summary_by_session.csv", per_session_template_summary)
    summary = {
        "metric": metric,
        "selected_center_offset_s": best_offset_row.center_offset_s,
        "best_pair": f"{best.reference_index}:{best.target_index}",
        "best_pair_separation": best.separation,
        "best_pair_blink_hit_rate": best.blink_hit_rate,
        "best_pair_background_trigger_rate": best.background_trigger_rate,
        "best_pair_negative_trigger_rate": best.negative_trigger_rate,
        "recommended_pair": f"{best.reference_index}:{best.target_index}" if recommended else None,
        "line_correlation_csv": "fmcw_pair_line_correlation_aggregate.csv" if line_aggregate else None,
        "best_line_pair": (
            f"{best_line.reference_index}:{best_line.target_index}" if best_line is not None else None
        ),
        "best_line_separation": best_line.separation if best_line is not None else 0.0,
        "best_line_blink_hit_rate": best_line.blink_hit_rate if best_line is not None else 0.0,
        "best_line_background_trigger_rate": (
            best_line.background_trigger_rate if best_line is not None else 0.0
        ),
        "best_line_negative_trigger_rate": best_line.negative_trigger_rate if best_line is not None else 0.0,
        "recommended_line_pair": recommended_line_pair,
        "recommended_line_cli": (
            f"--fmcw-fixed-trajectory-pair {recommended_line_pair}"
            if recommended_line_pair is not None
            else None
        ),
        "line_recommendation_reason": (
            "line template correlation passed hit-rate, separation, and session-coverage gates"
            if line_recommended
            else "no physical line passed template-correlation gates"
        ),
        "sessions": session_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_candidate_fusion_gate(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    sessions_dir = output_dir / "sessions"
    evaluations = []
    for raw_session_dir in args.session_dirs:
        session_dir = Path(raw_session_dir)
        evaluation = evaluate_candidate_fusion_session(
            session_dir,
            tolerance_s=float(args.tolerance),
            ignore_startup_s=float(args.ignore_startup),
            primary_min_score=float(args.primary_min_score),
            primary_min_ratio=float(args.primary_min_ratio),
            primary_max_score=float(args.primary_max_score),
            primary_refractory_s=float(args.primary_refractory),
            fallback_event_column=str(args.fallback_event_column),
            fallback_score_column=str(args.fallback_score_column),
            fallback_threshold=float(args.fallback_threshold),
            fallback_min_score=args.fallback_min_score,
            fallback_max_score=args.fallback_max_score,
            fallback_refractory_s=float(args.fallback_refractory),
            fallback_exclusion_s=float(args.fallback_exclusion),
            diagnostic_window_s=float(args.diagnostic_window),
        )
        write_candidate_fusion_outputs(evaluation, sessions_dir / session_dir.name)
        evaluations.append(evaluation)
    aggregate = aggregate_candidate_fusion_evaluations(evaluations)
    write_candidate_fusion_aggregate_outputs(aggregate, output_dir)
    strategy = {row.strategy: asdict(row) for row in aggregate.strategy_metrics}
    eligible = [row for row in aggregate.fmcw_sweep if row.recall >= 0.95]
    best_fmcw_sweep = None
    if eligible:
        best_fmcw_sweep = asdict(
            sorted(
                eligible,
                key=lambda row: (row.negative_conflict_total, -row.f1, -row.precision, row.false_positive),
            )[0]
        )
    return {
        "session_count": aggregate.session_count,
        "primary": strategy.get("primary", {}),
        "fallback": strategy.get("fallback", {}),
        "fused": strategy.get("fused", {}),
        "best_fmcw_sweep_recall_ge_0_95": best_fmcw_sweep,
    }


def _marker_times(rows: list[dict[str, str]], labels: tuple[str, ...]) -> list[float]:
    label_set = set(labels)
    times: list[float] = []
    for row in rows:
        if row.get("label") not in label_set:
            continue
        try:
            times.append(float(row["time_s"]))
        except (KeyError, ValueError):
            continue
    return times


def _write_marker_summaries(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__.keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


if __name__ == "__main__":
    raise SystemExit(main())
