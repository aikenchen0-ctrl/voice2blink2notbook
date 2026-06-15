#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.dsp import FmcwStreamProcessor
from hp_acoustic_wave.fmcw_pair_ranking import (
    AggregatePhasePairLineCorrelationRank,
    AggregatePhasePairRank,
    PHASE_PAIR_SCORE_METRICS,
    PhasePairLineCorrelationRank,
    PhasePairLineEvidence,
    PhasePairPeaks,
    PhasePairRank,
    aggregate_phase_pair_line_evidence,
    aggregate_phase_pair_peaks,
    collect_phase_pair_line_evidence,
    collect_phase_pair_peaks,
    phase_pair_template_points,
    rank_phase_pair_line_evidence,
    rank_phase_pair_peaks,
)
from hp_acoustic_wave.fmcw_session_analysis import fmcw_config_from_session, read_csv_rows
from hp_acoustic_wave.fmcw_template_diagnostics import summarize_template_rows


@dataclass(frozen=True)
class _SessionInput:
    session_dir: Path
    times: np.ndarray
    phase_matrix: np.ndarray
    phase_source: str
    blink_markers: tuple[float, ...]
    negative_markers: tuple[float, ...]
    valid_start: int
    valid_stop: int


@dataclass(frozen=True)
class OffsetSweepRow:
    center_offset_s: float
    best_reference_index: int
    best_target_index: int
    session_count: int
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    negative_trigger_rate: float
    blink_peak_median: float
    decision_threshold: float
    recommended: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate FMCW phase-pair ranking across labeled sessions.",
    )
    parser.add_argument("session_dirs", nargs="+", help="Session directories with features/manual markers.")
    parser.add_argument("--window", type=float, default=0.6, help="Marker-centered half-window in seconds.")
    parser.add_argument("--background-window", type=float, default=0.6, help="Background window length in seconds.")
    parser.add_argument("--center-offset", type=float, default=0.0, help="Seconds added to manual markers before windows are cut.")
    parser.add_argument(
        "--center-offset-sweep",
        default=None,
        help="Optional start:stop:step seconds, e.g. -0.4:0.4:0.1. Overrides --center-offset.",
    )
    parser.add_argument("--template-points", type=int, default=81, help="Resampled points per exported template curve.")
    parser.add_argument(
        "--metric",
        choices=PHASE_PAIR_SCORE_METRICS,
        default="shape",
        help="delta_peak uses adjacent-chirp jumps; shape scores returning open/close trajectory windows.",
    )
    parser.add_argument("--negative-labels", default="large_motion", help="Comma-separated labels treated as interference.")
    parser.add_argument("--min-recommend-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-recommend-separation", type=float, default=1.0)
    parser.add_argument("--min-recommend-line-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-recommend-line-separation", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=20, help="Rows to print.")
    parser.add_argument(
        "--output-dir",
        default="docs/experiments/fmcw_pair_rank_latest",
        help="Directory for aggregate and per-session CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    negative_labels = _parse_label_set(args.negative_labels)
    center_offsets = _parse_offset_sweep(args.center_offset_sweep, float(args.center_offset))

    session_inputs: list[_SessionInput] = []
    for raw_session_dir in args.session_dirs:
        session_dir = Path(raw_session_dir)
        try:
            times, phase_matrix, phase_source = load_phase_matrix(session_dir)
            markers = read_csv_rows(session_dir / "manual_markers.csv")
        except (OSError, ValueError) as exc:
            print(f"skip {session_dir}: {exc}", file=sys.stderr)
            continue

        blink_markers = _marker_times(markers, ("blink",))
        negative_markers = _marker_times(markers, negative_labels)
        if not blink_markers:
            print(f"skip {session_dir}: no blink markers", file=sys.stderr)
            continue

        config = fmcw_config_from_session(session_dir)
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

    if not session_inputs:
        print("no usable sessions", file=sys.stderr)
        return 2

    offset_results = [
        _evaluate_offset(
            session_inputs,
            center_offset_s=float(offset),
            metric=str(args.metric),
            marker_window_s=float(args.window),
            background_window_s=float(args.background_window),
            min_recommend_hit_rate=float(args.min_recommend_hit_rate),
            min_recommend_separation=float(args.min_recommend_separation),
        )
        for offset in center_offsets
    ]
    offset_results = [result for result in offset_results if result is not None]
    if not offset_results:
        print("no rankable phase pairs", file=sys.stderr)
        return 2
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
    is_recommended = (
        best.blink_hit_rate >= float(args.min_recommend_hit_rate)
        and best.separation >= float(args.min_recommend_separation)
    )
    recommended_pair = f"{best.reference_index}:{best.target_index}" if is_recommended else None
    line_aggregate, line_per_session_ranks = _evaluate_line_correlations(
        session_inputs,
        center_offset_s=float(best_offset_row.center_offset_s),
        marker_window_s=float(args.window),
        background_window_s=float(args.background_window),
        template_points=int(args.template_points),
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
        and best_line.blink_hit_rate >= float(args.min_recommend_line_hit_rate)
        and best_line.separation >= float(args.min_recommend_line_separation)
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
        marker_window_s=float(args.window),
        background_window_s=float(args.background_window),
        template_points=int(args.template_points),
    )
    if templates:
        _write_dict_rows(output_dir / "fmcw_pair_templates.csv", templates)
    per_session_template_summary = _per_session_template_summary_rows(
        per_session_ranks,
        center_offset_s=float(best_offset_row.center_offset_s),
        marker_window_s=float(args.window),
        background_window_s=float(args.background_window),
        template_points=int(args.template_points),
    )
    if per_session_template_summary:
        _write_dict_rows(output_dir / "fmcw_pair_template_summary_by_session.csv", per_session_template_summary)
    summary = {
        "session_count": len(session_inputs),
        "metric": str(args.metric),
        "center_offsets": list(center_offsets),
        "selected_center_offset_s": best_offset_row.center_offset_s,
        "sessions": session_summaries,
        "best_pair": f"{best.reference_index}:{best.target_index}",
        "best_pair_separation": best.separation,
        "best_pair_blink_hit_rate": best.blink_hit_rate,
        "best_pair_background_trigger_rate": best.background_trigger_rate,
        "best_pair_negative_trigger_rate": best.negative_trigger_rate,
        "recommended_pair": recommended_pair,
        "recommended_cli": (
            f"--fmcw-fixed-trajectory-pair {recommended_pair}" if recommended_pair is not None else None
        ),
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
        "recommendation_reason": (
            "passed minimum hit-rate and separation gates"
            if is_recommended
            else "no phase pair passed the minimum hit-rate and separation gates"
        ),
        "line_recommendation_reason": (
            "line template correlation passed hit-rate, separation, and session-coverage gates"
            if line_recommended
            else "no physical line passed template-correlation gates"
        ),
        "template_csv": "fmcw_pair_templates.csv" if templates else None,
        "per_session_template_summary_csv": (
            "fmcw_pair_template_summary_by_session.csv" if per_session_template_summary else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(
        f"metric={args.metric} offset={best_offset_row.center_offset_s:+.3f}s "
        f"best_pair={best.reference_index}:{best.target_index} "
        f"sep={best.separation:.3f} hit={best.blink_hit_rate:.2f} "
        f"bg={best.background_trigger_rate:.2f} neg={best.negative_trigger_rate:.2f} "
        f"recommended={'yes' if is_recommended else 'no'}"
    )
    if best_line is not None:
        print(
            f"line_pair={best_line.reference_index}:{best_line.target_index} "
            f"line_sep={best_line.separation:.3f} line_hit={best_line.blink_hit_rate:.2f} "
            f"line_bg={best_line.background_trigger_rate:.2f} line_neg={best_line.negative_trigger_rate:.2f} "
            f"line_recommended={'yes' if line_recommended else 'no'}"
        )
    print("rank  pair    sessions  sep    blink_med  threshold  blink_hit  bg_trigger  neg_trigger")
    for index, rank in enumerate(aggregate[: max(1, int(args.top))], 1):
        print(
            f"{index:>4}  {rank.reference_index:02d}:{rank.target_index:02d}  "
            f"{rank.session_count:>8}  {rank.separation:6.3f}  "
            f"{rank.blink_peak_median:9.5f}  {rank.decision_threshold:9.5f}  "
            f"{rank.blink_hit_rate:8.2f}  {rank.background_trigger_rate:10.2f}  "
            f"{rank.negative_trigger_rate:10.2f}"
        )
    return 0


def _evaluate_offset(
    session_inputs: list[_SessionInput],
    *,
    center_offset_s: float,
    metric: str,
    marker_window_s: float,
    background_window_s: float,
    min_recommend_hit_rate: float,
    min_recommend_separation: float,
) -> tuple[OffsetSweepRow, tuple[AggregatePhasePairRank, ...], list[tuple[_SessionInput, tuple[PhasePairRank, ...]]], list[dict[str, object]]] | None:
    all_peak_sets: list[tuple[PhasePairPeaks, ...]] = []
    per_session_ranks: list[tuple[_SessionInput, tuple[PhasePairRank, ...]]] = []
    session_summaries: list[dict[str, object]] = []
    for session_input in session_inputs:
        peak_set = collect_phase_pair_peaks(
            session_input.times,
            session_input.phase_matrix,
            session_input.blink_markers,
            negative_markers=session_input.negative_markers,
            valid_start=session_input.valid_start,
            valid_stop=session_input.valid_stop,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            metric=metric,
            center_offset_s=center_offset_s,
        )
        if not peak_set:
            continue
        ranks = rank_phase_pair_peaks(peak_set)
        if not ranks:
            continue
        all_peak_sets.append(peak_set)
        per_session_ranks.append((session_input, ranks))
        best = ranks[0]
        session_summaries.append(
            {
                "session": str(session_input.session_dir),
                "phase_source": session_input.phase_source,
                "blink_markers": len(session_input.blink_markers),
                "negative_markers": len(session_input.negative_markers),
                "phase_rows": int(session_input.phase_matrix.shape[0]),
                "phase_points": int(session_input.phase_matrix.shape[1]),
                "metric": metric,
                "center_offset_s": center_offset_s,
                "best_pair": f"{best.reference_index}:{best.target_index}",
                "best_separation": best.separation,
                "best_blink_hit_rate": best.blink_hit_rate,
            }
        )
    if not all_peak_sets:
        return None
    aggregate = aggregate_phase_pair_peaks(all_peak_sets)
    if not aggregate:
        return None
    best = aggregate[0]
    recommended = int(
        best.blink_hit_rate >= min_recommend_hit_rate and best.separation >= min_recommend_separation
    )
    return (
        OffsetSweepRow(
            center_offset_s=float(center_offset_s),
            best_reference_index=int(best.reference_index),
            best_target_index=int(best.target_index),
            session_count=int(len(all_peak_sets)),
            separation=float(best.separation),
            blink_hit_rate=float(best.blink_hit_rate),
            background_trigger_rate=float(best.background_trigger_rate),
            negative_trigger_rate=float(best.negative_trigger_rate),
            blink_peak_median=float(best.blink_peak_median),
            decision_threshold=float(best.decision_threshold),
            recommended=recommended,
        ),
        aggregate,
        per_session_ranks,
        session_summaries,
    )


def _evaluate_line_correlations(
    session_inputs: list[_SessionInput],
    *,
    center_offset_s: float,
    marker_window_s: float,
    background_window_s: float,
    template_points: int,
) -> tuple[
    tuple[AggregatePhasePairLineCorrelationRank, ...],
    list[tuple[_SessionInput, tuple[PhasePairLineCorrelationRank, ...]]],
]:
    all_evidence_sets: list[tuple[PhasePairLineEvidence, ...]] = []
    per_session_ranks: list[tuple[_SessionInput, tuple[PhasePairLineCorrelationRank, ...]]] = []
    for session_input in session_inputs:
        evidence = collect_phase_pair_line_evidence(
            session_input.times,
            session_input.phase_matrix,
            session_input.blink_markers,
            negative_markers=session_input.negative_markers,
            valid_start=session_input.valid_start,
            valid_stop=session_input.valid_stop,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            center_offset_s=center_offset_s,
            template_points=template_points,
        )
        if not evidence:
            continue
        ranks = rank_phase_pair_line_evidence(evidence)
        if not ranks:
            continue
        all_evidence_sets.append(evidence)
        per_session_ranks.append((session_input, ranks))
    if not all_evidence_sets:
        return tuple(), []
    return aggregate_phase_pair_line_evidence(all_evidence_sets), per_session_ranks


def _template_rows_for_pair(
    session_inputs: list[_SessionInput],
    *,
    reference_index: int,
    target_index: int,
    center_offset_s: float,
    marker_window_s: float,
    background_window_s: float,
    template_points: int,
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int], list[tuple[float, float, int]]] = {}
    for session_input in session_inputs:
        rows = phase_pair_template_points(
            session_input.times,
            session_input.phase_matrix,
            reference_index,
            target_index,
            session_input.blink_markers,
            negative_markers=session_input.negative_markers,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            center_offset_s=center_offset_s,
            template_points=template_points,
        )
        for row in rows:
            by_key.setdefault((row.group, row.point_index), []).append((row.mean, row.std, row.count))
    output: list[dict[str, object]] = []
    for group, point_index in sorted(by_key, key=lambda item: (item[0], item[1])):
        values = by_key[(group, point_index)]
        means = np.asarray([value[0] for value in values], dtype=np.float64)
        counts = [value[2] for value in values]
        relative_time = -1.0 + 2.0 * point_index / float(max(1, template_points - 1))
        output.append(
            {
                "group": group,
                "reference_index": reference_index,
                "target_index": target_index,
                "center_offset_s": f"{center_offset_s:.6f}",
                "point_index": point_index,
                "relative_time": f"{relative_time:.9f}",
                "mean": f"{float(np.mean(means)):.9f}",
                "std": f"{float(np.std(means)):.9f}",
                "session_count": len(values),
                "window_count": sum(counts),
            }
        )
    return output


def _per_session_template_summary_rows(
    per_session_ranks: list[tuple[_SessionInput, tuple[PhasePairRank, ...]]],
    *,
    center_offset_s: float,
    marker_window_s: float,
    background_window_s: float,
    template_points: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for session_input, ranks in per_session_ranks:
        if not ranks:
            continue
        best = ranks[0]
        template_rows = phase_pair_template_points(
            session_input.times,
            session_input.phase_matrix,
            int(best.reference_index),
            int(best.target_index),
            session_input.blink_markers,
            negative_markers=session_input.negative_markers,
            marker_window_s=marker_window_s,
            background_window_s=background_window_s,
            center_offset_s=center_offset_s,
            template_points=template_points,
        )
        csv_like_rows = [
            {
                "group": row.group,
                "reference_index": str(row.reference_index),
                "target_index": str(row.target_index),
                "center_offset_s": str(row.center_offset_s),
                "point_index": str(row.point_index),
                "relative_time": str(row.relative_time),
                "mean": str(row.mean),
                "std": str(row.std),
                "session_count": "1",
                "window_count": str(row.count),
            }
            for row in template_rows
        ]
        for summary in summarize_template_rows(csv_like_rows):
            output.append(
                {
                    "session": session_input.session_dir.name,
                    "phase_source": session_input.phase_source,
                    "best_pair": f"{best.reference_index}:{best.target_index}",
                    "best_separation": f"{best.separation:.9f}",
                    "best_blink_hit_rate": f"{best.blink_hit_rate:.9f}",
                    "group": summary.group,
                    "reference_index": summary.reference_index,
                    "target_index": summary.target_index,
                    "center_offset_s": f"{summary.center_offset_s:.6f}",
                    "point_count": summary.point_count,
                    "window_count": summary.window_count,
                    "peak_relative_time": f"{summary.peak_relative_time:.9f}",
                    "peak_value": f"{summary.peak_value:.9f}",
                    "span": f"{summary.span:.9f}",
                    "endpoint_delta": f"{summary.endpoint_delta:.9f}",
                    "sign": summary.sign,
                    "rms": f"{summary.rms:.9f}",
                    "blink_distance": f"{summary.blink_distance:.9f}",
                }
            )
    return output


def load_phase_matrix(session_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    feature_path = Path(session_dir) / "features.csv"
    if feature_path.exists():
        rows = read_csv_rows(feature_path)
        times: list[float] = []
        phases: list[tuple[float, ...]] = []
        row_length: int | None = None
        for row in rows:
            encoded = row.get("fmcw_phase_points", "")
            if not encoded:
                continue
            points = _parse_phase_points(encoded)
            if row_length is None:
                row_length = len(points)
            if len(points) != row_length:
                continue
            try:
                times.append(float(row["time_s"]))
            except (KeyError, ValueError):
                continue
            phases.append(points)
        if len(phases) >= 2:
            return (
                np.asarray(times, dtype=np.float64),
                np.asarray(phases, dtype=np.float64),
                "features.csv:fmcw_phase_points",
            )

    sample_rate, audio = _read_wav_mono(Path(session_dir) / "audio.wav")
    config = fmcw_config_from_session(session_dir)
    processor = FmcwStreamProcessor(config, sample_rate)
    sync_lag = _infer_sync_lag(session_dir)
    if sync_lag is not None:
        processor.period_start_offset = int(sync_lag)
    features = processor.process_block(audio, 0)
    if len(features) < 2:
        raise ValueError("not enough reconstructed FMCW phase rows")
    return (
        np.asarray([feature.time_s for feature in features], dtype=np.float64),
        np.asarray([feature.phase_points for feature in features], dtype=np.float64),
        "audio.wav:reconstructed",
    )


def _parse_phase_points(encoded: str) -> tuple[float, ...]:
    values = []
    for raw_part in encoded.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        values.append(float(part))
    return tuple(values)


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


def _parse_label_set(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    return labels or ("large_motion",)


def _parse_offset_sweep(value: str | None, default_offset: float) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return (float(default_offset),)
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("center-offset-sweep must look like start:stop:step")
    try:
        start, stop, step = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center-offset-sweep values must be numbers") from exc
    if step == 0.0:
        raise argparse.ArgumentTypeError("center-offset-sweep step must be non-zero")
    offsets: list[float] = []
    current = start
    epsilon = abs(step) * 1e-6
    if step > 0.0:
        while current <= stop + epsilon:
            offsets.append(round(current, 9))
            current += step
    else:
        while current >= stop - epsilon:
            offsets.append(round(current, 9))
            current += step
    if not offsets:
        raise argparse.ArgumentTypeError("center-offset-sweep produced no offsets")
    return tuple(offsets)


def _offset_sort_key(row: OffsetSweepRow) -> tuple[float, float, float, float, float]:
    return (
        -float(row.separation),
        -float(row.blink_hit_rate),
        float(row.negative_trigger_rate),
        float(row.background_trigger_rate),
        abs(float(row.center_offset_s)),
    )


def _write_rows(path: Path, rows, row_type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got sample width {sample_width}")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return sample_rate, pcm.reshape(-1)


def _infer_sync_lag(session_dir: Path) -> int | None:
    values: list[int] = []
    for name in ("features.csv", "manual_markers.csv"):
        path = Path(session_dir) / name
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            value = row.get("fmcw_sync_lag_samples")
            if value in (None, ""):
                continue
            try:
                values.append(int(float(value)))
            except ValueError:
                continue
    if not values:
        return None
    return max(set(values), key=values.count)


if __name__ == "__main__":
    raise SystemExit(main())
