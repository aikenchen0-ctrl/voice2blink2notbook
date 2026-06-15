from __future__ import annotations

import csv
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

from hp_acoustic_wave.config import FmcwConfig
from hp_acoustic_wave.dsp import FmcwStreamProcessor
from hp_acoustic_wave.fmcw_blink import decode_phase_matrix_vote
from hp_acoustic_wave.fmcw_session_analysis import fmcw_config_from_session, read_csv_rows


@dataclass(frozen=True)
class FmcwWindowVoteRow:
    session: str
    window_id: int
    source: str
    label: str
    key: str
    center_time_s: float
    window_s: float
    row_count: int
    pattern: str
    confidence: float
    vote_score: int
    candidate_count: int
    group_winners: str
    accepted_pattern: bool
    max_track_delta_rms: float
    median_track_delta_rms: float
    high_delta_duration_s: float
    max_blink_score: float
    max_blink_score_ratio: float
    max_fmcw_candidate_score: float
    max_fmcw_candidate_ratio: float
    primary_blink_event_count: int
    fmcw_candidate_event_count: int
    fmcw_confirmed_event_count: int
    is_positive: bool


@dataclass(frozen=True)
class FmcwWindowMetricRow:
    session: str
    confidence_threshold: float
    positive_total: int
    negative_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    recall: float
    precision: float
    false_positive_rate: float
    accuracy: float
    balanced_accuracy: float


@dataclass(frozen=True)
class FmcwWindowEvaluation:
    session: str
    window_s: float
    marker_exclusion_s: float
    background_step_s: float
    accepted_patterns: tuple[str, ...]
    positive_labels: tuple[str, ...]
    windows: tuple[FmcwWindowVoteRow, ...]
    metrics: tuple[FmcwWindowMetricRow, ...]


def evaluate_fmcw_session_windows(
    session_dir: Path,
    *,
    window_s: float = 0.8,
    background_step_s: float | None = None,
    marker_exclusion_s: float | None = None,
    max_background_windows: int | None = 40,
    ignore_startup_s: float = 0.0,
    confidence_thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
    accepted_patterns: Sequence[str] | None = None,
    positive_labels: Sequence[str] = ("blink",),
    high_delta_threshold: float | None = None,
) -> FmcwWindowEvaluation:
    session_dir = Path(session_dir)
    sample_rate, audio = read_wav_mono(session_dir / "audio.wav")
    config = fmcw_config_from_session(session_dir)
    processor = FmcwStreamProcessor(config, sample_rate)
    sync_lag = infer_sync_lag(session_dir)
    if sync_lag is not None:
        processor.period_start_offset = int(sync_lag)

    features = processor.process_block(audio, 0)
    times = np.asarray([feature.time_s for feature in features], dtype=np.float64)
    phase_matrix = np.asarray([feature.phase_points for feature in features], dtype=np.float64)
    delta_rms = np.asarray([feature.track_delta_rms for feature in features], dtype=np.float64)

    markers = read_csv_rows(session_dir / "manual_markers.csv")
    saved_features = read_csv_rows(session_dir / "features.csv") if (session_dir / "features.csv").exists() else ()
    accepted = tuple(accepted_patterns or config.confirm_single_blink_patterns)
    high_delta = (
        float(config.confirm_high_delta_rms)
        if high_delta_threshold is None
        else float(high_delta_threshold)
    )
    return evaluate_phase_windows(
        session_name=session_dir.name,
        times=times,
        phase_matrix=phase_matrix,
        track_delta_rms=delta_rms,
        saved_feature_rows=saved_features,
        markers=markers,
        config=config,
        window_s=window_s,
        background_step_s=background_step_s,
        marker_exclusion_s=marker_exclusion_s,
        max_background_windows=max_background_windows,
        ignore_startup_s=ignore_startup_s,
        confidence_thresholds=confidence_thresholds,
        accepted_patterns=accepted,
        positive_labels=positive_labels,
        high_delta_threshold=high_delta,
    )


def evaluate_phase_windows(
    *,
    session_name: str,
    times: np.ndarray,
    phase_matrix: np.ndarray,
    track_delta_rms: np.ndarray,
    saved_feature_rows: Sequence[dict[str, str]] = (),
    markers: Sequence[dict[str, str]],
    config: FmcwConfig,
    window_s: float = 0.8,
    background_step_s: float | None = None,
    marker_exclusion_s: float | None = None,
    max_background_windows: int | None = 40,
    ignore_startup_s: float = 0.0,
    confidence_thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
    accepted_patterns: Sequence[str] | None = None,
    positive_labels: Sequence[str] = ("blink",),
    high_delta_threshold: float = 0.10,
) -> FmcwWindowEvaluation:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    deltas = np.asarray(track_delta_rms, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("phase_matrix must be 2D")
    if times.size != matrix.shape[0] or deltas.size != matrix.shape[0]:
        raise ValueError("times, phase_matrix, and track_delta_rms must have matching row counts")

    positive = tuple(positive_labels)
    accepted = tuple(accepted_patterns or config.confirm_single_blink_patterns)
    step = float(background_step_s) if background_step_s is not None else max(float(window_s) * 2.0, 0.1)
    exclusion = (
        float(marker_exclusion_s)
        if marker_exclusion_s is not None
        else max(float(window_s) * 2.0, 0.1)
    )
    windows = _marker_window_specs(markers, positive_labels=positive)
    windows.extend(
        _background_window_specs(
            times,
            markers,
            window_s=float(window_s),
            step_s=step,
            marker_exclusion_s=exclusion,
            max_windows=max_background_windows,
            ignore_startup_s=float(ignore_startup_s),
        )
    )

    rows = tuple(
        _evaluate_one_window(
            session_name,
            index,
            spec,
            times,
            matrix,
            deltas,
            saved_feature_rows,
            config,
            window_s=float(window_s),
            accepted_patterns=accepted,
            positive_labels=positive,
            high_delta_threshold=float(high_delta_threshold),
        )
        for index, spec in enumerate(windows)
    )
    metrics = summarize_window_metrics(
        rows,
        confidence_thresholds=confidence_thresholds,
        session_name=session_name,
    )
    return FmcwWindowEvaluation(
        session=session_name,
        window_s=float(window_s),
        marker_exclusion_s=exclusion,
        background_step_s=step,
        accepted_patterns=accepted,
        positive_labels=positive,
        windows=rows,
        metrics=metrics,
    )


def summarize_window_metrics(
    windows: Sequence[FmcwWindowVoteRow],
    *,
    confidence_thresholds: Sequence[float],
    session_name: str,
) -> tuple[FmcwWindowMetricRow, ...]:
    rows: list[FmcwWindowMetricRow] = []
    for threshold in confidence_thresholds:
        predicted = [
            bool(row.accepted_pattern and row.confidence >= float(threshold))
            for row in windows
        ]
        positives = [row.is_positive for row in windows]
        tp = sum(p and y for p, y in zip(predicted, positives))
        fn = sum((not p) and y for p, y in zip(predicted, positives))
        fp = sum(p and (not y) for p, y in zip(predicted, positives))
        tn = sum((not p) and (not y) for p, y in zip(predicted, positives))
        recall = _ratio(tp, tp + fn)
        precision = _ratio(tp, tp + fp)
        false_positive_rate = _ratio(fp, fp + tn)
        accuracy = _ratio(tp + tn, len(windows))
        specificity = _ratio(tn, tn + fp)
        balanced = (recall + specificity) / 2.0 if windows else 0.0
        rows.append(
            FmcwWindowMetricRow(
                session=session_name,
                confidence_threshold=float(threshold),
                positive_total=int(tp + fn),
                negative_total=int(fp + tn),
                true_positive=int(tp),
                false_negative=int(fn),
                false_positive=int(fp),
                true_negative=int(tn),
                recall=float(recall),
                precision=float(precision),
                false_positive_rate=float(false_positive_rate),
                accuracy=float(accuracy),
                balanced_accuracy=float(balanced),
            )
        )
    return tuple(rows)


def write_window_evaluation_outputs(evaluation: FmcwWindowEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "window_votes.csv", evaluation.windows)
    _write_dataclass_csv(output_dir / "metrics.csv", evaluation.metrics)
    payload = asdict(evaluation)
    payload["windows"] = [asdict(row) for row in evaluation.windows]
    payload["metrics"] = [asdict(row) for row in evaluation.metrics]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return sample_rate, pcm.reshape(-1)


def infer_sync_lag(session_dir: Path) -> int | None:
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


def _marker_window_specs(
    markers: Sequence[dict[str, str]],
    *,
    positive_labels: Sequence[str],
) -> list[dict[str, str]]:
    specs = []
    positive = set(positive_labels)
    for marker in markers:
        label = marker.get("label", "")
        if not label:
            continue
        source = "marker" if label in positive else "negative_marker"
        specs.append(
            {
                "source": source,
                "label": label,
                "key": marker.get("key", ""),
                "time_s": marker.get("time_s", ""),
            }
        )
    return specs


def _background_window_specs(
    times: np.ndarray,
    markers: Sequence[dict[str, str]],
    *,
    window_s: float,
    step_s: float,
    marker_exclusion_s: float,
    max_windows: int | None,
    ignore_startup_s: float = 0.0,
) -> list[dict[str, str]]:
    if times.size == 0:
        return []
    marker_times = []
    for marker in markers:
        value = marker.get("time_s")
        if value in (None, ""):
            continue
        marker_times.append(float(value))

    centers = []
    current = float(times[0]) + max(float(window_s), float(ignore_startup_s))
    end = float(times[-1]) - window_s
    while current <= end:
        if all(abs(current - marker_time) > marker_exclusion_s for marker_time in marker_times):
            centers.append(current)
        current += max(float(step_s), 1e-6)

    if max_windows is not None and len(centers) > int(max_windows):
        indices = np.linspace(0, len(centers) - 1, int(max_windows)).round().astype(int)
        centers = [centers[int(index)] for index in indices]

    return [
        {
            "source": "background",
            "label": "background",
            "key": "",
            "time_s": f"{center:.9f}",
        }
        for center in centers
    ]


def _evaluate_one_window(
    session_name: str,
    window_id: int,
    spec: dict[str, str],
    times: np.ndarray,
    phase_matrix: np.ndarray,
    track_delta_rms: np.ndarray,
    saved_feature_rows: Sequence[dict[str, str]],
    config: FmcwConfig,
    *,
    window_s: float,
    accepted_patterns: Sequence[str],
    positive_labels: Sequence[str],
    high_delta_threshold: float,
) -> FmcwWindowVoteRow:
    center = float(spec.get("time_s") or 0.0)
    mask = (times >= center - window_s) & (times <= center + window_s)
    rows = phase_matrix[mask]
    deltas = track_delta_rms[mask]
    saved_metrics = _saved_feature_window_metrics(saved_feature_rows, center, window_s)
    decision = decode_phase_matrix_vote(rows, config)
    times_in_window = times[mask]
    label = spec.get("label", "")
    return FmcwWindowVoteRow(
        session=session_name,
        window_id=int(window_id),
        source=spec.get("source", ""),
        label=label,
        key=spec.get("key", ""),
        center_time_s=float(center),
        window_s=float(window_s),
        row_count=int(rows.shape[0]),
        pattern=decision.pattern,
        confidence=float(decision.confidence),
        vote_score=int(decision.score),
        candidate_count=int(decision.candidate_count),
        group_winners="|".join(decision.group_winners),
        accepted_pattern=decision.pattern in tuple(accepted_patterns),
        max_track_delta_rms=float(np.max(deltas)) if deltas.size else 0.0,
        median_track_delta_rms=float(median(deltas)) if deltas.size else 0.0,
        high_delta_duration_s=_duration_above(times_in_window, deltas, high_delta_threshold),
        max_blink_score=saved_metrics["max_blink_score"],
        max_blink_score_ratio=saved_metrics["max_blink_score_ratio"],
        max_fmcw_candidate_score=saved_metrics["max_fmcw_candidate_score"],
        max_fmcw_candidate_ratio=saved_metrics["max_fmcw_candidate_ratio"],
        primary_blink_event_count=int(saved_metrics["primary_blink_event_count"]),
        fmcw_candidate_event_count=int(saved_metrics["fmcw_candidate_event_count"]),
        fmcw_confirmed_event_count=int(saved_metrics["fmcw_confirmed_event_count"]),
        is_positive=label in tuple(positive_labels),
    )


def _duration_above(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    if times.size < 2 or values.size < 2:
        return 0.0
    duration = 0.0
    for index in range(times.size - 1):
        if values[index] >= threshold:
            duration += max(0.0, float(times[index + 1]) - float(times[index]))
    return float(duration)


def _saved_feature_window_metrics(
    rows: Sequence[dict[str, str]],
    center_time_s: float,
    window_s: float,
) -> dict[str, float]:
    nearby = [
        row
        for row in rows
        if abs(_row_float(row, "time_s") - float(center_time_s)) <= float(window_s)
    ]
    if not nearby:
        return {
            "max_blink_score": 0.0,
            "max_blink_score_ratio": 0.0,
            "max_fmcw_candidate_score": 0.0,
            "max_fmcw_candidate_ratio": 0.0,
            "primary_blink_event_count": 0.0,
            "fmcw_candidate_event_count": 0.0,
            "fmcw_confirmed_event_count": 0.0,
        }

    blink_score = [_row_float(row, "blink_score") for row in nearby]
    blink_ratio = [
        _row_float(row, "blink_score") / max(_row_float(row, "blink_threshold"), 1e-9)
        for row in nearby
        if _row_float(row, "blink_threshold") > 0.0
    ]
    fmcw_score = [_row_float(row, "fmcw_candidate_score") for row in nearby]
    fmcw_ratio = [
        _row_float(row, "fmcw_candidate_score") / max(_row_float(row, "fmcw_candidate_threshold"), 1e-9)
        for row in nearby
        if _row_float(row, "fmcw_candidate_threshold") > 0.0
    ]
    return {
        "max_blink_score": max(blink_score, default=0.0),
        "max_blink_score_ratio": max(blink_ratio, default=0.0),
        "max_fmcw_candidate_score": max(fmcw_score, default=0.0),
        "max_fmcw_candidate_ratio": max(fmcw_ratio, default=0.0),
        "primary_blink_event_count": float(
            sum(_looks_like_primary_blink_event(row) for row in nearby)
        ),
        "fmcw_candidate_event_count": float(
            sum(str(row.get("fmcw_candidate_is_event", "")) in ("1", "1.0", "True") for row in nearby)
        ),
        "fmcw_confirmed_event_count": float(
            sum(str(row.get("fmcw_confirm_state", "")) == "confirmed_blink" for row in nearby)
        ),
    }


def _looks_like_primary_blink_event(row: dict[str, str]) -> bool:
    method = str(row.get("detector_method", ""))
    is_event = str(row.get("is_event", "")) in ("1", "1.0", "True")
    if not is_event:
        return False
    return method.startswith("blink") or method.startswith("twinkle") or method.startswith("fmcw_primary")


def _row_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
