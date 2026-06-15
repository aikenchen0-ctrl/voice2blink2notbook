import csv

import numpy as np

from hp_acoustic_wave.config import FmcwConfig
from hp_acoustic_wave.fmcw_window_evaluation import (
    FmcwWindowEvaluation,
    FmcwWindowVoteRow,
    evaluate_phase_windows,
    summarize_window_metrics,
    write_window_evaluation_outputs,
)


def test_summarize_window_metrics_counts_recall_and_false_positive_rate():
    rows = (
        _row(label="blink", pattern="1", confidence=0.8, accepted=True, positive=True),
        _row(label="blink", pattern="", confidence=0.0, accepted=False, positive=True),
        _row(label="background", pattern="1", confidence=0.7, accepted=True, positive=False),
        _row(label="background", pattern="3", confidence=0.9, accepted=False, positive=False),
    )

    metrics = summarize_window_metrics(rows, confidence_thresholds=(0.6,), session_name="s")

    assert metrics[0].true_positive == 1
    assert metrics[0].false_negative == 1
    assert metrics[0].false_positive == 1
    assert metrics[0].true_negative == 1
    assert metrics[0].recall == 0.5
    assert metrics[0].precision == 0.5
    assert metrics[0].false_positive_rate == 0.5


def test_evaluate_phase_windows_adds_marker_and_background_windows():
    chirps = 320
    points = 30
    times = np.arange(chirps, dtype=np.float64) * 512.0 / 48_000.0
    phase = np.tile(np.linspace(0.0, 3.0, points, dtype=np.float64), (chirps, 1))
    blink_shape = np.exp(-0.5 * np.square((np.arange(chirps) - 140.0) / 4.0))
    phase[:, 16:25] += blink_shape[:, None] * np.linspace(0.05, 0.45, 9)
    deltas = np.zeros(chirps, dtype=np.float64)
    markers = [{"time_s": f"{times[140]:.6f}", "label": "blink", "key": "b"}]
    config = FmcwConfig(
        valid_start=15,
        valid_stop=25,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    evaluation = evaluate_phase_windows(
        session_name="synthetic",
        times=times,
        phase_matrix=phase,
        track_delta_rms=deltas,
        saved_feature_rows=(),
        markers=markers,
        config=config,
        window_s=0.4,
        background_step_s=0.8,
        marker_exclusion_s=0.8,
        ignore_startup_s=0.5,
        max_background_windows=3,
        confidence_thresholds=(0.5,),
    )

    assert evaluation.session == "synthetic"
    assert any(row.source == "marker" and row.label == "blink" for row in evaluation.windows)
    assert any(row.source == "background" for row in evaluation.windows)
    assert all(row.candidate_count in (0, 45) for row in evaluation.windows)
    assert evaluation.metrics[0].positive_total == 1
    assert evaluation.metrics[0].negative_total >= 1


def test_write_window_evaluation_outputs(tmp_path):
    rows = (_row(label="blink", pattern="1", confidence=0.8, accepted=True, positive=True),)
    metrics = summarize_window_metrics(rows, confidence_thresholds=(0.6,), session_name="s")
    evaluation = FmcwWindowEvaluation(
        session="s",
        window_s=0.8,
        marker_exclusion_s=1.6,
        background_step_s=1.6,
        accepted_patterns=("1",),
        positive_labels=("blink",),
        windows=rows,
        metrics=metrics,
    )

    write_window_evaluation_outputs(evaluation, tmp_path)

    with (tmp_path / "window_votes.csv").open(newline="", encoding="utf-8") as handle:
        vote_rows = list(csv.DictReader(handle))
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))

    assert vote_rows[0]["label"] == "blink"
    assert metric_rows[0]["recall"] == "1.0"
    assert (tmp_path / "summary.json").exists()


def _row(*, label: str, pattern: str, confidence: float, accepted: bool, positive: bool):
    return FmcwWindowVoteRow(
        session="s",
        window_id=0,
        source="marker" if positive else "background",
        label=label,
        key="",
        center_time_s=0.0,
        window_s=0.8,
        row_count=64,
        pattern=pattern,
        confidence=confidence,
        vote_score=10,
        candidate_count=45,
        group_winners=pattern,
        accepted_pattern=accepted,
        max_track_delta_rms=0.0,
        median_track_delta_rms=0.0,
        high_delta_duration_s=0.0,
        max_blink_score=0.0,
        max_blink_score_ratio=0.0,
        max_fmcw_candidate_score=0.0,
        max_fmcw_candidate_ratio=0.0,
        primary_blink_event_count=0,
        fmcw_candidate_event_count=0,
        fmcw_confirmed_event_count=0,
        is_positive=positive,
    )
