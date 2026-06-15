import csv

from hp_acoustic_wave.primary_blink_evaluation import (
    detect_primary_blink_peaks,
    evaluate_primary_blink_session,
    write_primary_blink_outputs,
)


def test_detect_primary_blink_peaks_uses_local_max_score_ratio_and_refractory():
    rows = [
        {"time_s": "0.0", "blink_score": "0.01", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.1", "blink_score": "0.07", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.2", "blink_score": "0.03", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.3", "blink_score": "0.08", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.4", "blink_score": "0.02", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.0", "blink_score": "0.09", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.1", "blink_score": "0.04", "blink_threshold": "0.05", "blink_method": "twinkle"},
    ]

    events = detect_primary_blink_peaks(
        session_name="s",
        feature_rows=rows,
        min_score=0.06,
        min_ratio=1.2,
        refractory_s=0.5,
    )

    assert [event.time_s for event in events] == [0.1, 1.0]
    assert round(events[0].ratio, 6) == 1.4


def test_detect_primary_blink_peaks_applies_optional_max_score_gate():
    rows = [
        {"time_s": "0.0", "blink_score": "0.01", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.1", "blink_score": "0.40", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.2", "blink_score": "0.03", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.4", "blink_score": "0.10", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "0.5", "blink_score": "0.02", "blink_threshold": "0.05", "blink_method": "twinkle"},
    ]

    rejected = detect_primary_blink_peaks(
        session_name="s",
        feature_rows=rows,
        min_score=0.06,
        min_ratio=1.0,
        max_score=0.25,
        refractory_s=0.5,
    )
    accepted = detect_primary_blink_peaks(
        session_name="s",
        feature_rows=rows,
        min_score=0.06,
        min_ratio=1.0,
        max_score=0.0,
        refractory_s=0.5,
    )

    assert rejected == ()
    assert [event.time_s for event in accepted] == [0.1]


def test_evaluate_primary_blink_session_writes_outputs(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    with (session / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_s",
                "blink_score",
                "blink_threshold",
                "blink_method",
                "fmcw_track_delta_rms",
                "fmcw_confirm_window_pattern",
                "fmcw_confirm_window_confidence",
                "fmcw_confirm_window_vote_score",
                "fmcw_confirm_window_candidate_count",
                "fmcw_candidate_score",
                "fmcw_candidate_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "time_s": "0.0",
                    "blink_score": "0.01",
                    "blink_threshold": "0.05",
                    "blink_method": "twinkle",
                    "fmcw_track_delta_rms": "0.01",
                    "fmcw_confirm_window_pattern": "",
                    "fmcw_confirm_window_confidence": "0",
                    "fmcw_confirm_window_vote_score": "0",
                    "fmcw_confirm_window_candidate_count": "0",
                    "fmcw_candidate_score": "0.01",
                    "fmcw_candidate_threshold": "0.05",
                },
                {
                    "time_s": "1.0",
                    "blink_score": "0.08",
                    "blink_threshold": "0.05",
                    "blink_method": "twinkle",
                    "fmcw_track_delta_rms": "0.04",
                    "fmcw_confirm_window_pattern": "1",
                    "fmcw_confirm_window_confidence": "0.8",
                    "fmcw_confirm_window_vote_score": "12",
                    "fmcw_confirm_window_candidate_count": "45",
                    "fmcw_candidate_score": "0.04",
                    "fmcw_candidate_threshold": "0.05",
                },
                {
                    "time_s": "1.1",
                    "blink_score": "0.03",
                    "blink_threshold": "0.05",
                    "blink_method": "twinkle",
                    "fmcw_track_delta_rms": "0.04",
                    "fmcw_confirm_window_pattern": "1",
                    "fmcw_confirm_window_confidence": "0.8",
                    "fmcw_confirm_window_vote_score": "12",
                    "fmcw_confirm_window_candidate_count": "45",
                    "fmcw_candidate_score": "0.04",
                    "fmcw_candidate_threshold": "0.05",
                },
            ]
        )
    with (session / "manual_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "label", "key"])
        writer.writeheader()
        writer.writerow({"time_s": "1.05", "label": "blink", "key": "b"})

    evaluation = evaluate_primary_blink_session(
        session,
        min_score=0.06,
        min_ratio=1.2,
        refractory_s=0.5,
        tolerance_s=0.2,
        ignore_startup_s=0.0,
    )
    write_primary_blink_outputs(evaluation, tmp_path / "out")

    assert evaluation.event_evaluation.metrics[0].true_positive == 1
    assert (tmp_path / "out" / "primary_blink_events.csv").exists()
    assert (tmp_path / "out" / "primary_blink_raw_peaks.csv").exists()
    assert (tmp_path / "out" / "primary_blink_marker_diagnostics.csv").exists()
    assert (tmp_path / "out" / "primary_blink_windows.csv").exists()
    assert (tmp_path / "out" / "primary_blink_event_diagnostics.csv").exists()
    assert (tmp_path / "out" / "primary_blink_sweep.csv").exists()
    assert (tmp_path / "out" / "primary_blink_confirm_sweep.csv").exists()
    assert (tmp_path / "out" / "primary_blink_shape_sweep.csv").exists()
    assert (tmp_path / "out" / "summary.json").exists()


def test_confirm_sweep_can_filter_high_fmcw_delta_false_positive(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    fieldnames = [
        "time_s",
        "blink_score",
        "blink_threshold",
        "blink_method",
        "fmcw_track_delta_rms",
        "fmcw_confirm_window_pattern",
        "fmcw_confirm_window_confidence",
        "fmcw_confirm_window_vote_score",
        "fmcw_confirm_window_candidate_count",
        "fmcw_candidate_score",
        "fmcw_candidate_threshold",
    ]
    rows = []
    for time_s, score, delta in (
        (0.9, 0.01, 0.02),
        (1.0, 0.08, 0.04),
        (1.1, 0.02, 0.04),
        (3.9, 0.01, 0.02),
        (4.0, 0.18, 0.20),
        (4.1, 0.02, 0.20),
    ):
        rows.append(
            {
                "time_s": str(time_s),
                "blink_score": str(score),
                "blink_threshold": "0.05",
                "blink_method": "twinkle",
                "fmcw_track_delta_rms": str(delta),
                "fmcw_confirm_window_pattern": "1",
                "fmcw_confirm_window_confidence": "0.8",
                "fmcw_confirm_window_vote_score": "12",
                "fmcw_confirm_window_candidate_count": "45",
                "fmcw_candidate_score": str(delta),
                "fmcw_candidate_threshold": "0.05",
            }
        )
    with (session / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (session / "manual_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "label", "key"])
        writer.writeheader()
        writer.writerow({"time_s": "1.0", "label": "blink", "key": "b"})

    evaluation = evaluate_primary_blink_session(
        session,
        min_score=0.06,
        min_ratio=1.0,
        refractory_s=0.5,
        tolerance_s=0.2,
        ignore_startup_s=0.0,
    )

    raw_metric = evaluation.event_evaluation.metrics[0]
    filtered = [
        row
        for row in evaluation.confirm_sweep
        if row.max_delta_rms == 0.06
        and row.max_high_delta_duration_s == 999.0
        and not row.require_pattern
    ][0]
    assert raw_metric.true_positive == 1
    assert raw_metric.false_positive == 1
    assert [row.classification for row in evaluation.event_diagnostics] == [
        "true_positive",
        "false_positive_isolated",
    ]
    assert filtered.true_positive == 1
    assert filtered.false_positive == 0


def test_event_diagnostics_classifies_large_motion_and_burst_false_positives(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    fieldnames = [
        "time_s",
        "blink_score",
        "blink_threshold",
        "blink_method",
        "fmcw_track_delta_rms",
        "fmcw_confirm_window_pattern",
        "fmcw_confirm_window_confidence",
        "fmcw_confirm_window_vote_score",
        "fmcw_confirm_window_candidate_count",
        "fmcw_candidate_score",
        "fmcw_candidate_threshold",
    ]
    rows = []
    for time_s, score, delta in (
        (0.9, 0.01, 0.02),
        (1.0, 0.08, 0.04),
        (1.1, 0.01, 0.02),
        (2.0, 0.09, 0.30),
        (2.1, 0.01, 0.30),
        (4.0, 0.09, 0.04),
        (4.1, 0.01, 0.04),
        (4.4, 0.10, 0.04),
        (4.5, 0.01, 0.04),
        (4.8, 0.11, 0.04),
        (4.9, 0.01, 0.04),
    ):
        rows.append(
            {
                "time_s": str(time_s),
                "blink_score": str(score),
                "blink_threshold": "0.05",
                "blink_method": "twinkle",
                "fmcw_track_delta_rms": str(delta),
                "fmcw_confirm_window_pattern": "",
                "fmcw_confirm_window_confidence": "0",
                "fmcw_confirm_window_vote_score": "0",
                "fmcw_confirm_window_candidate_count": "0",
                "fmcw_candidate_score": str(delta),
                "fmcw_candidate_threshold": "0.05",
            }
        )
    with (session / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (session / "manual_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "label", "key"])
        writer.writeheader()
        writer.writerow({"time_s": "1.0", "label": "blink", "key": "b"})
        writer.writerow({"time_s": "2.0", "label": "large_motion", "key": "w"})

    evaluation = evaluate_primary_blink_session(
        session,
        min_score=0.06,
        min_ratio=1.0,
        refractory_s=0.2,
        tolerance_s=0.2,
        ignore_startup_s=0.0,
    )

    classifications = {row.time_s: row.classification for row in evaluation.event_diagnostics}
    assert classifications[1.0] == "true_positive"
    assert classifications[2.0] == "false_positive_near_large_motion"
    assert classifications[4.4] == "false_positive_burst"


def test_shape_sweep_can_filter_broad_blink_score_false_positive(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    fieldnames = [
        "time_s",
        "blink_score",
        "blink_threshold",
        "blink_method",
        "fmcw_track_delta_rms",
        "fmcw_confirm_window_pattern",
        "fmcw_confirm_window_confidence",
        "fmcw_confirm_window_vote_score",
        "fmcw_confirm_window_candidate_count",
        "fmcw_candidate_score",
        "fmcw_candidate_threshold",
    ]
    rows = []
    for time_s, score in (
        (0.6, 0.01),
        (0.8, 0.03),
        (1.0, 0.08),
        (1.2, 0.03),
        (1.4, 0.01),
        (3.2, 0.01),
        (3.6, 0.06),
        (4.0, 0.08),
        (4.4, 0.06),
        (4.8, 0.01),
    ):
        rows.append(
            {
                "time_s": str(time_s),
                "blink_score": str(score),
                "blink_threshold": "0.05",
                "blink_method": "twinkle",
                "fmcw_track_delta_rms": "0.02",
                "fmcw_confirm_window_pattern": "",
                "fmcw_confirm_window_confidence": "0",
                "fmcw_confirm_window_vote_score": "0",
                "fmcw_confirm_window_candidate_count": "0",
                "fmcw_candidate_score": "0.02",
                "fmcw_candidate_threshold": "0.05",
            }
        )
    with (session / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (session / "manual_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "label", "key"])
        writer.writeheader()
        writer.writerow({"time_s": "1.0", "label": "blink", "key": "b"})

    evaluation = evaluate_primary_blink_session(
        session,
        min_score=0.06,
        min_ratio=1.0,
        refractory_s=0.5,
        tolerance_s=0.25,
        ignore_startup_s=0.0,
    )

    raw_metric = evaluation.event_evaluation.metrics[0]
    widths = {row.time_s: row.half_width_s for row in evaluation.event_diagnostics}
    filtered = [
        row
        for row in evaluation.shape_sweep
        if row.min_prominence == 0.0
        and row.min_prominence_ratio == 0.0
        and row.max_half_width_s == 0.4
        and row.max_abs_baseline_slope == 999.0
        and row.min_pre_post_symmetry == 0.0
        and row.max_delta_rms == 999.0
    ][0]

    assert raw_metric.true_positive == 1
    assert raw_metric.false_positive == 1
    assert widths[1.0] < 0.4
    assert widths[4.0] > 0.4
    assert filtered.true_positive == 1
    assert filtered.false_positive == 0


def test_marker_diagnostics_explains_ratio_and_refractory_misses(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    fieldnames = ["time_s", "blink_score", "blink_threshold", "blink_method"]
    rows = [
        {"time_s": "0.0", "blink_score": "0.01", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.0", "blink_score": "0.08", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.1", "blink_score": "0.03", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.5", "blink_score": "0.07", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "1.6", "blink_score": "0.02", "blink_threshold": "0.05", "blink_method": "twinkle"},
        {"time_s": "3.0", "blink_score": "0.07", "blink_threshold": "0.08", "blink_method": "twinkle"},
        {"time_s": "3.1", "blink_score": "0.02", "blink_threshold": "0.08", "blink_method": "twinkle"},
    ]
    with (session / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (session / "manual_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "label", "key"])
        writer.writeheader()
        writer.writerow({"time_s": "1.0", "label": "blink", "key": "b"})
        writer.writerow({"time_s": "1.5", "label": "blink", "key": "b"})
        writer.writerow({"time_s": "3.0", "label": "blink", "key": "b"})

    evaluation = evaluate_primary_blink_session(
        session,
        min_score=0.06,
        min_ratio=1.0,
        refractory_s=1.05,
        tolerance_s=0.2,
        ignore_startup_s=0.0,
    )

    reasons = {row.marker_time_s: row.reason for row in evaluation.marker_diagnostics}
    assert reasons[1.0] == "matched"
    assert reasons[1.5] == "refractory_suppressed"
    assert reasons[3.0] == "ratio_below_min"
