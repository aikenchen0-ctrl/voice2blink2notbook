import csv

from hp_acoustic_wave.candidate_fusion_evaluation import (
    aggregate_candidate_fusion_evaluations,
    evaluate_candidate_fusion_session,
    fuse_primary_and_fallback_events,
)
from hp_acoustic_wave.logged_gate_evaluation import LoggedGateEventRow
from hp_acoustic_wave.primary_blink_evaluation import PrimaryBlinkEventRow


def test_fuse_primary_and_fallback_skips_fallback_near_primary():
    primary = (
        PrimaryBlinkEventRow("s", 1, 1.0, 0.08, 0.05, 1.6, "twinkle"),
    )
    fallback = (
        LoggedGateEventRow("s", 1, 1.1, "twinkle_candidate_peak", "logged:twinkle_candidate_peak", 0.08, 1.0),
        LoggedGateEventRow("s", 2, 3.0, "twinkle_candidate_peak", "logged:twinkle_candidate_peak", 0.07, 1.0),
    )

    fused = fuse_primary_and_fallback_events(
        session_name="s",
        primary_events=primary,
        fallback_events=fallback,
        fallback_exclusion_s=0.8,
    )

    assert [(event.source, event.time_s, event.event_id) for event in fused] == [
        ("primary", 1.0, 1),
        ("fallback", 3.0, 2),
    ]


def test_candidate_fusion_session_reports_rescued_marker_and_added_fp(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    feature_rows = [
        _feature_row(0.0, 0.01, 0),
        _feature_row(1.0, 0.08, 0),
        _feature_row(1.1, 0.02, 0),
        _feature_row(3.0, 0.07, 1, vote=0.7, trajectory=0.4, pair="14:19"),
        _feature_row(3.1, 0.02, 0, vote=0.6, trajectory=0.3, pair="14:19"),
        _feature_row(6.0, 0.09, 1, vote=0.1, trajectory=0.05, pair=""),
    ]
    marker_rows = [
        {"time_s": "1.0", "label": "blink", "key": "b"},
        {"time_s": "3.0", "label": "blink", "key": "b"},
        {"time_s": "6.0", "label": "large_motion", "key": "w"},
    ]
    _write_csv(session / "features.csv", feature_rows)
    _write_csv(session / "manual_markers.csv", marker_rows)

    evaluation = evaluate_candidate_fusion_session(
        session,
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        primary_min_score=0.075,
        primary_min_ratio=1.0,
        primary_max_score=0.25,
        primary_refractory_s=0.5,
        fallback_event_column="twinkle_candidate_peak",
        fallback_score_column="blink_score",
        fallback_max_score=0.20,
        fallback_refractory_s=0.5,
        fallback_exclusion_s=0.3,
    )

    primary_metric = evaluation.primary_evaluation.metrics[0]
    fused_metric = evaluation.fused_evaluation.metrics[0]
    assert primary_metric.true_positive == 1
    assert primary_metric.false_negative == 1
    assert fused_metric.true_positive == 2
    assert fused_metric.false_negative == 0
    assert fused_metric.false_positive == 1
    assert evaluation.summary.added_fallback_event_total == 2
    assert evaluation.summary.rescued_marker_total == 1
    assert evaluation.summary.added_fallback_false_positive_total == 1
    assert evaluation.summary.negative_marker_total == 1
    assert evaluation.summary.fused_negative_conflict_total == 1
    fallback_diagnostics = [row for row in evaluation.diagnostics if row.source == "fallback"]
    assert [(row.classification, row.matched_marker_index) for row in fallback_diagnostics] == [
        ("matched_blink", 1),
        ("false_positive", -1),
    ]
    assert fallback_diagnostics[0].max_fmcw_blink_vote_evidence == 0.7
    assert fallback_diagnostics[0].max_abs_fmcw_blink_trajectory_value == 0.4
    assert fallback_diagnostics[0].max_abs_fmcw_fixed_trajectory_distance_mm == 2.0
    assert abs(fallback_diagnostics[0].max_abs_fmcw_fixed_trajectory_phase_rad - 0.02) < 1e-12
    assert fallback_diagnostics[0].dominant_fmcw_blink_trajectory_pair == "14:19"
    assert fallback_diagnostics[0].fmcw_blink_trajectory_pair_stability > 0.0
    filtered = [
        row
        for row in evaluation.fmcw_sweep
        if row.min_vote_evidence == 0.2
        and row.min_abs_trajectory_value == 0.1
        and row.min_pair_stability == 0.5
    ][0]
    assert filtered.true_positive == 2
    assert filtered.false_positive == 0
    assert filtered.negative_marker_total == 1
    assert filtered.negative_conflict_total == 0


def test_aggregate_candidate_fusion_sums_sessions_and_negative_conflicts(tmp_path):
    session_a = _write_candidate_fusion_session(
        tmp_path / "session_a",
        feature_rows=[
            _feature_row(0.0, 0.01, 0),
            _feature_row(1.0, 0.08, 0),
            _feature_row(3.0, 0.07, 1, vote=0.7, trajectory=0.4, pair="14:19"),
            _feature_row(3.1, 0.02, 0, vote=0.6, trajectory=0.3, pair="14:19"),
            _feature_row(6.0, 0.09, 1, vote=0.1, trajectory=0.05, pair=""),
        ],
        marker_rows=[
            {"time_s": "1.0", "label": "blink", "key": "b"},
            {"time_s": "3.0", "label": "blink", "key": "b"},
            {"time_s": "6.0", "label": "large_motion", "key": "w"},
        ],
    )
    session_b = _write_candidate_fusion_session(
        tmp_path / "session_b",
        feature_rows=[
            _feature_row(0.0, 0.01, 0),
            _feature_row(1.0, 0.08, 0),
            _feature_row(1.2, 0.02, 0),
        ],
        marker_rows=[
            {"time_s": "1.0", "label": "blink", "key": "b"},
        ],
    )

    evaluations = [
        evaluate_candidate_fusion_session(
            session,
            tolerance_s=0.3,
            ignore_startup_s=0.0,
            primary_min_score=0.075,
            primary_min_ratio=1.0,
            primary_max_score=0.25,
            primary_refractory_s=0.5,
            fallback_event_column="twinkle_candidate_peak",
            fallback_score_column="blink_score",
            fallback_max_score=0.20,
            fallback_refractory_s=0.5,
            fallback_exclusion_s=0.3,
        )
        for session in (session_a, session_b)
    ]

    aggregate = aggregate_candidate_fusion_evaluations(evaluations)
    fused = [row for row in aggregate.strategy_metrics if row.strategy == "fused"][0]
    assert fused.session_count == 2
    assert fused.marker_total == 3
    assert fused.true_positive == 3
    assert fused.false_positive == 1
    assert fused.negative_marker_total == 1
    assert fused.negative_conflict_total == 1

    filtered = [
        row
        for row in aggregate.fmcw_sweep
        if row.min_vote_evidence == 0.2
        and row.min_abs_trajectory_value == 0.1
        and row.min_pair_stability == 0.5
    ][0]
    assert filtered.session_count == 2
    assert filtered.true_positive == 3
    assert filtered.false_positive == 0
    assert filtered.negative_marker_total == 1
    assert filtered.negative_conflict_total == 0


def _feature_row(time_s, blink_score, twinkle_peak, *, vote=0.0, trajectory=0.0, pair=""):
    return {
        "time_s": str(time_s),
        "blink_score": str(blink_score),
        "blink_threshold": "0.05",
        "blink_method": "twinkle",
        "twinkle_candidate_peak": str(twinkle_peak),
        "fmcw_blink_vote_evidence": str(vote),
        "fmcw_blink_trajectory_value": str(trajectory),
        "fmcw_fixed_trajectory_value": str(trajectory * 5.0),
        "fmcw_fixed_trajectory_phase_rad": str(trajectory * 0.05),
        "fmcw_fixed_trajectory_distance_mm": str(trajectory * 5.0),
        "fmcw_blink_trajectory_pair": pair,
        "fmcw_blink_trajectory_pattern": "1" if pair else "",
        "fmcw_blink_trajectory_criterion": "internal_similarity" if pair else "",
        "fmcw_track_delta_rms": "0.04",
        "fmcw_confirm_window_confidence": str(vote),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_fusion_session(path, *, feature_rows, marker_rows):
    path.mkdir()
    _write_csv(path / "features.csv", feature_rows)
    _write_csv(path / "manual_markers.csv", marker_rows)
    return path
