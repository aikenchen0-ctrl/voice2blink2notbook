import csv

from hp_acoustic_wave.logged_gate_evaluation import (
    detect_logged_gate_events,
    evaluate_logged_gate_session,
)


def test_logged_gate_events_use_rising_edges_not_every_active_row():
    rows = [
        {"time_s": "0.0", "gate": "0", "blink_score": "0.01"},
        {"time_s": "0.1", "gate": "1", "blink_score": "0.07"},
        {"time_s": "0.2", "gate": "1", "blink_score": "0.08"},
        {"time_s": "0.3", "gate": "0", "blink_score": "0.03"},
        {"time_s": "0.4", "gate": "1", "blink_score": "0.09"},
    ]

    events = detect_logged_gate_events(
        session_name="s",
        feature_rows=rows,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
    )

    assert [event.time_s for event in events] == [0.1, 0.4]
    assert [event.score for event in events] == [0.07, 0.09]


def test_logged_gate_events_apply_refractory_after_rising_edge():
    rows = [
        {"time_s": "0.0", "gate": "1", "blink_score": "0.07"},
        {"time_s": "0.1", "gate": "0", "blink_score": "0.02"},
        {"time_s": "0.2", "gate": "1", "blink_score": "0.08"},
        {"time_s": "1.2", "gate": "0", "blink_score": "0.01"},
        {"time_s": "1.3", "gate": "1", "blink_score": "0.09"},
    ]

    events = detect_logged_gate_events(
        session_name="s",
        feature_rows=rows,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
        refractory_s=1.0,
    )

    assert [event.time_s for event in events] == [0.0, 1.3]


def test_logged_gate_events_can_filter_by_score_range():
    rows = [
        {"time_s": "0.0", "gate": "1", "blink_score": "0.03"},
        {"time_s": "0.1", "gate": "0", "blink_score": "0.02"},
        {"time_s": "0.2", "gate": "1", "blink_score": "0.08"},
        {"time_s": "0.3", "gate": "0", "blink_score": "0.02"},
        {"time_s": "0.4", "gate": "1", "blink_score": "0.40"},
    ]

    events = detect_logged_gate_events(
        session_name="s",
        feature_rows=rows,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
        min_score=0.05,
        max_score=0.25,
    )

    assert [event.time_s for event in events] == [0.2]


def test_logged_gate_session_evaluates_against_blink_markers(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    feature_rows = [
        {"time_s": "0.0", "gate": "0", "blink_score": "0.01"},
        {"time_s": "1.0", "gate": "1", "blink_score": "0.08"},
        {"time_s": "1.1", "gate": "1", "blink_score": "0.07"},
        {"time_s": "2.0", "gate": "0", "blink_score": "0.02"},
        {"time_s": "3.0", "gate": "1", "blink_score": "0.09"},
    ]
    marker_rows = [
        {"time_s": "1.2", "label": "blink", "key": "b"},
        {"time_s": "4.0", "label": "blink", "key": "b"},
    ]
    _write_csv(session / "features.csv", feature_rows)
    _write_csv(session / "manual_markers.csv", marker_rows)

    evaluation = evaluate_logged_gate_session(
        session,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
        max_score=0.25,
        tolerance_s=0.5,
        ignore_startup_s=0.0,
    )

    metric = evaluation.event_evaluation.metrics[0]
    assert evaluation.max_score == 0.25
    assert evaluation.refractory_s == 0.0
    assert len(evaluation.events) == 2
    assert metric.true_positive == 1
    assert metric.false_negative == 1
    assert metric.false_positive == 1


def test_logged_gate_session_can_mask_events_to_visual_face_ranges(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    feature_rows = [
        {"time_s": "1.0", "gate": "1", "blink_score": "0.08"},
        {"time_s": "1.1", "gate": "0", "blink_score": "0.01"},
        {"time_s": "5.0", "gate": "1", "blink_score": "0.09"},
    ]
    marker_rows = [{"time_s": "1.0", "label": "blink", "key": "b"}]
    visual_rows = [
        {"time_s": "0.9", "available": "1", "face_found": "1"},
        {"time_s": "1.0", "available": "1", "face_found": "1"},
        {"time_s": "1.1", "available": "1", "face_found": "1"},
        {"time_s": "5.0", "available": "1", "face_found": "0"},
    ]
    _write_csv(session / "features.csv", feature_rows)
    _write_csv(session / "manual_markers.csv", marker_rows)
    _write_csv(session / "visual_features.csv", visual_rows)

    unmasked = evaluate_logged_gate_session(
        session,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
        tolerance_s=0.3,
        ignore_startup_s=0.0,
    )
    masked = evaluate_logged_gate_session(
        session,
        event_column="gate",
        score_column="blink_score",
        threshold=0.5,
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        require_visual_face=True,
    )

    assert unmasked.event_evaluation.metrics[0].false_positive == 1
    assert masked.event_evaluation.metrics[0].false_positive == 0


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
