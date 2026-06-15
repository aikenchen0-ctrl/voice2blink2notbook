from hp_acoustic_wave.negative_marker_evaluation import evaluate_negative_markers


def test_evaluate_negative_markers_counts_conflicts_and_suppression():
    markers = [
        {"time_s": "1.0", "label": "large_motion", "key": "w"},
        {"time_s": "3.0", "label": "large_motion", "key": "w"},
        {"time_s": "5.0", "label": "blink", "key": "b"},
    ]
    events = [
        {"time_s": "1.2", "label": "fmcw_confirmed_blink", "event_id": "7", "score": "0.2"},
        {"time_s": "3.1", "label": "fmcw_suppressed_motion", "event_id": "2", "score": "0.8"},
        {"time_s": "5.1", "label": "fmcw_confirmed_blink", "event_id": "8", "score": "0.1"},
    ]

    evaluation = evaluate_negative_markers(
        session_name="s",
        markers=markers,
        events=events,
        tolerance_s=0.5,
    )

    metric = evaluation.metrics[0]
    assert metric.negative_total == 2
    assert metric.conflict_total == 1
    assert metric.suppressed_total == 1
    assert metric.conflict_rate == 0.5
    assert metric.suppression_rate == 0.5
    assert evaluation.conflicts[0].conflict
    assert not evaluation.conflicts[0].suppressed
    assert evaluation.conflicts[1].suppressed
