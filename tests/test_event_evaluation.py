import csv

from hp_acoustic_wave.event_evaluation import (
    evaluate_events,
    infer_event_labels,
    summarize_event_evaluations,
    write_event_aggregate_outputs,
    write_event_evaluation_outputs,
)


def test_event_evaluation_matches_nearest_events_once():
    markers = [
        {"time_s": "1.00", "label": "blink", "key": "b"},
        {"time_s": "3.00", "label": "blink", "key": "b"},
    ]
    events = [
        {"time_s": "1.10", "label": "blink_candidate", "event_id": "1", "method": "twinkle", "score": "0.4"},
        {"time_s": "1.20", "label": "blink_candidate", "event_id": "2", "method": "twinkle", "score": "0.3"},
        {"time_s": "3.70", "label": "blink_candidate", "event_id": "3", "method": "twinkle", "score": "0.5"},
    ]

    evaluation = evaluate_events(
        session_name="s",
        markers=markers,
        events=events,
        tolerance_s=0.5,
        event_labels=("blink_candidate",),
    )

    metric = evaluation.metrics[0]
    assert metric.true_positive == 1
    assert metric.false_negative == 1
    assert metric.false_positive == 2
    assert metric.recall == 0.5
    assert round(metric.precision, 6) == round(1 / 3, 6)
    assert evaluation.marker_matches[0].event_id == "1"


def test_event_evaluation_uses_final_fmcw_event_labels_and_ignores_candidates():
    markers = [{"time_s": "5.00", "label": "blink", "key": "b"}]
    events = [
        {"time_s": "5.00", "label": "fmcw_candidate", "event_id": "1", "method": "fmcw_candidate", "score": "0.2"},
        {"time_s": "5.20", "label": "fmcw_confirmed_blink", "event_id": "1", "method": "fmcw_confirm:1", "score": "0.2"},
    ]

    evaluation = evaluate_events(
        session_name="s",
        markers=markers,
        events=events,
        event_labels=("fmcw_confirmed_blink",),
        tolerance_s=0.8,
    )

    metric = evaluation.metrics[0]
    assert metric.event_total == 1
    assert metric.true_positive == 1
    assert metric.false_positive == 0


def test_event_evaluation_ignore_startup_filters_markers_and_events():
    markers = [
        {"time_s": "1.00", "label": "blink", "key": "b"},
        {"time_s": "4.00", "label": "blink", "key": "b"},
    ]
    events = [
        {"time_s": "1.10", "label": "blink_candidate", "event_id": "1", "method": "twinkle", "score": "0.4"},
        {"time_s": "4.10", "label": "blink_candidate", "event_id": "2", "method": "twinkle", "score": "0.4"},
    ]

    evaluation = evaluate_events(
        session_name="s",
        markers=markers,
        events=events,
        ignore_startup_s=2.0,
        tolerance_s=0.5,
    )

    metric = evaluation.metrics[0]
    assert metric.marker_total == 1
    assert metric.event_total == 1
    assert metric.true_positive == 1


def test_infer_event_labels_prefers_confirmed_fmcw_over_candidates():
    assert infer_event_labels([{"label": "fmcw_confirmed_blink"}, {"label": "fmcw_candidate"}]) == (
        "fmcw_confirmed_blink",
    )
    assert infer_event_labels([{"label": "blink_candidate"}]) == ("blink_candidate",)


def test_write_event_evaluation_outputs(tmp_path):
    evaluation = evaluate_events(
        session_name="s",
        markers=[{"time_s": "1.0", "label": "blink", "key": "b"}],
        events=[{"time_s": "1.1", "label": "blink_candidate", "event_id": "1", "method": "twinkle", "score": "0.4"}],
        tolerance_s=0.5,
    )

    write_event_evaluation_outputs(evaluation, tmp_path)

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    with (tmp_path / "marker_matches.csv").open(newline="", encoding="utf-8") as handle:
        matches = list(csv.DictReader(handle))

    assert metrics[0]["recall"] == "1.0"
    assert matches[0]["matched"] == "True"
    assert (tmp_path / "false_positives.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_summarize_event_evaluations_aggregates_counts_not_session_averages(tmp_path):
    first = evaluate_events(
        session_name="s1",
        markers=[
            {"time_s": "1.0", "label": "blink", "key": "b"},
            {"time_s": "3.0", "label": "blink", "key": "b"},
        ],
        events=[
            {"time_s": "1.1", "label": "blink_candidate", "event_id": "1", "method": "twinkle", "score": "0.4"},
            {"time_s": "9.0", "label": "blink_candidate", "event_id": "2", "method": "twinkle", "score": "0.4"},
        ],
        tolerance_s=0.5,
    )
    second = evaluate_events(
        session_name="s2",
        markers=[{"time_s": "2.0", "label": "blink", "key": "b"}],
        events=[{"time_s": "2.1", "label": "blink_candidate", "event_id": "3", "method": "twinkle", "score": "0.4"}],
        tolerance_s=0.5,
    )

    aggregate = summarize_event_evaluations([first, second])
    metric = aggregate.metrics[0]

    assert metric.marker_total == 3
    assert metric.event_total == 3
    assert metric.true_positive == 2
    assert metric.false_negative == 1
    assert metric.false_positive == 1
    assert round(metric.recall, 6) == round(2 / 3, 6)
    assert round(metric.precision, 6) == round(2 / 3, 6)

    write_event_aggregate_outputs(aggregate, tmp_path)
    with (tmp_path / "aggregate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert metrics[0]["session"] == "ALL"
    assert (tmp_path / "session_metrics.csv").exists()
    assert (tmp_path / "aggregate_summary.json").exists()
