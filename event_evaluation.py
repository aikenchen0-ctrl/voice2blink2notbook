from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EventMatchRow:
    session: str
    marker_index: int
    marker_time_s: float
    marker_label: str
    marker_key: str
    matched: bool
    event_index: int
    event_id: str
    event_time_s: float
    event_label: str
    event_method: str
    offset_s: float


@dataclass(frozen=True)
class FalsePositiveEventRow:
    session: str
    event_index: int
    event_id: str
    event_time_s: float
    event_label: str
    event_method: str
    score: float
    nearest_marker_offset_s: float


@dataclass(frozen=True)
class EventMetricRow:
    session: str
    tolerance_s: float
    marker_total: int
    event_total: int
    true_positive: int
    false_negative: int
    false_positive: int
    recall: float
    precision: float
    f1: float
    false_discovery_rate: float


@dataclass(frozen=True)
class EventEvaluation:
    session: str
    tolerance_s: float
    positive_labels: tuple[str, ...]
    event_labels: tuple[str, ...]
    marker_matches: tuple[EventMatchRow, ...]
    false_positives: tuple[FalsePositiveEventRow, ...]
    metrics: tuple[EventMetricRow, ...]


@dataclass(frozen=True)
class EventEvaluationAggregate:
    session: str
    tolerance_s: float
    evaluations: int
    metrics: tuple[EventMetricRow, ...]
    session_metrics: tuple[EventMetricRow, ...]


def evaluate_session_events(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    positive_labels: Sequence[str] = ("blink",),
    event_labels: Sequence[str] | None = None,
    ignore_startup_s: float = 0.0,
    require_visual_face: bool = False,
) -> EventEvaluation:
    session_dir = Path(session_dir)
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    events = read_csv_rows(session_dir / "events.csv")
    valid_time_ranges = None
    if require_visual_face:
        valid_time_ranges = visual_face_time_ranges(session_dir / "visual_features.csv")
    labels = tuple(event_labels or infer_event_labels(events))
    return evaluate_events(
        session_name=session_dir.name,
        markers=markers,
        events=events,
        tolerance_s=tolerance_s,
        positive_labels=positive_labels,
        event_labels=labels,
        ignore_startup_s=ignore_startup_s,
        valid_time_ranges=valid_time_ranges,
    )


def evaluate_events(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    events: Sequence[dict[str, str]],
    tolerance_s: float = 0.8,
    positive_labels: Sequence[str] = ("blink",),
    event_labels: Sequence[str] = ("blink_candidate",),
    ignore_startup_s: float = 0.0,
    valid_time_ranges: Sequence[tuple[float, float]] | None = None,
) -> EventEvaluation:
    positives = tuple(positive_labels)
    accepted_events = tuple(event_labels)
    ranges = tuple(valid_time_ranges or ())
    marker_rows = [
        (index, row)
        for index, row in enumerate(markers)
        if row.get("label", "") in positives and _float(row.get("time_s")) >= float(ignore_startup_s)
        and _time_in_ranges(_float(row.get("time_s")), ranges)
    ]
    event_rows = [
        (index, row)
        for index, row in enumerate(events)
        if row.get("label", "") in accepted_events and _float(row.get("time_s")) >= float(ignore_startup_s)
        and _time_in_ranges(_float(row.get("time_s")), ranges)
    ]

    candidate_pairs = []
    for marker_pos, (marker_index, marker) in enumerate(marker_rows):
        marker_time = _float(marker.get("time_s"))
        for event_pos, (event_index, event) in enumerate(event_rows):
            event_time = _float(event.get("time_s"))
            offset = event_time - marker_time
            if abs(offset) <= float(tolerance_s):
                candidate_pairs.append((abs(offset), marker_pos, event_pos, offset))

    matched_markers: dict[int, tuple[int, float]] = {}
    matched_events: dict[int, tuple[int, float]] = {}
    for _, marker_pos, event_pos, offset in sorted(candidate_pairs, key=lambda item: (item[0], item[1], item[2])):
        if marker_pos in matched_markers or event_pos in matched_events:
            continue
        matched_markers[marker_pos] = (event_pos, offset)
        matched_events[event_pos] = (marker_pos, offset)

    marker_matches = []
    for marker_pos, (marker_index, marker) in enumerate(marker_rows):
        marker_time = _float(marker.get("time_s"))
        if marker_pos in matched_markers:
            event_pos, offset = matched_markers[marker_pos]
            event_index, event = event_rows[event_pos]
            marker_matches.append(
                EventMatchRow(
                    session=session_name,
                    marker_index=int(marker_index),
                    marker_time_s=marker_time,
                    marker_label=marker.get("label", ""),
                    marker_key=marker.get("key", ""),
                    matched=True,
                    event_index=int(event_index),
                    event_id=event.get("event_id", ""),
                    event_time_s=_float(event.get("time_s")),
                    event_label=event.get("label", ""),
                    event_method=event.get("method", ""),
                    offset_s=float(offset),
                )
            )
        else:
            marker_matches.append(
                EventMatchRow(
                    session=session_name,
                    marker_index=int(marker_index),
                    marker_time_s=marker_time,
                    marker_label=marker.get("label", ""),
                    marker_key=marker.get("key", ""),
                    matched=False,
                    event_index=-1,
                    event_id="",
                    event_time_s=0.0,
                    event_label="",
                    event_method="",
                    offset_s=0.0,
                )
            )

    false_positives = []
    positive_marker_times = [_float(marker.get("time_s")) for _, marker in marker_rows]
    for event_pos, (event_index, event) in enumerate(event_rows):
        if event_pos in matched_events:
            continue
        event_time = _float(event.get("time_s"))
        nearest_offset = _nearest_offset(event_time, positive_marker_times)
        false_positives.append(
            FalsePositiveEventRow(
                session=session_name,
                event_index=int(event_index),
                event_id=event.get("event_id", ""),
                event_time_s=event_time,
                event_label=event.get("label", ""),
                event_method=event.get("method", ""),
                score=_float(event.get("score")),
                nearest_marker_offset_s=nearest_offset,
            )
        )

    tp = sum(row.matched for row in marker_matches)
    fn = len(marker_matches) - tp
    fp = len(false_positives)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    metric = EventMetricRow(
        session=session_name,
        tolerance_s=float(tolerance_s),
        marker_total=len(marker_matches),
        event_total=len(event_rows),
        true_positive=int(tp),
        false_negative=int(fn),
        false_positive=int(fp),
        recall=float(recall),
        precision=float(precision),
        f1=float(f1),
        false_discovery_rate=_ratio(fp, tp + fp),
    )
    return EventEvaluation(
        session=session_name,
        tolerance_s=float(tolerance_s),
        positive_labels=positives,
        event_labels=accepted_events,
        marker_matches=tuple(marker_matches),
        false_positives=tuple(false_positives),
        metrics=(metric,),
    )


def summarize_event_evaluations(
    evaluations: Sequence[EventEvaluation],
    *,
    session_name: str = "ALL",
) -> EventEvaluationAggregate:
    if not evaluations:
        metric = _event_metric_row(
            session=session_name,
            tolerance_s=0.0,
            marker_total=0,
            event_total=0,
            true_positive=0,
            false_negative=0,
            false_positive=0,
        )
        return EventEvaluationAggregate(
            session=session_name,
            tolerance_s=0.0,
            evaluations=0,
            metrics=(metric,),
            session_metrics=(),
        )

    tolerance = float(evaluations[0].tolerance_s)
    for evaluation in evaluations:
        if abs(float(evaluation.tolerance_s) - tolerance) > 1e-9:
            raise ValueError("all event evaluations must use the same tolerance_s")

    session_metrics = tuple(evaluation.metrics[0] for evaluation in evaluations)
    marker_total = sum(row.marker_total for row in session_metrics)
    event_total = sum(row.event_total for row in session_metrics)
    true_positive = sum(row.true_positive for row in session_metrics)
    false_negative = sum(row.false_negative for row in session_metrics)
    false_positive = sum(row.false_positive for row in session_metrics)
    metric = _event_metric_row(
        session=session_name,
        tolerance_s=tolerance,
        marker_total=marker_total,
        event_total=event_total,
        true_positive=true_positive,
        false_negative=false_negative,
        false_positive=false_positive,
    )
    return EventEvaluationAggregate(
        session=session_name,
        tolerance_s=tolerance,
        evaluations=len(evaluations),
        metrics=(metric,),
        session_metrics=session_metrics,
    )


def infer_event_labels(events: Sequence[dict[str, str]]) -> tuple[str, ...]:
    labels = {row.get("label", "") for row in events}
    if "fmcw_confirmed_blink" in labels:
        return ("fmcw_confirmed_blink",)
    if "blink_candidate" in labels:
        return ("blink_candidate",)
    if "blink" in labels:
        return ("blink",)
    return ("blink_candidate",)


def write_event_evaluation_outputs(evaluation: EventEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "marker_matches.csv", evaluation.marker_matches)
    _write_dataclass_csv(output_dir / "false_positives.csv", evaluation.false_positives)
    _write_dataclass_csv(output_dir / "metrics.csv", evaluation.metrics)
    payload = asdict(evaluation)
    payload["marker_matches"] = [asdict(row) for row in evaluation.marker_matches]
    payload["false_positives"] = [asdict(row) for row in evaluation.false_positives]
    payload["metrics"] = [asdict(row) for row in evaluation.metrics]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_event_aggregate_outputs(aggregate: EventEvaluationAggregate, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "aggregate_metrics.csv", aggregate.metrics)
    _write_dataclass_csv(output_dir / "session_metrics.csv", aggregate.session_metrics)
    payload = asdict(aggregate)
    payload["metrics"] = [asdict(row) for row in aggregate.metrics]
    payload["session_metrics"] = [asdict(row) for row in aggregate.session_metrics]
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def visual_face_time_ranges(
    visual_features_path: Path,
    *,
    max_gap_s: float = 0.35,
    pad_s: float = 0.05,
) -> tuple[tuple[float, float], ...]:
    path = Path(visual_features_path)
    if not path.exists() or path.stat().st_size <= 0:
        return tuple()
    rows = read_csv_rows(path)
    valid_times = [
        _float(row.get("time_s"))
        for row in rows
        if _bool(row.get("available")) and _bool(row.get("face_found"))
    ]
    if not valid_times:
        return tuple()
    valid_times = sorted(valid_times)
    ranges: list[tuple[float, float]] = []
    start = valid_times[0]
    previous = valid_times[0]
    for current in valid_times[1:]:
        if float(current) - float(previous) > float(max_gap_s):
            ranges.append((max(0.0, float(start) - float(pad_s)), float(previous) + float(pad_s)))
            start = current
        previous = current
    ranges.append((max(0.0, float(start) - float(pad_s)), float(previous) + float(pad_s)))
    return tuple(ranges)


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _nearest_offset(event_time: float, marker_times: Sequence[float]) -> float:
    if not marker_times:
        return 0.0
    nearest = min(marker_times, key=lambda marker_time: abs(float(event_time) - float(marker_time)))
    return float(event_time) - float(nearest)


def _time_in_ranges(time_s: float, ranges: Sequence[tuple[float, float]]) -> bool:
    if not ranges:
        return True
    value = float(time_s)
    return any(float(start) <= value <= float(end) for start, end in ranges)


def _event_metric_row(
    *,
    session: str,
    tolerance_s: float,
    marker_total: int,
    event_total: int,
    true_positive: int,
    false_negative: int,
    false_positive: int,
) -> EventMetricRow:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return EventMetricRow(
        session=session,
        tolerance_s=float(tolerance_s),
        marker_total=int(marker_total),
        event_total=int(event_total),
        true_positive=int(true_positive),
        false_negative=int(false_negative),
        false_positive=int(false_positive),
        recall=float(recall),
        precision=float(precision),
        f1=float(f1),
        false_discovery_rate=_ratio(false_positive, true_positive + false_positive),
    )


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
