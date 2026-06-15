from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hp_acoustic_wave.event_evaluation import EventEvaluation, evaluate_events, write_event_evaluation_outputs


@dataclass(frozen=True)
class LoggedGateEventRow:
    session: str
    event_id: int
    time_s: float
    label: str
    method: str
    score: float
    gate_value: float


@dataclass(frozen=True)
class LoggedGateEvaluation:
    session: str
    event_column: str
    score_column: str
    threshold: float
    min_score: float | None
    max_score: float | None
    refractory_s: float
    events: tuple[LoggedGateEventRow, ...]
    event_evaluation: EventEvaluation


def evaluate_logged_gate_session(
    session_dir: Path,
    *,
    event_column: str = "twinkle_candidate_accepted",
    score_column: str = "blink_score",
    threshold: float = 0.5,
    min_score: float | None = None,
    max_score: float | None = None,
    refractory_s: float = 0.0,
    tolerance_s: float = 0.8,
    positive_labels: Sequence[str] = ("blink",),
    ignore_startup_s: float = 2.0,
) -> LoggedGateEvaluation:
    session_dir = Path(session_dir)
    features = read_csv_rows(session_dir / "features.csv")
    markers = read_csv_rows(session_dir / "manual_markers.csv")
    events = detect_logged_gate_events(
        session_name=session_dir.name,
        feature_rows=features,
        event_column=event_column,
        score_column=score_column,
        threshold=threshold,
        min_score=min_score,
        max_score=max_score,
        refractory_s=refractory_s,
        ignore_startup_s=ignore_startup_s,
    )
    evaluation = evaluate_events(
        session_name=session_dir.name,
        markers=markers,
        events=logged_gate_events_as_dicts(events),
        tolerance_s=tolerance_s,
        positive_labels=positive_labels,
        event_labels=(event_column,),
        ignore_startup_s=ignore_startup_s,
    )
    return LoggedGateEvaluation(
        session=session_dir.name,
        event_column=str(event_column),
        score_column=str(score_column),
        threshold=float(threshold),
        min_score=None if min_score is None else float(min_score),
        max_score=None if max_score is None else float(max_score),
        refractory_s=float(refractory_s),
        events=events,
        event_evaluation=evaluation,
    )


def detect_logged_gate_events(
    *,
    session_name: str,
    feature_rows: Sequence[dict[str, str]],
    event_column: str,
    score_column: str,
    threshold: float,
    min_score: float | None = None,
    max_score: float | None = None,
    refractory_s: float = 0.0,
    ignore_startup_s: float = 0.0,
) -> tuple[LoggedGateEventRow, ...]:
    events: list[LoggedGateEventRow] = []
    previous_active = False
    last_event_time_s = -1e9
    for row in feature_rows:
        time_s = _float(row.get("time_s"))
        gate_value = _float(row.get(event_column))
        active = gate_value > float(threshold)
        is_rising_edge = active and not previous_active
        previous_active = active
        if time_s < float(ignore_startup_s) or not is_rising_edge:
            continue
        score = _float(row.get(score_column))
        if min_score is not None and score < float(min_score):
            continue
        if max_score is not None and score > float(max_score):
            continue
        if time_s - last_event_time_s < float(refractory_s):
            continue
        last_event_time_s = time_s
        events.append(
            LoggedGateEventRow(
                session=session_name,
                event_id=len(events) + 1,
                time_s=time_s,
                label=str(event_column),
                method=f"logged:{event_column}",
                score=score,
                gate_value=gate_value,
            )
        )
    return tuple(events)


def logged_gate_events_as_dicts(events: Sequence[LoggedGateEventRow]) -> list[dict[str, str]]:
    return [
        {
            "event_id": str(event.event_id),
            "time_s": f"{event.time_s:.6f}",
            "label": event.label,
            "method": event.method,
            "score": f"{event.score:.9f}",
        }
        for event in events
    ]


def write_logged_gate_outputs(evaluation: LoggedGateEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "logged_gate_events.csv", evaluation.events)
    write_event_evaluation_outputs(evaluation.event_evaluation, output_dir)
    payload = asdict(evaluation)
    payload["events"] = [asdict(row) for row in evaluation.events]
    payload["event_evaluation"] = asdict(evaluation.event_evaluation)
    with (output_dir / "logged_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
