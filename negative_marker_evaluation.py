from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class NegativeMarkerConflictRow:
    session: str
    marker_index: int
    marker_time_s: float
    marker_label: str
    marker_key: str
    conflict: bool
    nearest_conflict_event_index: int
    nearest_conflict_event_id: str
    nearest_conflict_event_time_s: float
    nearest_conflict_offset_s: float
    nearest_conflict_score: float
    suppressed: bool
    nearest_suppressed_event_index: int
    nearest_suppressed_event_id: str
    nearest_suppressed_event_time_s: float
    nearest_suppressed_offset_s: float
    nearest_suppressed_score: float


@dataclass(frozen=True)
class NegativeMarkerMetricRow:
    session: str
    tolerance_s: float
    negative_total: int
    conflict_total: int
    suppressed_total: int
    conflict_rate: float
    suppression_rate: float


@dataclass(frozen=True)
class NegativeMarkerEvaluation:
    session: str
    tolerance_s: float
    negative_labels: tuple[str, ...]
    conflict_event_labels: tuple[str, ...]
    suppressed_event_labels: tuple[str, ...]
    conflicts: tuple[NegativeMarkerConflictRow, ...]
    metrics: tuple[NegativeMarkerMetricRow, ...]


def evaluate_negative_markers_session(
    session_dir: Path,
    *,
    tolerance_s: float = 0.8,
    negative_labels: Sequence[str] = ("large_motion",),
    conflict_event_labels: Sequence[str] = ("fmcw_confirmed_blink",),
    suppressed_event_labels: Sequence[str] = ("fmcw_suppressed_motion",),
    ignore_startup_s: float = 0.0,
) -> NegativeMarkerEvaluation:
    session_dir = Path(session_dir)
    return evaluate_negative_markers(
        session_name=session_dir.name,
        markers=read_csv_rows(session_dir / "manual_markers.csv"),
        events=read_csv_rows(session_dir / "events.csv"),
        tolerance_s=tolerance_s,
        negative_labels=negative_labels,
        conflict_event_labels=conflict_event_labels,
        suppressed_event_labels=suppressed_event_labels,
        ignore_startup_s=ignore_startup_s,
    )


def evaluate_negative_markers(
    *,
    session_name: str,
    markers: Sequence[dict[str, str]],
    events: Sequence[dict[str, str]],
    tolerance_s: float = 0.8,
    negative_labels: Sequence[str] = ("large_motion",),
    conflict_event_labels: Sequence[str] = ("fmcw_confirmed_blink",),
    suppressed_event_labels: Sequence[str] = ("fmcw_suppressed_motion",),
    ignore_startup_s: float = 0.0,
) -> NegativeMarkerEvaluation:
    negative_set = tuple(negative_labels)
    conflict_set = tuple(conflict_event_labels)
    suppressed_set = tuple(suppressed_event_labels)
    marker_rows = [
        (index, row)
        for index, row in enumerate(markers)
        if row.get("label", "") in negative_set and _float(row.get("time_s")) >= float(ignore_startup_s)
    ]
    event_rows = [
        (index, row)
        for index, row in enumerate(events)
        if _float(row.get("time_s")) >= float(ignore_startup_s)
    ]

    conflicts = []
    for marker_index, marker in marker_rows:
        marker_time = _float(marker.get("time_s"))
        conflict = _nearest_event(
            marker_time,
            event_rows,
            accepted_labels=conflict_set,
            tolerance_s=tolerance_s,
        )
        suppressed = _nearest_event(
            marker_time,
            event_rows,
            accepted_labels=suppressed_set,
            tolerance_s=tolerance_s,
        )
        conflicts.append(
            NegativeMarkerConflictRow(
                session=session_name,
                marker_index=int(marker_index),
                marker_time_s=float(marker_time),
                marker_label=marker.get("label", ""),
                marker_key=marker.get("key", ""),
                conflict=conflict is not None,
                nearest_conflict_event_index=int(conflict[0]) if conflict else -1,
                nearest_conflict_event_id=conflict[1].get("event_id", "") if conflict else "",
                nearest_conflict_event_time_s=_float(conflict[1].get("time_s")) if conflict else 0.0,
                nearest_conflict_offset_s=_float(conflict[1].get("time_s")) - marker_time if conflict else 0.0,
                nearest_conflict_score=_float(conflict[1].get("score")) if conflict else 0.0,
                suppressed=suppressed is not None,
                nearest_suppressed_event_index=int(suppressed[0]) if suppressed else -1,
                nearest_suppressed_event_id=suppressed[1].get("event_id", "") if suppressed else "",
                nearest_suppressed_event_time_s=_float(suppressed[1].get("time_s")) if suppressed else 0.0,
                nearest_suppressed_offset_s=_float(suppressed[1].get("time_s")) - marker_time if suppressed else 0.0,
                nearest_suppressed_score=_float(suppressed[1].get("score")) if suppressed else 0.0,
            )
        )

    conflict_total = sum(row.conflict for row in conflicts)
    suppressed_total = sum(row.suppressed for row in conflicts)
    metric = NegativeMarkerMetricRow(
        session=session_name,
        tolerance_s=float(tolerance_s),
        negative_total=len(conflicts),
        conflict_total=int(conflict_total),
        suppressed_total=int(suppressed_total),
        conflict_rate=_ratio(conflict_total, len(conflicts)),
        suppression_rate=_ratio(suppressed_total, len(conflicts)),
    )
    return NegativeMarkerEvaluation(
        session=session_name,
        tolerance_s=float(tolerance_s),
        negative_labels=negative_set,
        conflict_event_labels=conflict_set,
        suppressed_event_labels=suppressed_set,
        conflicts=tuple(conflicts),
        metrics=(metric,),
    )


def write_negative_marker_outputs(evaluation: NegativeMarkerEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(output_dir / "negative_marker_conflicts.csv", evaluation.conflicts)
    _write_dataclass_csv(output_dir / "negative_marker_metrics.csv", evaluation.metrics)
    payload = asdict(evaluation)
    payload["conflicts"] = [asdict(row) for row in evaluation.conflicts]
    payload["metrics"] = [asdict(row) for row in evaluation.metrics]
    with (output_dir / "negative_marker_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nearest_event(
    marker_time: float,
    event_rows: Sequence[tuple[int, dict[str, str]]],
    *,
    accepted_labels: Sequence[str],
    tolerance_s: float,
) -> tuple[int, dict[str, str]] | None:
    candidates = [
        (abs(_float(row.get("time_s")) - float(marker_time)), index, row)
        for index, row in event_rows
        if row.get("label", "") in accepted_labels
        and abs(_float(row.get("time_s")) - float(marker_time)) <= float(tolerance_s)
    ]
    if not candidates:
        return None
    _, index, row = min(candidates, key=lambda item: (item[0], item[1]))
    return index, row


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


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
