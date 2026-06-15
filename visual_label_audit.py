from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class VisualLabelAuditRow:
    session: str
    source: str
    row_index: int
    time_s: float
    label: str
    key: str
    available: bool
    face_found: bool
    left_ear: float
    right_ear: float
    left_closed: bool
    right_closed: bool
    is_blink_event: bool
    blink_count: int
    inference_ms: float
    error: str
    valid_event: bool
    issue: str


@dataclass(frozen=True)
class VisualLabelAuditSummary:
    session: str
    source: str
    row_total: int
    visual_event_total: int
    valid_visual_event_total: int
    invalid_visual_event_total: int
    visual_marker_total: int
    visual_event_marker_mismatch: int
    error_total: int
    unavailable_total: int
    no_face_total: int
    event_without_face_total: int
    event_not_both_closed_total: int
    available_rate: float
    face_found_rate: float
    valid_visual_event_rate: float
    median_left_ear: float
    median_right_ear: float
    blink_count_monotonic: bool


@dataclass(frozen=True)
class VisualLabelAudit:
    session: str
    rows: tuple[VisualLabelAuditRow, ...]
    summary: VisualLabelAuditSummary


def audit_visual_labels(
    session_dir: Path,
    *,
    visual_key: str = "v",
    ear_threshold: float = 0.22,
) -> VisualLabelAudit:
    session_dir = Path(session_dir)
    visual_path = session_dir / "visual_features.csv"
    marker_rows = read_csv_rows(session_dir / "manual_markers.csv")
    visual_marker_total = sum(1 for row in marker_rows if row.get("key", "") == visual_key)
    if visual_path.exists() and visual_path.stat().st_size > 0:
        rows = _audit_visual_feature_rows(
            session_dir.name,
            read_csv_rows(visual_path),
            ear_threshold=ear_threshold,
        )
        source = "visual_features"
    else:
        rows = _audit_visual_marker_rows(
            session_dir.name,
            marker_rows,
            visual_key=visual_key,
            ear_threshold=ear_threshold,
        )
        source = "manual_markers"

    summary = summarize_visual_label_rows(
        session_dir.name,
        rows,
        source=source,
        visual_marker_total=visual_marker_total,
    )
    return VisualLabelAudit(session=session_dir.name, rows=tuple(rows), summary=summary)


def summarize_visual_label_rows(
    session_name: str,
    rows: Sequence[VisualLabelAuditRow],
    *,
    source: str,
    visual_marker_total: int,
) -> VisualLabelAuditSummary:
    row_total = len(rows)
    event_rows = [row for row in rows if row.is_blink_event]
    valid_events = [row for row in event_rows if row.valid_event]
    errors = [row for row in rows if row.error]
    unavailable = [row for row in rows if not row.available]
    no_face = [row for row in rows if row.available and not row.face_found]
    event_without_face = [row for row in event_rows if not row.face_found]
    event_not_both_closed = [row for row in event_rows if not (row.left_closed and row.right_closed)]
    left_ears = [row.left_ear for row in rows if row.left_ear > 0.0]
    right_ears = [row.right_ear for row in rows if row.right_ear > 0.0]
    blink_counts = [row.blink_count for row in rows]
    blink_count_monotonic = all(
        later >= earlier for earlier, later in zip(blink_counts, blink_counts[1:])
    )
    return VisualLabelAuditSummary(
        session=session_name,
        source=source,
        row_total=row_total,
        visual_event_total=len(event_rows),
        valid_visual_event_total=len(valid_events),
        invalid_visual_event_total=len(event_rows) - len(valid_events),
        visual_marker_total=int(visual_marker_total),
        visual_event_marker_mismatch=abs(len(event_rows) - int(visual_marker_total)),
        error_total=len(errors),
        unavailable_total=len(unavailable),
        no_face_total=len(no_face),
        event_without_face_total=len(event_without_face),
        event_not_both_closed_total=len(event_not_both_closed),
        available_rate=_ratio(sum(row.available for row in rows), row_total),
        face_found_rate=_ratio(sum(row.face_found for row in rows), row_total),
        valid_visual_event_rate=_ratio(len(valid_events), len(event_rows)),
        median_left_ear=float(median(left_ears)) if left_ears else 0.0,
        median_right_ear=float(median(right_ears)) if right_ears else 0.0,
        blink_count_monotonic=bool(blink_count_monotonic),
    )


def write_visual_label_audit_outputs(audit: VisualLabelAudit, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "visual_label_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(audit.rows[0]).keys()) if audit.rows else _row_fields())
        writer.writeheader()
        for row in audit.rows:
            writer.writerow(asdict(row))
    with (output_dir / "visual_label_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(audit.summary).keys()))
        writer.writeheader()
        writer.writerow(asdict(audit.summary))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(audit.summary), handle, indent=2, ensure_ascii=False)


def _audit_visual_feature_rows(
    session_name: str,
    rows: Sequence[dict[str, str]],
    *,
    ear_threshold: float,
) -> tuple[VisualLabelAuditRow, ...]:
    return tuple(
        _audit_row(
            session_name,
            row,
            source="visual_features",
            row_index=index,
            label="blink" if _bool(row.get("is_blink_event")) else "",
            key="v" if _bool(row.get("is_blink_event")) else "",
            ear_threshold=ear_threshold,
        )
        for index, row in enumerate(rows)
    )


def _audit_visual_marker_rows(
    session_name: str,
    rows: Sequence[dict[str, str]],
    *,
    visual_key: str,
    ear_threshold: float,
) -> tuple[VisualLabelAuditRow, ...]:
    visual_rows = [(index, row) for index, row in enumerate(rows) if row.get("key", "") == visual_key]
    return tuple(
        _audit_row(
            session_name,
            row,
            source="manual_markers",
            row_index=index,
            label=row.get("label", ""),
            key=row.get("key", ""),
            ear_threshold=ear_threshold,
        )
        for index, row in visual_rows
    )


def _audit_row(
    session_name: str,
    row: dict[str, str],
    *,
    source: str,
    row_index: int,
    label: str,
    key: str,
    ear_threshold: float,
) -> VisualLabelAuditRow:
    available = _bool(_first(row, "available", "visual_available"))
    face_found = _bool(_first(row, "face_found", "visual_face_found"))
    left_ear = _float(_first(row, "left_ear", "visual_left_ear"))
    right_ear = _float(_first(row, "right_ear", "visual_right_ear"))
    left_closed = _bool(_first(row, "left_closed", "visual_left_closed"))
    right_closed = _bool(_first(row, "right_closed", "visual_right_closed"))
    is_blink_event = _bool(_first(row, "is_blink_event", "visual_is_blink_event"))
    blink_count = _int(_first(row, "blink_count", "visual_blink_count"))
    inference_ms = _float(_first(row, "inference_ms", "visual_inference_ms"))
    error = str(_first(row, "error", "visual_error") or "")
    issue = _visual_issue(
        available=available,
        face_found=face_found,
        left_ear=left_ear,
        right_ear=right_ear,
        left_closed=left_closed,
        right_closed=right_closed,
        is_blink_event=is_blink_event,
        error=error,
        ear_threshold=ear_threshold,
    )
    valid_event = bool(is_blink_event and issue == "")
    return VisualLabelAuditRow(
        session=session_name,
        source=source,
        row_index=int(row_index),
        time_s=_float(row.get("time_s")),
        label=label,
        key=key,
        available=available,
        face_found=face_found,
        left_ear=left_ear,
        right_ear=right_ear,
        left_closed=left_closed,
        right_closed=right_closed,
        is_blink_event=is_blink_event,
        blink_count=blink_count,
        inference_ms=inference_ms,
        error=error,
        valid_event=valid_event,
        issue=issue,
    )


def _visual_issue(
    *,
    available: bool,
    face_found: bool,
    left_ear: float,
    right_ear: float,
    left_closed: bool,
    right_closed: bool,
    is_blink_event: bool,
    error: str,
    ear_threshold: float,
) -> str:
    if error:
        return "error"
    if not available:
        return "unavailable"
    if not is_blink_event:
        return ""
    if not face_found:
        return "event_without_face"
    if not (left_closed and right_closed):
        return "event_not_both_closed"
    if left_ear >= float(ear_threshold) or right_ear >= float(ear_threshold):
        return "event_ear_above_threshold"
    return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _row_fields() -> list[str]:
    return [
        "session",
        "source",
        "row_index",
        "time_s",
        "label",
        "key",
        "available",
        "face_found",
        "left_ear",
        "right_ear",
        "left_closed",
        "right_closed",
        "is_blink_event",
        "blink_count",
        "inference_ms",
        "error",
        "valid_event",
        "issue",
    ]
