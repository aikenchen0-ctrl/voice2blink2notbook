from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemplateGroupSummary:
    group: str
    reference_index: int
    target_index: int
    center_offset_s: float
    point_count: int
    window_count: int
    peak_relative_time: float
    peak_value: float
    span: float
    endpoint_delta: float
    sign: str
    rms: float
    blink_distance: float


def summarize_template_rows(rows: Iterable[dict[str, str]]) -> tuple[TemplateGroupSummary, ...]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group = str(row.get("group", ""))
        if not group:
            continue
        groups.setdefault(group, []).append(row)
    curves = {group: _curve_from_rows(items) for group, items in groups.items()}
    blink_curve = curves.get("blink")

    summaries: list[TemplateGroupSummary] = []
    for group, curve in sorted(curves.items()):
        means = curve["mean"]
        relative_times = curve["relative_time"]
        if means.size == 0:
            continue
        peak_index = int(np.argmax(np.abs(means)))
        span = float(np.ptp(means))
        endpoint_delta = float(means[-1] - means[0]) if means.size >= 2 else 0.0
        peak_value = float(means[peak_index])
        sign = "flat"
        if abs(peak_value) > 1e-12:
            sign = "positive" if peak_value > 0.0 else "negative"
        blink_distance = 0.0
        if blink_curve is not None and group != "blink":
            blink_distance = _curve_distance(blink_curve["mean"], means)
        summaries.append(
            TemplateGroupSummary(
                group=group,
                reference_index=int(curve["reference_index"]),
                target_index=int(curve["target_index"]),
                center_offset_s=float(curve["center_offset_s"]),
                point_count=int(means.size),
                window_count=int(curve["window_count"]),
                peak_relative_time=float(relative_times[peak_index]),
                peak_value=peak_value,
                span=span,
                endpoint_delta=endpoint_delta,
                sign=sign,
                rms=float(np.sqrt(np.mean(np.square(means)))),
                blink_distance=float(blink_distance),
            )
        )
    return tuple(summaries)


def _curve_from_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    rows = sorted(rows, key=lambda row: int(float(row.get("point_index", 0))))
    means = np.asarray([float(row.get("mean", 0.0)) for row in rows], dtype=np.float64)
    relative_times = np.asarray([float(row.get("relative_time", 0.0)) for row in rows], dtype=np.float64)
    first = rows[0]
    return {
        "mean": means,
        "relative_time": relative_times,
        "reference_index": int(float(first.get("reference_index", 0))),
        "target_index": int(float(first.get("target_index", 0))),
        "center_offset_s": float(first.get("center_offset_s", 0.0)),
        "window_count": max(int(float(row.get("window_count", 0))) for row in rows),
    }


def _curve_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    size = min(left.size, right.size)
    if size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(left[:size] - right[:size]))))
