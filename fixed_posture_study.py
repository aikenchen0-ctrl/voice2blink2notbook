from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SessionMarkerSummary:
    session: str
    blink_markers: int
    negative_markers: int
    total_markers: int
    has_phase_point_log: bool


@dataclass(frozen=True)
class FixedPostureStudyDecision:
    status: str
    marker_requirements_met: bool
    phase_point_logging_met: bool
    fmcw_physical_line_met: bool
    fusion_recall_met: bool
    fusion_precision_met: bool
    false_positive_met: bool
    negative_conflict_met: bool
    recommendation: str


def summarize_session_markers(
    session_dir: Path,
    *,
    negative_labels: Sequence[str] = ("large_motion",),
) -> SessionMarkerSummary:
    session_dir = Path(session_dir)
    rows = _read_csv_rows(session_dir / "manual_markers.csv")
    negative_label_set = set(negative_labels)
    blink_markers = sum(1 for row in rows if row.get("label") == "blink")
    negative_markers = sum(1 for row in rows if row.get("label") in negative_label_set)
    return SessionMarkerSummary(
        session=session_dir.name,
        blink_markers=int(blink_markers),
        negative_markers=int(negative_markers),
        total_markers=int(len(rows)),
        has_phase_point_log=has_phase_point_log(session_dir),
    )


def has_phase_point_log(session_dir: Path) -> bool:
    feature_path = Path(session_dir) / "features.csv"
    if not feature_path.exists():
        return False
    for row in _read_csv_rows(feature_path):
        if row.get("fmcw_phase_points", "").strip():
            return True
    return False


def decide_fixed_posture_study(
    *,
    blink_marker_total: int,
    negative_marker_total: int,
    session_count: int,
    phase_point_log_session_count: int,
    recommended_pair: str | None,
    fused_recall: float,
    fused_precision: float,
    fused_false_positives: int,
    fused_negative_conflicts: int,
    min_blink_markers: int,
    min_negative_markers: int,
    target_recall: float,
    min_precision: float,
    max_false_positives: int,
    max_negative_conflicts: int,
) -> FixedPostureStudyDecision:
    marker_requirements_met = (
        int(blink_marker_total) >= int(min_blink_markers)
        and int(negative_marker_total) >= int(min_negative_markers)
    )
    phase_point_logging_met = int(session_count) > 0 and int(phase_point_log_session_count) == int(session_count)
    fmcw_physical_line_met = bool(recommended_pair)
    fusion_recall_met = float(fused_recall) >= float(target_recall)
    fusion_precision_met = float(fused_precision) >= float(min_precision)
    false_positive_met = int(fused_false_positives) <= int(max_false_positives)
    negative_conflict_met = int(fused_negative_conflicts) <= int(max_negative_conflicts)
    misjudgment_met = fusion_precision_met and false_positive_met and negative_conflict_met

    if not phase_point_logging_met:
        status = "need_phase_point_logging"
        recommendation = "重新采集并保留 fmcw_phase_points，不要使用 --fmcw-no-phase-point-log。"
    elif not misjudgment_met:
        status = "needs_false_positive_iteration"
        recommendation = (
            "不能只看召回；当前 precision、false positive 或 w 冲突未达标，"
            "先迭代误报压制，再判断是否需要继续补标。"
        )
    elif not marker_requirements_met:
        status = "need_more_fixed_posture_labels"
        recommendation = "误报门槛当前未失败，但样本不足；继续固定姿态采集，补足 blink/w 标注数量。"
    elif not fmcw_physical_line_met:
        status = "fmcw_physical_line_not_supported"
        recommendation = "当前数据不支持固定 FMCW 眨眼物理线；继续固定姿态复采或将 FMCW 降级为干扰压制。"
    elif not fusion_recall_met:
        status = "needs_recall_iteration"
        recommendation = "误报门槛已过，但召回未达标；继续离线调 candidate/fusion 参数。"
    else:
        status = "ready_for_realtime_validation"
        recommendation = "离线门槛已达标，可以把参数接入实时页面做人工复测。"

    return FixedPostureStudyDecision(
        status=status,
        marker_requirements_met=marker_requirements_met,
        phase_point_logging_met=phase_point_logging_met,
        fmcw_physical_line_met=fmcw_physical_line_met,
        fusion_recall_met=fusion_recall_met,
        fusion_precision_met=fusion_precision_met,
        false_positive_met=false_positive_met,
        negative_conflict_met=negative_conflict_met,
        recommendation=recommendation,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
