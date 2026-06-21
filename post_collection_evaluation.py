from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hp_acoustic_wave.blink_layer_evaluation import (
    evaluate_blink_layers_session,
    write_blink_layer_outputs,
)
from hp_acoustic_wave.candidate_fusion_evaluation import (
    evaluate_candidate_fusion_session,
    write_candidate_fusion_outputs,
)
from hp_acoustic_wave.candidate_fusion_sweep import (
    best_sweep_rows,
    sweep_candidate_fusion_session,
    write_candidate_fusion_sweep_outputs,
)
from hp_acoustic_wave.fixed_posture_study import summarize_session_markers
from hp_acoustic_wave.visual_label_audit import (
    audit_visual_labels,
    write_visual_label_audit_outputs,
)


@dataclass(frozen=True)
class PostCollectionSummaryRow:
    session: str
    blink_markers: int
    negative_markers: int
    visual_events: int
    valid_visual_events: int
    visual_valid_event_rate: float
    visual_available_rate: float
    visual_face_found_rate: float
    layer_best_name: str
    layer_best_recall: float
    layer_best_precision: float
    layer_best_f1: float
    fused_recall: float
    fused_precision: float
    fused_f1: float
    fused_false_positive: int
    fused_negative_conflicts: int
    sweep_min_recall: float
    sweep_best_recall: float
    sweep_best_precision: float
    sweep_best_f1: float
    sweep_best_false_positive: int
    needs_negative_labels: bool
    reaches_target_recall: bool
    reaches_min_precision: bool
    decision_status: str
    recommendation: str


@dataclass(frozen=True)
class PostCollectionEvaluation:
    session: str
    summary: PostCollectionSummaryRow


def evaluate_post_collection_session(
    session_dir: Path,
    *,
    output_dir: Path,
    tolerance_s: float = 0.8,
    ignore_startup_s: float = 2.0,
    require_visual_face: bool = True,
    min_visual_face_found_rate: float = 0.50,
    min_negative_markers: int = 20,
    target_recall: float = 0.95,
    min_precision: float = 0.80,
    sweep_min_recall: float = 0.85,
) -> PostCollectionEvaluation:
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_summary = summarize_session_markers(
        session_dir,
        negative_labels=("large_motion", "wave", "w"),
    )

    visual_audit = audit_visual_labels(session_dir)
    visual_dir = output_dir / "visual_label_audit"
    write_visual_label_audit_outputs(visual_audit, visual_dir)

    layer_eval = evaluate_blink_layers_session(
        session_dir,
        tolerance_s=float(tolerance_s),
        ignore_startup_s=float(ignore_startup_s),
        require_visual_face=bool(require_visual_face),
    )
    layer_dir = output_dir / "blink_layers"
    write_blink_layer_outputs(layer_eval, layer_dir)
    layer_best = max(layer_eval.rows, key=lambda row: (row.f1, row.recall, row.precision))

    fusion_eval = evaluate_candidate_fusion_session(
        session_dir,
        tolerance_s=float(tolerance_s),
        ignore_startup_s=float(ignore_startup_s),
        require_visual_face=bool(require_visual_face),
    )
    fusion_dir = output_dir / "candidate_fusion"
    write_candidate_fusion_outputs(fusion_eval, fusion_dir)
    fused_metric = fusion_eval.fused_evaluation.metrics[0]

    sweep_eval = sweep_candidate_fusion_session(
        session_dir,
        tolerance_s=float(tolerance_s),
        ignore_startup_s=float(ignore_startup_s),
        require_visual_face=bool(require_visual_face),
    )
    sweep_dir = output_dir / "candidate_fusion_parameter_sweep"
    write_candidate_fusion_sweep_outputs(sweep_eval, sweep_dir)
    best_sweep = best_sweep_rows(sweep_eval.rows, min_recall=float(sweep_min_recall), limit=1)
    sweep_best = best_sweep[0] if best_sweep else None

    needs_negative_labels = int(marker_summary.negative_markers) < int(min_negative_markers)
    reaches_target_recall = float(fused_metric.recall) >= float(target_recall)
    reaches_min_precision = float(fused_metric.precision) >= float(min_precision)
    needs_visual_face = (
        bool(require_visual_face)
        and int(visual_audit.summary.row_total) > 0
        and float(visual_audit.summary.face_found_rate) < float(min_visual_face_found_rate)
    )
    decision_status, recommendation = decide_post_collection_next_step(
        needs_visual_face=needs_visual_face,
        needs_negative_labels=needs_negative_labels,
        reaches_target_recall=reaches_target_recall,
        reaches_min_precision=reaches_min_precision,
        fused_false_positive=int(fused_metric.false_positive),
        fused_negative_conflicts=int(fusion_eval.summary.fused_negative_conflict_total),
    )
    summary = PostCollectionSummaryRow(
        session=session_dir.name,
        blink_markers=int(marker_summary.blink_markers),
        negative_markers=int(marker_summary.negative_markers),
        visual_events=int(visual_audit.summary.visual_event_total),
        valid_visual_events=int(visual_audit.summary.valid_visual_event_total),
        visual_valid_event_rate=float(visual_audit.summary.valid_visual_event_rate),
        visual_available_rate=float(visual_audit.summary.available_rate),
        visual_face_found_rate=float(visual_audit.summary.face_found_rate),
        layer_best_name=str(layer_best.layer),
        layer_best_recall=float(layer_best.recall),
        layer_best_precision=float(layer_best.precision),
        layer_best_f1=float(layer_best.f1),
        fused_recall=float(fused_metric.recall),
        fused_precision=float(fused_metric.precision),
        fused_f1=float(fused_metric.f1),
        fused_false_positive=int(fused_metric.false_positive),
        fused_negative_conflicts=int(fusion_eval.summary.fused_negative_conflict_total),
        sweep_min_recall=float(sweep_min_recall),
        sweep_best_recall=0.0 if sweep_best is None else float(sweep_best.recall),
        sweep_best_precision=0.0 if sweep_best is None else float(sweep_best.precision),
        sweep_best_f1=0.0 if sweep_best is None else float(sweep_best.f1),
        sweep_best_false_positive=0 if sweep_best is None else int(sweep_best.false_positive),
        needs_negative_labels=needs_negative_labels,
        reaches_target_recall=reaches_target_recall,
        reaches_min_precision=reaches_min_precision,
        decision_status=decision_status,
        recommendation=recommendation,
    )
    evaluation = PostCollectionEvaluation(session=session_dir.name, summary=summary)
    write_post_collection_outputs(evaluation, output_dir)
    return evaluation


def decide_post_collection_next_step(
    *,
    needs_negative_labels: bool,
    reaches_target_recall: bool,
    reaches_min_precision: bool,
    fused_false_positive: int,
    fused_negative_conflicts: int,
    needs_visual_face: bool = False,
) -> tuple[str, str]:
    if bool(needs_visual_face):
        return (
            "need_visual_face",
            "视觉标注不可用：摄像头/MediaPipe 已运行，但没有稳定看到脸；先调整取景、光照和距离，再重新采集。",
        )
    if bool(needs_negative_labels):
        return (
            "need_negative_labels",
            "继续固定姿态采集；眨眼照常，挥手/转头/大动作时按终端 w，先补足负样本。",
        )
    if not bool(reaches_target_recall):
        return (
            "needs_recall_iteration",
            "负样本已够，但召回未达 95%；优先调主候选/阈值，避免先做强压制。",
        )
    if not bool(reaches_min_precision) or int(fused_false_positive) > 0 or int(fused_negative_conflicts) > 0:
        return (
            "needs_false_positive_iteration",
            "召回已够但误报/负样本冲突未达标；优先用分层诊断定位误报来源并压制。",
        )
    return (
        "ready_for_realtime_validation",
        "离线快评估已达标；把参数接入实时页面，进入人工复测。",
    )


def write_post_collection_outputs(evaluation: PostCollectionEvaluation, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "post_collection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(evaluation.summary).keys()))
        writer.writeheader()
        writer.writerow(asdict(evaluation.summary))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(evaluation.summary), handle, indent=2, ensure_ascii=False)
