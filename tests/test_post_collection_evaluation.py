import csv
from dataclasses import dataclass

from hp_acoustic_wave.post_collection_evaluation import (
    PostCollectionSummaryRow,
    best_dataset_fmcw_sweep_row,
    decide_post_collection_next_step,
    evaluate_post_collection_session,
    summarize_post_collection_dataset,
)


def test_post_collection_evaluation_writes_summary_bundle(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_csv(
        session / "features.csv",
        [
            _feature_row(0.0, 0.01, 0),
            _feature_row(1.0, 0.12, 1),
            _feature_row(1.1, 0.01, 0),
        ],
    )
    _write_csv(session / "manual_markers.csv", [{"time_s": "1.0", "label": "blink", "key": "b"}])
    _write_csv(
        session / "visual_features.csv",
        [
            {
                "time_s": "1.0",
                "timestamp_ms": "1000",
                "available": "1",
                "face_found": "1",
                "left_ear": "0.10",
                "right_ear": "0.11",
                "left_closed": "1",
                "right_closed": "1",
                "is_blink_event": "1",
                "blink_count": "1",
                "inference_ms": "8.0",
                "error": "",
            }
        ],
    )

    evaluation = evaluate_post_collection_session(
        session,
        output_dir=tmp_path / "out",
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        require_visual_face=True,
        sweep_min_recall=0.0,
    )

    assert evaluation.summary.blink_markers == 1
    assert evaluation.summary.visual_events == 1
    assert evaluation.summary.valid_visual_events == 1
    assert evaluation.summary.visual_face_found_rate == 1.0
    assert evaluation.summary.decision_status == "need_negative_labels"
    assert (tmp_path / "out" / "post_collection_summary.csv").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "visual_label_audit" / "visual_label_summary.csv").exists()
    assert (tmp_path / "out" / "blink_layers" / "blink_layer_metrics.csv").exists()
    assert (tmp_path / "out" / "candidate_fusion" / "fused_candidate_events.csv").exists()
    assert (
        tmp_path
        / "out"
        / "candidate_fusion_parameter_sweep"
        / "candidate_fusion_parameter_sweep.csv"
    ).exists()


def test_decide_post_collection_next_step_orders_gates():
    assert (
        decide_post_collection_next_step(
            needs_visual_face=True,
            needs_negative_labels=True,
            reaches_target_recall=True,
            reaches_min_precision=True,
            fused_false_positive=0,
            fused_negative_conflicts=0,
        )[0]
        == "need_visual_face"
    )
    assert (
        decide_post_collection_next_step(
            needs_blink_labels=True,
            needs_negative_labels=True,
            reaches_target_recall=True,
            reaches_min_precision=True,
            fused_false_positive=0,
            fused_negative_conflicts=0,
        )[0]
        == "need_blink_labels"
    )
    assert (
        decide_post_collection_next_step(
            needs_negative_labels=True,
            reaches_target_recall=True,
            reaches_min_precision=True,
            fused_false_positive=0,
            fused_negative_conflicts=0,
        )[0]
        == "need_negative_labels"
    )
    assert (
        decide_post_collection_next_step(
            needs_negative_labels=False,
            reaches_target_recall=False,
            reaches_min_precision=True,
            fused_false_positive=0,
            fused_negative_conflicts=0,
        )[0]
        == "needs_recall_iteration"
    )
    assert (
        decide_post_collection_next_step(
            needs_negative_labels=False,
            reaches_target_recall=True,
            reaches_min_precision=False,
            fused_false_positive=3,
            fused_negative_conflicts=1,
        )[0]
        == "needs_false_positive_iteration"
    )
    assert (
        decide_post_collection_next_step(
            needs_negative_labels=False,
            reaches_target_recall=True,
            reaches_min_precision=True,
            fused_false_positive=0,
            fused_negative_conflicts=0,
        )[0]
        == "ready_for_realtime_validation"
    )


def test_post_collection_evaluation_prioritizes_missing_visual_face(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_csv(
        session / "features.csv",
        [
            _feature_row(0.0, 0.01, 0),
            _feature_row(1.0, 0.12, 1),
        ],
    )
    _write_csv(session / "manual_markers.csv", [{"time_s": "1.0", "label": "blink", "key": "b"}])
    _write_csv(
        session / "visual_features.csv",
        [
            {
                "time_s": "1.0",
                "timestamp_ms": "1000",
                "available": "1",
                "face_found": "0",
                "left_ear": "0.0",
                "right_ear": "0.0",
                "left_closed": "0",
                "right_closed": "0",
                "is_blink_event": "0",
                "blink_count": "0",
                "inference_ms": "8.0",
                "error": "",
            }
        ],
    )

    evaluation = evaluate_post_collection_session(
        session,
        output_dir=tmp_path / "out",
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        require_visual_face=True,
    )

    assert evaluation.summary.visual_available_rate == 1.0
    assert evaluation.summary.visual_face_found_rate == 0.0
    assert evaluation.summary.decision_status == "need_visual_face"


def test_summarize_post_collection_dataset_aggregates_counts_and_gates():
    summary = summarize_post_collection_dataset(
        [
            _summary_row(
                session="a",
                blink_markers=30,
                negative_markers=15,
                visual_face_found_rate=1.0,
                fused_true_positive=29,
                fused_false_negative=1,
                fused_false_positive=0,
            ),
            _summary_row(
                session="b",
                blink_markers=15,
                negative_markers=10,
                visual_face_found_rate=1.0,
                fused_true_positive=14,
                fused_false_negative=1,
                fused_false_positive=0,
            ),
        ],
        best_sweep=_sweep_row(
            min_vote_evidence=0.4,
            min_abs_trajectory_value=0.2,
            min_pair_stability=0.5,
            recall=0.96,
            precision=0.92,
            f1=0.94,
            false_positive=3,
            negative_conflict_total=0,
        ),
        sweep_min_recall=0.95,
        min_blink_markers=40,
        min_negative_markers=20,
        target_recall=0.95,
        min_precision=0.80,
    )

    assert summary.session_count == 2
    assert summary.blink_markers == 45
    assert summary.negative_markers == 25
    assert summary.fused_true_positive == 43
    assert summary.fused_false_negative == 2
    assert summary.fused_false_positive == 0
    assert round(summary.fused_recall, 6) == round(43 / 45, 6)
    assert summary.fused_precision == 1.0
    assert summary.best_sweep_min_vote_evidence == 0.4
    assert summary.best_sweep_precision == 0.92
    assert summary.best_sweep_false_positive == 3
    assert summary.decision_status == "ready_for_realtime_validation"


def test_summarize_post_collection_dataset_blocks_false_positives():
    summary = summarize_post_collection_dataset(
        [
            _summary_row(
                session="a",
                blink_markers=40,
                negative_markers=20,
                visual_face_found_rate=1.0,
                fused_true_positive=40,
                fused_false_positive=1,
            )
        ],
        min_blink_markers=40,
        min_negative_markers=20,
        target_recall=0.95,
        min_precision=0.80,
    )

    assert summary.fused_recall == 1.0
    assert summary.decision_status == "needs_false_positive_iteration"


def test_summarize_post_collection_dataset_prioritizes_visual_problem():
    summary = summarize_post_collection_dataset(
        [
            _summary_row(
                session="bad_visual",
                blink_markers=40,
                negative_markers=20,
                visual_face_found_rate=0.0,
                fused_true_positive=40,
            )
        ]
    )

    assert summary.visual_face_problem_sessions == 1
    assert summary.decision_status == "need_visual_face"


def test_best_dataset_fmcw_sweep_row_prefers_no_negative_conflict_then_f1():
    best = best_dataset_fmcw_sweep_row(
        [
            _sweep_row(
                min_vote_evidence=0.0,
                min_abs_trajectory_value=0.0,
                min_pair_stability=0.0,
                recall=0.99,
                precision=0.80,
                f1=0.88,
                false_positive=4,
                negative_conflict_total=1,
            ),
            _sweep_row(
                min_vote_evidence=0.4,
                min_abs_trajectory_value=0.2,
                min_pair_stability=0.5,
                recall=0.96,
                precision=0.93,
                f1=0.945,
                false_positive=1,
                negative_conflict_total=0,
            ),
            _sweep_row(
                min_vote_evidence=0.6,
                min_abs_trajectory_value=0.3,
                min_pair_stability=0.8,
                recall=0.90,
                precision=1.00,
                f1=0.947,
                false_positive=0,
                negative_conflict_total=0,
            ),
        ],
        min_recall=0.95,
    )

    assert best is not None
    assert best.min_vote_evidence == 0.4
    assert best.negative_conflict_total == 0


@dataclass(frozen=True)
class _SweepRow:
    min_vote_evidence: float
    min_abs_trajectory_value: float
    min_pair_stability: float
    recall: float
    precision: float
    f1: float
    false_positive: int
    negative_conflict_total: int


def _feature_row(time_s, blink_score, twinkle_peak):
    return {
        "time_s": str(time_s),
        "blink_score": str(blink_score),
        "blink_threshold": "0.05",
        "blink_method": "twinkle",
        "twinkle_candidate_peak": str(twinkle_peak),
        "twinkle_candidate_accepted": str(twinkle_peak),
        "fmcw_blink_vote_evidence": "0.0",
        "fmcw_blink_trajectory_value": "0.0",
        "fmcw_fixed_trajectory_distance_mm": "0.0",
        "fmcw_fixed_trajectory_phase_rad": "0.0",
        "fmcw_track_delta_rms": "0.01",
        "fmcw_confirm_window_confidence": "0.0",
    }


def _sweep_row(
    *,
    min_vote_evidence,
    min_abs_trajectory_value,
    min_pair_stability,
    recall,
    precision,
    f1,
    false_positive,
    negative_conflict_total,
):
    return _SweepRow(
        min_vote_evidence=min_vote_evidence,
        min_abs_trajectory_value=min_abs_trajectory_value,
        min_pair_stability=min_pair_stability,
        recall=recall,
        precision=precision,
        f1=f1,
        false_positive=false_positive,
        negative_conflict_total=negative_conflict_total,
    )


def _summary_row(
    *,
    session,
    blink_markers,
    negative_markers,
    visual_face_found_rate,
    fused_true_positive=0,
    fused_false_negative=0,
    fused_false_positive=0,
    fused_negative_conflicts=0,
):
    fused_event_total = int(fused_true_positive) + int(fused_false_positive)
    fused_marker_total = int(fused_true_positive) + int(fused_false_negative)
    precision = fused_true_positive / fused_event_total if fused_event_total else 0.0
    recall = fused_true_positive / fused_marker_total if fused_marker_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return PostCollectionSummaryRow(
        session=session,
        blink_markers=blink_markers,
        negative_markers=negative_markers,
        visual_events=blink_markers,
        valid_visual_events=blink_markers,
        visual_valid_event_rate=1.0,
        visual_available_rate=1.0,
        visual_face_found_rate=visual_face_found_rate,
        layer_best_name="fused",
        layer_best_recall=recall,
        layer_best_precision=precision,
        layer_best_f1=f1,
        fused_recall=recall,
        fused_precision=precision,
        fused_f1=f1,
        fused_marker_total=fused_marker_total,
        fused_event_total=fused_event_total,
        fused_true_positive=fused_true_positive,
        fused_false_negative=fused_false_negative,
        fused_false_positive=fused_false_positive,
        fused_negative_conflicts=fused_negative_conflicts,
        sweep_min_recall=0.85,
        sweep_best_recall=recall,
        sweep_best_precision=precision,
        sweep_best_f1=f1,
        sweep_best_false_positive=fused_false_positive,
        needs_negative_labels=negative_markers < 20,
        reaches_target_recall=recall >= 0.95,
        reaches_min_precision=precision >= 0.80,
        decision_status="ready_for_realtime_validation",
        recommendation="ok",
    )


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
