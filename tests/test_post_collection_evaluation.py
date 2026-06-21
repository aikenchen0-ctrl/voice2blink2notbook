import csv

from hp_acoustic_wave.post_collection_evaluation import (
    decide_post_collection_next_step,
    evaluate_post_collection_session,
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


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
