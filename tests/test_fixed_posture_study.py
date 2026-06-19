from hp_acoustic_wave.fixed_posture_study import decide_fixed_posture_study
from scripts.evaluate_fixed_posture_study import (
    _phase_pair_gate_has_enough_labels,
    _write_skipped_phase_pair_summary,
)


def test_decide_fixed_posture_study_requires_enough_labels_first():
    decision = decide_fixed_posture_study(
        blink_marker_total=20,
        negative_marker_total=5,
        session_count=1,
        phase_point_log_session_count=1,
        recommended_pair="24:25",
        fused_recall=0.99,
        fused_precision=0.8,
        fused_false_positives=0,
        fused_negative_conflicts=0,
        min_blink_markers=40,
        min_negative_markers=20,
        target_recall=0.95,
        min_precision=0.5,
        max_false_positives=0,
        max_negative_conflicts=0,
    )

    assert decision.status == "need_more_fixed_posture_labels"
    assert not decision.marker_requirements_met


def test_decide_fixed_posture_study_prioritizes_false_positive_failures():
    decision = decide_fixed_posture_study(
        blink_marker_total=20,
        negative_marker_total=5,
        session_count=1,
        phase_point_log_session_count=1,
        recommended_pair="24:25",
        fused_recall=1.0,
        fused_precision=0.56,
        fused_false_positives=22,
        fused_negative_conflicts=5,
        min_blink_markers=40,
        min_negative_markers=20,
        target_recall=0.95,
        min_precision=0.8,
        max_false_positives=0,
        max_negative_conflicts=0,
    )

    assert decision.status == "needs_false_positive_iteration"
    assert decision.fusion_recall_met
    assert not decision.marker_requirements_met
    assert not decision.fusion_precision_met
    assert not decision.false_positive_met
    assert not decision.negative_conflict_met
    assert "不能只看召回" in decision.recommendation


def test_decide_fixed_posture_study_passes_when_all_gates_are_met():
    decision = decide_fixed_posture_study(
        blink_marker_total=45,
        negative_marker_total=20,
        session_count=2,
        phase_point_log_session_count=2,
        recommended_pair="24:25",
        fused_recall=0.96,
        fused_precision=0.55,
        fused_false_positives=0,
        fused_negative_conflicts=0,
        min_blink_markers=40,
        min_negative_markers=20,
        target_recall=0.95,
        min_precision=0.5,
        max_false_positives=0,
        max_negative_conflicts=0,
    )

    assert decision.status == "ready_for_realtime_validation"
    assert decision.marker_requirements_met
    assert decision.phase_point_logging_met
    assert decision.fmcw_physical_line_met


def test_phase_pair_gate_waits_for_blink_and_negative_labels():
    assert not _phase_pair_gate_has_enough_labels(
        blink_marker_total=80,
        negative_marker_total=0,
        min_blink_markers=40,
        min_negative_markers=20,
    )
    assert _phase_pair_gate_has_enough_labels(
        blink_marker_total=80,
        negative_marker_total=20,
        min_blink_markers=40,
        min_negative_markers=20,
    )


def test_skipped_phase_pair_summary_is_machine_readable(tmp_path):
    summary = _write_skipped_phase_pair_summary(
        tmp_path,
        reason="insufficient_fixed_posture_labels",
        blink_marker_total=187,
        negative_marker_total=0,
        min_blink_markers=40,
        min_negative_markers=20,
    )

    assert summary["skipped"] is True
    assert summary["recommended_line_pair"] is None
    assert summary["negative_marker_total"] == 0
    assert (tmp_path / "summary.json").exists()
