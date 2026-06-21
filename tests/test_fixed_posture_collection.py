import argparse
from pathlib import Path

import pytest

from scripts import run_fixed_posture_collection
from scripts.run_fixed_posture_collection import fixed_posture_detector_argv


def test_fixed_posture_collection_keeps_phase_point_logging_by_default():
    args = argparse.Namespace(
        session_root="sessions",
        input_device=0,
        output_device=1,
        amplitude=0.02,
        camera_width=320,
        camera_height=240,
        camera_fps=10.0,
        record_video=False,
        no_profile=False,
        no_post_eval=False,
        post_eval_output_root="docs/experiments",
        detector_args=[],
    )

    argv = fixed_posture_detector_argv(args)

    assert argv[:4] == ["--mode", "fmcw", "--fmcw-band-preset", "hybrid-split"]
    assert "--fmcw-no-phase-point-log" not in argv
    assert "--no-video-record" in argv
    assert "--profile" in argv
    assert argv[argv.index("--collection-target-blinks") + 1] == "40"
    assert argv[argv.index("--collection-target-negatives") + 1] == "20"


def test_fixed_posture_collection_rejects_disabling_phase_point_log():
    args = argparse.Namespace(
        session_root="sessions",
        input_device=0,
        output_device=1,
        amplitude=0.02,
        camera_width=320,
        camera_height=240,
        camera_fps=10.0,
        record_video=False,
        no_profile=False,
        no_post_eval=False,
        post_eval_output_root="docs/experiments",
        detector_args=["--", "--fmcw-no-phase-point-log"],
    )

    with pytest.raises(SystemExit):
        fixed_posture_detector_argv(args)


def test_fixed_posture_collection_runs_post_eval_after_detector(monkeypatch, tmp_path):
    session = tmp_path / "sessions" / "hp_fmcw_test"
    session.mkdir(parents=True)
    calls = {}

    def fake_detector_main(argv):
        calls["detector_argv"] = argv
        return session

    def fake_evaluate_post_collection_session(session_dir, *, output_dir, min_negative_markers):
        calls["post_eval"] = {
            "session_dir": Path(session_dir),
            "output_dir": Path(output_dir),
            "min_negative_markers": min_negative_markers,
        }

        class Evaluation:
            class summary:
                blink_markers = 40
                negative_markers = 20
                needs_negative_labels = False
                fused_recall = 0.95
                fused_precision = 0.85
                fused_false_positive = 0
                fused_negative_conflicts = 0
                sweep_best_recall = 0.96
                sweep_best_precision = 0.86
                sweep_best_false_positive = 0

        return Evaluation()

    monkeypatch.setattr(run_fixed_posture_collection, "detector_main", fake_detector_main)
    monkeypatch.setattr(
        run_fixed_posture_collection,
        "evaluate_post_collection_session",
        fake_evaluate_post_collection_session,
    )

    rc = run_fixed_posture_collection.main(
        [
            "--session-root",
            str(tmp_path / "sessions"),
            "--post-eval-output-root",
            str(tmp_path / "evals"),
        ]
    )

    assert rc == 0
    assert calls["post_eval"]["session_dir"] == session
    assert calls["post_eval"]["output_dir"] == tmp_path / "evals" / "hp_fmcw_test_post_collection_eval"
    assert calls["post_eval"]["min_negative_markers"] == 20
