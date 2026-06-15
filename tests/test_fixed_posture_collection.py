import argparse

import pytest

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
        detector_args=["--", "--fmcw-no-phase-point-log"],
    )

    with pytest.raises(SystemExit):
        fixed_posture_detector_argv(args)
