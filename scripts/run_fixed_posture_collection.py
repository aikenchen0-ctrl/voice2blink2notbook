#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (PACKAGE_PARENT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hp_acoustic_wave.cli import main as detector_main


MIN_BLINK_MARKERS = 40
MIN_NEGATIVE_MARKERS = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-posture FMCW collection with phase-point logging enabled.",
    )
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--input-device", type=int, default=0)
    parser.add_argument("--output-device", type=int, default=1)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--record-video", action="store_true", help="Keep camera.mp4; default skips video encoding.")
    parser.add_argument("--no-profile", action="store_true", help="Disable runtime timing stats.")
    parser.add_argument(
        "detector_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to scripts/run_hp_wave_detector.py after '--'.",
    )
    return parser


def fixed_posture_detector_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--mode",
        "fmcw",
        "--fmcw-band-preset",
        "hybrid-split",
        "--session-root",
        str(args.session_root),
        "--input-device",
        str(args.input_device),
        "--output-device",
        str(args.output_device),
        "--amplitude",
        str(args.amplitude),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--camera-fps",
        str(args.camera_fps),
        "--collection-target-blinks",
        str(MIN_BLINK_MARKERS),
        "--collection-target-negatives",
        str(MIN_NEGATIVE_MARKERS),
    ]
    if not bool(args.record_video):
        argv.append("--no-video-record")
    if not bool(args.no_profile):
        argv.append("--profile")
    extras = list(args.detector_args or [])
    if extras and extras[0] == "--":
        extras = extras[1:]
    if "--fmcw-no-phase-point-log" in extras:
        raise SystemExit("fixed-posture collection must keep fmcw_phase_points; remove --fmcw-no-phase-point-log")
    return argv + extras


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector_argv = fixed_posture_detector_argv(args)
    print("Fixed-posture collection protocol:")
    print(f"  target: >= {MIN_BLINK_MARKERS} blink labels and >= {MIN_NEGATIVE_MARKERS} w labels")
    print("  visual blink auto labels: enabled by default as key=v when camera/MediaPipe are available")
    print("  keep head/device posture fixed until collection ends")
    print("  fmcw_phase_points logging: enabled")
    print("  keys: b=manual blink, w=large_motion, q/Esc=quit")
    return detector_main(detector_argv)


if __name__ == "__main__":
    raise SystemExit(main())
