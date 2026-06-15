from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from hp_acoustic_wave.app import RealtimeHandWaveApp
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig, build_blink_detector
from hp_acoustic_wave.config import AppConfig


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _phase_pairs(value: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError("pairs must look like 19:29,20:29")
        left, right = part.split(":", 1)
        try:
            reference = int(left)
            target = int(right)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("phase pair indices must be integers") from exc
        if reference < 0 or target < 0 or reference == target:
            raise argparse.ArgumentTypeError("phase pair indices must be non-negative and different")
        pairs.append((reference, target))
    if not pairs:
        raise argparse.ArgumentTypeError("at least one phase pair is required")
    return tuple(pairs)


def _phase_pair(value: str) -> tuple[int, int]:
    pairs = _phase_pairs(value)
    if len(pairs) != 1:
        raise argparse.ArgumentTypeError("exactly one phase pair is required")
    return pairs[0]


def build_parser() -> argparse.ArgumentParser:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(
        description="Run the HP/macOS realtime acoustic hand-wave or blink-candidate detector.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print PortAudio input/output devices and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=("wave", "blink", "fmcw"),
        default=defaults.mode,
        help="Detection mode. wave is the hand-wave baseline; blink enables blink-candidate detectors; fmcw shows multi-track phase-difference trajectories.",
    )
    parser.add_argument(
        "--session-root",
        default=defaults.session_root,
        help="Directory where session outputs are written.",
    )
    parser.add_argument(
        "--duration",
        "--max-duration",
        dest="max_duration_s",
        type=_positive_float,
        default=None,
        help="Optional auto-stop duration in seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run audio capture/detection without opening a camera or OpenCV window.",
    )
    parser.add_argument(
        "--no-video-record",
        action="store_true",
        help="Keep camera/UI enabled but skip camera.mp4 encoding and writing.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print one-line per-second runtime timing stats for stutter diagnosis.",
    )
    parser.add_argument(
        "--ui-fps",
        type=_positive_float,
        default=defaults.ui_fps,
        help="Target OpenCV UI refresh FPS; detection still runs on every audio block.",
    )
    parser.add_argument(
        "--ui-text-fps",
        type=_positive_float,
        default=defaults.ui_text_fps,
        help="Target overlay text refresh FPS; lower values reduce Chinese font rendering load.",
    )
    parser.add_argument(
        "--collection-target-blinks",
        type=int,
        default=defaults.collection_target_blinks,
        help="Optional UI target count for b/blink manual labels; 0 hides the target.",
    )
    parser.add_argument(
        "--collection-target-negatives",
        type=int,
        default=defaults.collection_target_negatives,
        help="Optional UI target count for w/large_motion manual labels; 0 hides the target.",
    )

    audio = defaults.audio
    parser.add_argument("--sample-rate", type=int, default=audio.sample_rate)
    parser.add_argument("--chunk-size", type=int, default=audio.chunk_size)
    parser.add_argument("--frequency", "--tone-hz", dest="tone_hz", type=_positive_float, default=audio.tone_hz)
    parser.add_argument("--amplitude", type=_nonnegative_float, default=audio.output_amplitude)
    parser.add_argument("--input-device", type=int, default=audio.input_device)
    parser.add_argument("--output-device", type=int, default=audio.output_device)

    camera = defaults.camera
    parser.add_argument("--no-camera", action="store_true", help="Disable webcam capture; the UI still opens.")
    parser.add_argument("--camera-index", type=int, default=camera.index)
    parser.add_argument("--camera-width", type=int, default=camera.width)
    parser.add_argument("--camera-height", type=int, default=camera.height)
    parser.add_argument("--camera-fps", type=float, default=camera.fps)

    visual = defaults.visual_blink
    parser.add_argument(
        "--visual-blink",
        action=argparse.BooleanOptionalAction,
        default=visual.enabled,
        help="Enable MediaPipe visual blink detection on the same camera frame.",
    )
    parser.add_argument(
        "--visual-blink-auto-mark",
        action=argparse.BooleanOptionalAction,
        default=visual.auto_mark_blinks,
        help="Write visual blink events into manual_markers.csv as label=blink,key=v.",
    )
    parser.add_argument("--visual-blink-threshold", type=_nonnegative_float, default=visual.threshold)
    parser.add_argument("--visual-blink-refractory", type=_nonnegative_float, default=visual.refractory_s)
    parser.add_argument("--visual-blink-model", default=visual.model_path)
    parser.add_argument("--visual-blink-max-fps", type=_positive_float, default=visual.max_fps)
    parser.add_argument(
        "--visual-blink-min-face-detection-confidence",
        type=_nonnegative_float,
        default=visual.min_face_detection_confidence,
    )
    parser.add_argument(
        "--visual-blink-min-face-presence-confidence",
        type=_nonnegative_float,
        default=visual.min_face_presence_confidence,
    )
    parser.add_argument(
        "--visual-blink-min-tracking-confidence",
        type=_nonnegative_float,
        default=visual.min_tracking_confidence,
    )

    detector = defaults.detector
    parser.add_argument("--threshold-k", type=float, default=detector.threshold_k)
    parser.add_argument("--min-energy", type=_nonnegative_float, default=detector.min_energy)
    parser.add_argument("--refractory", dest="refractory_s", type=_nonnegative_float, default=detector.refractory_s)
    parser.add_argument(
        "--baseline-freeze",
        dest="baseline_freeze_s",
        type=_nonnegative_float,
        default=detector.baseline_freeze_s,
    )

    blink = defaults.blink
    parser.add_argument(
        "--blink-method",
        choices=("blinklistener", "twinkle", "both"),
        default=blink.method,
        help="Blink-candidate detector used when --mode blink.",
    )
    parser.add_argument("--blink-threshold-k", type=float, default=blink.threshold_k)
    parser.add_argument("--blink-min-score", type=_nonnegative_float, default=blink.min_score)
    parser.add_argument("--blink-startup-ignore", type=_nonnegative_float, default=blink.startup_ignore_s)
    parser.add_argument("--blink-release-ratio", type=_nonnegative_float, default=blink.release_ratio)
    parser.add_argument("--blink-phase-step-floor", type=_nonnegative_float, default=blink.phase_step_floor)
    parser.add_argument("--blink-peak-min-ratio", type=_nonnegative_float, default=blink.twinkle_peak_min_ratio)
    parser.add_argument("--blink-min-peak-score", type=_nonnegative_float, default=blink.twinkle_min_peak_score)
    parser.add_argument("--blink-max-peak-score", type=_nonnegative_float, default=blink.twinkle_max_peak_score)
    parser.add_argument("--blink-min-motion-energy", type=_nonnegative_float, default=blink.twinkle_min_motion_energy)
    parser.add_argument("--blink-max-motion-energy", type=_nonnegative_float, default=blink.twinkle_max_motion_energy)
    parser.add_argument("--blink-min-sign-changes", type=int, default=blink.twinkle_min_sign_changes)
    parser.add_argument("--blink-max-sign-changes", type=int, default=blink.twinkle_max_sign_changes)

    fmcw = defaults.fmcw
    # FMCW 参数分三层：发射/取相位、多轨投票、candidate->confirm 的实时眨眼判定。
    parser.add_argument(
        "--fmcw-band-preset",
        choices=("paper", "hybrid-split"),
        default="paper",
        help="paper keeps the 18-22 kHz TwinkleTwinkle chirp; hybrid-split moves chirp above the 18.5 kHz primary blink tone.",
    )
    parser.add_argument("--fmcw-f0", type=_positive_float, default=None)
    parser.add_argument("--fmcw-f1", type=_positive_float, default=None)
    parser.add_argument("--fmcw-period-samples", type=int, default=fmcw.period_samples)
    parser.add_argument("--fmcw-chirp-samples", type=int, default=fmcw.chirp_samples)
    parser.add_argument("--fmcw-guard-samples", type=int, default=fmcw.guard_samples)
    parser.add_argument("--fmcw-decimation", type=int, default=fmcw.decimation_factor)
    parser.add_argument("--fmcw-no-raw-highpass", action="store_true", help="Disable raw received-signal high-pass before FMCW demodulation.")
    parser.add_argument("--fmcw-raw-highpass-hz", type=_nonnegative_float, default=fmcw.raw_highpass_hz)
    parser.add_argument("--fmcw-raw-highpass-taps", type=int, default=fmcw.raw_highpass_taps)
    parser.add_argument("--fmcw-decimation-filter-taps-per-phase", type=int, default=fmcw.decimation_filter_taps_per_phase)
    parser.add_argument("--fmcw-valid-start", type=int, default=fmcw.valid_start)
    parser.add_argument(
        "--fmcw-valid-stop",
        type=int,
        default=None,
        help="Exclusive stop index for valid active-chirp FMCW phase points. Defaults to chirp_samples // decimation.",
    )
    parser.add_argument("--fmcw-track-count", type=int, default=fmcw.track_count)
    parser.add_argument(
        "--fmcw-track-pairs",
        type=_phase_pairs,
        default=(),
        help="Comma-separated realtime FMCW display/candidate phase pairs, e.g. 19:29,20:29,27:29.",
    )
    parser.add_argument(
        "--fmcw-fixed-trajectory-pair",
        type=_phase_pair,
        default=fmcw.fixed_trajectory_pair,
        help="Single diagnostic phase pair for the magenta physical trajectory line, e.g. 23:29.",
    )
    parser.add_argument("--fmcw-no-track-gap-normalization", action="store_true", help="Keep raw phase-difference scaling for realtime FMCW tracks.")
    parser.add_argument("--fmcw-phase-window-size", type=int, default=fmcw.phase_window_size)
    parser.add_argument(
        "--fmcw-trajectory-detrend-window",
        type=int,
        default=fmcw.trajectory_detrend_window,
        help="Engineering A/B option for trajectory detrending before FMCW vote. Paper-style default is 1 (disabled).",
    )
    parser.add_argument("--fmcw-trajectory-smoothing-window", type=int, default=fmcw.trajectory_smoothing_window)
    parser.add_argument(
        "--fmcw-segmentation-group-gap",
        type=_nonnegative_float,
        default=fmcw.segmentation_group_gap_s,
        help="Seconds between adjacent edge starts for grouping one blink segment. Paper default is 0.5s.",
    )
    parser.add_argument("--fmcw-vote-update-periods", type=int, default=fmcw.vote_update_periods)
    parser.add_argument("--fmcw-vote-min-delta-rms", type=_nonnegative_float, default=fmcw.vote_min_delta_rms)
    parser.add_argument(
        "--fmcw-confirm-window-vote-update-periods",
        type=int,
        default=fmcw.confirm_window_vote_update_periods,
        help="Throttle expensive pending-window FMCW vote refreshes; final confirmation still refreshes once.",
    )
    parser.add_argument(
        "--fmcw-no-phase-point-log",
        action="store_true",
        help="Do not write full FMCW phase point arrays into features.csv during realtime runs.",
    )
    # candidate 仍用旧版 rolling median/MAD 风格，负责灵敏地抓疑似动作。
    parser.add_argument("--fmcw-candidate-threshold-k", type=float, default=fmcw.candidate_threshold_k)
    parser.add_argument("--fmcw-candidate-min-score", type=_nonnegative_float, default=fmcw.candidate_min_score)
    parser.add_argument("--fmcw-candidate-refractory", type=_nonnegative_float, default=fmcw.candidate_refractory_s)
    parser.add_argument(
        "--fmcw-candidate-startup-ignore",
        type=_nonnegative_float,
        default=fmcw.candidate_startup_ignore_s,
    )
    parser.add_argument(
        "--fmcw-candidate-score-source",
        choices=("track_delta_rms", "track0_delta", "track1_delta", "track2_delta", "track3_delta", "track4_delta", "max_track_delta", "mean_track_delta"),
        default=fmcw.candidate_score_source,
    )
    # confirm 是最终层；默认先不强制 vote，避免未调好的多轨投票压低旧版召回。
    parser.add_argument("--fmcw-confirm-min-delta-rms", type=_nonnegative_float, default=fmcw.confirm_min_delta_rms)
    parser.add_argument("--fmcw-confirm-large-motion-delta-rms", type=_nonnegative_float, default=fmcw.confirm_large_motion_delta_rms)
    parser.add_argument("--fmcw-confirm-large-motion-duration", type=_nonnegative_float, default=fmcw.confirm_large_motion_duration_s)
    parser.add_argument("--fmcw-confirm-high-delta-rms", type=_nonnegative_float, default=fmcw.confirm_high_delta_rms)
    parser.add_argument("--fmcw-confirm-require-vote", action="store_true", default=fmcw.confirm_require_vote)
    parser.add_argument("--fmcw-confirm-vote-min-confidence", type=_nonnegative_float, default=fmcw.confirm_vote_min_confidence)
    parser.add_argument("--fmcw-confirm-vote-min-stability", type=_nonnegative_float, default=fmcw.confirm_vote_min_stability)
    parser.add_argument("--fmcw-no-primary-blink", action="store_true", help="Disable the parallel single-tone blink detector in FMCW mode.")
    parser.add_argument("--fmcw-primary-blink-tone-ratio", type=_nonnegative_float, default=fmcw.primary_blink_tone_ratio)
    parser.add_argument("--fmcw-primary-blink-chirp-ratio", type=_nonnegative_float, default=fmcw.primary_blink_chirp_ratio)
    parser.add_argument("--fmcw-no-primary-blink-peak", action="store_true", help="Disable the local-peak gate on the primary blink score curve.")
    parser.add_argument("--fmcw-primary-blink-peak-min-score", type=_nonnegative_float, default=fmcw.primary_blink_peak_min_score)
    parser.add_argument("--fmcw-primary-blink-peak-min-ratio", type=_nonnegative_float, default=fmcw.primary_blink_peak_min_ratio)
    parser.add_argument("--fmcw-primary-blink-peak-max-score", type=_nonnegative_float, default=fmcw.primary_blink_peak_max_score)
    parser.add_argument("--fmcw-primary-blink-peak-refractory", type=_nonnegative_float, default=fmcw.primary_blink_peak_refractory_s)
    parser.add_argument(
        "--fmcw-no-primary-blink-tone-cleanup",
        action="store_true",
        help="Keep the parallel single-tone component in FMCW phase extraction for A/B testing.",
    )
    parser.add_argument("--fmcw-no-sync", action="store_true", help="Disable automatic received-chirp period synchronization in FMCW mode.")
    parser.add_argument("--fmcw-sync-warmup-blocks", type=int, default=fmcw.sync_warmup_blocks)
    parser.add_argument("--fmcw-sync-min-confidence", type=_nonnegative_float, default=fmcw.sync_min_confidence)
    return parser


def _list_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Run: python -m pip install -r requirements_hp_acoustic_wave.txt", file=sys.stderr)
        return 2

    print("Default device [input, output]:", sd.default.device)
    print()
    print(sd.query_devices())
    return 0


def config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config = replace(
        config,
        mode=args.mode,
        session_root=args.session_root,
        max_duration_s=args.max_duration_s,
        headless=bool(args.headless),
        record_video=not bool(args.no_video_record),
        profile_performance=bool(args.profile),
        ui_fps=float(args.ui_fps),
        ui_text_fps=float(args.ui_text_fps),
        collection_target_blinks=max(0, int(args.collection_target_blinks)),
        collection_target_negatives=max(0, int(args.collection_target_negatives)),
    )
    config.audio.sample_rate = int(args.sample_rate)
    config.audio.chunk_size = int(args.chunk_size)
    config.audio.tone_hz = float(args.tone_hz)
    config.audio.output_amplitude = float(args.amplitude)
    config.audio.input_device = args.input_device
    config.audio.output_device = args.output_device

    config.camera.enabled = not args.no_camera and not args.headless
    config.camera.index = int(args.camera_index)
    config.camera.width = int(args.camera_width)
    config.camera.height = int(args.camera_height)
    config.camera.fps = float(args.camera_fps)

    config.visual_blink.enabled = bool(args.visual_blink) and config.camera.enabled
    config.visual_blink.auto_mark_blinks = bool(args.visual_blink_auto_mark)
    config.visual_blink.threshold = float(args.visual_blink_threshold)
    config.visual_blink.refractory_s = float(args.visual_blink_refractory)
    config.visual_blink.model_path = str(args.visual_blink_model)
    config.visual_blink.max_fps = float(args.visual_blink_max_fps)
    config.visual_blink.min_face_detection_confidence = float(
        args.visual_blink_min_face_detection_confidence
    )
    config.visual_blink.min_face_presence_confidence = float(
        args.visual_blink_min_face_presence_confidence
    )
    config.visual_blink.min_tracking_confidence = float(args.visual_blink_min_tracking_confidence)

    config.detector.threshold_k = float(args.threshold_k)
    config.detector.min_energy = float(args.min_energy)
    config.detector.refractory_s = float(args.refractory_s)
    config.detector.baseline_freeze_s = float(args.baseline_freeze_s)

    config.blink.method = args.blink_method
    config.blink.threshold_k = float(args.blink_threshold_k)
    config.blink.min_score = float(args.blink_min_score)
    config.blink.startup_ignore_s = float(args.blink_startup_ignore)
    config.blink.release_ratio = float(args.blink_release_ratio)
    config.blink.phase_step_floor = float(args.blink_phase_step_floor)
    config.blink.twinkle_peak_min_ratio = float(args.blink_peak_min_ratio)
    config.blink.twinkle_min_peak_score = float(args.blink_min_peak_score)
    config.blink.twinkle_max_peak_score = float(args.blink_max_peak_score)
    config.blink.twinkle_min_motion_energy = float(args.blink_min_motion_energy)
    config.blink.twinkle_max_motion_energy = float(args.blink_max_motion_energy)
    config.blink.twinkle_min_sign_changes = int(args.blink_min_sign_changes)
    config.blink.twinkle_max_sign_changes = int(args.blink_max_sign_changes)

    if args.fmcw_band_preset == "hybrid-split":
        preset_f0 = 19_500.0
        preset_f1 = 22_500.0
    else:
        preset_f0 = config.fmcw.f0_hz
        preset_f1 = config.fmcw.f1_hz
    config.fmcw.f0_hz = float(args.fmcw_f0 if args.fmcw_f0 is not None else preset_f0)
    config.fmcw.f1_hz = float(args.fmcw_f1 if args.fmcw_f1 is not None else preset_f1)
    config.fmcw.period_samples = int(args.fmcw_period_samples)
    config.fmcw.chirp_samples = int(args.fmcw_chirp_samples)
    config.fmcw.guard_samples = int(args.fmcw_guard_samples)
    config.fmcw.decimation_factor = int(args.fmcw_decimation)
    config.fmcw.raw_highpass_enabled = not bool(args.fmcw_no_raw_highpass)
    config.fmcw.raw_highpass_hz = float(args.fmcw_raw_highpass_hz)
    config.fmcw.raw_highpass_taps = int(args.fmcw_raw_highpass_taps)
    config.fmcw.decimation_filter_taps_per_phase = int(args.fmcw_decimation_filter_taps_per_phase)
    config.fmcw.valid_start = int(args.fmcw_valid_start)
    if args.fmcw_valid_stop is None:
        config.fmcw.valid_stop = int(config.fmcw.chirp_samples // config.fmcw.decimation_factor)
    else:
        config.fmcw.valid_stop = int(args.fmcw_valid_stop)
    config.fmcw.track_count = int(args.fmcw_track_count)
    config.fmcw.track_pairs = tuple(args.fmcw_track_pairs)
    config.fmcw.fixed_trajectory_pair = tuple(args.fmcw_fixed_trajectory_pair)
    config.fmcw.track_gap_normalization = not bool(args.fmcw_no_track_gap_normalization)
    config.fmcw.phase_window_size = int(args.fmcw_phase_window_size)
    config.fmcw.trajectory_detrend_window = int(args.fmcw_trajectory_detrend_window)
    config.fmcw.trajectory_smoothing_window = int(args.fmcw_trajectory_smoothing_window)
    config.fmcw.segmentation_group_gap_s = float(args.fmcw_segmentation_group_gap)
    config.fmcw.segmentation_sample_rate_hz = int(config.audio.sample_rate)
    config.fmcw.vote_update_periods = int(args.fmcw_vote_update_periods)
    config.fmcw.vote_min_delta_rms = float(args.fmcw_vote_min_delta_rms)
    config.fmcw.confirm_window_vote_update_periods = int(args.fmcw_confirm_window_vote_update_periods)
    config.fmcw.log_phase_points = not bool(args.fmcw_no_phase_point_log)
    config.fmcw.candidate_threshold_k = float(args.fmcw_candidate_threshold_k)
    config.fmcw.candidate_min_score = float(args.fmcw_candidate_min_score)
    config.fmcw.candidate_refractory_s = float(args.fmcw_candidate_refractory)
    config.fmcw.candidate_startup_ignore_s = float(args.fmcw_candidate_startup_ignore)
    config.fmcw.candidate_score_source = str(args.fmcw_candidate_score_source)
    config.fmcw.confirm_min_delta_rms = float(args.fmcw_confirm_min_delta_rms)
    config.fmcw.confirm_large_motion_delta_rms = float(args.fmcw_confirm_large_motion_delta_rms)
    config.fmcw.confirm_large_motion_duration_s = float(args.fmcw_confirm_large_motion_duration)
    config.fmcw.confirm_high_delta_rms = float(args.fmcw_confirm_high_delta_rms)
    config.fmcw.confirm_require_vote = bool(args.fmcw_confirm_require_vote)
    config.fmcw.confirm_vote_min_confidence = float(args.fmcw_confirm_vote_min_confidence)
    config.fmcw.confirm_vote_min_stability = float(args.fmcw_confirm_vote_min_stability)
    config.fmcw.primary_blink_enabled = not bool(args.fmcw_no_primary_blink)
    config.fmcw.primary_blink_tone_ratio = float(args.fmcw_primary_blink_tone_ratio)
    config.fmcw.primary_blink_chirp_ratio = float(args.fmcw_primary_blink_chirp_ratio)
    config.fmcw.primary_blink_peak_enabled = not bool(args.fmcw_no_primary_blink_peak)
    config.fmcw.primary_blink_peak_min_score = float(args.fmcw_primary_blink_peak_min_score)
    config.fmcw.primary_blink_peak_min_ratio = float(args.fmcw_primary_blink_peak_min_ratio)
    config.fmcw.primary_blink_peak_max_score = float(args.fmcw_primary_blink_peak_max_score)
    config.fmcw.primary_blink_peak_refractory_s = float(args.fmcw_primary_blink_peak_refractory)
    config.fmcw.primary_blink_tone_cleanup_enabled = not bool(args.fmcw_no_primary_blink_tone_cleanup)
    config.fmcw.primary_blink_tone_hz = float(config.audio.tone_hz)
    config.fmcw.sync_enabled = not bool(args.fmcw_no_sync)
    config.fmcw.sync_warmup_blocks = int(args.fmcw_sync_warmup_blocks)
    config.fmcw.sync_min_confidence = float(args.fmcw_sync_min_confidence)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        return _list_devices()

    config = config_from_args(args)
    if config.mode == "blink":
        build_blink_detector(BlinkDetectionConfig(**config.blink.__dict__))
    session_dir = RealtimeHandWaveApp(config).run()
    print(f"Session saved to: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
