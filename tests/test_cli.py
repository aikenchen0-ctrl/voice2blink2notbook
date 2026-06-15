from hp_acoustic_wave.cli import build_parser, config_from_args


def test_cli_defaults_match_safe_macos_audio_profile():
    args = build_parser().parse_args([])
    config = config_from_args(args)

    assert config.audio.sample_rate == 48_000
    assert config.audio.tone_hz == 18_500.0
    assert config.audio.output_amplitude == 0.02
    assert config.mode == "wave"
    assert not config.headless
    assert config.camera.enabled
    assert config.visual_blink.enabled
    assert config.visual_blink.auto_mark_blinks


def test_cli_headless_disables_camera_and_keeps_duration():
    args = build_parser().parse_args(["--headless", "--duration", "3", "--input-device", "0", "--output-device", "1"])
    config = config_from_args(args)

    assert config.headless
    assert not config.camera.enabled
    assert not config.visual_blink.enabled
    assert config.max_duration_s == 3.0
    assert config.audio.input_device == 0
    assert config.audio.output_device == 1


def test_cli_can_disable_default_visual_blink():
    args = build_parser().parse_args(["--no-visual-blink", "--no-visual-blink-auto-mark"])
    config = config_from_args(args)

    assert not config.visual_blink.enabled
    assert not config.visual_blink.auto_mark_blinks


def test_cli_visual_blink_options_map_to_config():
    args = build_parser().parse_args(
        [
            "--visual-blink-threshold",
            "0.24",
            "--visual-blink-refractory",
            "0.4",
            "--visual-blink-model",
            "assets/custom.task",
            "--visual-blink-max-fps",
            "8",
            "--visual-blink-min-face-detection-confidence",
            "0.6",
            "--visual-blink-min-face-presence-confidence",
            "0.7",
            "--visual-blink-min-tracking-confidence",
            "0.8",
        ]
    )
    config = config_from_args(args)

    assert config.visual_blink.enabled
    assert config.visual_blink.threshold == 0.24
    assert config.visual_blink.refractory_s == 0.4
    assert config.visual_blink.model_path == "assets/custom.task"
    assert config.visual_blink.max_fps == 8.0
    assert config.visual_blink.min_face_detection_confidence == 0.6
    assert config.visual_blink.min_face_presence_confidence == 0.7
    assert config.visual_blink.min_tracking_confidence == 0.8


def test_cli_collection_targets_map_to_config():
    args = build_parser().parse_args(
        [
            "--collection-target-blinks",
            "40",
            "--collection-target-negatives",
            "20",
        ]
    )
    config = config_from_args(args)

    assert config.collection_target_blinks == 40
    assert config.collection_target_negatives == 20


def test_cli_blink_options_map_to_config():
    args = build_parser().parse_args(
        [
            "--mode",
            "blink",
            "--blink-method",
            "both",
            "--blink-threshold-k",
            "2.1",
            "--blink-startup-ignore",
            "1.5",
            "--blink-phase-step-floor",
            "0.02",
            "--blink-min-peak-score",
            "0.04",
            "--blink-max-motion-energy",
            "0.12",
            "--blink-max-sign-changes",
            "3",
        ]
    )
    config = config_from_args(args)

    assert config.mode == "blink"
    assert config.blink.method == "both"
    assert config.blink.threshold_k == 2.1
    assert config.blink.startup_ignore_s == 1.5
    assert config.blink.phase_step_floor == 0.02
    assert config.blink.twinkle_min_peak_score == 0.04
    assert config.blink.twinkle_max_motion_energy == 0.12
    assert config.blink.twinkle_max_sign_changes == 3


def test_cli_fmcw_options_map_to_config():
    args = build_parser().parse_args(
        [
            "--mode",
            "fmcw",
            "--fmcw-f0",
            "18000",
            "--fmcw-f1",
            "22000",
            "--fmcw-period-samples",
            "512",
            "--fmcw-chirp-samples",
            "480",
            "--fmcw-guard-samples",
            "32",
            "--fmcw-no-raw-highpass",
            "--fmcw-raw-highpass-hz",
            "16500",
            "--fmcw-raw-highpass-taps",
            "33",
            "--fmcw-decimation-filter-taps-per-phase",
            "6",
            "--fmcw-track-count",
            "5",
            "--fmcw-fixed-trajectory-pair",
            "23:29",
            "--fmcw-no-track-gap-normalization",
            "--fmcw-phase-window-size",
            "96",
            "--fmcw-trajectory-detrend-window",
            "31",
            "--fmcw-trajectory-smoothing-window",
            "5",
            "--fmcw-segmentation-group-gap",
            "0.4",
            "--fmcw-valid-stop",
            "26",
            "--fmcw-vote-update-periods",
            "10",
            "--fmcw-vote-min-delta-rms",
            "0.2",
            "--fmcw-candidate-threshold-k",
            "4.5",
            "--fmcw-candidate-min-score",
            "0.04",
            "--fmcw-candidate-refractory",
            "0.6",
            "--fmcw-candidate-startup-ignore",
            "3.0",
            "--fmcw-candidate-score-source",
            "track2_delta",
            "--fmcw-confirm-min-delta-rms",
            "0.03",
            "--fmcw-confirm-large-motion-delta-rms",
            "0.06",
            "--fmcw-confirm-large-motion-duration",
            "0.08",
            "--fmcw-confirm-high-delta-rms",
            "0.04",
            "--fmcw-confirm-require-vote",
            "--fmcw-confirm-vote-min-confidence",
            "0.25",
            "--fmcw-confirm-vote-min-stability",
            "0.5",
            "--fmcw-no-primary-blink",
            "--fmcw-primary-blink-tone-ratio",
            "0.7",
            "--fmcw-primary-blink-chirp-ratio",
            "0.6",
            "--fmcw-no-primary-blink-peak",
            "--fmcw-primary-blink-peak-min-score",
            "0.07",
            "--fmcw-primary-blink-peak-min-ratio",
            "1.3",
            "--fmcw-primary-blink-peak-max-score",
            "0.4",
            "--fmcw-primary-blink-peak-refractory",
            "0.8",
            "--fmcw-no-primary-blink-tone-cleanup",
            "--fmcw-no-sync",
            "--fmcw-sync-warmup-blocks",
            "3",
            "--fmcw-sync-min-confidence",
            "0.2",
        ]
    )
    config = config_from_args(args)

    assert config.mode == "fmcw"
    assert config.fmcw.f0_hz == 18_000.0
    assert config.fmcw.f1_hz == 22_000.0
    assert config.fmcw.period_samples == 512
    assert config.fmcw.chirp_samples == 480
    assert config.fmcw.guard_samples == 32
    assert not config.fmcw.raw_highpass_enabled
    assert config.fmcw.raw_highpass_hz == 16_500.0
    assert config.fmcw.raw_highpass_taps == 33
    assert config.fmcw.decimation_filter_taps_per_phase == 6
    assert config.fmcw.valid_stop == 26
    assert config.fmcw.track_count == 5
    assert config.fmcw.fixed_trajectory_pair == (23, 29)
    assert not config.fmcw.track_gap_normalization
    assert config.fmcw.phase_window_size == 96
    assert config.fmcw.trajectory_detrend_window == 31
    assert config.fmcw.trajectory_smoothing_window == 5
    assert config.fmcw.segmentation_group_gap_s == 0.4
    assert config.fmcw.segmentation_sample_rate_hz == 48_000
    assert config.fmcw.vote_update_periods == 10
    assert config.fmcw.vote_min_delta_rms == 0.2
    assert config.fmcw.candidate_threshold_k == 4.5
    assert config.fmcw.candidate_min_score == 0.04
    assert config.fmcw.candidate_refractory_s == 0.6
    assert config.fmcw.candidate_startup_ignore_s == 3.0
    assert config.fmcw.candidate_score_source == "track2_delta"
    assert config.fmcw.confirm_min_delta_rms == 0.03
    assert config.fmcw.confirm_large_motion_delta_rms == 0.06
    assert config.fmcw.confirm_large_motion_duration_s == 0.08
    assert config.fmcw.confirm_high_delta_rms == 0.04
    assert config.fmcw.confirm_require_vote
    assert config.fmcw.confirm_vote_min_confidence == 0.25
    assert config.fmcw.confirm_vote_min_stability == 0.5
    assert not config.fmcw.primary_blink_enabled
    assert config.fmcw.primary_blink_tone_ratio == 0.7
    assert config.fmcw.primary_blink_chirp_ratio == 0.6
    assert not config.fmcw.primary_blink_peak_enabled
    assert config.fmcw.primary_blink_peak_min_score == 0.07
    assert config.fmcw.primary_blink_peak_min_ratio == 1.3
    assert config.fmcw.primary_blink_peak_max_score == 0.4
    assert config.fmcw.primary_blink_peak_refractory_s == 0.8
    assert not config.fmcw.primary_blink_tone_cleanup_enabled
    assert config.fmcw.primary_blink_tone_hz == 18_500.0
    assert not config.fmcw.sync_enabled
    assert config.fmcw.sync_warmup_blocks == 3
    assert config.fmcw.sync_min_confidence == 0.2


def test_cli_fmcw_valid_stop_defaults_to_decimated_chirp_length():
    defaults = config_from_args(build_parser().parse_args(["--mode", "fmcw"]))
    long_chirp = config_from_args(
        build_parser().parse_args(
            [
                "--mode",
                "fmcw",
                "--fmcw-period-samples",
                "1024",
                "--fmcw-chirp-samples",
                "960",
                "--fmcw-guard-samples",
                "64",
                "--fmcw-decimation",
                "16",
            ]
        )
    )

    assert defaults.fmcw.valid_stop == 30
    assert defaults.fmcw.primary_blink_peak_min_score == 0.04
    assert defaults.fmcw.primary_blink_peak_min_ratio == 0.8
    assert defaults.fmcw.primary_blink_peak_refractory_s == 1.2
    assert long_chirp.fmcw.valid_stop == 60


def test_cli_fmcw_hybrid_split_preset_moves_chirp_above_primary_tone():
    config = config_from_args(build_parser().parse_args(["--mode", "fmcw", "--fmcw-band-preset", "hybrid-split"]))

    assert config.fmcw.f0_hz == 19_500.0
    assert config.fmcw.f1_hz == 22_500.0
    assert config.fmcw.primary_blink_tone_hz == 18_500.0
    assert config.fmcw.primary_blink_tone_hz < config.fmcw.f0_hz


def test_cli_fmcw_explicit_band_overrides_preset():
    config = config_from_args(
        build_parser().parse_args(
            [
                "--mode",
                "fmcw",
                "--fmcw-band-preset",
                "hybrid-split",
                "--fmcw-f0",
                "19000",
                "--fmcw-f1",
                "22000",
            ]
        )
    )

    assert config.fmcw.f0_hz == 19_000.0
    assert config.fmcw.f1_hz == 22_000.0


def test_cli_fmcw_paper_style_trajectory_preprocessing_is_default():
    config = config_from_args(build_parser().parse_args(["--mode", "fmcw"]))

    assert config.fmcw.trajectory_detrend_window == 1
    assert config.fmcw.trajectory_smoothing_window == 3
