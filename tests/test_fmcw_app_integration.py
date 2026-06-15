from collections import deque

import numpy as np

from hp_acoustic_wave.app import RealtimeHandWaveApp
from hp_acoustic_wave.config import AppConfig
from hp_acoustic_wave.dsp import FmcwFeature, periodic_signal_chunk
from hp_acoustic_wave.fmcw_blink import FmcwTrajectoryEvidence
from hp_acoustic_wave.visual_blink_detector import VisualBlinkResult


def test_app_updates_fmcw_vote_from_rolling_phase_matrix():
    config = AppConfig(mode="fmcw")
    config.fmcw.phase_window_size = 64
    config.fmcw.trajectory_detrend_window = 1
    config.fmcw.trajectory_smoothing_window = 1
    config.fmcw.vote_min_delta_rms = 0.0
    app = RealtimeHandWaveApp(config)

    chirps = 64
    points = 30
    x = np.arange(chirps, dtype=np.float64)
    phase = np.tile(np.linspace(0.0, 3.0, points, dtype=np.float64), (chirps, 1))
    blink_shape = np.exp(-0.5 * np.square((x - 30.0) / 4.0))
    phase[:, 16:25] += blink_shape[:, None] * np.linspace(0.1, 0.6, 9)
    app.fmcw_processor.phase_history = deque(
        (row.copy() for row in phase),
        maxlen=config.fmcw.phase_window_size,
    )

    app._update_fmcw_vote()

    assert app.latest_fmcw_candidate_count == 45
    assert app.latest_fmcw_vote_score >= 0
    assert 0.0 <= app.latest_fmcw_vote_confidence <= 1.0


def test_app_suppresses_fmcw_vote_below_motion_floor():
    config = AppConfig(mode="fmcw")
    app = RealtimeHandWaveApp(config)

    app.latest_fmcw_pattern = "1"
    app.latest_fmcw_vote_confidence = 0.9
    app.latest_fmcw_vote_score = 40
    app.latest_fmcw_blink_vote_evidence = 0.9
    app.latest_fmcw_candidate_count = 45
    app._update_fmcw_vote(period_index=0, motion_score=0.0)

    assert app.latest_fmcw_pattern == ""
    assert app.latest_fmcw_vote_confidence == 0.0
    assert app.latest_fmcw_blink_vote_evidence == 0.0
    assert app.latest_fmcw_candidate_count == 0


def test_app_exposes_fmcw_blink_vote_evidence_only_for_blink_patterns():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app.latest_fmcw_candidate_count = 45
    app.latest_fmcw_vote_confidence = 0.73

    app.latest_fmcw_pattern = "1"
    assert app._fmcw_blink_vote_evidence() == 0.73

    app.latest_fmcw_pattern = "2"
    assert app._fmcw_blink_vote_evidence() == 0.0

    app.latest_fmcw_pattern = "1"
    app.latest_fmcw_candidate_count = 0
    assert app._fmcw_blink_vote_evidence() == 0.0


def test_app_tracks_manual_blink_and_large_motion_progress_in_title():
    config = AppConfig(mode="fmcw", collection_target_blinks=2, collection_target_negatives=1)
    app = RealtimeHandWaveApp(config)

    app._write_marker_for_key(ord("b"))
    app._write_marker_for_key(ord("w"))
    app._write_marker_for_key(ord("b"))

    assert app.manual_marker_count == 3
    assert app.manual_blink_marker_count == 2
    assert app.manual_large_motion_marker_count == 1

    _, _, _, title, help_text = app._overlay_texts()

    assert "b眨眼/blink=2/2" in title
    assert "w大动作/motion=1/1" in title
    assert "B blink" in help_text
    assert "W motion" in help_text


def test_app_marker_buttons_write_same_labels_as_keys():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    layout = app._marker_button_layout(320, 240)
    app._marker_button_regions = {key: region for key, region, _, _ in layout}

    assert app._handle_marker_click(20, layout[0][1][1] + 5)
    assert app._handle_marker_click(layout[1][1][0] + 5, layout[1][1][1] + 5)

    assert app.manual_marker_count == 2
    assert app.manual_blink_marker_count == 1
    assert app.manual_large_motion_marker_count == 1


def test_app_visual_blink_auto_mark_writes_blink_key_v():
    config = AppConfig(mode="fmcw")
    config.visual_blink.enabled = True
    config.visual_blink.auto_mark_blinks = True
    app = RealtimeHandWaveApp(config)
    written = []

    class Writer:
        def write_manual_marker(self, time_s, label, key, feature_snapshot, event_id):
            written.append(
                {
                    "time_s": time_s,
                    "label": label,
                    "key": key,
                    "feature_snapshot": feature_snapshot,
                    "event_id": event_id,
                }
            )

    class FakeVisualDetector:
        def process_frame(self, frame):
            return VisualBlinkResult(
                enabled=True,
                available=True,
                face_found=True,
                left_ear=0.12,
                right_ear=0.11,
                left_closed=True,
                right_closed=True,
                is_blink_event=True,
                blink_count=1,
                inference_ms=3.5,
            )

    app.writer = Writer()
    app.visual_blink_detector = FakeVisualDetector()

    app._process_visual_blink_frame(None, np.zeros((80, 120, 3), dtype=np.uint8))

    assert app.visual_blink_auto_marker_count == 1
    assert app.manual_blink_marker_count == 1
    assert written[0]["label"] == "blink"
    assert written[0]["key"] == "v"
    assert written[0]["feature_snapshot"]["visual_left_ear"] == 0.12
    assert written[0]["feature_snapshot"]["visual_is_blink_event"] == 1


def test_app_visual_progress_distinguishes_initializing_and_error():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))

    assert "initializing" in app._visual_blink_progress_text()

    app.latest_visual_blink_result = VisualBlinkResult(
        enabled=True,
        available=False,
        error="mediapipe failed",
    )
    assert "error" in app._visual_blink_progress_text()

    app.latest_visual_blink_result = VisualBlinkResult(
        enabled=True,
        available=True,
        face_found=False,
        blink_count=3,
    )
    assert "no-face" in app._visual_blink_progress_text()


def test_fmcw_overlay_uses_taller_split_diagnostic_panel():
    assert RealtimeHandWaveApp(AppConfig(mode="fmcw"))._overlay_plot_height() == 360
    assert RealtimeHandWaveApp(AppConfig(mode="blink"))._overlay_plot_height() == 160


def test_app_estimates_fmcw_received_period_sync_offset():
    config = AppConfig(mode="fmcw")
    config.fmcw.primary_blink_enabled = False
    config.fmcw.sync_warmup_blocks = 1
    config.fmcw.sync_min_confidence = 0.05
    app = RealtimeHandWaveApp(config)
    lag = 137
    samples = periodic_signal_chunk(app.fmcw_period, -lag, config.audio.chunk_size * 4)

    assert app._fmcw_sync_ready(samples)

    assert app.fmcw_sync_lag_samples == lag
    assert app.fmcw_processor.period_start_offset == lag
    assert app.fmcw_sync_confidence >= config.fmcw.sync_min_confidence


def _fmcw_feature(index: int, delta_rms: float) -> FmcwFeature:
    return FmcwFeature(
        time_s=index * 0.05,
        sample_index=index * 512,
        period_index=index,
        phase_point_count=30,
        track_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        track_delta_rms=delta_rms,
        phase_std=0.0,
        rms=0.0,
        peak_abs=0.0,
        pairs=((14, 15),),
        track_deltas=(delta_rms, delta_rms, delta_rms, delta_rms, delta_rms),
    )


def _fmcw_phase_feature(index: int, phase_points) -> FmcwFeature:
    return FmcwFeature(
        time_s=index * 512 / 48_000.0,
        sample_index=index * 512,
        period_index=index,
        phase_point_count=len(phase_points),
        track_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        track_delta_rms=0.08,
        phase_std=float(np.std(phase_points)),
        rms=0.0,
        peak_abs=0.0,
        pairs=((14, 15),),
        track_deltas=(0.08, 0.08, 0.08, 0.08, 0.08),
        phase_points=tuple(float(value) for value in phase_points),
    )


def _phase_window(chirps: int = 64, points: int = 30):
    for chirp in range(chirps):
        phase = np.linspace(0.0, 2.0, points, dtype=np.float64)
        blink_shape = np.exp(-0.5 * np.square((chirp - chirps / 2) / 4.0))
        phase[15:] += blink_shape * np.linspace(0.05, 0.45, points - 15)
        yield phase


def test_fmcw_mode_can_use_primary_blink_detector_for_main_event(tmp_path):
    config = AppConfig(mode="fmcw")
    config.session_root = str(tmp_path)
    config.fmcw.primary_blink_peak_enabled = False
    app = RealtimeHandWaveApp(config)
    detection = type(
        "Detection",
        (),
        {
            "is_event": True,
            "event_id": 3,
            "score": 0.12,
            "threshold": 0.04,
            "baseline": 0.01,
            "mad": 0.002,
            "method": "twinkle",
        },
    )()
    app.fmcw_primary_blink_detector = type("Detector", (), {"update": lambda self, feature: detection})()

    app._process_fmcw_primary_blink_samples(0, np.zeros(1024, dtype=np.float32))

    assert app.latest_event_id == 3
    assert app.latest_fmcw_primary_blink_is_event
    assert app.last_detection_method == "fmcw_primary_blink"
    assert app._detected_label() == "眨眼候选 / Blink candidate"


def test_fmcw_primary_blink_peak_gate_can_open_confirm_window(tmp_path):
    config = AppConfig(mode="fmcw")
    config.session_root = str(tmp_path)
    config.blink.startup_ignore_s = 0.0
    config.fmcw.primary_blink_peak_min_score = 0.06
    config.fmcw.primary_blink_peak_min_ratio = 1.0
    config.fmcw.primary_blink_peak_refractory_s = 0.2
    app = RealtimeHandWaveApp(config)
    scores = iter([0.02, 0.08, 0.03])

    class Detector:
        def update(self, feature):
            score = next(scores)
            return type(
                "Detection",
                (),
                {
                    "is_event": False,
                    "event_id": 0,
                    "score": score,
                    "threshold": 0.05,
                    "baseline": 0.04,
                    "mad": 0.002,
                    "method": "twinkle",
                },
            )()

    app.fmcw_primary_blink_detector = Detector()

    for index in range(3):
        app._process_fmcw_primary_blink_samples(index * 1024, np.zeros(1024, dtype=np.float32))

    assert app.latest_fmcw_primary_blink_is_event
    assert app.latest_fmcw_primary_blink_event_id == 1
    assert app.pending_fmcw_candidate_id == 1
    assert app.pending_fmcw_candidate_source == "primary_blink"
    assert app.last_detection_method == "fmcw_primary_blink"


def test_app_fmcw_candidate_detector_triggers_after_quiet_baseline():
    config = AppConfig(mode="fmcw")
    config.fmcw.candidate_min_history = 5
    config.fmcw.candidate_threshold_k = 3.0
    config.fmcw.candidate_min_score = 0.03
    config.fmcw.candidate_refractory_s = 0.1
    config.fmcw.candidate_startup_ignore_s = 0.0
    app = RealtimeHandWaveApp(config)

    for index in range(8):
        detection = app._update_fmcw_candidate_detector(_fmcw_feature(index, 0.02))
        assert not detection.is_event

    detection = app._update_fmcw_candidate_detector(_fmcw_feature(8, 0.12))
    row = app._fmcw_feature_row(_fmcw_feature(8, 0.12))

    assert detection.is_event
    assert app.latest_fmcw_candidate_is_event
    assert app.latest_fmcw_candidate_event_id == 1
    assert float(row["fmcw_candidate_score"]) == 0.12
    assert row["fmcw_candidate_is_event"] == 1
    assert row["fmcw_candidate_event_id"] == 1
    assert row["is_event"] == 1
    assert row["event_id"] == 1


def test_fmcw_feature_row_logs_primary_blink_detector_metrics():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app.latest_fmcw_primary_blink_metrics = {
        "phase_pair_delta": 0.12,
        "trajectory_span": 0.34,
        "twinkle_candidate_accepted": 1.0,
        "twinkle_reject_large_motion": 0.0,
    }

    row = app._fmcw_feature_row(_fmcw_feature(8, 0.04))

    assert row["twinkle_phase_pair_delta"] == "0.120000000"
    assert row["twinkle_trajectory_span"] == "0.340000000"
    assert row["twinkle_candidate_accepted"] == "1.000000000"
    assert row["twinkle_reject_large_motion"] == "0.000000000"


def test_fmcw_feature_row_logs_blink_vote_evidence():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app.latest_fmcw_pattern = "1"
    app.latest_fmcw_vote_confidence = 0.625
    app.latest_fmcw_candidate_count = 45
    app.latest_fmcw_blink_vote_evidence = app._fmcw_blink_vote_evidence()

    row = app._fmcw_feature_row(_fmcw_feature(8, 0.04))

    assert row["fmcw_blink_vote_evidence"] == "0.625000000"


def test_fmcw_feature_row_logs_representative_blink_trajectory():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app._set_fmcw_blink_trajectory_evidence(
        FmcwTrajectoryEvidence(
            value=-0.35,
            pattern="1",
            criterion="internal_similarity",
            reference_index=14,
            target_index=19,
            span=1.4,
            trajectory=(-0.35,),
        )
    )

    row = app._fmcw_feature_row(_fmcw_feature(8, 0.04))

    assert row["fmcw_blink_trajectory_value"] == "-0.350000000"
    assert row["fmcw_blink_trajectory_pattern"] == "1"
    assert row["fmcw_blink_trajectory_pair"] == "14:19"
    assert row["fmcw_blink_trajectory_criterion"] == "internal_similarity"


def test_app_updates_fixed_fmcw_trajectory_from_rolling_phase_pair():
    config = AppConfig(mode="fmcw")
    config.fmcw.trajectory_detrend_window = 1
    config.fmcw.trajectory_smoothing_window = 1
    app = RealtimeHandWaveApp(config)
    rows = []
    for value in np.linspace(0.0, 1.0, 8):
        phase_points = np.zeros(30, dtype=np.float64)
        phase_points[15] = value
        rows.append(phase_points)
    app.fmcw_processor.phase_history = deque(rows, maxlen=config.fmcw.phase_window_size)

    app._update_fmcw_fixed_trajectory(_fmcw_feature(8, 0.04))

    assert app.latest_fmcw_fixed_trajectory_pair == "14:15"
    assert app.latest_fmcw_fixed_trajectory_phase_rad > 0.4
    assert app.latest_fmcw_fixed_trajectory_distance_mm > 80.0
    assert app.latest_fmcw_fixed_trajectory_value == app.latest_fmcw_fixed_trajectory_distance_mm


def test_app_updates_fixed_fmcw_trajectory_from_configured_pair():
    config = AppConfig(mode="fmcw")
    config.fmcw.fixed_trajectory_pair = (23, 29)
    config.fmcw.trajectory_detrend_window = 1
    config.fmcw.trajectory_smoothing_window = 1
    app = RealtimeHandWaveApp(config)
    rows = []
    for value in np.linspace(0.0, 1.0, 8):
        phase_points = np.zeros(32, dtype=np.float64)
        phase_points[29] = value
        rows.append(phase_points)
    app.fmcw_processor.phase_history = deque(rows, maxlen=config.fmcw.phase_window_size)

    app._update_fmcw_fixed_trajectory(_fmcw_feature(8, 0.04))

    assert app.latest_fmcw_fixed_trajectory_pair == "23:29"
    assert app.latest_fmcw_fixed_trajectory_phase_rad > 0.4
    assert app.latest_fmcw_fixed_trajectory_distance_mm > 0.0


def test_fmcw_feature_row_logs_fixed_fmcw_trajectory():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app.latest_fmcw_fixed_trajectory_value = 0.42
    app.latest_fmcw_fixed_trajectory_phase_rad = 0.12
    app.latest_fmcw_fixed_trajectory_distance_mm = 0.42
    app.latest_fmcw_fixed_trajectory_pair = "14:15"

    row = app._fmcw_feature_row(_fmcw_feature(8, 0.04))

    assert row["fmcw_fixed_trajectory_value"] == "0.420000000"
    assert row["fmcw_fixed_trajectory_phase_rad"] == "0.120000000"
    assert row["fmcw_fixed_trajectory_distance_mm"] == "0.420000000"
    assert row["fmcw_fixed_trajectory_pair"] == "14:15"


def test_app_fmcw_candidate_detector_respects_startup_ignore():
    config = AppConfig(mode="fmcw")
    config.fmcw.candidate_min_history = 5
    config.fmcw.candidate_min_score = 0.03
    config.fmcw.candidate_startup_ignore_s = 1.0
    app = RealtimeHandWaveApp(config)

    for index in range(8):
        app._update_fmcw_candidate_detector(_fmcw_feature(index, 0.02))

    early = app._update_fmcw_candidate_detector(_fmcw_feature(8, 0.12))
    late = app._update_fmcw_candidate_detector(_fmcw_feature(24, 0.12))

    assert not early.is_event
    assert late.is_event


def test_app_fmcw_candidate_score_source_can_use_single_track_or_rms():
    config = AppConfig(mode="fmcw")
    app = RealtimeHandWaveApp(config)
    feature = FmcwFeature(
        time_s=0.0,
        sample_index=0,
        period_index=0,
        phase_point_count=30,
        track_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        track_delta_rms=0.30,
        phase_std=0.0,
        rms=0.0,
        peak_abs=0.0,
        pairs=((14, 15),),
        track_deltas=(0.08, 0.12, 0.30, 0.02, 0.01),
    )

    assert app._fmcw_candidate_score(feature) == 0.08
    config.fmcw.candidate_score_source = "track_delta_rms"
    assert app._fmcw_candidate_score(feature) == 0.30
    config.fmcw.candidate_score_source = "max_track_delta"
    assert app._fmcw_candidate_score(feature) == 0.30


def test_app_fmcw_candidate_detector_requires_release_before_next_event():
    config = AppConfig(mode="fmcw")
    config.fmcw.candidate_min_history = 5
    config.fmcw.candidate_threshold_k = 3.0
    config.fmcw.candidate_min_score = 0.03
    config.fmcw.candidate_refractory_s = 0.1
    config.fmcw.candidate_release_ratio = 0.4
    config.fmcw.candidate_startup_ignore_s = 0.0
    app = RealtimeHandWaveApp(config)

    for index in range(8):
        app._update_fmcw_candidate_detector(_fmcw_feature(index, 0.02))

    first = app._update_fmcw_candidate_detector(_fmcw_feature(8, 0.12))
    still_high = app._update_fmcw_candidate_detector(_fmcw_feature(12, 0.11))
    released = app._update_fmcw_candidate_detector(_fmcw_feature(16, 0.02))
    second = app._update_fmcw_candidate_detector(_fmcw_feature(20, 0.12))

    assert first.is_event
    assert not still_high.is_event
    assert not released.is_event
    assert second.is_event
    assert second.event_id == 2


def test_app_fmcw_candidate_event_does_not_replace_pending_confirm_window():
    config = AppConfig(mode="fmcw")
    app = RealtimeHandWaveApp(config)

    first = type("Detection", (), {"is_event": True, "event_id": 1})()
    second = type("Detection", (), {"is_event": True, "event_id": 2})()
    app.latest_fmcw_candidate_score = 0.12
    app.latest_fmcw_candidate_threshold = 0.05

    app._handle_fmcw_candidate_event(_fmcw_feature(20, 0.12), first)
    app._handle_fmcw_candidate_event(_fmcw_feature(21, 0.15), second)

    assert app.pending_fmcw_candidate_id == 1
    assert app.pending_fmcw_candidate_time_s == _fmcw_feature(20, 0.12).time_s
    assert app.latest_event_id == 2


def test_app_fmcw_vote_candidate_can_open_confirm_window_without_delta_event():
    config = AppConfig(mode="fmcw")
    config.fmcw.vote_candidate_enabled = True
    config.fmcw.vote_candidate_min_confidence = 0.6
    app = RealtimeHandWaveApp(config)
    feature = _fmcw_feature(40, 0.04)
    app.latest_fmcw_pattern = "11"
    app.latest_fmcw_vote_confidence = 0.8

    assert app._fmcw_vote_candidate_ready(feature.time_s)

    app._handle_fmcw_vote_candidate_event(feature)

    assert app.pending_fmcw_candidate_id == 1
    assert app.pending_fmcw_candidate_time_s == feature.time_s
    assert app.latest_event_id == 1
    assert app.last_detection_method == "fmcw_vote_candidate"
    assert app._detected_label() == "投票候选 / Vote candidate"


def test_app_fmcw_vote_candidate_is_disabled_by_default():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))
    app.latest_fmcw_pattern = "11"
    app.latest_fmcw_vote_confidence = 0.95

    assert not app._fmcw_vote_candidate_ready(10.0)


def test_fmcw_primary_blink_event_opens_confirm_window():
    config = AppConfig(mode="fmcw")
    config.fmcw.primary_blink_peak_enabled = False
    app = RealtimeHandWaveApp(config)
    detection = type(
        "Detection",
        (),
        {
            "is_event": True,
            "event_id": 4,
            "score": 0.12,
            "threshold": 0.04,
            "baseline": 0.01,
            "mad": 0.002,
            "method": "twinkle",
        },
    )()
    app.fmcw_primary_blink_detector = type("Detector", (), {"update": lambda self, feature: detection})()

    app._process_fmcw_primary_blink_samples(0, np.zeros(1024, dtype=np.float32))

    assert app.pending_fmcw_candidate_id == 4
    assert app.pending_fmcw_candidate_source == "primary_blink"
    assert app.latest_fmcw_confirm_state == "pending"


def test_fmcw_mode_labels_candidate_confirmed_and_suppressed_states():
    app = RealtimeHandWaveApp(AppConfig(mode="fmcw"))

    assert app._detected_label() == "眨眼候选 / Blink candidate"
    app.last_detection_method = "fmcw_confirmed_blink"
    assert app._detected_label() == "确认眨眼 / Blink confirmed"
    app.last_detection_method = "fmcw_suppressed_motion"
    assert app._detected_label() == "大动作已压制 / Motion suppressed"


def test_app_fmcw_confirm_state_confirms_blink_like_candidate():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_min_delta_rms = 0.05
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    config.fmcw.confirm_large_motion_duration_s = 0.25
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 7
    app.pending_fmcw_candidate_time_s = 1.0

    for index, score in enumerate((0.04, 0.09, 0.08, 0.03)):
        feature = _fmcw_feature(20 + index, score)
        feature = FmcwFeature(
            time_s=1.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app.latest_fmcw_pattern = "1"
        app.latest_fmcw_vote_confidence = 0.6
        app.latest_fmcw_vote_score = 27
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(1.25)

    assert event == (7, "fmcw_confirmed_blink", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "confirmed_blink"
    assert app.latest_fmcw_confirm_is_event
    assert app.latest_fmcw_final_blink_event_id == 1
    assert app.latest_fmcw_confirm_pattern == "1"
    assert app.latest_fmcw_final_pattern == "1"
    assert app.latest_fmcw_confirm_pattern_stability == 1.0


def test_app_fmcw_confirm_window_recomputes_paper_style_vote_from_phase_points():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.trajectory_detrend_window = 1
    config.fmcw.trajectory_smoothing_window = 1
    app = RealtimeHandWaveApp(config)

    for index, phase_points in enumerate(_phase_window()):
        app.latest_fmcw_pattern = "1"
        app.latest_fmcw_vote_confidence = 0.5
        app.latest_fmcw_vote_score = 20
        app._append_fmcw_confirm_observation(_fmcw_phase_feature(index, phase_points))

    app._refresh_fmcw_confirm_metrics(0.0, 0.7)
    row = app._fmcw_feature_row(_fmcw_phase_feature(65, np.zeros(30)))

    assert app.latest_fmcw_confirm_window_candidate_count == 45
    assert 0.0 <= app.latest_fmcw_confirm_window_confidence <= 1.0
    assert row["fmcw_confirm_window_candidate_count"] == 45
    assert row["fmcw_confirm_window_confidence"] != ""


def test_app_fmcw_confirm_state_suppresses_large_motion_candidate():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 9
    app.pending_fmcw_candidate_time_s = 2.0

    for index, score in enumerate((0.08, 0.24, 0.28, 0.12)):
        feature = _fmcw_feature(40 + index, score)
        feature = FmcwFeature(
            time_s=2.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(2.25)

    assert event == (9, "fmcw_suppressed_motion", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "suppressed_motion"
    assert app.latest_fmcw_confirm_is_event
    assert app.latest_fmcw_suppressed_event_id == 1


def test_app_fmcw_confirm_state_confirms_primary_blink_without_fmcw_delta_support():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_min_delta_rms = 0.05
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    config.fmcw.confirm_require_vote = True
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 16
    app.pending_fmcw_candidate_time_s = 4.0
    app.pending_fmcw_candidate_source = "primary_blink"

    for index, score in enumerate((0.01, 0.02, 0.015, 0.01)):
        feature = _fmcw_feature(80 + index, score)
        feature = FmcwFeature(
            time_s=4.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app.latest_fmcw_pattern = ""
        app.latest_fmcw_vote_confidence = 0.0
        app.latest_fmcw_vote_score = 0
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(4.25)

    assert event == (16, "fmcw_confirmed_blink", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "confirmed_blink"
    assert app.latest_fmcw_confirm_is_event


def test_app_fmcw_confirm_state_still_suppresses_primary_blink_during_large_motion():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 17
    app.pending_fmcw_candidate_time_s = 5.0
    app.pending_fmcw_candidate_source = "primary_blink"

    for index, score in enumerate((0.03, 0.24, 0.26, 0.04)):
        feature = _fmcw_feature(100 + index, score)
        feature = FmcwFeature(
            time_s=5.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(5.25)

    assert event == (17, "fmcw_suppressed_motion", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "suppressed_motion"


def test_app_fmcw_confirm_state_suppresses_vote_candidate_during_large_motion():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 12
    app.pending_fmcw_candidate_time_s = 2.0
    app.pending_fmcw_candidate_source = "vote"

    for index, score in enumerate((1.20, 1.40, 1.10, 0.90)):
        feature = _fmcw_feature(40 + index, score)
        feature = FmcwFeature(
            time_s=2.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app.latest_fmcw_pattern = "11"
        app.latest_fmcw_vote_confidence = 0.9
        app.latest_fmcw_vote_score = 40
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(2.25)

    assert event == (12, "fmcw_suppressed_motion", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "suppressed_motion"
    assert app.latest_fmcw_final_pattern == ""


def test_app_fmcw_confirm_state_suppresses_strong_window_vote_during_large_motion():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.7
    config.fmcw.trajectory_detrend_window = 1
    config.fmcw.trajectory_smoothing_window = 1
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    config.fmcw.confirm_vote_min_confidence = 0.20
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 14
    app.pending_fmcw_candidate_time_s = 0.0
    app.pending_fmcw_candidate_source = "primary_blink"

    for index, phase_points in enumerate(_phase_window()):
        feature = _fmcw_phase_feature(index, phase_points)
        feature = FmcwFeature(
            time_s=feature.time_s,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=1.40,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
            track_deltas=(1.40, 1.40, 1.40, 1.40, 1.40),
            phase_points=feature.phase_points,
        )
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(0.75)

    assert event == (14, "fmcw_suppressed_motion", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "suppressed_motion"
    assert app.latest_fmcw_confirm_window_candidate_count == 45
    assert app.latest_fmcw_final_pattern == ""


def test_app_fmcw_confirm_state_defaults_to_blink_when_vote_support_is_missing():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_min_delta_rms = 0.05
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 10
    app.pending_fmcw_candidate_time_s = 3.0

    for index, score in enumerate((0.06, 0.09, 0.08, 0.03)):
        feature = _fmcw_feature(60 + index, score)
        feature = FmcwFeature(
            time_s=3.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app.latest_fmcw_pattern = ""
        app.latest_fmcw_vote_confidence = 0.0
        app.latest_fmcw_vote_score = 0
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(3.25)

    assert event == (10, "fmcw_confirmed_blink", app.latest_fmcw_confirm_max_delta_rms)
    assert app.latest_fmcw_confirm_state == "confirmed_blink"
    assert app.latest_fmcw_confirm_is_event


def test_app_fmcw_confirm_state_can_require_vote_support():
    config = AppConfig(mode="fmcw")
    config.fmcw.confirm_window_s = 0.2
    config.fmcw.confirm_min_delta_rms = 0.05
    config.fmcw.confirm_large_motion_delta_rms = 0.20
    config.fmcw.confirm_require_vote = True
    app = RealtimeHandWaveApp(config)
    app.pending_fmcw_candidate_id = 10
    app.pending_fmcw_candidate_time_s = 3.0

    for index, score in enumerate((0.06, 0.09, 0.08, 0.03)):
        feature = _fmcw_feature(60 + index, score)
        feature = FmcwFeature(
            time_s=3.0 + index * 0.08,
            sample_index=feature.sample_index,
            period_index=feature.period_index,
            phase_point_count=feature.phase_point_count,
            track_values=feature.track_values,
            track_delta_rms=feature.track_delta_rms,
            phase_std=feature.phase_std,
            rms=feature.rms,
            peak_abs=feature.peak_abs,
            pairs=feature.pairs,
        )
        app.latest_fmcw_pattern = ""
        app.latest_fmcw_vote_confidence = 0.0
        app.latest_fmcw_vote_score = 0
        app._append_fmcw_confirm_observation(feature)

    event = app._update_fmcw_confirm_state(3.25)

    assert event is None
    assert app.latest_fmcw_confirm_state == "rejected_vote"
    assert not app.latest_fmcw_confirm_is_event
