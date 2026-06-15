import csv
import json

from hp_acoustic_wave.config import FmcwConfig
from hp_acoustic_wave.fmcw_session_analysis import CandidateDetectorParams, compute_confirm_feature, fmcw_config_from_session, marker_trajectory_details, replay_candidate_detector
from hp_acoustic_wave.fmcw_session_analysis import analyze_session, paper_style_vote_from_rows, write_analysis_outputs
from hp_acoustic_wave.session_io import SessionWriter


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_session_summarizes_marker_windows(tmp_path):
    session = tmp_path / "hp_fmcw_test"
    session.mkdir()
    feature_fields = {
        "time_s": "",
        "fmcw_track_delta_rms": "",
        "fmcw_pattern": "",
        "fmcw_vote_confidence": "",
        "fmcw_track_0": "",
        "fmcw_track_1": "",
        "fmcw_track_2": "",
        "fmcw_track_3": "",
        "fmcw_track_4": "",
    }
    features = []
    for time_s, delta, pattern, confidence in [
        (0.0, 0.01, "", 0.0),
        (0.5, 0.12, "1", 0.6),
        (1.0, 0.03, "", 0.0),
        (2.0, 0.40, "3", 0.8),
    ]:
        row = dict(feature_fields)
        row.update(
            {
                "time_s": f"{time_s:.3f}",
                "fmcw_track_delta_rms": f"{delta:.3f}",
                "fmcw_pattern": pattern,
                "fmcw_vote_confidence": f"{confidence:.3f}",
            }
        )
        features.append(row)
    markers = [
        {"time_s": "0.500", "label": "blink", "key": "b", "fmcw_track_delta_rms": "0.120", "fmcw_pattern": "1"},
        {"time_s": "2.000", "label": "large_motion", "key": "w", "fmcw_track_delta_rms": "0.400", "fmcw_pattern": "3"},
    ]
    _write_csv(session / "features.csv", features)
    _write_csv(session / "manual_markers.csv", markers)

    analysis = analyze_session(session, window_s=0.25, thresholds=(0.10, 0.30))

    assert analysis.feature_rows == 4
    assert analysis.marker_rows == 2
    assert analysis.marker_counts == {"blink": 1, "large_motion": 1}
    assert analysis.marker_windows[0].dominant_pattern == "1"
    assert analysis.marker_windows[1].max_delta_rms == 0.4
    assert len(analysis.confirm_features) == 2
    assert analysis.confirm_features[0].label == "blink"
    assert analysis.confirm_features[1].large_motion_suppressed
    assert analysis.candidate_summary.blink_total == 1
    assert analysis.threshold_sweep[0].blink_recall == 1.0
    assert analysis.threshold_sweep[1].blink_recall == 0.0
    assert analysis.threshold_sweep[1].large_motion_trigger_rate == 1.0


def test_candidate_detector_params_match_realtime_defaults():
    params = CandidateDetectorParams()

    assert params.refractory_s == 1.05
    assert params.release_ratio == 0.4
    assert params.score_source == "track0_delta"


def test_fmcw_config_from_session_uses_recorded_metadata(tmp_path):
    session = tmp_path / "hp_fmcw_test"
    session.mkdir()
    (session / "metadata.json").write_text(
        json.dumps({"fmcw": {"trajectory_detrend_window": 31, "trajectory_smoothing_window": 5}}),
        encoding="utf-8",
    )

    config = fmcw_config_from_session(session)

    assert config.trajectory_detrend_window == 31
    assert config.trajectory_smoothing_window == 5


def test_fmcw_config_from_session_supports_nested_app_metadata(tmp_path):
    session = tmp_path / "hp_fmcw_test"
    session.mkdir()
    (session / "metadata.json").write_text(
        json.dumps({"config": {"fmcw": {"valid_stop": 25, "trajectory_detrend_window": 31}}}),
        encoding="utf-8",
    )

    config = fmcw_config_from_session(session)

    assert config.valid_stop == 25
    assert config.trajectory_detrend_window == 31


def test_replay_candidate_detector_uses_single_track_delta_source():
    rows = [
        {"time_s": "0.0", "fmcw_track_0": "1.00", "fmcw_track_delta_rms": "0.90"},
        {"time_s": "0.1", "fmcw_track_0": "1.08", "fmcw_track_delta_rms": "0.90"},
    ]
    params = CandidateDetectorParams(min_history=20, score_source="track0_delta")

    points = replay_candidate_detector(rows, params)

    assert points[0].score == 0.0
    assert round(points[1].score, 6) == 0.08


def test_replay_candidate_detector_unwraps_single_track_delta_source():
    rows = [
        {"time_s": "0.0", "fmcw_track_0": "3.121592654", "fmcw_track_delta_rms": "0.90"},
        {"time_s": "0.1", "fmcw_track_0": "-3.121592654", "fmcw_track_delta_rms": "0.90"},
    ]
    params = CandidateDetectorParams(min_history=20, score_source="track0_delta")

    points = replay_candidate_detector(rows, params)

    assert points[0].score == 0.0
    assert round(points[1].score, 6) == 0.04


def test_write_analysis_outputs_creates_csv_and_json(tmp_path):
    session = tmp_path / "hp_fmcw_test"
    output = tmp_path / "report"
    session.mkdir()
    _write_csv(
        session / "features.csv",
        [{"time_s": "0.0", "fmcw_track_delta_rms": "0.1", "fmcw_pattern": "", "fmcw_vote_confidence": "0"}],
    )
    _write_csv(
        session / "manual_markers.csv",
        [{"time_s": "0.0", "label": "blink", "key": "b", "fmcw_track_delta_rms": "0.1", "fmcw_pattern": ""}],
    )

    analysis = analyze_session(session)
    write_analysis_outputs(analysis, output)

    assert (output / "marker_windows.csv").exists()
    assert (output / "confirm_features.csv").exists()
    assert (output / "marker_trajectory_details.csv").exists()
    with (output / "marker_trajectory_details.csv").open(newline="", encoding="utf-8") as handle:
        assert "trajectory_index" in handle.readline()
    assert (output / "drms_threshold_sweep.csv").exists()
    assert (output / "candidate_detector_summary.csv").exists()
    assert (output / "candidate_detector_sweep.csv").exists()
    assert (output / "summary.json").exists()


def test_session_writer_marker_includes_final_pattern_and_primary_blink_fields(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    writer = SessionWriter(session, sample_rate=48_000)
    writer.open()

    writer.write_manual_marker(
        1.0,
        label="blink",
        key="b",
        feature_snapshot={
            "fmcw_final_pattern": "1",
            "blink_score": 0.12,
            "blink_threshold": 0.04,
            "blink_event_id": 7,
            "blink_is_event": 1,
            "blink_method": "twinkle",
            "fmcw_fixed_trajectory_value": -0.25,
            "fmcw_fixed_trajectory_phase_rad": -0.01,
            "fmcw_fixed_trajectory_distance_mm": -0.25,
            "fmcw_fixed_trajectory_pair": "14:15",
        },
        event_id=7,
    )
    writer.close()

    rows = list(csv.DictReader((session / "manual_markers.csv").open(newline="", encoding="utf-8")))

    assert rows[0]["fmcw_final_pattern"] == "1"
    assert rows[0]["blink_score"] == "0.120000000"
    assert rows[0]["blink_threshold"] == "0.040000000"
    assert rows[0]["blink_event_id"] == "7"
    assert rows[0]["blink_is_event"] == "1"
    assert rows[0]["blink_method"] == "twinkle"
    assert rows[0]["fmcw_fixed_trajectory_value"] == "-0.250000000"
    assert rows[0]["fmcw_fixed_trajectory_phase_rad"] == "-0.010000000"
    assert rows[0]["fmcw_fixed_trajectory_distance_mm"] == "-0.250000000"
    assert rows[0]["fmcw_fixed_trajectory_pair"] == "14:15"


def test_session_writer_writes_visual_features_for_label_audit(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    writer = SessionWriter(session, sample_rate=48_000)
    writer.open()

    writer.write_visual_feature(
        {
            "time_s": "1.230000",
            "timestamp_ms": 1234,
            "available": 1,
            "face_found": 1,
            "left_ear": "0.210000000",
            "right_ear": "0.190000000",
            "left_closed": 1,
            "right_closed": 1,
            "is_blink_event": 1,
            "blink_count": 3,
            "inference_ms": "12.500000000",
            "error": "",
        }
    )
    writer.close()

    rows = list(csv.DictReader((session / "visual_features.csv").open(newline="", encoding="utf-8")))

    assert rows[0]["time_s"] == "1.230000"
    assert rows[0]["timestamp_ms"] == "1234"
    assert rows[0]["left_ear"] == "0.210000000"
    assert rows[0]["is_blink_event"] == "1"
    assert rows[0]["blink_count"] == "3"


def test_session_writer_feature_includes_full_fmcw_phase_points(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    writer = SessionWriter(session, sample_rate=48_000)
    writer.open()

    writer.write_feature(
        {
            "time_s": "0.000000",
            "fmcw_phase_point_count": 3,
            "fmcw_phase_points": "0.100000000;0.200000000;0.300000000",
        }
    )
    writer.close()

    rows = list(csv.DictReader((session / "features.csv").open(newline="", encoding="utf-8")))

    assert rows[0]["fmcw_phase_point_count"] == "3"
    assert rows[0]["fmcw_phase_points"] == "0.100000000;0.200000000;0.300000000"


def test_paper_style_vote_uses_saved_phase_points_for_45_trajectories():
    rows = []
    chirps = 64
    points = 30
    for chirp in range(chirps):
        phase = [index * 0.03 for index in range(points)]
        blink_shape = 0.35 if 26 <= chirp <= 34 else 0.0
        for index in range(15, points):
            phase[index] += blink_shape * (index - 14) / 15.0
        rows.append(
            {
                "time_s": f"{chirp * 512 / 48000.0:.6f}",
                "fmcw_phase_points": ";".join(f"{value:.9f}" for value in phase),
            }
        )

    summary = paper_style_vote_from_rows(rows)

    assert summary.candidate_count == 45
    assert len(summary.group_winners) == 3
    assert 0.0 <= summary.confidence <= 1.0


def test_confirm_feature_prefers_paper_vote_before_large_motion_suppression():
    rows = []
    chirps = 64
    points = 30
    for chirp in range(chirps):
        phase = [index * 0.03 for index in range(points)]
        blink_shape = 0.35 if 26 <= chirp <= 34 else 0.0
        for index in range(15, points):
            phase[index] += blink_shape * (index - 14) / 15.0
        rows.append(
            {
                "time_s": f"{chirp * 512 / 48000.0:.6f}",
                "fmcw_track_delta_rms": "1.400000000",
                "fmcw_pattern": "",
                "fmcw_vote_confidence": "0.000000000",
                "fmcw_track_0": "0.000000000",
                "fmcw_track_1": "0.000000000",
                "fmcw_track_2": "0.000000000",
                "fmcw_track_3": "0.000000000",
                "fmcw_track_4": "0.000000000",
                "fmcw_phase_points": ";".join(f"{value:.9f}" for value in phase),
            }
        )
    candidate = type("Candidate", (), {"time_s": 0.34, "is_event": True})()
    config = FmcwConfig(trajectory_detrend_window=1, trajectory_smoothing_window=1)

    feature = compute_confirm_feature(
        0,
        {"time_s": "0.341333", "label": "blink", "key": "b"},
        rows,
        [candidate],
        window_s=0.8,
        config=config,
    )

    assert feature.paper_vote_candidate_count == 45
    assert feature.paper_vote_pattern in config.confirm_single_blink_patterns
    assert feature.fmcw_confirmed
    assert not feature.large_motion_suppressed


def test_marker_trajectory_details_exports_each_paper_candidate_trajectory():
    rows = []
    chirps = 64
    points = 30
    for chirp in range(chirps):
        phase = [index * 0.03 for index in range(points)]
        blink_shape = 0.35 if 26 <= chirp <= 34 else 0.0
        for index in range(15, points):
            phase[index] += blink_shape * (index - 14) / 15.0
        rows.append(
            {
                "time_s": f"{chirp * 512 / 48000.0:.6f}",
                "fmcw_phase_points": ";".join(f"{value:.9f}" for value in phase),
            }
        )

    details = marker_trajectory_details(
        0,
        {"time_s": "0.341333", "label": "blink", "key": "b"},
        rows,
        window_s=0.8,
    )

    assert len(details) == 45
    assert {detail.vote_group for detail in details} == {0, 1, 2}
    assert {detail.criterion for detail in details} == {
        "internal_similarity",
        "slope_stability",
        "same_point_temporal_consistency",
    }
    assert all(detail.label == "blink" for detail in details)
    assert all(detail.row_count == 64 for detail in details)
