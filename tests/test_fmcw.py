import numpy as np

from hp_acoustic_wave.config import FmcwConfig
from hp_acoustic_wave.dsp import (
    FmcwStreamProcessor,
    _candidate_interval_slope_variance,
    _candidate_interval_stability_score,
    _same_point_temporal_difference_score,
    decimate_baseband_lowpass,
    detrend_trajectory,
    fmcw_chirp_period,
    highpass_filter,
    normalize_trajectory,
    periodic_signal_chunk,
    phase_difference_to_distance_meters,
    phase_difference_trajectory,
    remove_known_tone,
    select_fmcw_candidate_bundles,
    smooth_trajectory,
)


def test_fmcw_chirp_period_has_active_chirp_and_zero_guard():
    config = FmcwConfig()
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)

    assert period.shape == (512,)
    assert np.max(np.abs(period[:480])) > 0.0
    assert np.allclose(period[480:], 0.0)
    assert np.max(np.abs(period)) <= 0.021


def test_phase_difference_to_distance_uses_eq8_sample_gap():
    config = FmcwConfig(
        f0_hz=18_000.0,
        f1_hz=22_000.0,
        chirp_samples=480,
        decimation_factor=16,
        sound_speed_m_s=343.0,
    )

    distance = phase_difference_to_distance_meters(np.asarray([1.0]), 14, 15, config)

    expected = 480.0 * 343.0 / (4.0 * np.pi * 4_000.0 * 16.0)
    assert np.allclose(distance, [expected])


def test_fmcw_chirp_period_uses_paper_cosine_phase_when_unwindowed():
    config = FmcwConfig(chirp_samples=8, period_samples=8, guard_samples=0, tukey_alpha=0.0)
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)

    assert np.isclose(period[0], 0.02)


def test_periodic_signal_chunk_wraps_across_period_boundary():
    period = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    chunk = periodic_signal_chunk(period, start_sample=2, num_samples=6)

    assert np.allclose(chunk, [2.0, 3.0, 0.0, 1.0, 2.0, 3.0])


def test_decimate_baseband_lowpass_keeps_expected_point_count():
    samples = np.ones(512, dtype=np.complex128)

    points = decimate_baseband_lowpass(samples, factor=16, taps_per_phase=8)

    assert points.shape == (32,)
    assert np.allclose(points[3:-3], 1.0, atol=0.05)


def test_decimate_baseband_lowpass_suppresses_alternating_noise():
    index = np.arange(480, dtype=np.float64)
    samples = 1.0 + 0.4 * np.sin(0.73 * np.pi * index)

    filtered = decimate_baseband_lowpass(samples.astype(np.complex128), factor=16, taps_per_phase=8)
    block_average = decimate_baseband_lowpass(samples.astype(np.complex128), factor=16, taps_per_phase=0)

    assert np.std(filtered[3:-3].real) < np.std(block_average[3:-3].real)
    assert np.max(np.abs(filtered[3:-3].real - 1.0)) < 0.05


def test_highpass_filter_suppresses_audible_noise_before_fmcw_demodulation():
    sample_rate = 48_000
    index = np.arange(480, dtype=np.float64)
    audible = 0.4 * np.sin(2.0 * np.pi * 2_000.0 * index / sample_rate)
    ultrasonic = 0.08 * np.sin(2.0 * np.pi * 19_000.0 * index / sample_rate)
    mixed = audible + ultrasonic

    filtered = highpass_filter(mixed, sample_rate=sample_rate, cutoff_hz=17_000.0, tap_count=65)

    audible_basis = np.exp(-1j * 2.0 * np.pi * 2_000.0 * index / sample_rate)
    ultrasonic_basis = np.exp(-1j * 2.0 * np.pi * 19_000.0 * index / sample_rate)
    audible_before = abs(np.mean(mixed * audible_basis))
    audible_after = abs(np.mean(filtered * audible_basis))
    ultrasonic_before = abs(np.mean(mixed * ultrasonic_basis))
    ultrasonic_after = abs(np.mean(filtered * ultrasonic_basis))
    assert audible_after < audible_before * 0.15
    assert ultrasonic_after > ultrasonic_before * 0.45


def test_remove_known_tone_suppresses_engineering_primary_blink_tone():
    sample_rate = 48_000
    start_sample = 123
    index = np.arange(480, dtype=np.float64)
    tone = 0.30 * np.sin(2.0 * np.pi * 18_500.0 * (start_sample + index) / sample_rate + 0.4)
    chirp_like = 0.03 * np.sin(2.0 * np.pi * (18_000.0 * index / sample_rate + 0.2 * index**2 / sample_rate**2))
    mixed = tone + chirp_like

    cleaned = remove_known_tone(
        mixed,
        sample_rate=sample_rate,
        tone_hz=18_500.0,
        start_sample=start_sample,
    )

    basis = np.exp(-1j * 2.0 * np.pi * 18_500.0 * (start_sample + index) / sample_rate)
    before = abs(np.mean(mixed * basis))
    after = abs(np.mean(cleaned * basis))
    assert cleaned.shape == mixed.shape
    assert after < before * 0.10
    assert np.std(cleaned) > 0.0


def test_fmcw_stream_processor_extracts_track_values_from_periods():
    config = FmcwConfig(track_count=5)
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)
    samples = np.tile(period, 4)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)

    features = processor.process_block(samples, start_sample=0)

    assert len(features) == 4
    assert features[0].phase_point_count == 32
    assert len(features[0].track_values) == 5
    assert features[0].pairs[0] == (15, 16)
    assert len(features[0].phase_points) == 32
    assert features[-1].period_index == 3


def test_fmcw_stream_processor_cleans_primary_tone_before_phase_extraction():
    config = FmcwConfig(
        primary_blink_tone_cleanup_enabled=True,
        primary_blink_tone_hz=18_500.0,
        raw_highpass_enabled=False,
    )
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)
    start_sample = config.period_samples * 5
    contaminated = period + 0.30 * np.sin(
        2.0
        * np.pi
        * 18_500.0
        * np.arange(start_sample, start_sample + config.period_samples, dtype=np.float64)
        / 48_000.0
    ).astype(np.float32)
    clean_processor = FmcwStreamProcessor(config, sample_rate=48_000)
    dirty_config = FmcwConfig(
        primary_blink_tone_cleanup_enabled=False,
        primary_blink_tone_hz=18_500.0,
        raw_highpass_enabled=False,
    )
    dirty_processor = FmcwStreamProcessor(dirty_config, sample_rate=48_000)

    clean = clean_processor.process_frame(contaminated, start_sample)
    dirty = dirty_processor.process_frame(contaminated, start_sample)
    reference = clean_processor.extract_phase_points(period, frame_start_sample=start_sample)

    clean_error = np.linalg.norm(np.asarray(clean.phase_points) - reference)
    dirty_error = np.linalg.norm(np.asarray(dirty.phase_points) - reference)
    assert clean_error < dirty_error


def test_fmcw_stream_processor_highpass_removes_audible_noise_before_phase_extraction():
    clean_config = FmcwConfig(primary_blink_enabled=False, raw_highpass_enabled=True, raw_highpass_hz=17_000.0)
    dirty_config = FmcwConfig(primary_blink_enabled=False, raw_highpass_enabled=False)
    period = fmcw_chirp_period(clean_config, sample_rate=48_000, amplitude=0.02)
    start_sample = clean_config.period_samples * 3
    audible = 0.30 * np.sin(
        2.0
        * np.pi
        * 2_000.0
        * np.arange(start_sample, start_sample + clean_config.period_samples, dtype=np.float64)
        / 48_000.0
    ).astype(np.float32)
    contaminated = period + audible
    clean_processor = FmcwStreamProcessor(clean_config, sample_rate=48_000)
    dirty_processor = FmcwStreamProcessor(dirty_config, sample_rate=48_000)

    clean = clean_processor.process_frame(contaminated, start_sample)
    dirty = dirty_processor.process_frame(contaminated, start_sample)
    reference = clean_processor.extract_phase_points(period, frame_start_sample=start_sample)

    clean_error = np.linalg.norm(np.asarray(clean.phase_points) - reference)
    dirty_error = np.linalg.norm(np.asarray(dirty.phase_points) - reference)
    assert clean_error < dirty_error


def test_fmcw_stream_processor_normalizes_realtime_track_by_phase_point_gap():
    config = FmcwConfig(track_count=5, trajectory_detrend_window=1, trajectory_smoothing_window=1)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)
    phases = np.linspace(0.0, 2.9, 30, dtype=np.float64)
    processor.extract_phase_points = lambda frame, **kwargs: phases
    frame = np.zeros(config.period_samples, dtype=np.float32)

    feature = processor.process_frame(frame, 0)

    assert np.allclose(feature.track_values, [0.1] * 5)


def test_fmcw_stream_processor_can_keep_raw_phase_gap_scaling():
    config = FmcwConfig(track_count=5, track_gap_normalization=False)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)
    phases = np.linspace(0.0, 2.9, 30, dtype=np.float64)
    processor.extract_phase_points = lambda frame, **kwargs: phases
    frame = np.zeros(config.period_samples, dtype=np.float32)

    feature = processor.process_frame(frame, 0)

    assert np.allclose(feature.track_values, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_fmcw_stream_processor_uses_configured_track_pairs():
    config = FmcwConfig(track_pairs=((19, 29), (20, 29)), track_count=5)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)
    phases = np.linspace(0.0, 3.1, 32, dtype=np.float64)
    processor.extract_phase_points = lambda frame, **kwargs: phases
    frame = np.zeros(config.period_samples, dtype=np.float32)

    feature = processor.process_frame(frame, 0)

    assert feature.pairs == ((19, 29), (20, 29))
    assert len(feature.track_values) == 2
    assert np.allclose(feature.track_values, [0.1, 0.1])


def test_fmcw_stream_processor_buffers_partial_periods():
    config = FmcwConfig(track_count=5)
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)
    samples = np.tile(period, 3)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)

    first = processor.process_block(samples[:700], start_sample=0)
    second = processor.process_block(samples[700:1100], start_sample=700)
    third = processor.process_block(samples[1100:], start_sample=1100)

    features = first + second + third
    assert [feature.period_index for feature in features] == [0, 1, 2]


def test_fmcw_stream_processor_can_align_periods_with_sync_offset():
    config = FmcwConfig(track_count=5)
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)
    lag = 137
    samples = np.concatenate((np.zeros(lag, dtype=np.float32), np.tile(period, 3)))
    processor = FmcwStreamProcessor(config, sample_rate=48_000)
    processor.period_start_offset = lag

    features = processor.process_block(samples, start_sample=0)

    assert len(features) == 3
    assert features[0].sample_index == lag
    assert [feature.period_index for feature in features] == [0, 1, 2]
    assert np.allclose(features[0].phase_points, processor.extract_phase_points(period, frame_start_sample=lag))


def test_fmcw_stream_processor_keeps_rolling_phase_matrix():
    config = FmcwConfig(track_count=5, phase_window_size=3)
    period = fmcw_chirp_period(config, sample_rate=48_000, amplitude=0.02)
    samples = np.tile(period, 5)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)

    first = processor.process_block(samples[:1300], start_sample=0)
    second = processor.process_block(samples[1300:], start_sample=1300)

    features = first + second
    matrix = processor.rolling_phase_matrix()

    assert len(features) == 5
    assert matrix.shape == (3, 32)
    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix[-1], processor.extract_phase_points(period))


def test_fmcw_stream_processor_unwraps_track_delta_across_periods():
    config = FmcwConfig(track_count=1)
    processor = FmcwStreamProcessor(config, sample_rate=48_000)
    processor.pairs = ((0, 1),)
    processor.extract_phase_points = lambda frame, **kwargs: np.asarray(frame[:2], dtype=np.float64)
    frame_one = np.zeros(config.period_samples, dtype=np.float32)
    frame_two = np.zeros(config.period_samples, dtype=np.float32)
    frame_one[:2] = [0.0, np.pi - 0.02]
    frame_two[:2] = [0.0, -np.pi + 0.02]

    first = processor.process_frame(frame_one, 0)
    second = processor.process_frame(frame_two, config.period_samples)

    assert first.track_delta_rms == 0.0
    assert abs(second.track_deltas[0] - 0.04) < 1e-6
    assert abs(second.track_delta_rms - 0.04) < 1e-6


def test_select_fmcw_candidate_bundles_returns_45_trajectories():
    chirps = 64
    points = 30
    x = np.arange(chirps, dtype=np.float64)
    phase = np.tile(np.linspace(0.0, 3.0, points, dtype=np.float64), (chirps, 1))
    blink_shape = np.exp(-0.5 * np.square((x - 28.0) / 4.0))
    phase[:, 16:25] += blink_shape[:, None] * np.linspace(0.05, 0.45, 9)
    config = FmcwConfig(
        valid_start=15,
        valid_stop=25,
        candidate_interval_length=5,
        candidate_intervals_per_criterion=3,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)

    assert len(bundles) == 9
    assert {bundle.criterion for bundle in bundles} == {
        "internal_similarity",
        "slope_stability",
        "same_point_temporal_consistency",
    }
    assert sum(len(bundle.trajectories) for bundle in bundles) == 45
    assert all(len(bundle.target_indices) == 5 for bundle in bundles)
    assert all(len(trajectory) == chirps for bundle in bundles for trajectory in bundle.trajectories)


def test_default_fmcw_candidate_intervals_exclude_guard_phase_points():
    phase = np.tile(np.linspace(0.0, 3.1, 32, dtype=np.float64), (64, 1))
    config = FmcwConfig(
        candidate_interval_length=5,
        candidate_intervals_per_criterion=3,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)

    assert bundles
    assert all(max(bundle.target_indices) < config.valid_stop for bundle in bundles)
    assert config.valid_stop == config.chirp_samples // config.decimation_factor
    assert phase.shape[1] == config.period_samples // config.decimation_factor


def test_select_fmcw_candidate_bundles_keeps_three_groups_of_fifteen():
    chirps = 80
    points = 30
    phase = np.tile(np.linspace(0.0, 2.0, points, dtype=np.float64), (chirps, 1))
    phase[:, 17:22] += np.sin(np.linspace(0.0, np.pi, chirps))[:, None] * 0.2
    config = FmcwConfig(
        valid_start=15,
        valid_stop=25,
        candidate_interval_length=5,
        candidate_intervals_per_criterion=3,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)

    assert sum(bundle.criterion == "internal_similarity" for bundle in bundles) == 3
    assert sum(bundle.criterion == "slope_stability" for bundle in bundles) == 3
    assert sum(bundle.criterion == "same_point_temporal_consistency" for bundle in bundles) == 3
    assert sum(len(bundle.trajectories) for bundle in bundles) == 45


def test_select_fmcw_candidate_bundles_changes_reference_for_repeated_intervals():
    chirps = 64
    points = 30
    phase = np.tile(np.linspace(0.0, 1.0, points, dtype=np.float64), (chirps, 1))
    config = FmcwConfig(
        valid_start=15,
        valid_stop=30,
        candidate_interval_length=5,
        candidate_intervals_per_criterion=3,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)
    by_target: dict[tuple[int, ...], set[int]] = {}
    for bundle in bundles:
        by_target.setdefault(bundle.target_indices, set()).add(bundle.reference_index)

    assert len(bundles) == 9
    assert any(len(reference_indices) > 1 for reference_indices in by_target.values())
    assert len({(bundle.reference_index, bundle.target_indices) for bundle in bundles}) == len(bundles)


def test_select_fmcw_candidate_bundles_keeps_alternate_references_after_valid_boundary():
    chirps = 64
    points = 32
    phase = np.tile(np.linspace(0.0, 1.0, points, dtype=np.float64), (chirps, 1))
    config = FmcwConfig(
        valid_start=16,
        valid_stop=30,
        candidate_interval_length=5,
        candidate_intervals_per_criterion=3,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)
    by_target: dict[tuple[int, ...], set[int]] = {}
    for bundle in bundles:
        by_target.setdefault(bundle.target_indices, set()).add(bundle.reference_index)

    assert bundles
    assert all(bundle.reference_index >= config.valid_start - 1 for bundle in bundles)
    assert any(
        target_indices[0] > config.valid_start and len(reference_indices) > 1
        for target_indices, reference_indices in by_target.items()
    )


def test_select_fmcw_candidate_bundles_falls_back_when_boundary_reference_is_used():
    chirps = 32
    points = 24
    phase = np.tile(np.linspace(0.0, 1.0, points, dtype=np.float64), (chirps, 1))
    config = FmcwConfig(
        valid_start=16,
        valid_stop=23,
        candidate_interval_length=5,
        candidate_intervals_per_criterion=1,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)
    by_criterion = {bundle.criterion: bundle for bundle in bundles}

    assert by_criterion["internal_similarity"].target_indices[0] == 16
    assert by_criterion["slope_stability"].target_indices[0] == 17
    assert by_criterion["slope_stability"].reference_index == 16
    assert all(bundle.reference_index >= config.valid_start - 1 for bundle in bundles)


def test_select_fmcw_candidate_bundles_avoids_unstable_phase_point():
    chirps = 48
    points = 30
    phase = np.tile(np.linspace(0.0, 2.0, points, dtype=np.float64), (chirps, 1))
    phase[:, 18] = np.tile([0.0, 5.0, -5.0, 4.0], chirps // 4)
    config = FmcwConfig(
        valid_start=15,
        valid_stop=25,
        candidate_interval_length=2,
        candidate_intervals_per_criterion=1,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)

    assert bundles
    assert all(18 not in bundle.target_indices for bundle in bundles)


def test_combined_candidate_stability_score_includes_cross_chirp_temporal_variation():
    chirps = 48
    points = 30
    phase = np.tile(np.linspace(0.0, 2.0, points, dtype=np.float64), (chirps, 1))
    phase[:, 18] += np.tile([0.0, 0.8, -0.8, 0.6], chirps // 4)
    temporal_phase = np.unwrap(phase, axis=0)

    stable = _candidate_interval_stability_score(phase, temporal_phase, (15, 16))
    unstable = _candidate_interval_stability_score(phase, temporal_phase, (17, 18))

    assert unstable > stable


def test_select_fmcw_candidate_bundles_scores_slope_and_temporal_groups_independently():
    chirps = 48
    points = 30
    phase = np.tile(np.linspace(0.0, 2.0, points, dtype=np.float64), (chirps, 1))
    phase[:, 16] += np.tile([0.0, 0.4, -0.4, 0.3], chirps // 4)
    phase[:, 18] += np.tile([0.0, 0.3, -0.3, 0.2], chirps // 4)
    phase[:, 20] += np.linspace(0.0, 1.5, chirps)
    config = FmcwConfig(
        valid_start=15,
        valid_stop=21,
        candidate_interval_length=2,
        candidate_intervals_per_criterion=4,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)
    temporal_phase = np.unwrap(np.unwrap(phase, axis=1), axis=0)

    slope_bundles = [bundle for bundle in bundles if bundle.criterion == "slope_stability"]
    temporal_bundles = [
        bundle for bundle in bundles if bundle.criterion == "same_point_temporal_consistency"
    ]
    assert len(slope_bundles) == 4
    assert len(temporal_bundles) == 4
    assert any(
        _same_point_temporal_difference_score(temporal_phase, bundle.target_indices) > 0.0
        for bundle in slope_bundles
    )
    assert all(
        np.isclose(
            bundle.score,
            _candidate_interval_slope_variance(phase, bundle.target_indices),
        )
        for bundle in slope_bundles
    )
    assert all(
        np.isclose(
            bundle.score,
            _same_point_temporal_difference_score(temporal_phase, bundle.target_indices),
        )
        for bundle in temporal_bundles
    )


def test_same_point_temporal_consistency_penalizes_linear_drift():
    chirps = 48
    phase = np.zeros((chirps, 30), dtype=np.float64)
    phase[:, 18] = np.linspace(0.0, 2.0, chirps)

    stable = _same_point_temporal_difference_score(phase, (16,))
    drifting = _same_point_temporal_difference_score(phase, (18,))

    assert stable == 0.0
    assert drifting > 1.9


def test_select_fmcw_slope_stability_group_avoids_cross_chirp_unstable_interval():
    chirps = 48
    points = 30
    phase = np.tile(np.linspace(0.0, 2.0, points, dtype=np.float64), (chirps, 1))
    phase[:, 18] += np.tile([0.0, 0.8, -0.8, 0.6], chirps // 4)
    config = FmcwConfig(
        valid_start=15,
        valid_stop=22,
        candidate_interval_length=2,
        candidate_intervals_per_criterion=1,
        trajectory_detrend_window=1,
        trajectory_smoothing_window=1,
    )

    bundles = select_fmcw_candidate_bundles(phase, config)
    slope_bundle = next(bundle for bundle in bundles if bundle.criterion == "slope_stability")

    assert 18 not in slope_bundle.target_indices


def test_same_point_temporal_consistency_unwraps_across_chirps():
    phase = np.zeros((4, 30), dtype=np.float64)
    phase[:, 18] = [np.pi - 0.06, np.pi - 0.02, -np.pi + 0.02, -np.pi + 0.06]

    score = _same_point_temporal_difference_score(phase, (18,))

    assert abs(score - 0.12) < 1e-12


def test_fmcw_trajectory_preprocessing_removes_drift_and_normalizes_shape():
    x = np.arange(81, dtype=np.float64)
    drift = 0.01 * x
    blink = np.exp(-0.5 * np.square((x - 40.0) / 5.0))
    raw = drift + blink

    detrended = detrend_trajectory(raw, window=31)
    normalized = normalize_trajectory(detrended)
    smoothed = smooth_trajectory(normalized, window=3)

    assert abs(float(np.mean(normalized))) < 1e-9
    assert np.max(np.abs(normalized)) <= 1.0
    assert int(np.argmax(smoothed)) in range(36, 45)
    assert np.ptp(smoothed[5:20]) < np.ptp(raw[5:20])


def test_smooth_trajectory_uses_edge_padding_not_zero_padding():
    values = np.ones(7, dtype=np.float64)

    smoothed = smooth_trajectory(values, window=3)

    assert np.allclose(smoothed, values)


def test_phase_difference_trajectory_defaults_to_paper_style_without_detrend():
    chirps = 41
    phase = np.zeros((chirps, 30), dtype=np.float64)
    slow_open_close = np.sin(np.linspace(0.0, np.pi, chirps))
    phase[:, 18] = phase[:, 14] + slow_open_close

    paper_style = phase_difference_trajectory(
        phase,
        reference_index=14,
        target_index=18,
        smoothing_window=1,
        detrend_window=1,
        normalize=False,
    )
    detrended = phase_difference_trajectory(
        phase,
        reference_index=14,
        target_index=18,
        smoothing_window=1,
        detrend_window=31,
        normalize=False,
    )

    assert np.allclose(paper_style, slow_open_close)
    assert float(np.ptp(detrended)) < float(np.ptp(paper_style))


def test_phase_difference_trajectory_uses_reference_and_target_columns():
    chirps = 40
    phase = np.zeros((chirps, 30), dtype=np.float64)
    phase[:, 14] = np.linspace(0.0, 1.0, chirps)
    phase[:, 18] = phase[:, 14] + np.sin(np.linspace(0.0, np.pi, chirps))

    trajectory = phase_difference_trajectory(
        phase,
        reference_index=14,
        target_index=18,
        smoothing_window=1,
        detrend_window=1,
        normalize=False,
    )

    assert np.allclose(trajectory, phase[:, 18] - phase[:, 14])
    assert int(np.argmax(trajectory)) in range(18, 22)


def test_phase_difference_trajectory_unwraps_across_chirps():
    phase = np.zeros((2, 30), dtype=np.float64)
    phase[:, 14] = 0.0
    phase[:, 18] = [np.pi - 0.02, -np.pi + 0.02]

    trajectory = phase_difference_trajectory(
        phase,
        reference_index=14,
        target_index=18,
        smoothing_window=1,
        detrend_window=1,
        normalize=False,
    )

    assert abs(float(trajectory[1] - trajectory[0]) - 0.04) < 1e-6


def test_phase_difference_trajectory_unwraps_within_chirp_before_subtraction():
    phase = np.zeros((2, 30), dtype=np.float64)
    phase[:, 14] = np.pi - 0.03
    phase[:, 15] = -np.pi + 0.03

    trajectory = phase_difference_trajectory(
        phase,
        reference_index=14,
        target_index=15,
        smoothing_window=1,
        detrend_window=1,
        normalize=False,
    )

    assert np.allclose(trajectory, 0.06, atol=1e-6)
