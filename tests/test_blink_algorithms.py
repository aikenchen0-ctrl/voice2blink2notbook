import math
from typing import Optional

import numpy as np

from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig,
    BlinkListenerBlinkDetector,
    CompositeBlinkDetector,
    TwinkleTwinkleBlinkDetector,
)
from hp_acoustic_wave.dsp import ChunkFeature, extract_chunk_feature


def _feature(
    index: int,
    i_value: float = 0.0,
    q_value: float = 0.0,
    phase: float = 0.0,
    motion_energy: float = 0.0,
    amplitude: Optional[float] = None,
) -> ChunkFeature:
    if amplitude is None:
        amplitude = math.hypot(i_value, q_value)
    return ChunkFeature(
        time_s=index * 0.05,
        sample_index=index,
        i_value=i_value,
        q_value=q_value,
        amplitude=amplitude,
        amplitude_delta=0.0,
        phase=phase,
        phase_delta=0.0,
        motion_energy=motion_energy,
        rms=0.0,
        peak_abs=0.0,
    )


def _config(**overrides) -> BlinkDetectionConfig:
    values = {
        "history_size": 80,
        "min_history": 8,
        "threshold_k": 3.0,
        "min_score": 0.01,
        "refractory_s": 0.4,
        "baseline_freeze_s": 0.35,
        "short_window": 7,
        "phase_pair_lag": 3,
    }
    values.update(overrides)
    return BlinkDetectionConfig(**values)


def test_extract_chunk_feature_downmixes_stereo_instead_of_discarding_channel():
    sample_rate = 48_000
    tone_hz = 18_000
    samples = np.zeros((256, 2), dtype=np.float32)
    samples[:, 1] = np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate)

    feature = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None)

    assert feature.rms > 0.2
    assert feature.amplitude > 0.05


def test_motion_energy_prioritizes_amplitude_change_over_phase_jump():
    sample_rate = 48_000
    tone_hz = 18_000
    samples = np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate).astype(np.float32)
    previous = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None)
    phase_jump = extract_chunk_feature(
        np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate + math.pi / 2).astype(np.float32),
        sample_rate,
        tone_hz,
        0,
        previous,
    )
    amp_change = extract_chunk_feature(
        np.float32(1.45) * samples,
        sample_rate,
        tone_hz,
        0,
        previous,
    )

    assert amp_change.motion_energy > phase_jump.motion_energy


def test_blinklistener_detects_iq_bump_after_quiet_baseline():
    detector = BlinkListenerBlinkDetector(_config())
    events = []

    for index in range(12):
        events.append(detector.update(_feature(index, i_value=1.0, q_value=0.05)).is_event)
    for index, i_value in enumerate([1.04, 1.16, 1.32, 1.15, 1.03], start=12):
        events.append(detector.update(_feature(index, i_value=i_value, q_value=0.05)).is_event)

    assert any(events[12:])


def test_blinklistener_ignores_pure_phase_jump_when_amplitude_is_stable():
    detector = BlinkListenerBlinkDetector(_config(min_score=0.02))
    events = []

    for index in range(12):
        phase = 0.01 * index
        events.append(
            detector.update(
                _feature(index, i_value=math.cos(phase), q_value=math.sin(phase), phase=phase)
            ).is_event
        )
    for index, phase in enumerate([0.9, 1.4, 2.0, 2.5], start=12):
        events.append(
            detector.update(
                _feature(index, i_value=math.cos(phase), q_value=math.sin(phase), phase=phase)
            ).is_event
        )

    assert not any(events[12:])


def test_twinkle_proxy_detects_reversal_trajectory_not_monotonic_drift():
    drift_detector = TwinkleTwinkleBlinkDetector(_config())
    blink_detector = TwinkleTwinkleBlinkDetector(_config())

    drift_events = [
        drift_detector.update(_feature(index, phase=index * 0.03, amplitude=1.0)).is_event
        for index in range(24)
    ]
    blink_phases = [index * 0.01 for index in range(12)] + [0.20, 0.35, 0.53, 0.36, 0.18, 0.08]
    blink_events = [
        blink_detector.update(_feature(index, phase=phase, amplitude=1.0)).is_event
        for index, phase in enumerate(blink_phases)
    ]

    assert not any(drift_events[12:])
    assert any(blink_events[12:])


def test_twinkle_peak_gate_rejects_low_score_candidates():
    detector = TwinkleTwinkleBlinkDetector(_config(twinkle_min_peak_score=0.12))
    phases = [index * 0.005 for index in range(12)] + [0.04, 0.07, 0.10, 0.07, 0.04]
    results = [
        detector.update(_feature(index, phase=phase, motion_energy=0.02, amplitude=1.0))
        for index, phase in enumerate(phases)
    ]

    assert not any(result.is_event for result in results[12:])
    assert any(result.metrics.get("twinkle_reject_low_score") == 1.0 for result in results[12:])


def test_twinkle_peak_gate_rejects_large_motion_candidates():
    detector = TwinkleTwinkleBlinkDetector(_config(twinkle_max_motion_energy=0.08))
    phases = [index * 0.01 for index in range(12)] + [0.20, 0.35, 0.53, 0.36, 0.18, 0.08]
    results = [
        detector.update(_feature(index, phase=phase, motion_energy=0.18, amplitude=1.0))
        for index, phase in enumerate(phases)
    ]

    assert not any(result.is_event for result in results[12:])
    assert any(result.metrics.get("twinkle_reject_large_motion") == 1.0 for result in results[12:])


def test_twinkle_peak_gate_releases_near_baseline_not_near_zero():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.05,
            release_ratio=0.4,
            twinkle_min_peak_score=0.06,
            refractory_s=0.4,
        )
    )
    results = []
    phases = [index * 0.005 for index in range(12)]
    phases += [0.16, 0.28, 0.42, 0.30, 0.18, 0.06]
    phases += [0.055 + index * 0.001 for index in range(12)]
    phases += [0.22, 0.38, 0.55, 0.34, 0.16, 0.04]
    for index, phase in enumerate(phases):
        results.append(detector.update(_feature(index, phase=phase, motion_energy=0.02, amplitude=1.0)))

    event_indices = [index for index, result in enumerate(results) if result.is_event]

    assert len(event_indices) >= 2
    assert results[event_indices[0] + 3].metrics["twinkle_release_level"] > 0.0


def test_composite_detector_applies_shared_refractory_across_child_detectors():
    detector = CompositeBlinkDetector(_config(method="both", min_score=0.01))
    event_ids = []

    for index in range(12):
        result = detector.update(_feature(index, i_value=1.0, q_value=0.0, phase=index * 0.01))
        event_ids.append(result.event_id)
    for index, (i_value, phase) in enumerate(
        [(1.05, 0.12), (1.28, 0.30), (1.45, 0.55), (1.18, 0.32), (1.02, 0.10)],
        start=12,
    ):
        result = detector.update(_feature(index, i_value=i_value, q_value=0.0, phase=phase))
        event_ids.append(result.event_id)

    assert max(event_ids) == 1
