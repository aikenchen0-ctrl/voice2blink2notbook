from hp_acoustic_wave.config import DetectorConfig
from hp_acoustic_wave.detector import AdaptiveWaveDetector


def test_adaptive_detector_ignores_events_before_startup_window():
    detector = AdaptiveWaveDetector(
        DetectorConfig(
            min_history=5,
            threshold_k=3.0,
            min_energy=0.03,
            refractory_s=0.1,
            startup_ignore_s=1.0,
        )
    )

    for index in range(8):
        detector.update(index * 0.1, 0.02)

    early = detector.update(0.8, 0.12)
    late = detector.update(1.2, 0.12)

    assert not early.is_event
    assert late.is_event


def test_adaptive_detector_keeps_warmup_outlier_out_of_baseline():
    detector = AdaptiveWaveDetector(
        DetectorConfig(
            min_history=9,
            threshold_k=3.0,
            min_energy=0.03,
            startup_ignore_s=0.0,
        )
    )

    for index in range(3):
        detector.update(index * 0.1, 0.02)
    detector.update(0.3, 0.90)
    for index in range(4, 10):
        detector.update(index * 0.1, 0.02)

    result = detector.update(1.0, 0.12)

    assert result.threshold < 0.12
    assert result.is_event
