import numpy as np

from hp_acoustic_wave.fmcw_pair_ranking import (
    aggregate_phase_pair_line_evidence,
    aggregate_phase_pair_peaks,
    collect_phase_pair_line_evidence,
    collect_phase_pair_peaks,
    phase_pair_template_points,
    rank_phase_pair_line_correlations,
    rank_phase_pairs,
)


def test_rank_phase_pairs_prefers_blink_target_inside_valid_region():
    times = np.arange(100, dtype=np.float64) * 0.1
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += _pulse(times, 5.0, 0.15) * 1.0
    phase[:, 6] += _pulse(times, 8.0, 0.15) * 2.0

    ranks = rank_phase_pairs(
        times,
        phase,
        [5.0],
        negative_markers=[8.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.3,
        background_window_s=0.3,
    )

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].blink_hit_rate == 1.0
    assert all(rank.target_index >= 5 for rank in ranks)


def test_aggregate_phase_pair_peaks_keeps_session_count():
    times = np.arange(100, dtype=np.float64) * 0.1
    first = np.zeros((times.size, 8), dtype=np.float64)
    second = np.zeros((times.size, 8), dtype=np.float64)
    first[:, 5] += _pulse(times, 5.0, 0.15)
    second[:, 5] += _pulse(times, 5.0, 0.15) * 0.8

    first_peaks = collect_phase_pair_peaks(
        times,
        first,
        [5.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.3,
        background_window_s=0.3,
    )
    second_peaks = collect_phase_pair_peaks(
        times,
        second,
        [5.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.3,
        background_window_s=0.3,
    )

    ranks = aggregate_phase_pair_peaks([first_peaks, second_peaks])

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].session_count == 2
    assert ranks[0].blink_peak_count == 2


def test_shape_metric_prefers_returning_open_close_trajectory():
    times = np.arange(120, dtype=np.float64) * 0.05
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += _pulse(times, 3.0, 0.18) * 1.0
    phase[:, 6] += _window_ramp(times, 3.0, 0.6) * 1.8
    phase[:, 6] += _pulse(times, 5.0, 0.18) * 2.0

    ranks = rank_phase_pairs(
        times,
        phase,
        [3.0],
        negative_markers=[5.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.45,
        background_window_s=0.45,
        metric="shape",
    )

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].blink_hit_rate == 1.0
    assert ranks[0].negative_trigger_rate == 0.0


def test_center_offset_recovers_delayed_manual_marker():
    times = np.arange(120, dtype=np.float64) * 0.05
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += _pulse(times, 3.0, 0.10)

    ranks = rank_phase_pairs(
        times,
        phase,
        [3.3],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.12,
        background_window_s=0.3,
        metric="shape",
        center_offset_s=-0.3,
    )

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].blink_hit_rate == 1.0


def test_phase_pair_template_points_exports_group_curves():
    times = np.arange(120, dtype=np.float64) * 0.05
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += _pulse(times, 3.0, 0.18)
    phase[:, 5] += _pulse(times, 5.0, 0.18) * 2.0

    rows = phase_pair_template_points(
        times,
        phase,
        4,
        5,
        [3.0],
        negative_markers=[5.0],
        marker_window_s=0.4,
        background_window_s=0.4,
        template_points=9,
    )

    assert {row.group for row in rows} >= {"blink", "background", "negative"}
    assert {row.point_index for row in rows if row.group == "blink"} == set(range(9))
    blink_midpoint = [row for row in rows if row.group == "blink" and row.point_index == 4][0]
    assert blink_midpoint.count == 1


def test_rank_phase_pair_line_correlations_prefers_stable_blink_line():
    times = np.arange(240, dtype=np.float64) * 0.05
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += _pulse(times, 3.0, 0.16)
    phase[:, 5] += _pulse(times, 7.0, 0.16)
    phase[:, 6] += _pulse(times, 5.0, 0.16) * 2.0

    ranks = rank_phase_pair_line_correlations(
        times,
        phase,
        [3.0, 7.0],
        negative_markers=[5.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.4,
        background_window_s=0.4,
        template_points=21,
    )

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].blink_hit_rate == 1.0
    assert ranks[0].negative_trigger_rate == 0.0


def test_aggregate_phase_pair_line_evidence_keeps_session_count():
    times = np.arange(240, dtype=np.float64) * 0.05
    first = np.zeros((times.size, 8), dtype=np.float64)
    second = np.zeros((times.size, 8), dtype=np.float64)
    first[:, 5] += _pulse(times, 3.0, 0.16)
    first[:, 5] += _pulse(times, 7.0, 0.16)
    second[:, 5] += _pulse(times, 3.0, 0.16) * 0.8
    second[:, 5] += _pulse(times, 7.0, 0.16) * 0.8

    first_evidence = collect_phase_pair_line_evidence(
        times,
        first,
        [3.0, 7.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.4,
        background_window_s=0.4,
        template_points=21,
    )
    second_evidence = collect_phase_pair_line_evidence(
        times,
        second,
        [3.0, 7.0],
        valid_start=5,
        valid_stop=7,
        marker_window_s=0.4,
        background_window_s=0.4,
        template_points=21,
    )

    ranks = aggregate_phase_pair_line_evidence([first_evidence, second_evidence])

    assert ranks
    assert ranks[0].target_index == 5
    assert ranks[0].session_count == 2
    assert ranks[0].blink_window_count == 4


def _pulse(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * np.square((times - float(center)) / float(width)))


def _window_ramp(times: np.ndarray, center: float, width: float) -> np.ndarray:
    start = float(center) - float(width) / 2.0
    end = float(center) + float(width) / 2.0
    ramp = (times - start) / max(float(width), 1e-9)
    return np.clip(ramp, 0.0, 1.0)
