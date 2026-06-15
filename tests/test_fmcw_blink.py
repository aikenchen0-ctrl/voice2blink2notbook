import numpy as np

from hp_acoustic_wave.dsp import FmcwCandidateBundle
from hp_acoustic_wave.fmcw_blink import (
    FmcwEdge,
    FmcwSegment,
    build_edges,
    coarse_segments,
    count_high_low_high_variance_motifs,
    decode_candidate_bundles,
    decode_phase_matrix_vote_with_evidence,
    decode_segment,
    decode_trajectory_pattern,
    detect_extrema,
    edge_representative_difference,
    fallback_count_from_variance,
    filter_edges,
    fine_select_segments,
    group_edges,
    hierarchical_vote,
    hierarchical_vote_by_groups,
    over_pruning_correction,
    segmentation_max_gap,
    trim_edge_starts,
    variance_profile,
)
from hp_acoustic_wave.config import FmcwConfig


def _blink_trajectory(blink_count: int, *, gap: int = 8) -> np.ndarray:
    values = [0.0] * gap
    for _ in range(blink_count):
        values.extend([0.0, 0.18, 0.55, 0.90, 0.55, 0.18, 0.0])
        values.extend([0.0] * 3)
    values.extend([0.0] * gap)
    return np.asarray(values, dtype=np.float64)


def test_fmcw_segmentation_decodes_single_blink():
    trajectory = _blink_trajectory(1)

    assert decode_trajectory_pattern(trajectory, max_gap=12) == "1"


def test_fmcw_segmentation_decodes_two_consecutive_blinks():
    trajectory = _blink_trajectory(2)

    assert decode_trajectory_pattern(trajectory, max_gap=12) == "2"


def test_fmcw_segmentation_ignores_slow_monotonic_drift():
    trajectory = np.linspace(0.0, 1.0, 80, dtype=np.float64)

    assert decode_trajectory_pattern(trajectory, max_gap=12) == ""


def test_fmcw_detect_extrema_uses_middle_of_flat_peak_and_valley():
    trajectory = np.asarray([0.0, 0.4, 1.0, 1.0, 1.0, 0.2, -0.5, -0.5, 0.1], dtype=np.float64)

    extrema = detect_extrema(trajectory)

    assert extrema == [0, 3, 6, 8]


def test_fmcw_segmentation_gap_uses_paper_half_second():
    default = FmcwConfig(period_samples=512, segmentation_sample_rate_hz=48_000)
    long_period = FmcwConfig(period_samples=1024, segmentation_sample_rate_hz=48_000)

    assert segmentation_max_gap(default) == 46
    assert segmentation_max_gap(long_period) == 23


def test_fmcw_group_edges_treats_exact_half_second_as_new_segment():
    max_gap = segmentation_max_gap(FmcwConfig(period_samples=512, segmentation_sample_rate_hz=48_000))
    edges = [
        FmcwEdge(0, 3, 1, 3, 0.5),
        FmcwEdge(47, 50, -1, 3, 0.5),
    ]

    segments = group_edges(edges, max_gap=max_gap)

    assert len(segments) == 2


def test_fmcw_build_edges_keeps_lines_with_more_than_two_samples():
    trajectory = np.asarray([0.0, 0.4, 0.8], dtype=np.float64)

    edges = build_edges(trajectory, [0, 2])

    assert len(edges) == 1
    assert edges[0].length == 2


def test_fmcw_build_edges_discards_lines_with_only_two_samples():
    trajectory = np.asarray([0.0, 0.8], dtype=np.float64)

    edges = build_edges(trajectory, [0, 1])

    assert edges == []


def test_fmcw_edge_representative_difference_uses_top_three_adjacent_diffs():
    trajectory = np.asarray([0.0, 0.1, 1.1, 1.3, 3.3, 3.4, 6.4], dtype=np.float64)

    representative = edge_representative_difference(trajectory, 0, len(trajectory) - 1)

    assert abs(representative - 2.0) < 1e-12


def test_fmcw_filter_edges_uses_longest_edge_per_direction():
    trajectory = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    edges = [
        FmcwEdge(0, 10, 1, 10, 1.0),
        FmcwEdge(10, 12, -1, 2, 1.0),
        FmcwEdge(12, 20, 1, 8, 1.0),
    ]

    filtered = filter_edges(
        trajectory,
        edges,
        same_direction_min_ratio=0.20,
        representative_diff_min_ratio=0.20,
        trim_ratio=0.0,
    )

    assert any(edge.direction == -1 and edge.end == 12 for edge in filtered)


def test_fmcw_variance_fallback_decodes_exceptional_direction_record():
    trajectory = np.asarray(
        [
            0.0,
            0.4,
            0.9,
            0.4,
            0.0,
            0.02,
            0.01,
            0.0,
            0.0,
            -0.4,
            -0.9,
            -0.4,
            0.0,
            0.03,
            0.01,
            0.0,
            0.0,
            0.5,
            1.0,
            0.5,
            0.0,
        ],
        dtype=np.float64,
    )
    segment = FmcwSegment(
        start=0,
        end=len(trajectory) - 1,
        edges=(
            FmcwEdge(0, 3, 1, 3, 0.3),
            FmcwEdge(3, 6, -1, 3, 0.3),
            FmcwEdge(6, 9, 1, 3, 0.3),
            FmcwEdge(9, 12, -1, 3, 0.3),
            FmcwEdge(12, 15, 1, 3, 0.3),
            FmcwEdge(15, 18, -1, 3, 0.3),
            FmcwEdge(18, 20, 1, 2, 0.3),
        ),
    )

    assert fallback_count_from_variance(trajectory, segment) == 2
    assert decode_segment(segment, trajectory) == 2


def test_fmcw_variance_profile_uses_paper_window_and_step():
    values = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    profile = variance_profile(values, window=4, step=2)

    assert np.allclose(profile, [0.25, 0.1875])


def test_fmcw_variance_fallback_counts_local_high_low_high_runs():
    variances = np.asarray([0.90, 0.80, 0.05, 0.04, 0.70, 0.60, 0.06, 0.05, 0.85], dtype=np.float64)

    assert count_high_low_high_variance_motifs(variances) == 2


def test_fmcw_variance_fallback_rejects_one_sided_or_flat_shapes():
    one_sided = np.asarray([0.95, 0.80, 0.05, 0.04, 0.06, 0.07], dtype=np.float64)
    flat = np.asarray([0.10, 0.10, 0.10, 0.10], dtype=np.float64)

    assert count_high_low_high_variance_motifs(one_sided) == 0
    assert count_high_low_high_variance_motifs(flat) == 0


def test_fmcw_trim_edge_start_uses_sample_after_threshold_crossing():
    trajectory = np.asarray([0.0, 0.01, 0.02, 0.50, 0.80, 1.00, 1.10], dtype=np.float64)
    edge = FmcwEdge(0, 6, 1, 6, 0.30)

    trimmed = trim_edge_starts(trajectory, [edge], trim_ratio=0.25)

    assert trimmed[0].start == 3
    assert trimmed[0].length == 3


def test_fmcw_trim_edge_start_requires_strictly_larger_than_threshold():
    trajectory = np.asarray([0.0, 0.25, 0.51, 1.51, 1.61, 1.70], dtype=np.float64)
    edge = FmcwEdge(0, 5, 1, 5, 0.50)

    trimmed = trim_edge_starts(trajectory, [edge], trim_ratio=0.25)

    assert trimmed[0].start == 2


def test_fmcw_direction_decode_does_not_guess_from_five_edges_without_fallback():
    segment = FmcwSegment(
        start=0,
        end=15,
        edges=(
            FmcwEdge(0, 3, 1, 3, 0.3),
            FmcwEdge(3, 6, -1, 3, 0.3),
            FmcwEdge(6, 9, 1, 3, 0.3),
            FmcwEdge(9, 12, -1, 3, 0.3),
            FmcwEdge(12, 15, 1, 3, 0.3),
        ),
    )

    assert decode_segment(segment) == 0


def test_fmcw_over_pruning_keeps_edges_inside_refined_segment():
    trajectory = _blink_trajectory(2)
    coarse = coarse_segments(trajectory, max_gap=12)
    refined = fine_select_segments(coarse, max_gap=12)
    corrected = over_pruning_correction(coarse, refined, max_gap=12)

    assert corrected
    assert sum(len(segment.edges) for segment in corrected) >= sum(len(segment.edges) for segment in refined)


def test_fmcw_fine_selection_regroups_edges_after_pruning():
    segments = [
        FmcwSegment(
            start=0,
            end=34,
            edges=(
                FmcwEdge(0, 5, 1, 5, 1.0),
                FmcwEdge(5, 10, -1, 5, 0.10),
                FmcwEdge(30, 34, 1, 4, 0.90),
            ),
        )
    ]

    refined = fine_select_segments(segments, max_gap=12)

    assert len(refined) == 2
    assert [segment.start for segment in refined] == [0, 30]


def test_fmcw_fine_selection_keeps_two_edge_segment_when_either_diff_is_large():
    two_edge_segment = FmcwSegment(
        start=0,
        end=8,
        edges=(
            FmcwEdge(0, 4, 1, 4, 0.40),
            FmcwEdge(4, 8, -1, 4, 0.20),
        ),
    )
    reference_segment = FmcwSegment(
        start=30,
        end=50,
        edges=(
            FmcwEdge(30, 50, 1, 20, 1.0),
        ),
    )

    refined = fine_select_segments([two_edge_segment, reference_segment], max_gap=12)

    selected_edges = [edge for segment in refined for edge in segment.edges]
    assert two_edge_segment.edges[0] in selected_edges
    assert two_edge_segment.edges[1] in selected_edges


def test_fmcw_over_pruning_only_restores_edges_inside_refined_range():
    coarse = [
        FmcwSegment(
            start=0,
            end=30,
            edges=(
                FmcwEdge(0, 5, 1, 5, 0.8),
                FmcwEdge(10, 15, -1, 5, 0.7),
                FmcwEdge(25, 30, 1, 5, 0.8),
            ),
        )
    ]
    refined = [FmcwSegment(start=0, end=15, edges=(coarse[0].edges[0],))]

    corrected = over_pruning_correction(coarse, refined, max_gap=12)

    restored = [edge for segment in corrected for edge in segment.edges]
    assert coarse[0].edges[1] in restored
    assert coarse[0].edges[2] not in restored


def test_fmcw_over_pruning_considers_edges_overlapping_refined_range():
    coarse = [
        FmcwSegment(
            start=0,
            end=30,
            edges=(
                FmcwEdge(0, 6, 1, 6, 0.8),
                FmcwEdge(5, 11, -1, 6, 0.7),
                FmcwEdge(24, 30, 1, 6, 0.8),
            ),
        )
    ]
    refined = [FmcwSegment(start=6, end=18, edges=(coarse[0].edges[0],))]

    corrected = over_pruning_correction(coarse, refined, max_gap=12)

    restored = [edge for segment in corrected for edge in segment.edges]
    assert coarse[0].edges[1] in restored
    assert coarse[0].edges[2] not in restored


def test_fmcw_hierarchical_vote_prefers_group_majorities():
    patterns = tuple(["1"] * 10 + [""] * 5 + ["2"] * 9 + ["1"] * 6 + ["1"] * 8 + ["3"] * 7)

    decision = hierarchical_vote(patterns)

    assert decision.pattern == "1"
    assert decision.group_winners == ("1", "2", "1")
    assert decision.score == 10
    assert 0.6 < decision.confidence < 0.7


def test_fmcw_hierarchical_vote_uses_strongest_group_winner():
    patterns = tuple(["1"] * 9 + ["2"] * 6 + ["2"] * 8 + ["1"] * 7 + ["3"] * 14 + [""] * 1)

    decision = hierarchical_vote(patterns)

    assert decision.pattern == "3"
    assert decision.group_winners == ("1", "2", "3")
    assert decision.score == 14
    assert decision.confidence == 14 / 15


def test_fmcw_hierarchical_vote_by_groups_tracks_criterion_labels():
    decision = hierarchical_vote_by_groups(
        (
            ("internal_similarity", ("1", "1", "2")),
            ("slope_stability", ("2", "2", "1")),
            ("same_point_temporal_consistency", ("1", "1", "3")),
        )
    )

    assert decision.pattern == "1"
    assert decision.group_winners == ("1", "2", "1")
    assert decision.group_labels == (
        "internal_similarity",
        "slope_stability",
        "same_point_temporal_consistency",
    )
    assert decision.score == 2
    assert decision.confidence == 2 / 3


def test_fmcw_decode_candidate_bundles_votes_across_trajectories():
    single = _blink_trajectory(1)
    double = _blink_trajectory(2)
    grouped_trajectories = (
        [single] * 5,
        [single] * 3 + [double] * 2,
        [single] * 4 + [double] * 1,
    )
    bundles = [
        FmcwCandidateBundle(
            criterion=criterion,
            score=0.0,
            reference_index=14,
            target_indices=tuple(range(15, 20)),
            trajectories=tuple(grouped_trajectories[group]),
        )
        for group, criterion in enumerate(
            [
                "internal_similarity",
                "slope_stability",
                "same_point_temporal_consistency",
            ]
        )
    ]

    patterns, decision = decode_candidate_bundles(bundles, max_gap=12)

    assert len(patterns) == 15
    assert decision.pattern == "1"
    assert decision.score == 5
    assert decision.group_winners == ("1", "1", "1")
    assert decision.confidence == 1.0


def test_fmcw_vote_with_evidence_returns_representative_phase_difference_trajectory(monkeypatch):
    strong_single = _blink_trajectory(1) * 2.0 + 0.25
    single = _blink_trajectory(1)
    double = _blink_trajectory(2)
    bundles = []
    for group, criterion in enumerate(
        [
            "internal_similarity",
            "slope_stability",
            "same_point_temporal_consistency",
        ]
    ):
        trajectories = tuple(([strong_single] if group == 0 else []) + [single] * 4 + [double])
        bundles.append(
            FmcwCandidateBundle(
                criterion=criterion,
                score=0.0,
                reference_index=14 + group,
                target_indices=tuple(range(15, 15 + len(trajectories))),
                trajectories=trajectories,
            )
        )
    monkeypatch.setattr(
        "hp_acoustic_wave.fmcw_blink.select_fmcw_candidate_bundles",
        lambda phase_matrix, config: tuple(bundles),
    )

    vote, evidence = decode_phase_matrix_vote_with_evidence(
        np.zeros((16, 30), dtype=np.float64),
        FmcwConfig(),
        blink_patterns=("1",),
    )

    assert vote.pattern == "1"
    assert evidence is not None
    assert evidence.pattern == "1"
    assert evidence.reference_index == 14
    assert evidence.target_index == 15
    assert evidence.value == 0.25
    assert evidence.span == np.ptp(strong_single)


def test_fmcw_decode_candidate_bundles_groups_by_criterion_not_bundle_order(monkeypatch):
    def fake_decode_trajectory_pattern(trajectory, *, max_gap):
        return str(int(trajectory[0]))

    monkeypatch.setattr(
        "hp_acoustic_wave.fmcw_blink.decode_trajectory_pattern",
        fake_decode_trajectory_pattern,
    )
    bundles = (
        FmcwCandidateBundle("internal_similarity", 0.0, 14, (15,), (np.asarray([1]), np.asarray([1]))),
        FmcwCandidateBundle("slope_stability", 0.0, 14, (15,), (np.asarray([2]), np.asarray([2]))),
        FmcwCandidateBundle("internal_similarity", 0.0, 15, (16,), (np.asarray([1]),)),
        FmcwCandidateBundle("same_point_temporal_consistency", 0.0, 14, (15,), (np.asarray([3]), np.asarray([3]))),
        FmcwCandidateBundle("slope_stability", 0.0, 15, (16,), (np.asarray([2]),)),
        FmcwCandidateBundle("same_point_temporal_consistency", 0.0, 15, (16,), (np.asarray([3]),)),
    )

    patterns, decision = decode_candidate_bundles(bundles, max_gap=12)

    assert patterns == ("1", "1", "2", "2", "1", "3", "3", "2", "3")
    assert decision.group_winners == ("1", "2", "3")
    assert decision.group_labels == (
        "internal_similarity",
        "slope_stability",
        "same_point_temporal_consistency",
    )
