from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np

from hp_acoustic_wave.config import FmcwConfig
from hp_acoustic_wave.dsp import FmcwCandidateBundle, select_fmcw_candidate_bundles


@dataclass(frozen=True)
class FmcwEdge:
    start: int
    end: int
    direction: int
    length: int
    representative_diff: float


@dataclass(frozen=True)
class FmcwSegment:
    start: int
    end: int
    edges: tuple[FmcwEdge, ...]


@dataclass(frozen=True)
class FmcwVoteDecision:
    pattern: str
    score: int
    group_winners: tuple[str, ...]
    confidence: float
    group_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class FmcwWindowVote:
    pattern: str
    confidence: float
    score: int
    candidate_count: int
    group_winners: tuple[str, ...]
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FmcwTrajectoryEvidence:
    value: float
    pattern: str
    criterion: str
    reference_index: int
    target_index: int
    span: float
    trajectory: tuple[float, ...]


@dataclass(frozen=True)
class _DecodedTrajectory:
    pattern: str
    criterion: str
    reference_index: int
    target_index: int
    trajectory: np.ndarray


def segmentation_max_gap(config: FmcwConfig) -> int:
    # 论文 4.3.1 写的是“小于 0.5 秒”才归为同一段；因此离散 chirp 行数要用严格小于阈值的最大整数。
    period = max(1, int(config.period_samples))
    sample_rate = max(1, int(config.segmentation_sample_rate_hz))
    gap_s = max(0.0, float(config.segmentation_group_gap_s))
    gap_periods = gap_s * float(sample_rate) / float(period)
    return max(0, int(math.ceil(gap_periods) - 1))


def detect_extrema(trajectory: np.ndarray) -> list[int]:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.size < 3:
        return list(range(values.size))

    extrema = [0]
    runs: list[tuple[int, int, float]] = []
    start = 0
    for index in range(1, values.size):
        if values[index] != values[start]:
            runs.append((start, index - 1, float(values[start])))
            start = index
    runs.append((start, values.size - 1, float(values[start])))

    for run_index in range(1, len(runs) - 1):
        left = runs[run_index][2] - runs[run_index - 1][2]
        right = runs[run_index + 1][2] - runs[run_index][2]
        if (left > 0.0 and right < 0.0) or (left < 0.0 and right > 0.0):
            run_start, run_end, _ = runs[run_index]
            extrema.append((run_start + run_end) // 2)
    if extrema[-1] != values.size - 1:
        extrema.append(values.size - 1)
    return extrema


def edge_representative_difference(values: np.ndarray, start: int, end: int) -> float:
    segment = np.asarray(values[start : end + 1], dtype=np.float64)
    if segment.size < 2:
        return 0.0
    diffs = np.sort(np.abs(np.diff(segment)))[::-1]
    return float(np.mean(diffs[: min(3, diffs.size)]))


def build_edges(trajectory: np.ndarray, extrema_indices: Iterable[int]) -> list[FmcwEdge]:
    values = np.asarray(trajectory, dtype=np.float64)
    extrema = list(extrema_indices)
    edges: list[FmcwEdge] = []
    for start, end in zip(extrema, extrema[1:]):
        length = int(end - start)
        # 论文 4.3.1 保留“more than two sampling points”的线段。
        # end-start 是相邻间隔数，采样点数是 length+1，因此只丢掉 1/2 个采样点的线段。
        if length <= 1:
            continue
        direction = 1 if values[end] >= values[start] else -1
        edges.append(
            FmcwEdge(
                start=int(start),
                end=int(end),
                direction=direction,
                length=length,
                representative_diff=edge_representative_difference(values, start, end),
            )
        )
    return edges


def trim_edge_starts(
    trajectory: np.ndarray,
    edges: Iterable[FmcwEdge],
    *,
    trim_ratio: float = 0.25,
) -> list[FmcwEdge]:
    values = np.asarray(trajectory, dtype=np.float64)
    trimmed: list[FmcwEdge] = []
    for edge in edges:
        diffs = np.abs(np.diff(values[edge.start : edge.end + 1]))
        if diffs.size == 0:
            trimmed.append(edge)
            continue
        threshold = float(np.max(diffs)) * trim_ratio
        offset = 0
        for index, value in enumerate(diffs):
            if value > threshold:
                # 论文 4.3.1：起点移到“与前一个点差值”超过阈值的第一个样本。
                # diff[index] 对应 sample[index] -> sample[index + 1]，所以新起点是 index + 1。
                offset = index + 1
                break
        start = edge.start + offset
        if edge.end - start <= 1:
            start = edge.start
        trimmed.append(
            replace(
                edge,
                start=int(start),
                length=int(edge.end - start),
                representative_diff=edge_representative_difference(values, start, edge.end),
            )
        )
    return trimmed


def filter_edges(
    trajectory: np.ndarray,
    edges: Iterable[FmcwEdge],
    *,
    same_direction_min_ratio: float = 0.20,
    representative_diff_min_ratio: float = 0.20,
    trim_ratio: float = 0.25,
) -> list[FmcwEdge]:
    values = np.asarray(trajectory, dtype=np.float64)
    edges = list(edges)
    if not edges:
        return []
    longest_by_direction = {
        direction: max((edge.length for edge in edges if edge.direction == direction), default=0)
        for direction in (-1, 1)
    }
    max_rep = max(edge.representative_diff for edge in edges)
    filtered = [
        edge
        for edge in edges
        if edge.length >= same_direction_min_ratio * longest_by_direction[edge.direction]
        and edge.representative_diff >= representative_diff_min_ratio * max_rep
    ]
    return trim_edge_starts(values, filtered, trim_ratio=trim_ratio)


def group_edges(edges: Iterable[FmcwEdge], *, max_gap: int) -> list[FmcwSegment]:
    edges = sorted(edges, key=lambda edge: edge.start)
    if not edges:
        return []
    groups: list[list[FmcwEdge]] = [[edges[0]]]
    for edge in edges[1:]:
        if edge.start - groups[-1][-1].start <= max_gap:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    return [
        FmcwSegment(start=group[0].start, end=group[-1].end, edges=tuple(group))
        for group in groups
    ]


def coarse_segments(trajectory: np.ndarray, *, max_gap: int = 47) -> list[FmcwSegment]:
    # 粗分割：先按局部极值切边，再用相对长度/相对变化量过滤明显不像眨眼的边。
    extrema = detect_extrema(trajectory)
    edges = build_edges(trajectory, extrema)
    filtered = filter_edges(trajectory, edges)
    return group_edges(filtered, max_gap=max_gap)


def fine_select_segments(segments: Iterable[FmcwSegment], *, max_gap: int = 47) -> list[FmcwSegment]:
    # 细分割：不用固定绝对阈值，而在当前轨迹内部比较 edge length 和 representative diff。
    # 这是论文“adaptive constraints”的核心，适配不同人的眨眼力度和干扰强度。
    segments = list(segments)
    all_edges = [edge for segment in segments for edge in segment.edges]
    if not all_edges:
        return []
    average_diff = float(np.mean([edge.representative_diff for edge in all_edges]))
    max_length = max(edge.length for edge in all_edges)
    max_diff = max(edge.representative_diff for edge in all_edges)

    selected_edges: list[FmcwEdge] = []
    for segment in segments:
        edges = list(segment.edges)
        segment_average = float(np.mean([edge.representative_diff for edge in edges]))
        if segment_average < average_diff * 0.50:
            continue

        kept: list[FmcwEdge] = []
        if len(edges) == 2:
            first, second = edges
            if (
                (first.length >= max_length / 2 and second.length >= max_length / 2)
                or first.representative_diff >= max_diff * 0.40
                or second.representative_diff >= max_diff * 0.40
            ):
                kept = edges
        else:
            local_max_length = max(edge.length for edge in edges)
            local_max_diff = max(edge.representative_diff for edge in edges)
            for edge in edges:
                if (
                    edge.length >= local_max_length * 0.50
                    and edge.representative_diff >= local_max_diff * 0.35
                ) or edge.representative_diff >= local_max_diff * 0.50:
                    kept.append(edge)

        selected_edges.extend(kept)
    # 论文在得到 S2 后，会用和粗分割相同的时间间隔规则重新分组。
    # 如果中间干扰边被删掉，两侧剩余边距离过远，就不应再被硬拼成一个 blink segment。
    return group_edges(selected_edges, max_gap=max_gap)


def over_pruning_correction(
    coarse: Iterable[FmcwSegment],
    refined: Iterable[FmcwSegment],
    *,
    max_gap: int = 47,
) -> list[FmcwSegment]:
    # 过剪枝修正：连续眨眼中间的边可能较弱，被细分割误删；用更宽松的相对约束补回。
    coarse = list(coarse)
    refined = list(refined)
    corrected: list[FmcwSegment] = []
    for refined_segment in refined:
        candidates = [
            edge
            for coarse_segment in coarse
            for edge in coarse_segment.edges
            if edge.end >= refined_segment.start and edge.start <= refined_segment.end
        ]
        refined_ids = {(edge.start, edge.end, edge.direction) for edge in refined_segment.edges}
        missing = [edge for edge in candidates if (edge.start, edge.end, edge.direction) not in refined_ids]
        lengths = sorted((edge.length for edge in candidates), reverse=True)
        diffs = sorted((edge.representative_diff for edge in candidates), reverse=True)
        second_length = lengths[1] if len(lengths) > 1 else (lengths[0] if lengths else 0)
        second_diff = diffs[1] if len(diffs) > 1 else (diffs[0] if diffs else 0.0)

        kept = list(refined_segment.edges)
        for edge in missing:
            if edge.length >= 0.40 * second_length or edge.representative_diff >= 0.50 * second_diff:
                kept.append(edge)
        corrected.extend(group_edges(kept, max_gap=max_gap))
    return corrected


def compress_directions(edges: Sequence[FmcwEdge]) -> list[int]:
    directions: list[int] = []
    for edge in edges:
        if not directions or directions[-1] != edge.direction:
            directions.append(edge.direction)
    return directions


def _decode_from_directions(directions: Sequence[int]) -> int:
    # 论文 4.3.3 明确列出的主路径是：
    # 1 次眨眼：单边、单方向，或一次开闭形成的 2/3 个方向；
    # 2 次眨眼：交替方向 4 个；
    # 3 次眨眼：交替方向 6 个。
    # 其他缺边/多边形态不要直接按长度猜，交给 variance fallback 处理。
    if len(directions) <= 1:
        return 1 if directions else 0
    if len(directions) in (2, 3):
        return 1
    if len(directions) == 4:
        return 2
    if len(directions) == 6:
        return 3
    return 0


def variance_profile(values: np.ndarray, *, window: int = 4, step: int = 2) -> np.ndarray:
    # 论文 4.3.3：异常方向序列用 window=4、step=2 计算采样点方差曲线。
    samples = np.asarray(values, dtype=np.float64)
    window = int(window)
    step = int(step)
    if window <= 0 or step <= 0 or samples.size < window:
        return np.asarray([], dtype=np.float64)

    variances = [
        float(np.var(samples[start : start + window]))
        for start in range(0, samples.size - window + 1, step)
    ]
    return np.asarray(variances, dtype=np.float64)


def count_high_low_high_variance_motifs(
    variances: np.ndarray,
    *,
    low_percentile: float = 35.0,
    high_percentile: float = 60.0,
    high_min_span_ratio: float = 0.30,
) -> int:
    # 论文只给出“高-低-高”的形态，没有给绝对阈值；这里用当前 segment 内的相对分位数。
    # 低方差 run 代表 tip 附近点更密，高方差 shoulder 代表两侧 edge 点更稀。
    series = np.asarray(variances, dtype=np.float64)
    if series.size < 3:
        return 0
    span = float(np.ptp(series))
    if span <= 0.0:
        return 0

    low_threshold = float(np.percentile(series, low_percentile))
    high_threshold = float(np.percentile(series, high_percentile))
    high_threshold = max(high_threshold, low_threshold + span * float(high_min_span_ratio))

    low_mask = series <= low_threshold
    motifs = 0
    index = 1
    while index < series.size - 1:
        if not low_mask[index]:
            index += 1
            continue

        run_start = index
        while index < series.size - 1 and low_mask[index]:
            index += 1
        run_end = index - 1

        previous_low = run_start - 1
        while previous_low >= 0 and not low_mask[previous_low]:
            previous_low -= 1
        next_low = run_end + 1
        while next_low < series.size and not low_mask[next_low]:
            next_low += 1

        left_shoulder = series[previous_low + 1 : run_start]
        right_shoulder = series[run_end + 1 : next_low]
        left_high = left_shoulder.size > 0 and float(np.max(left_shoulder)) >= high_threshold
        right_high = right_shoulder.size > 0 and float(np.max(right_shoulder)) >= high_threshold
        if left_high and right_high:
            motifs += 1

    return min(3, int(motifs))


def fallback_count_from_variance(
    trajectory: np.ndarray,
    segment: FmcwSegment,
    *,
    window: int = 4,
    step: int = 2,
    min_motion: float = 1e-4,
) -> int:
    values = np.asarray(trajectory[segment.start : segment.end + 1], dtype=np.float64)
    if values.size < max(3, int(window)) or float(np.ptp(values)) < float(min_motion):
        return 0

    return count_high_low_high_variance_motifs(variance_profile(values, window=window, step=step))


def decode_segment(segment: FmcwSegment, trajectory: np.ndarray | None = None) -> int:
    directions = compress_directions(segment.edges)
    blink_count = _decode_from_directions(directions)
    if blink_count:
        return blink_count
    if trajectory is None:
        return 0
    return fallback_count_from_variance(trajectory, segment)


def decode_trajectory_pattern(trajectory: np.ndarray, *, max_gap: int = 47) -> str:
    # 单条轨迹先分割成 blink segment，再按边方向变化估计 1/2/3 次眨眼节奏。
    coarse = coarse_segments(trajectory, max_gap=max_gap)
    refined = fine_select_segments(coarse, max_gap=max_gap)
    corrected = over_pruning_correction(coarse, refined, max_gap=max_gap)
    active = corrected or refined or coarse
    active = [
        segment
        for segment in active
        if segment.edges and min(edge.length for edge in segment.edges) <= max_gap
    ]
    return "".join(str(count) for count in (decode_segment(segment, trajectory) for segment in active) if count)


def hierarchical_vote(patterns: Sequence[str]) -> FmcwVoteDecision:
    # 论文不是把 45 条轨迹混在一起投票，而是每 15 条一组先投，再在三组赢家间投。
    if not patterns:
        return FmcwVoteDecision(pattern="", score=0, group_winners=tuple(), confidence=0.0)
    if len(patterns) % 3 != 0:
        raise ValueError("patterns length must be divisible by 3")

    group_size = len(patterns) // 3
    group_results: list[tuple[str, int]] = []
    for group_index in range(3):
        group = list(patterns[group_index * group_size : (group_index + 1) * group_size])
        counts = Counter(group)
        winner, score = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0]
        group_results.append((winner, int(score)))

    group_winners = tuple(pattern for pattern, _ in group_results)
    group_winner_counts = Counter(group_winners)
    overall_counts = Counter(patterns)
    winner, score = sorted(
        group_results,
        key=lambda item: (
            -item[1],
            -group_winner_counts[item[0]],
            -overall_counts[item[0]],
            len(item[0]),
            item[0],
        ),
    )[0]
    return FmcwVoteDecision(
        pattern=winner,
        score=score,
        group_winners=group_winners,
        confidence=float(score) / float(group_size),
    )


def hierarchical_vote_by_groups(grouped_patterns: Sequence[tuple[str, Sequence[str]]]) -> FmcwVoteDecision:
    # 论文第 5 节的“三组”来自 4.2 的三个 phase-interval 选择准则，而不是任意顺序切片。
    groups = [(label, list(patterns)) for label, patterns in grouped_patterns if patterns]
    if not groups:
        return FmcwVoteDecision(pattern="", score=0, group_winners=tuple(), confidence=0.0)

    group_results: list[tuple[str, str, int, int]] = []
    overall_patterns: list[str] = []
    for label, patterns in groups:
        counts = Counter(patterns)
        winner, score = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0]
        group_results.append((label, winner, int(score), len(patterns)))
        overall_patterns.extend(patterns)

    group_winners = tuple(pattern for _, pattern, _, _ in group_results)
    group_winner_counts = Counter(group_winners)
    overall_counts = Counter(overall_patterns)
    label, winner, score, group_size = sorted(
        group_results,
        key=lambda item: (
            -item[2],
            -group_winner_counts[item[1]],
            -overall_counts[item[1]],
            len(item[1]),
            item[1],
            item[0],
        ),
    )[0]
    return FmcwVoteDecision(
        pattern=winner,
        score=score,
        group_winners=group_winners,
        confidence=float(score) / float(group_size),
        group_labels=tuple(label for label, _, _, _ in group_results),
    )


def decode_candidate_bundles(
    bundles: Sequence[FmcwCandidateBundle],
    *,
    max_gap: int = 47,
) -> tuple[tuple[str, ...], FmcwVoteDecision]:
    decoded, decision = _decode_candidate_bundles_with_details(bundles, max_gap=max_gap)
    return tuple(item.pattern for item in decoded), decision


def _decode_candidate_bundles_with_details(
    bundles: Sequence[FmcwCandidateBundle],
    *,
    max_gap: int = 47,
) -> tuple[tuple[_DecodedTrajectory, ...], FmcwVoteDecision]:
    # 对 45 条候选轨迹逐条解码，再按 4.2 的三个 criterion 显式分组做第 5 节层级投票。
    decoded: list[_DecodedTrajectory] = []
    grouped: dict[str, list[str]] = {}
    group_order: list[str] = []
    for bundle in bundles:
        if bundle.criterion not in grouped:
            grouped[bundle.criterion] = []
            group_order.append(bundle.criterion)
        targets = tuple(int(index) for index in bundle.target_indices)
        fallback_target = targets[-1] if targets else int(bundle.reference_index)
        for trajectory_index, trajectory in enumerate(bundle.trajectories):
            target_index = targets[trajectory_index] if trajectory_index < len(targets) else fallback_target
            pattern = decode_trajectory_pattern(trajectory, max_gap=max_gap)
            decoded.append(
                _DecodedTrajectory(
                    pattern=pattern,
                    criterion=bundle.criterion,
                    reference_index=int(bundle.reference_index),
                    target_index=int(target_index),
                    trajectory=np.asarray(trajectory, dtype=np.float64),
                )
            )
            grouped[bundle.criterion].append(pattern)
    return tuple(decoded), hierarchical_vote_by_groups(
        tuple((criterion, tuple(grouped[criterion])) for criterion in group_order)
    )


def _select_representative_trajectory(
    decoded: Sequence[_DecodedTrajectory],
    *,
    vote_pattern: str,
    blink_patterns: Sequence[str],
) -> FmcwTrajectoryEvidence | None:
    # 只展示“当前投票赢家也认为是 blink”的代表轨迹，避免把零散噪声轨迹误当主证据。
    if not vote_pattern or vote_pattern not in tuple(blink_patterns):
        return None
    candidates = [item for item in decoded if item.pattern == vote_pattern]
    if not candidates:
        return None

    def rank(item: _DecodedTrajectory) -> tuple[float, int, int, str]:
        values = np.asarray(item.trajectory, dtype=np.float64)
        span = float(np.ptp(values)) if values.size else 0.0
        return (-span, item.reference_index, item.target_index, item.criterion)

    selected = sorted(candidates, key=rank)[0]
    trajectory = np.asarray(selected.trajectory, dtype=np.float64)
    span = float(np.ptp(trajectory)) if trajectory.size else 0.0
    value = float(trajectory[-1]) if trajectory.size else 0.0
    return FmcwTrajectoryEvidence(
        value=max(-1.0, min(1.0, value)),
        pattern=selected.pattern,
        criterion=selected.criterion,
        reference_index=selected.reference_index,
        target_index=selected.target_index,
        span=span,
        trajectory=tuple(float(value) for value in trajectory),
    )


def decode_phase_matrix_vote(
    phase_matrix: np.ndarray,
    config: FmcwConfig,
    *,
    minimum_rows: int | None = None,
) -> FmcwWindowVote:
    # 论文式窗口投票：一段 phase matrix -> 45 条 phase-pair trajectories -> 分段解码 -> 三组层级投票。
    vote, _ = decode_phase_matrix_vote_with_evidence(
        phase_matrix,
        config,
        minimum_rows=minimum_rows,
    )
    return vote


def decode_phase_matrix_vote_with_evidence(
    phase_matrix: np.ndarray,
    config: FmcwConfig,
    *,
    minimum_rows: int | None = None,
    blink_patterns: Sequence[str] = ("1", "11", "12", "21"),
) -> tuple[FmcwWindowVote, FmcwTrajectoryEvidence | None]:
    # 论文式窗口投票，同时返回支持当前 blink 投票赢家的代表相位差轨迹，供 UI/CSV 做“数据线”展示。
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return FmcwWindowVote("", 0.0, 0, 0, tuple()), None

    min_rows = (
        max(int(config.candidate_interval_length), int(config.trajectory_detrend_window), 16)
        if minimum_rows is None
        else int(minimum_rows)
    )
    if matrix.shape[0] < min_rows:
        return FmcwWindowVote("", 0.0, 0, 0, tuple()), None

    bundles = select_fmcw_candidate_bundles(matrix, config)
    if not bundles:
        return FmcwWindowVote("", 0.0, 0, 0, tuple()), None

    decoded, decision = _decode_candidate_bundles_with_details(
        bundles,
        max_gap=segmentation_max_gap(config),
    )
    patterns = tuple(item.pattern for item in decoded)
    vote = FmcwWindowVote(
        pattern=decision.pattern,
        confidence=float(decision.confidence),
        score=int(decision.score),
        candidate_count=len(patterns),
        group_winners=tuple(decision.group_winners),
        patterns=tuple(patterns),
    )
    evidence = _select_representative_trajectory(
        decoded,
        vote_pattern=decision.pattern,
        blink_patterns=blink_patterns,
    )
    return vote, evidence
