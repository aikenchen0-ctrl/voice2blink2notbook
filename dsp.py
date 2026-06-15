from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional

import numpy as np

from hp_acoustic_wave.config import FmcwConfig


@dataclass
class ChunkFeature:
    time_s: float
    sample_index: int
    i_value: float
    q_value: float
    amplitude: float
    amplitude_delta: float
    phase: float
    phase_delta: float
    motion_energy: float
    rms: float
    peak_abs: float


@dataclass
class FmcwFeature:
    time_s: float
    sample_index: int
    period_index: int
    phase_point_count: int
    track_values: tuple[float, ...]
    track_delta_rms: float
    phase_std: float
    rms: float
    peak_abs: float
    pairs: tuple[tuple[int, int], ...]
    track_deltas: tuple[float, ...] = ()
    phase_points: tuple[float, ...] = ()


@dataclass
class FmcwCandidateBundle:
    criterion: str
    score: float
    reference_index: int
    target_indices: tuple[int, ...]
    trajectories: tuple[np.ndarray, ...]


def generate_tone(
    num_samples: int,
    sample_rate: int,
    frequency_hz: float,
    start_sample: int,
    amplitude: float,
) -> np.ndarray:
    indices = np.arange(start_sample, start_sample + num_samples, dtype=np.float64)
    phase = 2.0 * math.pi * frequency_hz * indices / float(sample_rate)
    tone = amplitude * np.sin(phase)
    return tone.astype(np.float32)


def tukey_window(num_samples: int, alpha: float) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if alpha <= 0.0:
        return np.ones(num_samples, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(num_samples).astype(np.float64)

    x = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    window = np.ones(num_samples, dtype=np.float64)
    first = x < alpha / 2.0
    last = x >= 1.0 - alpha / 2.0
    window[first] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[first] / alpha - 1.0)))
    window[last] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[last] / alpha - 2.0 / alpha + 1.0)))
    return window


def fmcw_chirp_period(config: FmcwConfig, sample_rate: int, amplitude: float) -> np.ndarray:
    if config.chirp_samples + config.guard_samples != config.period_samples:
        raise ValueError("chirp_samples + guard_samples must equal period_samples")
    if config.f1_hz <= config.f0_hz:
        raise ValueError("f1_hz must be greater than f0_hz")
    if config.f1_hz > sample_rate / 2.0:
        raise ValueError("f1_hz must be at or below Nyquist")

    t = np.arange(config.chirp_samples, dtype=np.float64) / float(sample_rate)
    chirp_duration = config.chirp_samples / float(sample_rate)
    bandwidth = config.f1_hz - config.f0_hz
    phase = 2.0 * math.pi * (config.f0_hz * t + bandwidth * t * t / (2.0 * chirp_duration))
    # 论文 §3.1 定义发射 FMCW 为 A*cos(phi_t)。Tukey window 会把默认首尾压到 0，避免跳变声。
    active = float(amplitude) * tukey_window(config.chirp_samples, config.tukey_alpha) * np.cos(phase)
    period = np.zeros(config.period_samples, dtype=np.float32)
    period[: config.chirp_samples] = active.astype(np.float32)
    return period


def periodic_signal_chunk(period: np.ndarray, start_sample: int, num_samples: int) -> np.ndarray:
    period = np.asarray(period, dtype=np.float32).reshape(-1)
    if period.size == 0:
        raise ValueError("period must not be empty")
    offsets = (np.arange(start_sample, start_sample + num_samples, dtype=np.int64) % period.size)
    return period[offsets].astype(np.float32)


def decimate_baseband_lowpass(samples: np.ndarray, factor: int, taps_per_phase: int) -> np.ndarray:
    # 论文接收端是混频后低通，再 polyphase decimation。这里用 windowed-sinc FIR 做轻量实时近似。
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    factor = int(factor)
    taps_per_phase = int(taps_per_phase)
    if factor <= 0:
        raise ValueError("factor must be positive")
    if values.size == 0:
        return values.copy()
    if factor == 1:
        return values.copy()
    if taps_per_phase <= 0:
        usable = (values.size // factor) * factor
        return np.mean(values[:usable].reshape(-1, factor), axis=1) if usable else np.asarray([], dtype=np.complex128)

    tap_count = max(factor, int(factor * taps_per_phase))
    if tap_count % 2 == 0:
        tap_count += 1
    offsets = np.arange(tap_count, dtype=np.float64) - (tap_count - 1) / 2.0
    cutoff = 1.0 / float(factor)
    kernel = cutoff * np.sinc(cutoff * offsets)
    kernel *= np.hamming(tap_count)
    kernel_sum = float(np.sum(kernel))
    if abs(kernel_sum) > 0.0:
        kernel /= kernel_sum

    filtered = np.convolve(values, kernel.astype(np.complex128), mode="same")
    start = factor // 2
    decimated = filtered[start::factor]
    expected = values.size // factor
    return decimated[:expected].astype(np.complex128, copy=False)


def highpass_filter(samples: np.ndarray, *, sample_rate: int, cutoff_hz: float, tap_count: int) -> np.ndarray:
    # 论文 6.4.4 只说明先用 high-pass filter 去掉 audible noise，没有给具体滤波器。
    # [UNSPECIFIED] 这里用 Hamming-windowed sinc FIR；它只作用于 FMCW 相位提取，不影响旧版单频候选。
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if sample_rate <= 0 or cutoff_hz <= 0.0:
        return values.copy()
    if cutoff_hz >= sample_rate / 2.0:
        return np.zeros_like(values)

    taps = max(3, int(tap_count))
    if taps % 2 == 0:
        taps += 1
    offsets = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    normalized_cutoff = float(cutoff_hz) / float(sample_rate)
    lowpass = 2.0 * normalized_cutoff * np.sinc(2.0 * normalized_cutoff * offsets)
    lowpass *= np.hamming(taps)
    lowpass_sum = float(np.sum(lowpass))
    if abs(lowpass_sum) > 0.0:
        lowpass /= lowpass_sum

    highpass = -lowpass
    highpass[taps // 2] += 1.0
    return np.convolve(values, highpass, mode="same")


def remove_known_tone(
    samples: np.ndarray,
    *,
    sample_rate: int,
    tone_hz: float,
    start_sample: int,
) -> np.ndarray:
    # 工程混合模式会额外播放 18.5 kHz 单频作为旧版 blink 候选源。
    # [UNSPECIFIED] 论文没有混合单频，因此也没有这一步；这里仅为避免单频污染 FMCW baseband。
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if sample_rate <= 0 or tone_hz <= 0.0 or tone_hz >= sample_rate / 2.0:
        return values.copy()

    indices = np.arange(start_sample, start_sample + values.size, dtype=np.float64)
    phase = 2.0 * math.pi * float(tone_hz) * indices / float(sample_rate)
    basis = np.column_stack((np.sin(phase), np.cos(phase)))
    coeffs, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return values - basis @ coeffs


def unwrap_delta(current_phase: float, previous_phase: float) -> float:
    delta = current_phase - previous_phase
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _as_float32_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] == 0:
            return np.asarray([], dtype=np.float32)
        array = np.mean(array, axis=1, dtype=np.float32)
    return array.reshape(-1)


def smooth_trajectory(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detrend_trajectory(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return values - np.convolve(padded, kernel, mode="valid")


def normalize_trajectory(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values)) if values.size else values.copy()
    scale = float(np.max(np.abs(centered))) if centered.size else 0.0
    if scale <= 0.0:
        return centered
    return centered / scale


def phase_difference_trajectory(
    phase_matrix: np.ndarray,
    reference_index: int,
    target_index: int,
    *,
    smoothing_window: int,
    detrend_window: int,
    normalize: bool = True,
) -> np.ndarray:
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("phase_matrix must be 2D")
    if reference_index < 0 or target_index < 0:
        raise ValueError("indices must be non-negative")
    if reference_index >= matrix.shape[1] or target_index >= matrix.shape[1]:
        raise IndexError("phase index out of range")

    # 论文 §3.2 明确要求先 unwrap chirp samples，再做 phase subtraction。
    # 先相减再 unwrap 在 fast-time 相位跨 +/-pi 时会把很小相位差误看成大跳变。
    unwrapped = np.unwrap(matrix, axis=1)
    trajectory = np.unwrap(unwrapped[:, target_index] - unwrapped[:, reference_index])
    # 论文 §3.2 的主链路是 phase subtraction -> normalize -> small moving average。
    # detrend 是 [UNSPECIFIED] 工程 A/B 选项，默认关闭，避免改变眨眼的低频开闭形状。
    if detrend_window > 1:
        trajectory = detrend_trajectory(trajectory, detrend_window)
    if normalize:
        trajectory = normalize_trajectory(trajectory)
    if smoothing_window > 1:
        trajectory = smooth_trajectory(trajectory, smoothing_window)
    return trajectory


def phase_difference_to_distance_meters(
    phase_difference: np.ndarray | float,
    reference_index: int,
    target_index: int,
    config: FmcwConfig,
) -> np.ndarray:
    # TwinkleTwinkle Eq. 8: d = phase_diff * N * v / (4*pi*B*(N2-N1)).
    # 这里的 phase point 已经按 decimation_factor 抽取，所以 N2-N1 要换回原始采样点间隔。
    sample_gap = abs(int(target_index) - int(reference_index)) * int(config.decimation_factor)
    bandwidth = float(config.f1_hz) - float(config.f0_hz)
    if sample_gap <= 0 or bandwidth <= 0.0:
        return np.zeros_like(np.asarray(phase_difference, dtype=np.float64))
    scale = (
        float(config.chirp_samples)
        * float(config.sound_speed_m_s)
        / (4.0 * math.pi * bandwidth * float(sample_gap))
    )
    return np.asarray(phase_difference, dtype=np.float64) * scale


def select_fmcw_candidate_bundles(
    phase_matrix: np.ndarray,
    config: FmcwConfig,
) -> tuple[FmcwCandidateBundle, ...]:
    # 论文思路：同一段信号从多个 phase-pair 视角看，选更稳定/更相似的候选轨迹再投票。
    # 这里返回 3 个准则 * 每准则 3 个 interval * 每 interval 5 条轨迹 = 最多 45 条轨迹。
    matrix = np.asarray(phase_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("phase_matrix must be 2D")
    if matrix.shape[0] == 0:
        return tuple()
    # phase points 在单个 chirp 内先展开，避免相邻 fast-time 点的 +/-pi 跳变污染斜率。
    matrix = np.unwrap(matrix, axis=1)
    # 同一 phase point 跨 chirp 比较时，再沿时间方向展开；它只用于 temporal consistency 打分。
    temporal_matrix = np.unwrap(matrix, axis=0)

    interval_length = int(config.candidate_interval_length)
    intervals_per_criterion = int(config.candidate_intervals_per_criterion)
    valid_start = max(0, int(config.valid_start))
    valid_stop = min(int(config.valid_stop), int(matrix.shape[1]))
    max_start = valid_stop - valid_start - interval_length
    if interval_length <= 0 or intervals_per_criterion <= 0 or max_start < 0:
        return tuple()

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for target_start in range(valid_start, valid_start + max_start + 1):
        target_indices = tuple(target_start + delta for delta in range(interval_length))
        reference_index = target_start - 1
        if reference_index < 0:
            continue
        candidates.append((reference_index, target_indices))

    scored = {
        "internal_similarity": [],
        "slope_stability": [],
        "same_point_temporal_consistency": [],
    }
    for reference_index, target_indices in candidates:
        trajectories = _candidate_trajectories(matrix, reference_index, target_indices, config)
        scored["internal_similarity"].append(
            FmcwCandidateBundle(
                criterion="internal_similarity",
                score=_trajectory_distance_sum(trajectories),
                reference_index=reference_index,
                target_indices=target_indices,
                trajectories=trajectories,
            )
        )
        scored["slope_stability"].append(
            FmcwCandidateBundle(
                criterion="slope_stability",
                score=_candidate_interval_slope_variance(matrix, target_indices),
                reference_index=reference_index,
                target_indices=target_indices,
                trajectories=trajectories,
            )
        )
        scored["same_point_temporal_consistency"].append(
            FmcwCandidateBundle(
                criterion="same_point_temporal_consistency",
                score=_same_point_temporal_difference_score(temporal_matrix, target_indices),
                reference_index=reference_index,
                target_indices=target_indices,
                trajectories=trajectories,
            )
        )

    selected: list[FmcwCandidateBundle] = []
    used: set[tuple[int, tuple[int, ...]]] = set()
    reference_offsets = (1, 2, 3)
    min_reference_index = max(0, valid_start - 1)
    for criterion in ("internal_similarity", "slope_stability", "same_point_temporal_consistency"):
        ranked = sorted(
            scored[criterion],
            key=lambda item: (item.score, item.reference_index, item.target_indices),
        )
        chosen = 0
        for bundle in ranked:
            # 论文 4.2：多个准则选到同一 interval 时，优先把 reference 从 t2-1
            # 改成 t2-2/t2-3，保留同一 target interval 的不同 phase-pair 视角。
            # 若替代 reference 会越过有效边界或已被使用，则继续找下一个 unused interval，尽量保留每组 3 个 interval。
            for offset in reference_offsets:
                reference_index = bundle.target_indices[0] - offset
                signature = (reference_index, bundle.target_indices)
                if reference_index < min_reference_index or signature in used:
                    continue
                trajectories = _candidate_trajectories(matrix, reference_index, bundle.target_indices, config)
                selected.append(
                    FmcwCandidateBundle(
                        criterion=criterion,
                        score=bundle.score,
                        reference_index=reference_index,
                        target_indices=bundle.target_indices,
                        trajectories=trajectories,
                    )
                )
                used.add(signature)
                chosen += 1
                break
            if chosen >= intervals_per_criterion:
                break

    return tuple(selected)


def _candidate_trajectories(
    matrix: np.ndarray,
    reference_index: int,
    target_indices: tuple[int, ...],
    config: FmcwConfig,
) -> tuple[np.ndarray, ...]:
    return tuple(
        phase_difference_trajectory(
            matrix,
            reference_index,
            target_index,
            smoothing_window=int(config.trajectory_smoothing_window),
            detrend_window=int(config.trajectory_detrend_window),
        )
        for target_index in target_indices
    )


def _trajectory_distance_sum(trajectories: tuple[np.ndarray, ...]) -> float:
    arrays = [normalize_trajectory(trajectory) for trajectory in trajectories]
    total = 0.0
    for left_index in range(len(arrays)):
        for right_index in range(left_index + 1, len(arrays)):
            total += float(np.linalg.norm(arrays[left_index] - arrays[right_index]))
    return total


def _candidate_interval_slope_variance(
    phase_matrix: np.ndarray,
    target_indices: tuple[int, ...],
) -> float:
    matrix = np.unwrap(np.asarray(phase_matrix, dtype=np.float64), axis=1)
    window = matrix[:, target_indices]
    if window.shape[1] < 2:
        return 0.0
    slopes = np.diff(window, axis=1)
    return float(np.sum(np.var(slopes, axis=0)))


def _candidate_interval_stability_score(
    phase_matrix: np.ndarray,
    temporal_phase_matrix: np.ndarray,
    target_indices: tuple[int, ...],
) -> float:
    # 兼容旧测试/调试脚本的组合分数。论文 4.2 的 45 轨迹选择实际拆成
    # slope_stability 和 same_point_temporal_consistency 两组，以保持三组候选视角独立。
    return _candidate_interval_slope_variance(
        phase_matrix,
        target_indices,
    ) + _same_point_temporal_difference_score(
        temporal_phase_matrix,
        target_indices,
    )


def _same_point_temporal_difference_score(
    phase_matrix: np.ndarray,
    target_indices: tuple[int, ...],
) -> float:
    # 论文 4.2 写的是同一 phase index 在不同 chirp 上的 differences；
    # 用绝对相邻差值总和，避免大幅线性漂移因“步长方差为 0”被误当稳定。
    matrix = np.unwrap(np.asarray(phase_matrix, dtype=np.float64), axis=0)
    window = matrix[:, target_indices]
    if window.shape[0] < 2:
        return 0.0
    temporal_steps = np.diff(window, axis=0)
    return float(np.sum(np.abs(temporal_steps)))


def _within_chirp_slope_variance(
    phase_matrix: np.ndarray,
    reference_index: int,
    target_indices: tuple[int, ...],
) -> float:
    matrix = np.unwrap(np.asarray(phase_matrix, dtype=np.float64), axis=1)
    score = 0.0
    for target_index in target_indices:
        start = min(reference_index, target_index)
        end = max(reference_index, target_index)
        window = matrix[:, start : end + 1]
        if window.shape[1] < 2:
            continue
        slopes = np.diff(window, axis=1)
        score += float(np.sum(np.var(slopes, axis=0)))
    return score


def extract_chunk_feature(
    samples: np.ndarray,
    sample_rate: int,
    tone_hz: float,
    start_sample: int,
    previous: Optional[ChunkFeature],
) -> ChunkFeature:
    mono = _as_float32_mono(samples)
    if mono.size == 0:
        raise ValueError("samples must not be empty")

    indices = np.arange(start_sample, start_sample + mono.size, dtype=np.float64)
    mixer = np.exp(-1j * 2.0 * math.pi * tone_hz * indices / float(sample_rate))
    baseband = mono.astype(np.float64) * mixer
    complex_mean = np.mean(baseband)

    i_value = float(np.real(complex_mean))
    q_value = float(np.imag(complex_mean))
    amplitude = float(abs(complex_mean))
    phase = float(math.atan2(q_value, i_value))
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    peak_abs = float(np.max(np.abs(mono)))

    if previous is None:
        amplitude_delta = 0.0
        phase_delta = 0.0
    else:
        amplitude_delta = amplitude - previous.amplitude
        phase_delta = unwrap_delta(phase, previous.phase)

    reference_amplitude = previous.amplitude if previous is not None else amplitude
    relative_amp_delta = abs(amplitude_delta) / max(reference_amplitude, amplitude, 1e-3)
    phase_assist = 0.15 * min(math.pi, abs(phase_delta)) / math.pi
    motion_energy = float(min(5.0, relative_amp_delta) + phase_assist)

    return ChunkFeature(
        time_s=float(start_sample) / float(sample_rate),
        sample_index=int(start_sample),
        i_value=i_value,
        q_value=q_value,
        amplitude=amplitude,
        amplitude_delta=float(amplitude_delta),
        phase=phase,
        phase_delta=float(phase_delta),
        motion_energy=motion_energy,
        rms=rms,
        peak_abs=peak_abs,
    )


class FmcwStreamProcessor:
    def __init__(self, config: FmcwConfig, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate
        self.reference = self._reference()
        self.pairs = self._track_pairs()
        self.previous_tracks: Optional[np.ndarray] = None
        self.phase_history: Deque[np.ndarray] = deque(maxlen=max(1, int(config.phase_window_size)))
        self._buffer = np.asarray([], dtype=np.float32)
        self._buffer_start_sample: Optional[int] = None
        self.period_start_offset = 0

    def _reference(self) -> np.ndarray:
        t = np.arange(self.config.chirp_samples, dtype=np.float64) / float(self.sample_rate)
        chirp_duration = self.config.chirp_samples / float(self.sample_rate)
        bandwidth = self.config.f1_hz - self.config.f0_hz
        phase = 2.0 * math.pi * (
            self.config.f0_hz * t + bandwidth * t * t / (2.0 * chirp_duration)
        )
        reference = np.zeros(self.config.period_samples, dtype=np.complex128)
        reference[: self.config.chirp_samples] = np.exp(-1j * phase)
        return reference

    def _track_pairs(self) -> tuple[tuple[int, int], ...]:
        point_count = self.config.period_samples // self.config.decimation_factor
        configured_pairs = tuple(getattr(self.config, "track_pairs", ()) or ())
        if configured_pairs:
            pairs = []
            for reference, target in configured_pairs:
                reference = int(reference)
                target = int(target)
                if 0 <= reference < point_count and 0 <= target < point_count and reference != target:
                    pairs.append((reference, target))
            return tuple(pairs)
        reference = max(0, int(self.config.valid_start) - int(self.config.reference_offset))
        pairs = []
        for index in range(max(0, int(self.config.track_count))):
            target = int(self.config.valid_start) + index
            if reference < point_count and target < point_count:
                pairs.append((reference, target))
        return tuple(pairs)

    def process_block(self, samples: np.ndarray, start_sample: int) -> list[FmcwFeature]:
        mono = _as_float32_mono(samples)
        if mono.size == 0:
            return []

        period = int(self.config.period_samples)
        features: list[FmcwFeature] = []

        if self._buffer_start_sample is None:
            self._buffer_start_sample = int(start_sample)
            self._buffer = mono.copy()
        else:
            expected_start = self._buffer_start_sample + int(self._buffer.size)
            if start_sample < expected_start:
                overlap = expected_start - int(start_sample)
                if overlap >= mono.size:
                    return []
                mono = mono[overlap:]
                start_sample = expected_start
            elif start_sample > expected_start:
                self._buffer_start_sample = int(start_sample)
                self._buffer = np.asarray([], dtype=np.float32)

            self._buffer = np.concatenate((self._buffer, mono.astype(np.float32, copy=False)))

        while self._buffer.size >= period and self._buffer_start_sample is not None:
            misalignment = (self._buffer_start_sample - int(self.period_start_offset)) % period
            if misalignment:
                skip = min(period - misalignment, int(self._buffer.size))
                self._buffer = self._buffer[skip:]
                self._buffer_start_sample += skip
                if self._buffer.size < period:
                    break

            frame = self._buffer[:period]
            frame_start = self._buffer_start_sample
            features.append(self.process_frame(frame, frame_start))
            self._buffer = self._buffer[period:]
            self._buffer_start_sample += period

        return features

    def process_frame(self, frame: np.ndarray, frame_start_sample: int) -> FmcwFeature:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size < self.config.period_samples:
            raise ValueError("frame must contain one full FMCW period")

        phases = self.extract_phase_points(frame, frame_start_sample=frame_start_sample)
        self.phase_history.append(phases.copy())

        track_values = np.asarray(
            [self._track_value(phases, reference, target) for reference, target in self.pairs],
            dtype=np.float64,
        )
        if self.previous_tracks is None or self.previous_tracks.shape != track_values.shape:
            delta = np.zeros_like(track_values)
            delta_rms = 0.0
        else:
            # track_values 是相位差，跨 chirp 可能从 +pi 跳到 -pi；这里必须按圆周相位求最短差。
            delta = np.asarray(
                [
                    unwrap_delta(float(current), float(previous))
                    for current, previous in zip(track_values, self.previous_tracks)
                ],
                dtype=np.float64,
            )
            delta_rms = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
        self.previous_tracks = track_values.copy()

        return FmcwFeature(
            time_s=float(frame_start_sample) / float(self.sample_rate),
            sample_index=int(frame_start_sample),
            period_index=int((frame_start_sample - int(self.period_start_offset)) // self.config.period_samples),
            phase_point_count=int(phases.size),
            track_values=tuple(float(v) for v in track_values),
            track_delta_rms=delta_rms,
            phase_std=float(np.std(phases)) if phases.size else 0.0,
            rms=float(np.sqrt(np.mean(np.square(frame.astype(np.float64))))),
            peak_abs=float(np.max(np.abs(frame))) if frame.size else 0.0,
            pairs=self.pairs,
            track_deltas=tuple(float(value) for value in delta),
            phase_points=tuple(float(value) for value in phases),
        )

    def _track_value(self, phases: np.ndarray, reference: int, target: int) -> float:
        phase_delta = float(phases[target] - phases[reference])
        if not bool(self.config.track_gap_normalization):
            return phase_delta
        # Eq. 6-8: phase_delta is proportional to (N2-N1). Divide by the point
        # gap so tracks with wider target spacing remain comparable in realtime UI.
        point_gap = max(1, abs(int(target) - int(reference)))
        return phase_delta / float(point_gap)

    def extract_phase_points(self, frame: np.ndarray, frame_start_sample: int = 0) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size < self.config.period_samples:
            raise ValueError("frame must contain one full FMCW period")

        period = frame[: self.config.period_samples].astype(np.float64)
        if bool(self.config.raw_highpass_enabled):
            period = highpass_filter(
                period,
                sample_rate=int(self.sample_rate),
                cutoff_hz=float(self.config.raw_highpass_hz),
                tap_count=int(self.config.raw_highpass_taps),
            )
        if (
            bool(self.config.primary_blink_tone_cleanup_enabled)
            and bool(self.config.primary_blink_enabled)
            and float(self.config.primary_blink_tone_ratio) > 0.0
        ):
            period = remove_known_tone(
                period,
                sample_rate=int(self.sample_rate),
                tone_hz=float(self.config.primary_blink_tone_hz),
                start_sample=int(frame_start_sample),
            )
        mixed = period * self.reference
        factor = int(self.config.decimation_factor)
        baseband_points = decimate_baseband_lowpass(
            mixed,
            factor,
            int(self.config.decimation_filter_taps_per_phase),
        )
        return np.unwrap(np.angle(baseband_points))

    def rolling_phase_matrix(self) -> np.ndarray:
        if not self.phase_history:
            point_count = self.config.period_samples // self.config.decimation_factor
            return np.empty((0, point_count), dtype=np.float64)
        return np.vstack(tuple(self.phase_history)).astype(np.float64, copy=False)
