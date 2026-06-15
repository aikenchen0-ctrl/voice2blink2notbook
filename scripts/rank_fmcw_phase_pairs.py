#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()

from hp_acoustic_wave.config import AudioConfig
from hp_acoustic_wave.dsp import FmcwStreamProcessor
from hp_acoustic_wave.fmcw_pair_ranking import PHASE_PAIR_SCORE_METRICS, rank_phase_pairs
from hp_acoustic_wave.fmcw_session_analysis import fmcw_config_from_session, read_csv_rows


@dataclass(frozen=True)
class PairRank:
    reference_index: int
    target_index: int
    blink_peak_median: float
    background_peak_p95: float
    separation: float
    blink_hit_rate: float
    background_trigger_rate: float
    blink_peak_count: int
    background_window_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank FMCW phase-point pairs by blink-marker separation from a recorded session.",
    )
    parser.add_argument("session_dir", help="Session directory containing audio.wav and manual_markers.csv.")
    parser.add_argument("--window", type=float, default=0.6, help="Marker-centered blink window in seconds.")
    parser.add_argument("--background-window", type=float, default=0.6, help="Background window length in seconds.")
    parser.add_argument("--center-offset", type=float, default=0.0, help="Seconds added to manual markers before windows are cut.")
    parser.add_argument(
        "--metric",
        choices=PHASE_PAIR_SCORE_METRICS,
        default="shape",
        help="delta_peak uses adjacent-chirp jumps; shape scores returning open/close trajectory windows.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of ranked pairs to print.")
    parser.add_argument("--output", default=None, help="Optional CSV path for full ranking.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    markers = [
        float(row["time_s"])
        for row in read_csv_rows(session_dir / "manual_markers.csv")
        if row.get("label") == "blink"
    ]
    if not markers:
        print("No blink markers found.", file=sys.stderr)
        return 2

    sample_rate, audio = _read_wav_mono(session_dir / "audio.wav")
    config = fmcw_config_from_session(session_dir)
    processor = FmcwStreamProcessor(config, sample_rate)
    sync_lag = _infer_sync_lag(session_dir)
    if sync_lag is not None:
        processor.period_start_offset = int(sync_lag)
    features = processor.process_block(audio, 0)
    times = np.asarray([feature.time_s for feature in features], dtype=np.float64)
    phase_matrix = np.asarray([feature.phase_points for feature in features], dtype=np.float64)
    if phase_matrix.ndim != 2 or phase_matrix.shape[0] < 2:
        print("Not enough FMCW phase rows reconstructed.", file=sys.stderr)
        return 2

    ranks = rank_phase_pairs(
        times,
        phase_matrix,
        markers,
        valid_start=int(config.valid_start),
        valid_stop=int(config.valid_stop),
        marker_window_s=float(args.window),
        background_window_s=float(args.background_window),
        metric=str(args.metric),
        center_offset_s=float(args.center_offset),
    )
    if args.output:
        _write_ranks(Path(args.output), ranks)

    print(f"session={session_dir}")
    print(
        f"blink_markers={len(markers)} phase_rows={phase_matrix.shape[0]} "
        f"phase_points={phase_matrix.shape[1]} sync_lag={sync_lag if sync_lag is not None else 'none'}"
    )
    print(f"metric={args.metric} center_offset={args.center_offset:+.3f}s")
    print("rank  pair    sep    blink_med  bg_p95  blink_hit  bg_trigger")
    for index, rank in enumerate(ranks[: max(1, int(args.top))], 1):
        print(
            f"{index:>4}  {rank.reference_index:02d}:{rank.target_index:02d}  "
            f"{rank.separation:6.2f}  {rank.blink_peak_median:9.5f}  "
            f"{rank.background_peak_p95:7.5f}  {rank.blink_hit_rate:8.2f}  "
            f"{rank.background_trigger_rate:10.2f}"
        )
    return 0


def rank_pairs(
    times: np.ndarray,
    phase_matrix: np.ndarray,
    markers: list[float],
    *,
    valid_start: int,
    valid_stop: int,
    marker_window_s: float,
    background_window_s: float,
) -> list[PairRank]:
    matrix = np.unwrap(np.asarray(phase_matrix, dtype=np.float64), axis=1)
    valid_stop = min(int(valid_stop), int(matrix.shape[1]))
    valid_start = max(1, min(int(valid_start), valid_stop - 1))
    background_windows = _background_windows(
        float(times[0]),
        float(times[-1]),
        markers,
        marker_window_s=marker_window_s,
        background_window_s=background_window_s,
    )
    ranks: list[PairRank] = []
    for reference in range(max(0, valid_start - 8), valid_stop - 1):
        for target in range(max(reference + 1, valid_start), valid_stop):
            trajectory = _pair_trajectory(matrix, reference, target)
            delta = np.abs(np.diff(trajectory, prepend=trajectory[0]))
            blink_peaks = _window_peaks(times, delta, [(m - marker_window_s, m + marker_window_s) for m in markers])
            background_peaks = _window_peaks(times, delta, background_windows)
            if not blink_peaks or not background_peaks:
                continue
            blink_median = float(np.median(blink_peaks))
            bg_p95 = float(np.percentile(background_peaks, 95))
            threshold = max(bg_p95, 1e-9)
            blink_hit_rate = float(np.mean(np.asarray(blink_peaks) >= threshold))
            background_trigger_rate = float(np.mean(np.asarray(background_peaks) >= threshold))
            separation = (blink_median / threshold) * (0.5 + blink_hit_rate) / (0.5 + background_trigger_rate)
            ranks.append(
                PairRank(
                    reference_index=int(reference),
                    target_index=int(target),
                    blink_peak_median=blink_median,
                    background_peak_p95=bg_p95,
                    separation=float(separation),
                    blink_hit_rate=blink_hit_rate,
                    background_trigger_rate=background_trigger_rate,
                    blink_peak_count=len(blink_peaks),
                    background_window_count=len(background_peaks),
                )
            )
    return sorted(
        ranks,
        key=lambda row: (
            -row.separation,
            -row.blink_hit_rate,
            row.background_trigger_rate,
            row.reference_index,
            row.target_index,
        ),
    )


def _read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return sample_rate, pcm.reshape(-1)


def _infer_sync_lag(session_dir: Path) -> int | None:
    values: list[int] = []
    for name in ("features.csv", "manual_markers.csv"):
        path = Path(session_dir) / name
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            value = row.get("fmcw_sync_lag_samples")
            if value in (None, ""):
                continue
            try:
                values.append(int(float(value)))
            except ValueError:
                continue
    if not values:
        return None
    return max(set(values), key=values.count)


def _pair_trajectory(matrix: np.ndarray, reference: int, target: int) -> np.ndarray:
    trajectory = np.unwrap(matrix[:, target] - matrix[:, reference])
    gap = max(1, abs(int(target) - int(reference)))
    return trajectory / float(gap)


def _window_peaks(times: np.ndarray, values: np.ndarray, windows: list[tuple[float, float]]) -> list[float]:
    peaks: list[float] = []
    for start, end in windows:
        mask = (times >= float(start)) & (times <= float(end))
        if np.any(mask):
            peaks.append(float(np.max(values[mask])))
    return peaks


def _background_windows(
    start_time: float,
    end_time: float,
    markers: list[float],
    *,
    marker_window_s: float,
    background_window_s: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    step = max(float(background_window_s), 0.1)
    t = float(start_time)
    while t + background_window_s <= end_time:
        center = t + background_window_s / 2.0
        if all(abs(center - marker) > marker_window_s * 1.5 for marker in markers):
            windows.append((t, t + background_window_s))
        t += step
    return windows


def _write_ranks(path: Path, ranks: list[PairRank]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PairRank.__dataclass_fields__.keys()))
        writer.writeheader()
        for rank in ranks:
            writer.writerow(rank.__dict__)


if __name__ == "__main__":
    raise SystemExit(main())
