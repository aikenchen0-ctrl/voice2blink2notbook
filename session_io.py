import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import wave

import numpy as np


# features.csv 每个处理帧一行：同时保留旧版单分数、FMCW 多轨、candidate 和 confirm 字段。
# 这样回放时能分清误报发生在“候选触发”还是“最终眨眼确认”。
FEATURE_FIELDS = [
    "time_s",
    "sample_index",
    "i",
    "q",
    "amplitude",
    "amplitude_delta",
    "phase",
    "phase_delta",
    "motion_energy",
    "rms",
    "peak_abs",
    "fmcw_period_index",
    "fmcw_phase_point_count",
    "fmcw_phase_points",
    "fmcw_sync_lag_samples",
    "fmcw_sync_confidence",
    "fmcw_track_0",
    "fmcw_track_1",
    "fmcw_track_2",
    "fmcw_track_3",
    "fmcw_track_4",
    "fmcw_track_delta_rms",
    "fmcw_phase_std",
    "fmcw_pairs",
    "fmcw_pattern",
    "fmcw_vote_confidence",
    "fmcw_vote_score",
    "fmcw_blink_vote_evidence",
    "fmcw_blink_trajectory_value",
    "fmcw_blink_trajectory_pattern",
    "fmcw_blink_trajectory_pair",
    "fmcw_blink_trajectory_criterion",
    "fmcw_fixed_trajectory_value",
    "fmcw_fixed_trajectory_phase_rad",
    "fmcw_fixed_trajectory_distance_mm",
    "fmcw_fixed_trajectory_pair",
    "fmcw_group_winners",
    "fmcw_candidate_count",
    "fmcw_candidate_score",
    "fmcw_candidate_baseline",
    "fmcw_candidate_mad",
    "fmcw_candidate_threshold",
    "fmcw_candidate_is_event",
    "fmcw_candidate_event_id",
    "fmcw_confirm_state",
    "fmcw_confirm_is_event",
    "fmcw_confirm_event_id",
    "fmcw_confirm_max_delta_rms",
    "fmcw_confirm_high_delta_duration_s",
    "fmcw_confirm_pattern",
    "fmcw_confirm_confidence",
    "fmcw_confirm_vote_score",
    "fmcw_confirm_pattern_rows",
    "fmcw_confirm_pattern_stability",
    "fmcw_confirm_window_pattern",
    "fmcw_confirm_window_confidence",
    "fmcw_confirm_window_vote_score",
    "fmcw_confirm_window_candidate_count",
    "fmcw_confirm_window_group_winners",
    "fmcw_final_pattern",
    "baseline",
    "mad",
    "threshold",
    "detector_method",
    "blink_score",
    "blink_threshold",
    "blink_baseline",
    "blink_mad",
    "blinklistener_viewing_amplitude",
    "blinklistener_viewing_range",
    "blinklistener_raw_viewing_score",
    "blinklistener_relative_viewing_score",
    "blinklistener_center_i",
    "blinklistener_center_q",
    "twinkle_phase_pair_delta",
    "twinkle_trajectory_span",
    "twinkle_trajectory_rms",
    "twinkle_peak_score",
    "twinkle_peak_threshold",
    "twinkle_peak_motion_energy",
    "twinkle_peak_sign_changes",
    "twinkle_candidate_peak",
    "twinkle_candidate_accepted",
    "twinkle_candidate_local_peak",
    "twinkle_candidate_rising_edge",
    "twinkle_candidate_event_level",
    "twinkle_candidate_active",
    "twinkle_reject_low_score",
    "twinkle_reject_high_score",
    "twinkle_reject_low_motion",
    "twinkle_reject_large_motion",
    "twinkle_reject_few_reversals",
    "twinkle_reject_many_reversals",
    "twinkle_reject_suppressed",
    "twinkle_reject_refractory",
    "twinkle_reject_active",
    "twinkle_large_motion_suppressed",
    "is_event",
    "event_id",
]

# events.csv 只记录事件：fmcw_candidate 是疑似候选，fmcw_confirmed_blink 才是最终眨眼。
# fmcw_suppressed_motion 表示候选被大动作/干扰动作压制，不计作 blink。
EVENT_FIELDS = [
    "event_id",
    "time_s",
    "label",
    "method",
    "score",
    "motion_energy",
    "threshold",
]

# manual_markers.csv 是人工按键时刻的状态快照，用来把用户实测标记和检测器内部状态对齐。
MARKER_FIELDS = [
    "time_s",
    "label",
    "key",
    "event_id",
    "amplitude",
    "phase",
    "motion_energy",
    "fmcw_track_0",
    "fmcw_track_1",
    "fmcw_track_2",
    "fmcw_track_3",
    "fmcw_track_4",
    "fmcw_track_delta_rms",
    "fmcw_phase_std",
    "fmcw_phase_points",
    "fmcw_sync_lag_samples",
    "fmcw_sync_confidence",
    "fmcw_pairs",
    "fmcw_pattern",
    "fmcw_vote_confidence",
    "fmcw_vote_score",
    "fmcw_blink_vote_evidence",
    "fmcw_blink_trajectory_value",
    "fmcw_blink_trajectory_pattern",
    "fmcw_blink_trajectory_pair",
    "fmcw_blink_trajectory_criterion",
    "fmcw_fixed_trajectory_value",
    "fmcw_fixed_trajectory_phase_rad",
    "fmcw_fixed_trajectory_distance_mm",
    "fmcw_fixed_trajectory_pair",
    "fmcw_group_winners",
    "fmcw_candidate_count",
    "fmcw_candidate_score",
    "fmcw_candidate_threshold",
    "fmcw_candidate_is_event",
    "fmcw_candidate_event_id",
    "fmcw_confirm_state",
    "fmcw_confirm_is_event",
    "fmcw_confirm_event_id",
    "fmcw_confirm_max_delta_rms",
    "fmcw_confirm_high_delta_duration_s",
    "fmcw_confirm_pattern",
    "fmcw_confirm_confidence",
    "fmcw_confirm_vote_score",
    "fmcw_confirm_pattern_rows",
    "fmcw_confirm_pattern_stability",
    "fmcw_confirm_window_pattern",
    "fmcw_confirm_window_confidence",
    "fmcw_confirm_window_vote_score",
    "fmcw_confirm_window_candidate_count",
    "fmcw_confirm_window_group_winners",
    "fmcw_final_pattern",
    "blink_score",
    "blink_threshold",
    "blink_baseline",
    "blink_mad",
    "blink_event_id",
    "blink_is_event",
    "blink_method",
    "visual_enabled",
    "visual_available",
    "visual_face_found",
    "visual_left_ear",
    "visual_right_ear",
    "visual_left_closed",
    "visual_right_closed",
    "visual_is_blink_event",
    "visual_blink_count",
    "visual_auto_marker_count",
    "visual_inference_ms",
    "visual_error",
]


def create_session_dir(root: str, prefix: str = "hp_wave") -> Path:
    root_path = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root_path / f"{prefix}_{timestamp}"
    suffix = 1
    while session_dir.exists():
        session_dir = root_path / f"{prefix}_{timestamp}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


class SessionWriter:
    def __init__(self, session_dir: Path, sample_rate: int):
        self.session_dir = Path(session_dir)
        self.sample_rate = sample_rate
        self._wav: Optional[wave.Wave_write] = None
        self._features_handle = None
        self._events_handle = None
        self._markers_handle = None
        self._features_writer = None
        self._events_writer = None
        self._markers_writer = None

    def open(self) -> None:
        self._wav = wave.open(str(self.session_dir / "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)

        self._features_handle = open(self.session_dir / "features.csv", "w", newline="", encoding="utf-8")
        self._features_writer = csv.DictWriter(self._features_handle, fieldnames=FEATURE_FIELDS)
        self._features_writer.writeheader()

        self._events_handle = open(self.session_dir / "events.csv", "w", newline="", encoding="utf-8")
        self._events_writer = csv.DictWriter(self._events_handle, fieldnames=EVENT_FIELDS)
        self._events_writer.writeheader()

        self._markers_handle = open(self.session_dir / "manual_markers.csv", "w", newline="", encoding="utf-8")
        self._markers_writer = csv.DictWriter(self._markers_handle, fieldnames=MARKER_FIELDS)
        self._markers_writer.writeheader()

    def write_audio(self, samples: np.ndarray) -> None:
        if self._wav is None:
            raise RuntimeError("SessionWriter is not open")
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        clipped = np.clip(mono, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        self._wav.writeframes(pcm16.tobytes())

    def write_feature(self, row: Dict[str, float]) -> None:
        if self._features_writer is None:
            raise RuntimeError("SessionWriter is not open")
        self._features_writer.writerow({field: row.get(field, "") for field in FEATURE_FIELDS})

    def write_event(
        self,
        event_id: int,
        time_s: float,
        motion_energy: float,
        threshold: float,
        label: str = "wave",
        method: str = "wave",
        score: Optional[float] = None,
    ) -> None:
        if self._events_writer is None:
            raise RuntimeError("SessionWriter is not open")
        self._events_writer.writerow(
            {
                "event_id": event_id,
                "time_s": f"{time_s:.6f}",
                "label": label,
                "method": method,
                "score": "" if score is None else f"{score:.9f}",
                "motion_energy": f"{motion_energy:.9f}",
                "threshold": f"{threshold:.9f}",
            }
        )

    def write_manual_marker(
        self,
        time_s: float,
        label: str = "manual_wave",
        key: str = "m",
        feature_snapshot: Optional[Dict] = None,
        event_id: Optional[int] = None,
    ) -> None:
        if self._markers_writer is None:
            raise RuntimeError("SessionWriter is not open")
        snapshot = feature_snapshot or {}
        # marker 保存按键当下的快照，不重算历史；后续分析再按 time_s 与 features.csv 对齐。
        self._markers_writer.writerow(
            {
                "time_s": f"{time_s:.6f}",
                "label": label,
                "key": key,
                "event_id": "" if event_id is None else event_id,
                "amplitude": _format_optional_float(snapshot.get("amplitude")),
                "phase": _format_optional_float(snapshot.get("phase")),
                "motion_energy": _format_optional_float(snapshot.get("motion_energy")),
                "fmcw_track_0": _format_optional_float(snapshot.get("fmcw_track_0")),
                "fmcw_track_1": _format_optional_float(snapshot.get("fmcw_track_1")),
                "fmcw_track_2": _format_optional_float(snapshot.get("fmcw_track_2")),
                "fmcw_track_3": _format_optional_float(snapshot.get("fmcw_track_3")),
                "fmcw_track_4": _format_optional_float(snapshot.get("fmcw_track_4")),
                "fmcw_track_delta_rms": _format_optional_float(snapshot.get("fmcw_track_delta_rms")),
                "fmcw_phase_std": _format_optional_float(snapshot.get("fmcw_phase_std")),
                "fmcw_phase_points": _format_optional_sequence(snapshot.get("fmcw_phase_points")),
                "fmcw_sync_lag_samples": snapshot.get("fmcw_sync_lag_samples", ""),
                "fmcw_sync_confidence": _format_optional_float(snapshot.get("fmcw_sync_confidence")),
                "fmcw_pairs": snapshot.get("fmcw_pairs", ""),
                "fmcw_pattern": snapshot.get("fmcw_pattern", ""),
                "fmcw_vote_confidence": _format_optional_float(snapshot.get("fmcw_vote_confidence")),
                "fmcw_vote_score": snapshot.get("fmcw_vote_score", ""),
                "fmcw_blink_vote_evidence": _format_optional_float(snapshot.get("fmcw_blink_vote_evidence")),
                "fmcw_blink_trajectory_value": _format_optional_float(
                    snapshot.get("fmcw_blink_trajectory_value")
                ),
                "fmcw_blink_trajectory_pattern": snapshot.get("fmcw_blink_trajectory_pattern", ""),
                "fmcw_blink_trajectory_pair": snapshot.get("fmcw_blink_trajectory_pair", ""),
                "fmcw_blink_trajectory_criterion": snapshot.get("fmcw_blink_trajectory_criterion", ""),
                "fmcw_fixed_trajectory_value": _format_optional_float(
                    snapshot.get("fmcw_fixed_trajectory_value")
                ),
                "fmcw_fixed_trajectory_phase_rad": _format_optional_float(
                    snapshot.get("fmcw_fixed_trajectory_phase_rad")
                ),
                "fmcw_fixed_trajectory_distance_mm": _format_optional_float(
                    snapshot.get("fmcw_fixed_trajectory_distance_mm")
                ),
                "fmcw_fixed_trajectory_pair": snapshot.get("fmcw_fixed_trajectory_pair", ""),
                "fmcw_group_winners": snapshot.get("fmcw_group_winners", ""),
                "fmcw_candidate_count": snapshot.get("fmcw_candidate_count", ""),
                "fmcw_candidate_score": _format_optional_float(snapshot.get("fmcw_candidate_score")),
                "fmcw_candidate_threshold": _format_optional_float(snapshot.get("fmcw_candidate_threshold")),
                "fmcw_candidate_is_event": snapshot.get("fmcw_candidate_is_event", ""),
                "fmcw_candidate_event_id": snapshot.get("fmcw_candidate_event_id", ""),
                "fmcw_confirm_state": snapshot.get("fmcw_confirm_state", ""),
                "fmcw_confirm_is_event": snapshot.get("fmcw_confirm_is_event", ""),
                "fmcw_confirm_event_id": snapshot.get("fmcw_confirm_event_id", ""),
                "fmcw_confirm_max_delta_rms": _format_optional_float(snapshot.get("fmcw_confirm_max_delta_rms")),
                "fmcw_confirm_high_delta_duration_s": _format_optional_float(
                    snapshot.get("fmcw_confirm_high_delta_duration_s")
                ),
                "fmcw_confirm_pattern": snapshot.get("fmcw_confirm_pattern", ""),
                "fmcw_confirm_confidence": _format_optional_float(snapshot.get("fmcw_confirm_confidence")),
                "fmcw_confirm_vote_score": snapshot.get("fmcw_confirm_vote_score", ""),
                "fmcw_confirm_pattern_rows": snapshot.get("fmcw_confirm_pattern_rows", ""),
                "fmcw_confirm_pattern_stability": _format_optional_float(
                    snapshot.get("fmcw_confirm_pattern_stability")
                ),
                "fmcw_confirm_window_pattern": snapshot.get("fmcw_confirm_window_pattern", ""),
                "fmcw_confirm_window_confidence": _format_optional_float(
                    snapshot.get("fmcw_confirm_window_confidence")
                ),
                "fmcw_confirm_window_vote_score": snapshot.get("fmcw_confirm_window_vote_score", ""),
                "fmcw_confirm_window_candidate_count": snapshot.get("fmcw_confirm_window_candidate_count", ""),
                "fmcw_confirm_window_group_winners": snapshot.get("fmcw_confirm_window_group_winners", ""),
                "fmcw_final_pattern": snapshot.get("fmcw_final_pattern", ""),
                "blink_score": _format_optional_float(snapshot.get("blink_score")),
                "blink_threshold": _format_optional_float(snapshot.get("blink_threshold")),
                "blink_baseline": _format_optional_float(snapshot.get("blink_baseline")),
                "blink_mad": _format_optional_float(snapshot.get("blink_mad")),
                "blink_event_id": snapshot.get("blink_event_id", ""),
                "blink_is_event": snapshot.get("blink_is_event", ""),
                "blink_method": snapshot.get("blink_method", ""),
                "visual_enabled": snapshot.get("visual_enabled", ""),
                "visual_available": snapshot.get("visual_available", ""),
                "visual_face_found": snapshot.get("visual_face_found", ""),
                "visual_left_ear": _format_optional_float(snapshot.get("visual_left_ear")),
                "visual_right_ear": _format_optional_float(snapshot.get("visual_right_ear")),
                "visual_left_closed": snapshot.get("visual_left_closed", ""),
                "visual_right_closed": snapshot.get("visual_right_closed", ""),
                "visual_is_blink_event": snapshot.get("visual_is_blink_event", ""),
                "visual_blink_count": snapshot.get("visual_blink_count", ""),
                "visual_auto_marker_count": snapshot.get("visual_auto_marker_count", ""),
                "visual_inference_ms": _format_optional_float(snapshot.get("visual_inference_ms")),
                "visual_error": snapshot.get("visual_error", ""),
            }
        )

    def write_metadata(self, payload: Dict) -> None:
        with open(self.session_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def flush(self) -> None:
        for handle in (self._features_handle, self._events_handle, self._markers_handle):
            if handle is not None:
                handle.flush()

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        for attr in ("_features_handle", "_events_handle", "_markers_handle"):
            handle = getattr(self, attr)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)


def _format_optional_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.9f}"


def _format_optional_sequence(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ";".join(f"{float(item):.9f}" for item in value)
