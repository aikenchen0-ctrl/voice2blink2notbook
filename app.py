import platform
import queue
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hp_acoustic_wave.audio_devices import format_audio_device_selection, resolve_audio_device_selection
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig, build_blink_detector
from hp_acoustic_wave.config import AppConfig, DetectorConfig
from hp_acoustic_wave.detector import AdaptiveWaveDetector
from hp_acoustic_wave.dsp import (
    ChunkFeature,
    FmcwFeature,
    FmcwStreamProcessor,
    extract_chunk_feature,
    fmcw_chirp_period,
    generate_tone,
    phase_difference_trajectory,
    phase_difference_to_distance_meters,
    periodic_signal_chunk,
)
from hp_acoustic_wave.fmcw_blink import decode_phase_matrix_vote, decode_phase_matrix_vote_with_evidence
from hp_acoustic_wave.primary_blink_peak import PrimaryBlinkPeakConfig, PrimaryBlinkPeakGate
from hp_acoustic_wave.session_io import SessionWriter, create_session_dir
from hp_acoustic_wave.ui_state import detection_status
from hp_acoustic_wave.visual_blink_detector import VisualBlinkDetector, VisualBlinkResult


_CJK_FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
)


class _CameraFrameReader:
    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame = None
        self._latest_ok = False
        self._read_count = 0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="camera-frame-reader", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            ok, frame = self.camera.read()
            with self._lock:
                self._latest_ok = bool(ok)
                if ok:
                    self._latest_frame = frame
                    self._read_count += 1
            if not ok:
                time.sleep(0.02)

    def latest(self):
        with self._lock:
            if not self._latest_ok or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class _TerminalKeyReader:
    def __init__(self, on_key: Callable[[str], None]):
        self.on_key = on_key
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        if not sys.stdin or not sys.stdin.isatty():
            return False
        self._thread = threading.Thread(target=self._run, name="terminal-key-reader", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self) -> None:
        if platform.system() == "Windows":
            self._run_windows()
        else:
            self._run_posix()

    def _run_windows(self) -> None:
        try:
            import msvcrt
        except ImportError:
            return
        while not self._stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                self.on_key(key)
            else:
                time.sleep(0.03)

    def _run_posix(self) -> None:
        try:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            previous_attrs = termios.tcgetattr(fd)
        except Exception:
            return
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    key = sys.stdin.read(1)
                    if key:
                        self.on_key(key)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, previous_attrs)
            except Exception:
                pass


class RealtimeHandWaveApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_queue = queue.Queue(maxsize=64)
        self.terminal_key_queue = queue.Queue(maxsize=64)
        self.playback_sample_index = 0
        self.previous_feature: Optional[ChunkFeature] = None
        self.fmcw_processor = FmcwStreamProcessor(config.fmcw, config.audio.sample_rate)
        self.fmcw_period = fmcw_chirp_period(
            config.fmcw,
            config.audio.sample_rate,
            config.audio.output_amplitude * float(config.fmcw.primary_blink_chirp_ratio),
        )
        self.latest_fmcw_feature: Optional[FmcwFeature] = None
        self.fmcw_sync_buffer = np.asarray([], dtype=np.float32)
        self.fmcw_sync_block_count = 0
        self.fmcw_sync_lag_samples: Optional[int] = None
        self.fmcw_sync_confidence = 0.0
        self.fmcw_primary_blink_detector = (
            build_blink_detector(BlinkDetectionConfig(**config.blink.__dict__))
            if config.mode == "fmcw" and config.fmcw.primary_blink_enabled
            else None
        )
        self.fmcw_primary_blink_peak_gate = (
            PrimaryBlinkPeakGate(
                PrimaryBlinkPeakConfig(
                    min_score=config.fmcw.primary_blink_peak_min_score,
                    min_ratio=config.fmcw.primary_blink_peak_min_ratio,
                    max_score=config.fmcw.primary_blink_peak_max_score,
                    refractory_s=config.fmcw.primary_blink_peak_refractory_s,
                    startup_ignore_s=config.blink.startup_ignore_s,
                )
            )
            if config.mode == "fmcw"
            and config.fmcw.primary_blink_enabled
            and config.fmcw.primary_blink_peak_enabled
            else None
        )
        self.fmcw_primary_previous_feature: Optional[ChunkFeature] = None
        self.latest_fmcw_primary_blink_score = 0.0
        self.latest_fmcw_primary_blink_threshold = config.blink.min_score
        self.latest_fmcw_primary_blink_baseline = 0.0
        self.latest_fmcw_primary_blink_mad = 0.0
        self.latest_fmcw_primary_detector_event_id = 0
        self.latest_fmcw_primary_detector_is_event = False
        self.latest_fmcw_primary_detector_method = ""
        self.latest_fmcw_primary_peak_event_id = 0
        self.latest_fmcw_primary_peak_is_event = False
        self.latest_fmcw_primary_peak_score = 0.0
        self.latest_fmcw_primary_peak_threshold = 0.0
        self.latest_fmcw_primary_peak_ratio = 0.0
        self.latest_fmcw_primary_blink_event_id = 0
        self.latest_fmcw_primary_blink_is_event = False
        self.latest_fmcw_primary_blink_method = ""
        self.latest_fmcw_primary_blink_metrics: dict[str, float] = {}
        self.detector = self._build_detector()
        self.energy_history = deque(maxlen=240)
        self.threshold_history = deque(maxlen=240)
        self.fmcw_track_delta_history = deque(maxlen=240)
        self.fmcw_blink_vote_evidence_history = deque(maxlen=240)
        self.fmcw_blink_trajectory_history = deque(maxlen=240)
        self.fmcw_fixed_trajectory_history = deque(maxlen=240)
        self.fmcw_primary_blink_score_history = deque(maxlen=240)
        self.fmcw_primary_blink_threshold_history = deque(maxlen=240)
        self.fmcw_track_history = [deque(maxlen=240) for _ in range(max(0, config.fmcw.track_count))]
        # confirm_history 保存实时窗口里的投票摘要；它不是离线重算，而是连续 UI/事件流的状态。
        self.fmcw_confirm_history = deque(maxlen=max(4, int(config.fmcw.phase_window_size)))
        # confirm_phase_history 保存完整相位点，用于候选窗口内重算论文式 45 轨迹投票。
        self.fmcw_confirm_phase_history = deque(maxlen=max(4, int(config.fmcw.phase_window_size)))
        self.session_dir: Optional[Path] = None
        self.writer: Optional[SessionWriter] = None
        self.manual_marker_count = 0
        self.manual_blink_marker_count = 0
        self.manual_large_motion_marker_count = 0
        self.latest_event_id = 0
        self.latest_time_s = 0.0
        self.last_detection_time_s: Optional[float] = None
        self.last_detection_display_time_s: Optional[float] = None
        self.last_detection_energy = 0.0
        self.last_detection_method = "wave"
        self.last_detection_score = 0.0
        self.latest_score = 0.0
        self.latest_threshold = 0.0
        self.latest_detector_method = "wave"
        self.latest_fmcw_pattern = ""
        self.latest_fmcw_vote_confidence = 0.0
        self.latest_fmcw_vote_score = 0
        self.latest_fmcw_blink_vote_evidence = 0.0
        self.latest_fmcw_blink_trajectory_value = 0.0
        self.latest_fmcw_blink_trajectory_pattern = ""
        self.latest_fmcw_blink_trajectory_pair = ""
        self.latest_fmcw_blink_trajectory_criterion = ""
        self.latest_fmcw_fixed_trajectory_value = 0.0
        self.latest_fmcw_fixed_trajectory_phase_rad = 0.0
        self.latest_fmcw_fixed_trajectory_distance_mm = 0.0
        self.latest_fmcw_fixed_trajectory_pair = ""
        self.latest_fmcw_group_winners: tuple[str, ...] = tuple()
        self.latest_fmcw_candidate_count = 0
        self.fmcw_candidate_detector = AdaptiveWaveDetector(
            DetectorConfig(
                history_size=config.fmcw.candidate_history_size,
                min_history=config.fmcw.candidate_min_history,
                threshold_k=config.fmcw.candidate_threshold_k,
                min_energy=config.fmcw.candidate_min_score,
                refractory_s=config.fmcw.candidate_refractory_s,
                baseline_freeze_s=config.fmcw.candidate_baseline_freeze_s,
                detection_hold_s=config.detector.detection_hold_s,
                release_ratio=config.fmcw.candidate_release_ratio,
                startup_ignore_s=config.fmcw.candidate_startup_ignore_s,
            )
        )
        self.latest_fmcw_candidate_score = 0.0
        self.latest_fmcw_candidate_baseline = 0.0
        self.latest_fmcw_candidate_mad = 0.0
        self.latest_fmcw_candidate_threshold = config.fmcw.candidate_min_score
        self.latest_fmcw_candidate_is_event = False
        self.latest_fmcw_candidate_event_id = 0
        # candidate 是疑似触发；pending 表示正在等待 confirm_window_s 收集足够投票信息。
        self.pending_fmcw_candidate_id = 0
        self.pending_fmcw_candidate_time_s: Optional[float] = None
        self.pending_fmcw_candidate_source = ""
        self.latest_fmcw_vote_candidate_event_id = 0
        self.last_fmcw_vote_candidate_time_s = -1e9
        # confirm_state 的含义：idle/pending/confirmed_blink/suppressed_motion/rejected_vote/rejected_low_motion。
        self.latest_fmcw_confirm_state = "idle"
        self.latest_fmcw_confirm_is_event = False
        self.latest_fmcw_confirm_event_id = 0
        self.latest_fmcw_confirm_max_delta_rms = 0.0
        self.latest_fmcw_confirm_high_delta_duration_s = 0.0
        self.latest_fmcw_confirm_pattern = ""
        self.latest_fmcw_confirm_confidence = 0.0
        self.latest_fmcw_confirm_vote_score = 0
        self.latest_fmcw_confirm_pattern_rows = 0
        self.latest_fmcw_confirm_pattern_stability = 0.0
        self.latest_fmcw_confirm_window_pattern = ""
        self.latest_fmcw_confirm_window_confidence = 0.0
        self.latest_fmcw_confirm_window_vote_score = 0
        self.latest_fmcw_confirm_window_candidate_count = 0
        self.latest_fmcw_confirm_window_group_winners: tuple[str, ...] = tuple()
        self.latest_fmcw_final_pattern = ""
        self.latest_fmcw_final_blink_event_id = 0
        self.latest_fmcw_suppressed_event_id = 0
        self.last_fmcw_vote_period_index = -1_000_000
        self.last_fmcw_confirm_window_vote_period_index = -1_000_000
        self.running = False
        self.first_audio_time_monotonic: Optional[float] = None
        self.video_writer = None
        self.camera = None
        self.camera_reader: Optional[_CameraFrameReader] = None
        self.terminal_key_reader: Optional[_TerminalKeyReader] = None
        self.camera_enabled = False
        self.camera_open_seconds: Optional[float] = None
        self.visual_blink_detector = (
            VisualBlinkDetector(config.visual_blink) if config.visual_blink.enabled else None
        )
        self.latest_visual_blink_result = VisualBlinkResult(
            enabled=bool(config.visual_blink.enabled),
            available=False,
        )
        self.visual_blink_auto_marker_count = 0
        self.visual_blink_sample_count = 0
        self.visual_blink_face_found_count = 0
        self._last_visual_process_time = -1e9
        self.audio_device_selection = None
        self._profile_time: dict[str, float] = {}
        self._profile_count: dict[str, int] = {}
        self._profile_last_print = time.monotonic()
        self._text_image_cache: dict[
            tuple[str, int, tuple[int, int, int]], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._overlay_text_snapshot: Optional[tuple[bool, str, str, str, str]] = None
        self._overlay_text_last_update = 0.0
        self._marker_button_regions: dict[str, tuple[int, int, int, int]] = {}

    def _profile_add(self, name: str, elapsed_s: float) -> None:
        if not self.config.profile_performance:
            return
        self._profile_time[name] = self._profile_time.get(name, 0.0) + float(elapsed_s)

    def _profile_count_add(self, name: str, count: int = 1) -> None:
        if not self.config.profile_performance:
            return
        self._profile_count[name] = self._profile_count.get(name, 0) + int(count)

    def _profile_maybe_print(self) -> None:
        if not self.config.profile_performance:
            return
        now = time.monotonic()
        elapsed = now - self._profile_last_print
        if elapsed < 1.0:
            return

        def ms(name: str) -> float:
            return self._profile_time.get(name, 0.0) * 1000.0

        counts = " ".join(
            f"{name}={value}" for name, value in sorted(self._profile_count.items())
        )
        print(
            "[profile] "
            f"dt={elapsed:.2f}s q={self.audio_queue.qsize()} "
            f"{counts} "
            f"audio_ms={ms('audio_queue'):.1f} "
            f"audio_write_ms={ms('audio_write'):.1f} "
            f"primary_ms={ms('fmcw_primary'):.1f} "
            f"fmcw_block_ms={ms('fmcw_block'):.1f} "
            f"vote_ms={ms('fmcw_vote'):.1f} "
            f"confirm_ms={ms('fmcw_confirm'):.1f} "
            f"confirm_vote_ms={ms('fmcw_confirm_vote'):.1f} "
            f"feature_write_ms={ms('feature_write'):.1f} "
            f"camera_ms={ms('camera_read'):.1f} "
            f"visual_ms={ms('visual_blink'):.1f} "
            f"draw_ms={ms('draw'):.1f} "
            f"text_ms={ms('draw_text'):.1f} "
            f"tracks_ms={ms('draw_tracks'):.1f} "
            f"score_ms={ms('draw_score'):.1f} "
            f"video_ms={ms('video_write'):.1f} "
            f"imshow_ms={ms('imshow_wait'):.1f}",
            flush=True,
        )
        self._profile_time.clear()
        self._profile_count.clear()
        self._profile_last_print = now

    def _build_detector(self):
        if self.config.mode == "blink":
            blink_config = BlinkDetectionConfig(**self.config.blink.__dict__)
            return build_blink_detector(blink_config)
        if self.config.mode == "fmcw":
            return None
        return AdaptiveWaveDetector(self.config.detector)

    def _audio_callback(self, indata, outdata, frames, callback_time, status):
        start_sample = self.playback_sample_index
        if self.config.mode == "fmcw":
            signal = periodic_signal_chunk(self.fmcw_period, start_sample, frames)
            if self.config.fmcw.primary_blink_enabled and self.config.fmcw.primary_blink_tone_ratio > 0.0:
                signal = signal + generate_tone(
                    num_samples=frames,
                    sample_rate=self.config.audio.sample_rate,
                    frequency_hz=self.config.audio.tone_hz,
                    start_sample=start_sample,
                    amplitude=self.config.audio.output_amplitude
                    * float(self.config.fmcw.primary_blink_tone_ratio),
                )
        else:
            signal = generate_tone(
                num_samples=frames,
                sample_rate=self.config.audio.sample_rate,
                frequency_hz=self.config.audio.tone_hz,
                start_sample=start_sample,
                amplitude=self.config.audio.output_amplitude,
            )
        outdata[:, 0] = signal
        recorded = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait((start_sample, recorded, str(status)))
        except queue.Full:
            pass
        self.playback_sample_index += frames

    def _open_camera(self, cv2):
        if not self.config.camera.enabled:
            return

        started_at = time.monotonic()
        system = platform.system()
        if system == "Darwin":
            backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
        elif system == "Windows":
            backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        else:
            backends = (cv2.CAP_ANY,)
        camera = None
        for attempt in range(4):
            for backend in backends:
                candidate = cv2.VideoCapture(self.config.camera.index, backend)
                if candidate.isOpened():
                    camera = candidate
                    break
                candidate.release()
            if camera is not None:
                break
            time.sleep(0.75)

        self.camera_open_seconds = time.monotonic() - started_at
        if camera is None:
            return
        if not camera.isOpened():
            camera.release()
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        self.camera = camera
        self.camera_reader = _CameraFrameReader(camera)
        self.camera_reader.start()
        self.camera_enabled = True

    def _open_video_writer(self, cv2):
        if not self.config.record_video:
            return
        if not self.camera_enabled or self.session_dir is None:
            return
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.camera.width
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.camera.height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = str(self.session_dir / "camera.mp4")
        writer = cv2.VideoWriter(
            path,
            fourcc,
            self.config.camera.fps,
            (width, height + self._overlay_plot_height()),
        )
        if writer.isOpened():
            self.video_writer = writer
        else:
            writer.release()

    def _process_audio_queue(self):
        queue_started = time.monotonic()
        processed = 0
        while True:
            try:
                start_sample, samples, status_text = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if self.first_audio_time_monotonic is None:
                self.first_audio_time_monotonic = time.monotonic()
            if self.writer is not None:
                write_started = time.monotonic()
                self.writer.write_audio(samples)
                self._profile_add("audio_write", time.monotonic() - write_started)
            if self.config.mode == "fmcw":
                primary_started = time.monotonic()
                self._process_fmcw_primary_blink_samples(start_sample, samples)
                self._profile_add("fmcw_primary", time.monotonic() - primary_started)
                self._process_fmcw_samples(start_sample, samples)
                processed += 1
                continue
            feature = extract_chunk_feature(
                samples=samples,
                sample_rate=self.config.audio.sample_rate,
                tone_hz=self.config.audio.tone_hz,
                start_sample=start_sample,
                previous=self.previous_feature,
            )
            self.previous_feature = feature
            self.latest_time_s = feature.time_s
            detection = self._update_detector(feature)
            plot_value = self.latest_score if self.config.mode == "blink" else feature.motion_energy
            self.energy_history.append(plot_value)
            self.threshold_history.append(self.latest_threshold)
            if detection.is_event:
                self.latest_event_id = detection.event_id
                self.last_detection_time_s = feature.time_s
                self.last_detection_display_time_s = time.monotonic()
                self.last_detection_energy = feature.motion_energy
                self.last_detection_score = self.latest_score
                self.last_detection_method = self.latest_detector_method
                if self.writer is not None:
                    self.writer.write_event(
                        event_id=detection.event_id,
                        time_s=feature.time_s,
                        motion_energy=feature.motion_energy,
                        threshold=self.latest_threshold,
                        label="blink_candidate" if self.config.mode == "blink" else "wave",
                        method=self.latest_detector_method,
                        score=self.latest_score,
                    )
            if self.writer is not None:
                write_started = time.monotonic()
                self.writer.write_feature(self._feature_row(feature, detection))
                self._profile_add("feature_write", time.monotonic() - write_started)
            processed += 1
        if processed and self.writer is not None:
            self.writer.flush()
        if processed:
            self._profile_count_add("audio_blocks", processed)
        self._profile_add("audio_queue", time.monotonic() - queue_started)

    def _process_fmcw_samples(self, start_sample: int, samples: np.ndarray):
        if not self._fmcw_sync_ready(samples):
            return
        block_started = time.monotonic()
        features = self.fmcw_processor.process_block(samples, start_sample)
        self._profile_add("fmcw_block", time.monotonic() - block_started)
        self._profile_count_add("fmcw_features", len(features))
        for feature in features:
            self.latest_fmcw_feature = feature
            self.latest_time_s = feature.time_s
            self.latest_score = float(feature.track_delta_rms)
            self.latest_threshold = self.latest_fmcw_candidate_threshold
            self.latest_detector_method = "fmcw"
            # 实时链路分三步：candidate 高召回 -> 连续投票摘要 -> confirm/suppress 最终事件。
            candidate = self._update_fmcw_candidate_detector(feature)
            vote_started = time.monotonic()
            self._update_fmcw_vote(feature.period_index, feature.track_delta_rms)
            self._update_fmcw_fixed_trajectory(feature)
            self._profile_add("fmcw_vote", time.monotonic() - vote_started)
            self._append_fmcw_confirm_observation(feature)
            self.energy_history.append(self.latest_fmcw_candidate_score)
            self.threshold_history.append(self.latest_fmcw_candidate_threshold)
            self.fmcw_track_delta_history.append(float(feature.track_delta_rms))
            self.fmcw_blink_vote_evidence_history.append(float(self.latest_fmcw_blink_vote_evidence))
            self.fmcw_blink_trajectory_history.append(float(self.latest_fmcw_blink_trajectory_value))
            self.fmcw_fixed_trajectory_history.append(float(self.latest_fmcw_fixed_trajectory_value))
            self.fmcw_primary_blink_score_history.append(float(self.latest_fmcw_primary_blink_score))
            self.fmcw_primary_blink_threshold_history.append(float(self.latest_fmcw_primary_blink_threshold))
            if candidate.is_event:
                self._handle_fmcw_candidate_event(feature, candidate)
            elif self._fmcw_vote_candidate_ready(feature.time_s):
                self._handle_fmcw_vote_candidate_event(feature)
            confirm_started = time.monotonic()
            confirmed_event = self._update_fmcw_confirm_state(feature.time_s)
            self._profile_add("fmcw_confirm", time.monotonic() - confirm_started)
            if confirmed_event is not None and self.writer is not None:
                event_id, label, score = confirmed_event
                self.writer.write_event(
                    event_id=event_id,
                    time_s=feature.time_s,
                    motion_energy=score,
                    threshold=self.latest_fmcw_confirm_high_delta_duration_s,
                    label=label,
                    method=(
                        f"fmcw_confirm:{self.latest_fmcw_final_pattern}"
                        if label == "fmcw_confirmed_blink" and self.latest_fmcw_final_pattern
                        else "fmcw_confirm"
                    ),
                    score=score,
                )
            for index, value in enumerate(feature.track_values):
                if index < len(self.fmcw_track_history):
                    self.fmcw_track_history[index].append(value)
            if self.writer is not None:
                write_started = time.monotonic()
                self.writer.write_feature(self._fmcw_feature_row(feature))
                self._profile_add("feature_write", time.monotonic() - write_started)

    def _process_fmcw_primary_blink_samples(self, start_sample: int, samples: np.ndarray) -> None:
        if self.fmcw_primary_blink_detector is None:
            return

        feature = extract_chunk_feature(
            samples=samples,
            sample_rate=self.config.audio.sample_rate,
            tone_hz=self.config.audio.tone_hz,
            start_sample=start_sample,
            previous=self.fmcw_primary_previous_feature,
        )
        self.fmcw_primary_previous_feature = feature
        detection = self.fmcw_primary_blink_detector.update(feature)
        self.latest_fmcw_primary_blink_score = float(detection.score)
        self.latest_fmcw_primary_blink_threshold = float(detection.threshold)
        self.latest_fmcw_primary_blink_baseline = float(detection.baseline)
        self.latest_fmcw_primary_blink_mad = float(detection.mad)
        self.latest_fmcw_primary_detector_event_id = int(detection.event_id)
        self.latest_fmcw_primary_detector_is_event = bool(detection.is_event)
        self.latest_fmcw_primary_detector_method = str(detection.method)
        self.latest_fmcw_primary_blink_event_id = int(detection.event_id)
        self.latest_fmcw_primary_blink_is_event = bool(detection.is_event)
        self.latest_fmcw_primary_blink_method = detection.method
        self.latest_fmcw_primary_blink_metrics = dict(getattr(detection, "metrics", {}) or {})
        peak_result = None
        if self.fmcw_primary_blink_peak_gate is not None:
            peak_result = self.fmcw_primary_blink_peak_gate.update(
                feature.time_s,
                detection.score,
                detection.threshold,
            )
            self.latest_fmcw_primary_peak_event_id = int(peak_result.event_id)
            self.latest_fmcw_primary_peak_is_event = bool(peak_result.is_event)
            self.latest_fmcw_primary_peak_score = float(peak_result.score)
            self.latest_fmcw_primary_peak_threshold = float(peak_result.threshold)
            self.latest_fmcw_primary_peak_ratio = float(peak_result.ratio)
            self.latest_fmcw_primary_blink_event_id = int(peak_result.event_id)
            self.latest_fmcw_primary_blink_is_event = bool(peak_result.is_event)
        else:
            self.latest_fmcw_primary_peak_event_id = 0
            self.latest_fmcw_primary_peak_is_event = False
            self.latest_fmcw_primary_peak_score = 0.0
            self.latest_fmcw_primary_peak_threshold = 0.0
            self.latest_fmcw_primary_peak_ratio = 0.0
        is_event = bool(peak_result.is_event if peak_result is not None else detection.is_event)
        if not is_event:
            return

        event_id = int(peak_result.event_id if peak_result is not None else detection.event_id)
        event_time_s = float(peak_result.time_s if peak_result is not None else feature.time_s)
        event_score = float(peak_result.score if peak_result is not None else detection.score)
        event_threshold = float(peak_result.threshold if peak_result is not None else detection.threshold)
        event_method = str(peak_result.method if peak_result is not None else detection.method)
        self.latest_event_id = event_id
        self.last_detection_time_s = event_time_s
        self.last_detection_display_time_s = time.monotonic()
        self.last_detection_energy = float(feature.motion_energy)
        self.last_detection_score = event_score
        self.last_detection_method = "fmcw_primary_blink"
        if self.writer is not None:
            self.writer.write_event(
                event_id=event_id,
                time_s=event_time_s,
                motion_energy=feature.motion_energy,
                threshold=event_threshold,
                label="blink_candidate",
                method=f"fmcw_primary_{event_method}",
                score=event_score,
            )
        if self.pending_fmcw_candidate_time_s is None:
            self.pending_fmcw_candidate_id = event_id
            self.pending_fmcw_candidate_time_s = event_time_s
            self.pending_fmcw_candidate_source = "primary_blink"
            self.latest_fmcw_confirm_state = "pending"
            self.latest_fmcw_confirm_is_event = False

    def _update_detector(self, feature: ChunkFeature):
        if self.config.mode == "blink":
            detection = self.detector.update(feature)
            self.latest_score = float(detection.score)
            self.latest_threshold = float(detection.threshold)
            self.latest_detector_method = detection.method
            return detection

        detection = self.detector.update(feature.time_s, feature.motion_energy)
        self.latest_score = float(feature.motion_energy)
        self.latest_threshold = float(detection.threshold)
        self.latest_detector_method = "wave"
        return detection

    def _fmcw_feature_row(self, feature: FmcwFeature):
        tracks = list(feature.track_values)
        row = {
            "time_s": f"{feature.time_s:.6f}",
            "sample_index": feature.sample_index,
            "i": "",
            "q": "",
            "amplitude": "",
            "amplitude_delta": "",
            "phase": "",
            "phase_delta": "",
            "motion_energy": f"{feature.track_delta_rms:.9f}",
            "rms": f"{feature.rms:.9f}",
            "peak_abs": f"{feature.peak_abs:.9f}",
            "fmcw_period_index": feature.period_index,
            "fmcw_phase_point_count": feature.phase_point_count,
            "fmcw_phase_points": _format_sequence(feature.phase_points)
            if self.config.fmcw.log_phase_points
            else "",
            "fmcw_sync_lag_samples": "" if self.fmcw_sync_lag_samples is None else self.fmcw_sync_lag_samples,
            "fmcw_sync_confidence": f"{self.fmcw_sync_confidence:.9f}",
            "fmcw_track_0": _format_track(tracks, 0),
            "fmcw_track_1": _format_track(tracks, 1),
            "fmcw_track_2": _format_track(tracks, 2),
            "fmcw_track_3": _format_track(tracks, 3),
            "fmcw_track_4": _format_track(tracks, 4),
            "fmcw_track_delta_rms": f"{feature.track_delta_rms:.9f}",
            "fmcw_phase_std": f"{feature.phase_std:.9f}",
            "fmcw_pairs": ";".join(f"{a}:{b}" for a, b in feature.pairs),
            "fmcw_pattern": self.latest_fmcw_pattern,
            "fmcw_vote_confidence": f"{self.latest_fmcw_vote_confidence:.9f}",
            "fmcw_vote_score": self.latest_fmcw_vote_score,
            "fmcw_blink_vote_evidence": f"{self.latest_fmcw_blink_vote_evidence:.9f}",
            "fmcw_blink_trajectory_value": f"{self.latest_fmcw_blink_trajectory_value:.9f}",
            "fmcw_blink_trajectory_pattern": self.latest_fmcw_blink_trajectory_pattern,
            "fmcw_blink_trajectory_pair": self.latest_fmcw_blink_trajectory_pair,
            "fmcw_blink_trajectory_criterion": self.latest_fmcw_blink_trajectory_criterion,
            "fmcw_fixed_trajectory_value": f"{self.latest_fmcw_fixed_trajectory_value:.9f}",
            "fmcw_fixed_trajectory_phase_rad": f"{self.latest_fmcw_fixed_trajectory_phase_rad:.9f}",
            "fmcw_fixed_trajectory_distance_mm": f"{self.latest_fmcw_fixed_trajectory_distance_mm:.9f}",
            "fmcw_fixed_trajectory_pair": self.latest_fmcw_fixed_trajectory_pair,
            "fmcw_group_winners": "|".join(self.latest_fmcw_group_winners),
            "fmcw_candidate_count": self.latest_fmcw_candidate_count,
            "fmcw_candidate_score": f"{self.latest_fmcw_candidate_score:.9f}",
            "fmcw_candidate_baseline": f"{self.latest_fmcw_candidate_baseline:.9f}",
            "fmcw_candidate_mad": f"{self.latest_fmcw_candidate_mad:.9f}",
            "fmcw_candidate_threshold": f"{self.latest_fmcw_candidate_threshold:.9f}",
            "fmcw_candidate_is_event": int(self.latest_fmcw_candidate_is_event),
            "fmcw_candidate_event_id": self.latest_fmcw_candidate_event_id,
            "fmcw_confirm_state": self.latest_fmcw_confirm_state,
            "fmcw_confirm_is_event": int(self.latest_fmcw_confirm_is_event),
            "fmcw_confirm_event_id": self.latest_fmcw_confirm_event_id,
            "fmcw_confirm_max_delta_rms": f"{self.latest_fmcw_confirm_max_delta_rms:.9f}",
            "fmcw_confirm_high_delta_duration_s": f"{self.latest_fmcw_confirm_high_delta_duration_s:.9f}",
            "fmcw_confirm_pattern": self.latest_fmcw_confirm_pattern,
            "fmcw_confirm_confidence": f"{self.latest_fmcw_confirm_confidence:.9f}",
            "fmcw_confirm_vote_score": self.latest_fmcw_confirm_vote_score,
            "fmcw_confirm_pattern_rows": self.latest_fmcw_confirm_pattern_rows,
            "fmcw_confirm_pattern_stability": f"{self.latest_fmcw_confirm_pattern_stability:.9f}",
            "fmcw_confirm_window_pattern": self.latest_fmcw_confirm_window_pattern,
            "fmcw_confirm_window_confidence": f"{self.latest_fmcw_confirm_window_confidence:.9f}",
            "fmcw_confirm_window_vote_score": self.latest_fmcw_confirm_window_vote_score,
            "fmcw_confirm_window_candidate_count": self.latest_fmcw_confirm_window_candidate_count,
            "fmcw_confirm_window_group_winners": "|".join(self.latest_fmcw_confirm_window_group_winners),
            "fmcw_final_pattern": self.latest_fmcw_final_pattern,
            "baseline": f"{self.latest_fmcw_candidate_baseline:.9f}",
            "mad": f"{self.latest_fmcw_candidate_mad:.9f}",
            "threshold": f"{self.latest_fmcw_candidate_threshold:.9f}",
            "detector_method": self.latest_fmcw_primary_blink_method or "fmcw",
            "blink_score": f"{self.latest_fmcw_primary_blink_score:.9f}"
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_threshold": f"{self.latest_fmcw_primary_blink_threshold:.9f}"
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_baseline": f"{self.latest_fmcw_primary_blink_baseline:.9f}"
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_mad": f"{self.latest_fmcw_primary_blink_mad:.9f}"
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_detector_event_id": self.latest_fmcw_primary_detector_event_id
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_detector_is_event": int(self.latest_fmcw_primary_detector_is_event)
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_detector_method": self.latest_fmcw_primary_detector_method
            if self.fmcw_primary_blink_detector is not None
            else "",
            "blink_peak_event_id": self.latest_fmcw_primary_peak_event_id
            if self.fmcw_primary_blink_peak_gate is not None
            else "",
            "blink_peak_is_event": int(self.latest_fmcw_primary_peak_is_event)
            if self.fmcw_primary_blink_peak_gate is not None
            else "",
            "blink_peak_score": f"{self.latest_fmcw_primary_peak_score:.9f}"
            if self.fmcw_primary_blink_peak_gate is not None
            else "",
            "blink_peak_threshold": f"{self.latest_fmcw_primary_peak_threshold:.9f}"
            if self.fmcw_primary_blink_peak_gate is not None
            else "",
            "blink_peak_ratio": f"{self.latest_fmcw_primary_peak_ratio:.9f}"
            if self.fmcw_primary_blink_peak_gate is not None
            else "",
            "is_event": int(self.latest_fmcw_primary_blink_is_event or self.latest_fmcw_candidate_is_event),
            "event_id": self.latest_event_id,
        }
        row.update(_blink_metric_fields(self.latest_fmcw_primary_blink_metrics))
        return row

    def _update_fmcw_candidate_detector(self, feature: FmcwFeature):
        # 这里沿用旧版 rolling median/MAD 的优势，只产生候选触发，不直接声明眨眼。
        score = self._fmcw_candidate_score(feature)
        detection = self.fmcw_candidate_detector.update(feature.time_s, score)
        self.latest_fmcw_candidate_score = score
        self.latest_fmcw_candidate_baseline = float(detection.baseline)
        self.latest_fmcw_candidate_mad = float(detection.mad)
        self.latest_fmcw_candidate_threshold = float(detection.threshold)
        self.latest_fmcw_candidate_is_event = bool(detection.is_event)
        self.latest_fmcw_candidate_event_id = int(detection.event_id)
        if detection.is_event:
            self.latest_event_id = int(detection.event_id)
        self.latest_threshold = self.latest_fmcw_candidate_threshold
        return detection

    def _fmcw_candidate_score(self, feature: FmcwFeature) -> float:
        source = str(self.config.fmcw.candidate_score_source)
        deltas = tuple(float(value) for value in getattr(feature, "track_deltas", ()) or ())
        if source.startswith("track") and source.endswith("_delta"):
            raw_index = source[len("track") : -len("_delta")]
            if raw_index.isdigit():
                index = int(raw_index)
                if 0 <= index < len(deltas):
                    return abs(deltas[index])
        if source == "max_track_delta" and deltas:
            return max(abs(value) for value in deltas)
        if source == "mean_track_delta" and deltas:
            return float(sum(abs(value) for value in deltas) / float(len(deltas)))
        return float(feature.track_delta_rms)

    def _fmcw_sync_ready(self, samples: np.ndarray) -> bool:
        if not bool(self.config.fmcw.sync_enabled):
            self.fmcw_processor.period_start_offset = 0
            return True
        if self.fmcw_sync_lag_samples is None:
            self._update_fmcw_sync_estimate(samples)
        if self.fmcw_sync_lag_samples is not None:
            self.fmcw_processor.period_start_offset = int(self.fmcw_sync_lag_samples)
            return True
        return False

    def _update_fmcw_sync_estimate(self, samples: np.ndarray) -> None:
        # 论文实验会先等待同步；本机实时 I/O 需要估计录音里的 chirp 周期偏移。
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return

        self.fmcw_sync_block_count += 1
        self.fmcw_sync_buffer = np.concatenate((self.fmcw_sync_buffer, mono))
        max_samples = max(
            int(self.config.audio.chunk_size) * max(1, int(self.config.fmcw.sync_warmup_blocks)),
            int(self.config.fmcw.period_samples) * 4,
        )
        if self.fmcw_sync_buffer.size > max_samples:
            self.fmcw_sync_buffer = self.fmcw_sync_buffer[-max_samples:]

        if self.fmcw_sync_block_count < max(1, int(self.config.fmcw.sync_warmup_blocks)):
            return
        estimate = self._estimate_fmcw_sync(self.fmcw_sync_buffer)
        if estimate is None:
            return
        lag, confidence = estimate
        self.fmcw_sync_confidence = float(confidence)
        if confidence >= float(self.config.fmcw.sync_min_confidence):
            self.fmcw_sync_lag_samples = int(lag)

    def _estimate_fmcw_sync(self, samples: np.ndarray) -> Optional[tuple[int, float]]:
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        period = np.asarray(self.fmcw_period, dtype=np.float64).reshape(-1)
        if values.size < period.size * 2 or period.size == 0:
            return None

        period = period - float(np.mean(period))
        period_norm = float(np.linalg.norm(period))
        if period_norm <= 0.0:
            return None

        scores: list[float] = []
        for lag in range(period.size):
            usable = ((values.size - lag) // period.size) * period.size
            if usable < period.size:
                scores.append(0.0)
                continue
            window = values[lag : lag + usable].reshape(-1, period.size)
            averaged = np.mean(window, axis=0)
            averaged = averaged - float(np.mean(averaged))
            norm = float(np.linalg.norm(averaged)) * period_norm
            scores.append(0.0 if norm <= 0.0 else abs(float(np.dot(averaged, period))) / norm)

        if not scores:
            return None
        best_lag = int(np.argmax(scores))
        confidence = float(scores[best_lag])
        return best_lag, confidence

    def _handle_fmcw_candidate_event(self, feature: FmcwFeature, candidate) -> None:
        self.latest_event_id = int(candidate.event_id)
        self.last_detection_time_s = float(feature.time_s)
        self.last_detection_display_time_s = time.monotonic()
        self.last_detection_energy = self.latest_fmcw_candidate_score
        self.last_detection_score = self.latest_fmcw_candidate_score
        self.last_detection_method = "fmcw_candidate"
        if self.writer is not None:
            self.writer.write_event(
                event_id=candidate.event_id,
                time_s=feature.time_s,
                motion_energy=self.latest_fmcw_candidate_score,
                threshold=self.latest_fmcw_candidate_threshold,
                label="fmcw_candidate",
                method="fmcw_candidate",
                score=self.latest_fmcw_candidate_score,
            )
        if self.pending_fmcw_candidate_time_s is not None:
            # confirm_window_s 尚未收满时，不允许新 candidate 覆盖旧窗口；否则最终确认会被不断推迟。
            return
        self.pending_fmcw_candidate_id = int(candidate.event_id)
        self.pending_fmcw_candidate_time_s = float(feature.time_s)
        self.pending_fmcw_candidate_source = "delta"
        self.latest_fmcw_confirm_state = "pending"
        self.latest_fmcw_confirm_is_event = False

    def _fmcw_vote_candidate_ready(self, time_s: float) -> bool:
        if not bool(self.config.fmcw.vote_candidate_enabled):
            return False
        if self.pending_fmcw_candidate_time_s is not None:
            return False
        if self.latest_fmcw_pattern not in tuple(self.config.fmcw.confirm_single_blink_patterns):
            return False
        if self.latest_fmcw_vote_confidence < float(self.config.fmcw.vote_candidate_min_confidence):
            return False
        return (float(time_s) - self.last_fmcw_vote_candidate_time_s) >= float(
            self.config.fmcw.vote_candidate_refractory_s
        )

    def _handle_fmcw_vote_candidate_event(self, feature: FmcwFeature) -> None:
        self.latest_fmcw_vote_candidate_event_id += 1
        self.last_fmcw_vote_candidate_time_s = float(feature.time_s)
        self.latest_event_id = int(self.latest_fmcw_vote_candidate_event_id)
        self.last_detection_time_s = float(feature.time_s)
        self.last_detection_display_time_s = time.monotonic()
        self.last_detection_energy = self.latest_fmcw_vote_confidence
        self.last_detection_score = self.latest_fmcw_vote_confidence
        self.last_detection_method = "fmcw_vote_candidate"
        if self.writer is not None:
            self.writer.write_event(
                event_id=self.latest_fmcw_vote_candidate_event_id,
                time_s=feature.time_s,
                motion_energy=self.latest_fmcw_vote_confidence,
                threshold=float(self.config.fmcw.vote_candidate_min_confidence),
                label="fmcw_vote_candidate",
                method=f"fmcw_vote:{self.latest_fmcw_pattern}",
                score=self.latest_fmcw_vote_confidence,
            )
        self.pending_fmcw_candidate_id = int(self.latest_fmcw_vote_candidate_event_id)
        self.pending_fmcw_candidate_time_s = float(feature.time_s)
        self.pending_fmcw_candidate_source = "vote"
        self.latest_fmcw_confirm_state = "pending"
        self.latest_fmcw_confirm_is_event = False

    def _append_fmcw_confirm_observation(self, feature: FmcwFeature) -> None:
        # 每个 chirp 记录当前 vote 摘要，candidate 到期时从这个连续窗口里计算稳定性。
        self.fmcw_confirm_history.append(
            {
                "time_s": float(feature.time_s),
                "delta_rms": float(feature.track_delta_rms),
                "pattern": self.latest_fmcw_pattern,
                "confidence": float(self.latest_fmcw_vote_confidence),
                "vote_score": int(self.latest_fmcw_vote_score),
                "group_winners": "|".join(self.latest_fmcw_group_winners),
            }
        )
        self.fmcw_confirm_phase_history.append(
            {
                "time_s": float(feature.time_s),
                "phase_points": tuple(float(value) for value in feature.phase_points),
            }
        )

    def _update_fmcw_confirm_state(self, time_s: float):
        # 没有 pending candidate 时不做最终判定，避免把背景投票误当 blink。
        if self.pending_fmcw_candidate_time_s is None:
            self.latest_fmcw_confirm_is_event = False
            if self.latest_fmcw_confirm_state not in ("confirmed_blink", "suppressed_motion"):
                self.latest_fmcw_confirm_state = "idle"
            return None

        elapsed = float(time_s) - float(self.pending_fmcw_candidate_time_s)
        if elapsed < float(self.config.fmcw.confirm_window_s):
            # 候选刚发生时先显示 pending，等窗口收满再判定 confirmed/suppressed/rejected。
            self.latest_fmcw_confirm_state = "pending"
            self.latest_fmcw_confirm_is_event = False
            self._refresh_fmcw_confirm_metrics(
                self.pending_fmcw_candidate_time_s,
                time_s,
                current_period_index=None,
                force_window_vote=False,
            )
            return None

        self._refresh_fmcw_confirm_metrics(
            self.pending_fmcw_candidate_time_s,
            time_s,
            current_period_index=None,
            force_window_vote=True,
        )
        large_motion = (
            self.latest_fmcw_confirm_max_delta_rms >= float(self.config.fmcw.confirm_large_motion_delta_rms)
            or self.latest_fmcw_confirm_high_delta_duration_s >= float(
                self.config.fmcw.confirm_large_motion_duration_s
            )
        )
        enough_motion = self.latest_fmcw_confirm_max_delta_rms >= float(self.config.fmcw.confirm_min_delta_rms)
        confirm_pattern = self.latest_fmcw_confirm_window_pattern or self.latest_fmcw_confirm_pattern
        confirm_confidence = max(
            self.latest_fmcw_confirm_window_confidence,
            self.latest_fmcw_confirm_confidence,
        )
        window_vote_support = (
            self.latest_fmcw_confirm_window_pattern in tuple(self.config.fmcw.confirm_single_blink_patterns)
            and self.latest_fmcw_confirm_window_confidence >= float(self.config.fmcw.confirm_vote_min_confidence)
        )
        continuous_vote_support = (
            self.latest_fmcw_confirm_pattern in tuple(self.config.fmcw.confirm_single_blink_patterns)
            and self.latest_fmcw_confirm_confidence >= float(self.config.fmcw.confirm_vote_min_confidence)
            and self.latest_fmcw_confirm_pattern_stability >= float(self.config.fmcw.confirm_vote_min_stability)
        )
        # 45 轨迹窗口 vote 是论文路径；如果它稳定支持 blink，就不要先被幅度门控误压成 large motion。
        vote_support = window_vote_support or continuous_vote_support
        event_score = self.latest_fmcw_confirm_max_delta_rms
        candidate_id = int(self.pending_fmcw_candidate_id)
        candidate_source = str(self.pending_fmcw_candidate_source)
        self.pending_fmcw_candidate_time_s = None
        self.pending_fmcw_candidate_id = 0
        self.pending_fmcw_candidate_source = ""

        if large_motion:
            self.latest_fmcw_confirm_is_event = True
            self.latest_fmcw_final_pattern = ""
            self.latest_fmcw_suppressed_event_id += 1
            self.latest_fmcw_confirm_event_id = self.latest_fmcw_suppressed_event_id
            self.latest_fmcw_confirm_state = "suppressed_motion"
            self.last_detection_method = "fmcw_suppressed_motion"
            return candidate_id, "fmcw_suppressed_motion", event_score
        self.latest_fmcw_confirm_is_event = True
        if candidate_source == "primary_blink":
            # primary_blink 来自旧版单频 rolling median/MAD detector。
            # 在混合链路中它是主候选源；FMCW 窗口此时主要承担“大动作压制”，
            # 不再要求 FMCW 自己也出现足够大的 delta，否则会把轻微眨眼误拒。
            return self._confirm_fmcw_blink(candidate_id, confirm_pattern, event_score, time_s)
        if vote_support:
            return self._confirm_fmcw_blink(candidate_id, confirm_pattern, event_score, time_s)
        if enough_motion and (vote_support or not bool(self.config.fmcw.confirm_require_vote)):
            return self._confirm_fmcw_blink(candidate_id, confirm_pattern, event_score, time_s)

        self.latest_fmcw_confirm_state = "rejected_vote" if enough_motion else "rejected_low_motion"
        self.latest_fmcw_confirm_is_event = False
        return None

    def _confirm_fmcw_blink(
        self,
        candidate_id: int,
        pattern: str,
        event_score: float,
        time_s: float,
    ):
        self.latest_fmcw_final_pattern = pattern
        self.latest_fmcw_final_blink_event_id += 1
        self.latest_fmcw_confirm_event_id = self.latest_fmcw_final_blink_event_id
        self.latest_fmcw_confirm_state = "confirmed_blink"
        self.last_detection_time_s = float(time_s)
        self.last_detection_display_time_s = time.monotonic()
        self.last_detection_energy = event_score
        self.last_detection_score = event_score
        self.last_detection_method = "fmcw_confirmed_blink"
        return int(candidate_id), "fmcw_confirmed_blink", event_score

    def _refresh_fmcw_confirm_metrics(
        self,
        start_time_s: float,
        end_time_s: float,
        current_period_index: Optional[int] = None,
        force_window_vote: bool = False,
    ) -> None:
        # 从 candidate 窗口内汇总运动强度、持续时间和投票稳定性，供 UI/CSV/最终事件共享。
        window_start = float(start_time_s)
        window_end = float(end_time_s)
        points = [
            point
            for point in self.fmcw_confirm_history
            if window_start <= float(point["time_s"]) <= window_end
        ]
        if not points:
            self.latest_fmcw_confirm_max_delta_rms = 0.0
            self.latest_fmcw_confirm_high_delta_duration_s = 0.0
            self.latest_fmcw_confirm_pattern = ""
            self.latest_fmcw_confirm_confidence = 0.0
            self.latest_fmcw_confirm_vote_score = 0
            self.latest_fmcw_confirm_pattern_rows = 0
            self.latest_fmcw_confirm_pattern_stability = 0.0
            self._reset_fmcw_confirm_window_vote()
            return

        deltas = [float(point["delta_rms"]) for point in points]
        self.latest_fmcw_confirm_max_delta_rms = max(deltas)
        self.latest_fmcw_confirm_high_delta_duration_s = _duration_above(
            [float(point["time_s"]) for point in points],
            deltas,
            float(self.config.fmcw.confirm_high_delta_rms),
        )
        pattern_counts = {}
        for point in points:
            pattern = str(point["pattern"])
            if pattern:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        if pattern_counts:
            winner, count = sorted(
                pattern_counts.items(),
                key=lambda item: (-item[1], len(item[0]), item[0]),
            )[0]
            self.latest_fmcw_confirm_pattern = winner
            self.latest_fmcw_confirm_pattern_rows = sum(pattern_counts.values())
            self.latest_fmcw_confirm_pattern_stability = float(count) / float(
                max(1, self.latest_fmcw_confirm_pattern_rows)
            )
        else:
            self.latest_fmcw_confirm_pattern = ""
            self.latest_fmcw_confirm_pattern_rows = 0
            self.latest_fmcw_confirm_pattern_stability = 0.0
        self.latest_fmcw_confirm_confidence = max(float(point["confidence"]) for point in points)
        self.latest_fmcw_confirm_vote_score = max(int(point["vote_score"]) for point in points)
        if current_period_index is None:
            latest = self.latest_fmcw_feature
            current_period_index = latest.period_index if latest is not None else None
        if force_window_vote or self._fmcw_confirm_window_vote_due(current_period_index):
            vote_started = time.monotonic()
            self._refresh_fmcw_confirm_window_vote(window_start, window_end)
            self._profile_add("fmcw_confirm_vote", time.monotonic() - vote_started)

    def _fmcw_confirm_window_vote_due(self, period_index: Optional[int]) -> bool:
        if period_index is None:
            return True
        update_periods = max(1, int(self.config.fmcw.confirm_window_vote_update_periods))
        if period_index - self.last_fmcw_confirm_window_vote_period_index < update_periods:
            return False
        self.last_fmcw_confirm_window_vote_period_index = int(period_index)
        return True

    def _reset_fmcw_confirm_window_vote(self) -> None:
        self.latest_fmcw_confirm_window_pattern = ""
        self.latest_fmcw_confirm_window_confidence = 0.0
        self.latest_fmcw_confirm_window_vote_score = 0
        self.latest_fmcw_confirm_window_candidate_count = 0
        self.latest_fmcw_confirm_window_group_winners = tuple()

    def _refresh_fmcw_confirm_window_vote(self, start_time_s: float, end_time_s: float) -> None:
        phase_rows = [
            tuple(point["phase_points"])
            for point in self.fmcw_confirm_phase_history
            if float(start_time_s) <= float(point["time_s"]) <= float(end_time_s)
            and point.get("phase_points")
        ]
        if not phase_rows:
            self._reset_fmcw_confirm_window_vote()
            return

        row_length = len(phase_rows[0])
        phase_rows = [row for row in phase_rows if len(row) == row_length]
        if not phase_rows:
            self._reset_fmcw_confirm_window_vote()
            return

        decision = decode_phase_matrix_vote(np.asarray(phase_rows, dtype=np.float64), self.config.fmcw)
        self.latest_fmcw_confirm_window_pattern = decision.pattern
        self.latest_fmcw_confirm_window_confidence = float(decision.confidence)
        self.latest_fmcw_confirm_window_vote_score = int(decision.score)
        self.latest_fmcw_confirm_window_candidate_count = int(decision.candidate_count)
        self.latest_fmcw_confirm_window_group_winners = tuple(decision.group_winners)

    def _feature_row(self, feature: ChunkFeature, detection):
        row = {
            "time_s": f"{feature.time_s:.6f}",
            "sample_index": feature.sample_index,
            "i": f"{feature.i_value:.9f}",
            "q": f"{feature.q_value:.9f}",
            "amplitude": f"{feature.amplitude:.9f}",
            "amplitude_delta": f"{feature.amplitude_delta:.9f}",
            "phase": f"{feature.phase:.9f}",
            "phase_delta": f"{feature.phase_delta:.9f}",
            "motion_energy": f"{feature.motion_energy:.9f}",
            "rms": f"{feature.rms:.9f}",
            "peak_abs": f"{feature.peak_abs:.9f}",
            "baseline": f"{detection.baseline:.9f}",
            "mad": f"{detection.mad:.9f}",
            "threshold": f"{detection.threshold:.9f}",
            "detector_method": self.latest_detector_method,
            "blink_score": f"{self.latest_score:.9f}" if self.config.mode == "blink" else "",
            "blink_threshold": f"{self.latest_threshold:.9f}" if self.config.mode == "blink" else "",
            "blink_baseline": f"{detection.baseline:.9f}" if self.config.mode == "blink" else "",
            "blink_mad": f"{detection.mad:.9f}" if self.config.mode == "blink" else "",
            "is_event": int(detection.is_event),
            "event_id": detection.event_id,
        }
        metrics = getattr(detection, "metrics", {}) or {}
        row.update(
            {
                "blinklistener_viewing_amplitude": _format_metric(metrics.get("viewing_amplitude")),
                "blinklistener_viewing_range": _format_metric(metrics.get("viewing_range")),
                "blinklistener_raw_viewing_score": _format_metric(metrics.get("raw_viewing_score")),
                "blinklistener_relative_viewing_score": _format_metric(metrics.get("relative_viewing_score")),
                "blinklistener_center_i": _format_metric(metrics.get("center_i")),
                "blinklistener_center_q": _format_metric(metrics.get("center_q")),
                "twinkle_phase_pair_delta": _format_metric(metrics.get("phase_pair_delta")),
                "twinkle_trajectory_span": _format_metric(metrics.get("trajectory_span")),
                "twinkle_trajectory_rms": _format_metric(
                    metrics.get("acceleration_rms", metrics.get("trajectory_rms"))
                ),
                "twinkle_peak_score": _format_metric(metrics.get("twinkle_peak_score")),
                "twinkle_peak_threshold": _format_metric(metrics.get("twinkle_peak_threshold")),
                "twinkle_peak_motion_energy": _format_metric(metrics.get("twinkle_peak_motion_energy")),
                "twinkle_peak_sign_changes": _format_metric(metrics.get("twinkle_peak_sign_changes")),
                "twinkle_candidate_peak": _format_metric(metrics.get("twinkle_candidate_peak")),
                "twinkle_candidate_accepted": _format_metric(metrics.get("twinkle_candidate_accepted")),
                "twinkle_candidate_local_peak": _format_metric(metrics.get("twinkle_candidate_local_peak")),
                "twinkle_candidate_rising_edge": _format_metric(metrics.get("twinkle_candidate_rising_edge")),
                "twinkle_candidate_event_level": _format_metric(metrics.get("twinkle_candidate_event_level")),
                "twinkle_candidate_active": _format_metric(metrics.get("twinkle_candidate_active")),
                "twinkle_reject_low_score": _format_metric(metrics.get("twinkle_reject_low_score")),
                "twinkle_reject_high_score": _format_metric(metrics.get("twinkle_reject_high_score")),
                "twinkle_reject_low_motion": _format_metric(metrics.get("twinkle_reject_low_motion")),
                "twinkle_reject_large_motion": _format_metric(metrics.get("twinkle_reject_large_motion")),
                "twinkle_reject_few_reversals": _format_metric(metrics.get("twinkle_reject_few_reversals")),
                "twinkle_reject_many_reversals": _format_metric(metrics.get("twinkle_reject_many_reversals")),
                "twinkle_reject_suppressed": _format_metric(metrics.get("twinkle_reject_suppressed")),
                "twinkle_reject_refractory": _format_metric(metrics.get("twinkle_reject_refractory")),
                "twinkle_reject_active": _format_metric(metrics.get("twinkle_reject_active")),
                "twinkle_large_motion_suppressed": _format_metric(metrics.get("twinkle_large_motion_suppressed")),
            }
        )
        return row

    def _update_fmcw_vote(self, period_index: Optional[int] = None, motion_score: Optional[float] = None):
        # 实时版用节流后的滑动窗口投票摘要；它服务 confirm 层，不单独作为 blink 事件。
        if period_index is not None:
            update_periods = max(1, int(self.config.fmcw.vote_update_periods))
            if period_index - self.last_fmcw_vote_period_index < update_periods:
                return
            self.last_fmcw_vote_period_index = int(period_index)

        if motion_score is not None and motion_score < float(self.config.fmcw.vote_min_delta_rms):
            # 低运动时清空 pattern，避免静止噪声累积成“稳定投票”。
            self._clear_fmcw_vote_state()
            return

        phase_matrix = self.fmcw_processor.rolling_phase_matrix()
        minimum_rows = max(
            int(self.config.fmcw.candidate_interval_length),
            int(self.config.fmcw.trajectory_detrend_window),
            16,
        )
        if phase_matrix.shape[0] < minimum_rows:
            self._clear_fmcw_vote_state()
            return

        decision, trajectory_evidence = decode_phase_matrix_vote_with_evidence(
            phase_matrix,
            self.config.fmcw,
            minimum_rows=minimum_rows,
            blink_patterns=tuple(self.config.fmcw.confirm_single_blink_patterns),
        )
        if decision.candidate_count == 0:
            self._clear_fmcw_vote_state()
            return

        self.latest_fmcw_pattern = decision.pattern
        self.latest_fmcw_vote_confidence = decision.confidence
        self.latest_fmcw_vote_score = decision.score
        self.latest_fmcw_group_winners = decision.group_winners
        self.latest_fmcw_candidate_count = int(decision.candidate_count)
        self.latest_fmcw_blink_vote_evidence = self._fmcw_blink_vote_evidence()
        self._set_fmcw_blink_trajectory_evidence(trajectory_evidence)

    def _clear_fmcw_vote_state(self) -> None:
        self.latest_fmcw_pattern = ""
        self.latest_fmcw_vote_confidence = 0.0
        self.latest_fmcw_vote_score = 0
        self.latest_fmcw_blink_vote_evidence = 0.0
        self._set_fmcw_blink_trajectory_evidence(None)
        self.latest_fmcw_group_winners = tuple()
        self.latest_fmcw_candidate_count = 0

    def _set_fmcw_blink_trajectory_evidence(self, evidence) -> None:
        if evidence is None:
            self.latest_fmcw_blink_trajectory_value = 0.0
            self.latest_fmcw_blink_trajectory_pattern = ""
            self.latest_fmcw_blink_trajectory_pair = ""
            self.latest_fmcw_blink_trajectory_criterion = ""
            return
        self.latest_fmcw_blink_trajectory_value = float(evidence.value)
        self.latest_fmcw_blink_trajectory_pattern = str(evidence.pattern)
        self.latest_fmcw_blink_trajectory_pair = f"{evidence.reference_index}:{evidence.target_index}"
        self.latest_fmcw_blink_trajectory_criterion = str(evidence.criterion)

    def _update_fmcw_fixed_trajectory(self, feature: FmcwFeature) -> None:
        # 固定 phase-pair 诊断线：按论文的 phase-difference trajectory 处理同一对点，
        # 不经过候选选择/投票，专门用来判断 FMCW 原始轨迹是否和眨眼同步。
        configured_pair = tuple(getattr(self.config.fmcw, "fixed_trajectory_pair", ()) or ())
        if len(configured_pair) == 2:
            reference_index, target_index = int(configured_pair[0]), int(configured_pair[1])
        elif feature.pairs:
            reference_index, target_index = feature.pairs[0]
        else:
            self.latest_fmcw_fixed_trajectory_value = 0.0
            self.latest_fmcw_fixed_trajectory_phase_rad = 0.0
            self.latest_fmcw_fixed_trajectory_distance_mm = 0.0
            self.latest_fmcw_fixed_trajectory_pair = ""
            return
        phase_matrix = self.fmcw_processor.rolling_phase_matrix()
        minimum_rows = max(
            3,
            int(self.config.fmcw.candidate_interval_length),
            int(self.config.fmcw.trajectory_smoothing_window),
            int(self.config.fmcw.trajectory_detrend_window),
        )
        if phase_matrix.shape[0] < minimum_rows:
            self.latest_fmcw_fixed_trajectory_value = 0.0
            self.latest_fmcw_fixed_trajectory_phase_rad = 0.0
            self.latest_fmcw_fixed_trajectory_distance_mm = 0.0
            self.latest_fmcw_fixed_trajectory_pair = f"{reference_index}:{target_index}"
            return
        try:
            phase_trajectory = phase_difference_trajectory(
                phase_matrix,
                int(reference_index),
                int(target_index),
                smoothing_window=int(self.config.fmcw.trajectory_smoothing_window),
                detrend_window=1,
                normalize=False,
            )
        except (IndexError, ValueError):
            self.latest_fmcw_fixed_trajectory_value = 0.0
            self.latest_fmcw_fixed_trajectory_phase_rad = 0.0
            self.latest_fmcw_fixed_trajectory_distance_mm = 0.0
            self.latest_fmcw_fixed_trajectory_pair = ""
            return
        if phase_trajectory.size:
            relative_phase = phase_trajectory - float(np.median(phase_trajectory))
            distance_mm = (
                phase_difference_to_distance_meters(
                    relative_phase,
                    int(reference_index),
                    int(target_index),
                    self.config.fmcw,
                )
                * 1000.0
            )
            self.latest_fmcw_fixed_trajectory_phase_rad = float(relative_phase[-1])
            self.latest_fmcw_fixed_trajectory_distance_mm = float(distance_mm[-1])
            self.latest_fmcw_fixed_trajectory_value = self.latest_fmcw_fixed_trajectory_distance_mm
        else:
            self.latest_fmcw_fixed_trajectory_phase_rad = 0.0
            self.latest_fmcw_fixed_trajectory_distance_mm = 0.0
            self.latest_fmcw_fixed_trajectory_value = 0.0
        self.latest_fmcw_fixed_trajectory_pair = f"{reference_index}:{target_index}"

    def _fmcw_blink_vote_evidence(self) -> float:
        # 这是 UI/CSV 用的连续证据线，不直接触发事件：论文投票输出 blink pattern，
        # 这里仅把“单次眨眼 pattern 的置信度”转成 0..1 分数。
        if self.latest_fmcw_candidate_count <= 0:
            return 0.0
        if self.latest_fmcw_pattern not in tuple(self.config.fmcw.confirm_single_blink_patterns):
            return 0.0
        return max(0.0, min(1.0, float(self.latest_fmcw_vote_confidence)))

    def _blank_frame(self, cv2):
        frame = np.zeros((self.config.camera.height, self.config.camera.width, 3), dtype=np.uint8)
        self._draw_text(frame, "摄像头不可用 / Camera unavailable", (40, 50), 34, (0, 180, 255))
        return frame

    def _draw_text(
        self,
        canvas: np.ndarray,
        text: str,
        xy: tuple[int, int],
        size: int,
        color_bgr: tuple[int, int, int],
    ) -> None:
        if not text:
            return
        key = (str(text), int(size), tuple(int(value) for value in color_bgr))
        cached = self._text_image_cache.get(key)
        if cached is None:
            cached = _render_text_image(str(text), int(size), key[2])
            if len(self._text_image_cache) > 256:
                self._text_image_cache.clear()
            self._text_image_cache[key] = cached
        image, alpha = cached

        x = max(0, min(int(xy[0]), canvas.shape[1] - 1))
        y = max(0, min(int(xy[1]), canvas.shape[0] - 1))
        right = min(canvas.shape[1], x + image.shape[1])
        bottom = min(canvas.shape[0], y + image.shape[0])
        if right <= x or bottom <= y:
            return
        patch = image[: bottom - y, : right - x]
        patch_alpha = alpha[: bottom - y, : right - x, None].astype(np.float32) / 255.0
        region = canvas[y:bottom, x:right]
        region[:] = (patch.astype(np.float32) * patch_alpha + region.astype(np.float32) * (1.0 - patch_alpha)).astype(
            np.uint8
        )

    def _overlay_texts(self) -> tuple[bool, str, str, str, str]:
        energy = self.energy_history[-1] if self.energy_history else 0.0
        threshold = self.threshold_history[-1] if self.threshold_history else self.config.detector.min_energy
        status = detection_status(
            current_time_s=time.monotonic(),
            last_detection_time_s=self.last_detection_display_time_s,
            hold_s=self.config.detector.detection_hold_s,
            detected_label=self._detected_label(),
        )
        status_label = status.label
        if status.is_detected:
            if self.config.mode == "blink":
                detail = f"#{self.latest_event_id}  眨眼候选 / Blink candidate  S={self.last_detection_score:.4f}"
            elif self.config.mode == "fmcw":
                if self.last_detection_method == "fmcw_confirmed_blink":
                    detail = (
                        f"确认眨眼 / Confirmed blink  #{self.latest_fmcw_confirm_event_id}  "
                        f"M={self.latest_fmcw_confirm_max_delta_rms:.4f} "
                        f"D={self.latest_fmcw_confirm_high_delta_duration_s:.2f}s  "
                        f"win={self.latest_fmcw_confirm_window_pattern or '-'} "
                        f"conf={self.latest_fmcw_confirm_window_confidence:.2f} "
                        f"stab={self.latest_fmcw_confirm_pattern_stability:.2f}"
                    )
                elif self.last_detection_method == "fmcw_suppressed_motion":
                    detail = (
                        f"大动作压制 / Motion suppressed  "
                        f"M={self.latest_fmcw_confirm_max_delta_rms:.4f} "
                        f"D={self.latest_fmcw_confirm_high_delta_duration_s:.2f}s  "
                        f"win={self.latest_fmcw_confirm_window_pattern or '-'} "
                        f"cand=#{self.latest_event_id}"
                    )
                elif self.last_detection_method == "fmcw_primary_blink":
                    detail = (
                        f"主眨眼候选 / Main blink candidate  #{self.latest_fmcw_primary_blink_event_id}  "
                        f"B={self.latest_fmcw_primary_blink_score:.4f}/"
                        f"{self.latest_fmcw_primary_blink_threshold:.4f}  "
                        f"F={self.latest_fmcw_candidate_score:.4f}/"
                        f"{self.latest_fmcw_candidate_threshold:.4f}  "
                        f"confirm={self.latest_fmcw_confirm_state}"
                    )
                elif self.last_detection_method == "fmcw_vote_candidate":
                    detail = (
                        f"投票候选 / Vote candidate  #{self.latest_fmcw_vote_candidate_event_id}  "
                        f"pattern={self.latest_fmcw_pattern or '-'} "
                        f"conf={self.latest_fmcw_vote_confidence:.2f}  "
                        f"F={self.latest_fmcw_candidate_score:.4f}/"
                        f"{self.latest_fmcw_candidate_threshold:.4f}  "
                        f"confirm={self.latest_fmcw_confirm_state}"
                    )
                else:
                    detail = (
                        f"#{self.latest_event_id}  FMCW候选 / FMCW candidate  "
                        f"S={self.latest_fmcw_candidate_score:.4f} T={self.latest_fmcw_candidate_threshold:.4f}  "
                        f"confirm={self.latest_fmcw_confirm_state} win={self.latest_fmcw_confirm_window_pattern or '-'} "
                        f"conf={self.latest_fmcw_confirm_window_confidence:.2f} "
                        f"stab={self.latest_fmcw_confirm_pattern_stability:.2f}"
                    )
            else:
                detail = f"#{self.latest_event_id}  E={self.last_detection_energy:.2f}"
        else:
            if self._visual_blink_needs_face_attention():
                status_label = "调整摄像头 / No face"
            if self.config.mode == "blink":
                detail = f"眨眼检测 / Blink detection  S={energy:.4f}  T={threshold:.4f}"
            elif self.config.mode == "fmcw":
                detail = (
                    f"眨眼检测 / Blink detect  B={self.latest_fmcw_primary_blink_score:.4f}/"
                    f"{self.latest_fmcw_primary_blink_threshold:.4f}  "
                    f"F={energy:.4f}/{threshold:.4f}  "
                    f"confirm={self.latest_fmcw_confirm_state} win={self.latest_fmcw_confirm_window_pattern or '-'} "
                    f"conf={self.latest_fmcw_confirm_window_confidence:.2f} "
                    f"stab={self.latest_fmcw_confirm_pattern_stability:.2f}"
                )
            else:
                detail = f"E={energy:.2f}  T={threshold:.2f}"

        title = f"时间/Time={self.latest_time_s:6.2f}s  自动/Auto={self.latest_event_id}  {self._manual_marker_progress_text()}"
        if self.config.mode == "blink":
            help_text = "键盘/Click: b=B blink  w=W motion  m=M mark   q/Esc=quit"
        elif self.config.mode == "fmcw":
            help_text = "目标/Goal: blink + FMCW candidate/confirm   键盘或点击/Keys or click: B blink  W motion  M mark"
        else:
            help_text = "键盘或点击/Keys or click: M manual   q/Esc=quit"
        return bool(status.is_detected), status_label, detail, title, help_text

    def _visual_blink_needs_face_attention(self) -> bool:
        if not bool(self.config.visual_blink.enabled):
            return False
        result = self.latest_visual_blink_result
        return bool(result.available) and not bool(result.face_found)

    def _manual_marker_progress_text(self) -> str:
        blink_target = int(self.config.collection_target_blinks)
        negative_target = int(self.config.collection_target_negatives)
        if blink_target <= 0 and negative_target <= 0:
            return f"人工/Manual={self.manual_marker_count}  {self._visual_blink_progress_text()}"

        blink_text = (
            f"b眨眼/blink={self.manual_blink_marker_count}/{blink_target}"
            if blink_target > 0
            else f"b眨眼/blink={self.manual_blink_marker_count}"
        )
        negative_text = (
            f"w大动作/motion={self.manual_large_motion_marker_count}/{negative_target}"
            if negative_target > 0
            else f"w大动作/motion={self.manual_large_motion_marker_count}"
        )
        return (
            f"人工/Manual={self.manual_marker_count}  {blink_text}  {negative_text}  "
            f"{self._visual_blink_progress_text()}"
        )

    def _visual_blink_progress_text(self) -> str:
        if not bool(self.config.visual_blink.enabled):
            return "视觉/Visual=off"
        result = self.latest_visual_blink_result
        if not result.available:
            if result.error:
                return "视觉/Visual=error"
            return "视觉/Visual=initializing"
        auto_text = "auto" if bool(self.config.visual_blink.auto_mark_blinks) else "view"
        face_text = "face" if result.face_found else "no-face"
        face_rate = (
            self.visual_blink_face_found_count / self.visual_blink_sample_count
            if self.visual_blink_sample_count > 0
            else 0.0
        )
        return (
            f"视觉/Visual blink={result.blink_count} auto={self.visual_blink_auto_marker_count} "
            f"{face_text} face={face_rate:.0%} {auto_text}"
        )

    def _overlay_text_snapshot_for_frame(self) -> tuple[bool, str, str, str, str]:
        now = time.monotonic()
        interval = 1.0 / max(1.0, float(self.config.ui_text_fps))
        if self._overlay_text_snapshot is None or now - self._overlay_text_last_update >= interval:
            self._overlay_text_snapshot = self._overlay_texts()
            self._overlay_text_last_update = now
        return self._overlay_text_snapshot

    def _draw_overlay(self, cv2, frame):
        height, width = frame.shape[:2]
        plot_h = self._overlay_plot_height()
        canvas = np.zeros((height + plot_h, width, 3), dtype=np.uint8)
        canvas[:height, :width] = frame
        panel_y = height

        is_detected, status_label, detail, title, help_text = self._overlay_text_snapshot_for_frame()
        status_color = (35, 35, 230) if is_detected else (35, 120, 35)
        text_color = (255, 255, 255)
        cv2.rectangle(canvas, (0, 0), (width, 58), status_color, -1)
        text_started = time.monotonic()
        self._draw_text(canvas, status_label, (20, 12), 34, text_color)
        self._draw_text(canvas, detail, (330, 17), 20, text_color)
        self._draw_text(canvas, title, (20, 68), 22, (255, 255, 255))
        self._draw_text(canvas, help_text, (20, 102), 20, (180, 220, 255))
        self._draw_marker_buttons(cv2, canvas, width, height)
        self._profile_add("draw_text", time.monotonic() - text_started)

        values = list(self.energy_history)
        thresholds = list(self.threshold_history)
        if self.config.mode == "fmcw":
            score_started = time.monotonic()
            self._draw_fmcw_score_view(cv2, canvas, panel_y, width, height, plot_h)
            self._profile_add("draw_score", time.monotonic() - score_started)
        elif values:
            max_value = max(max(values), max(thresholds) if thresholds else 0.0, self.config.detector.min_energy)
            max_value = max(max_value, 1e-6)
            left = 20
            right = width - 20
            top = panel_y + 20
            bottom = height + plot_h - 20
            cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
            for series, color in ((values, (0, 255, 0)), (thresholds, (0, 180, 255))):
                if len(series) < 2:
                    continue
                points = []
                for idx, val in enumerate(series):
                    x = int(left + idx * (right - left) / max(1, len(series) - 1))
                    y = int(bottom - min(val / max_value, 1.0) * (bottom - top))
                    points.append((x, y))
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2)
        return canvas

    def _process_visual_blink_frame(self, cv2, frame: np.ndarray) -> np.ndarray:
        if self.visual_blink_detector is None:
            return frame

        now = time.monotonic()
        interval_s = 1.0 / max(1.0, float(self.config.visual_blink.max_fps))
        if now - self._last_visual_process_time >= interval_s:
            self._last_visual_process_time = now
            result = self.visual_blink_detector.process_frame(frame)
            self.latest_visual_blink_result = result
            if result.available:
                self.visual_blink_sample_count += 1
                if result.face_found:
                    self.visual_blink_face_found_count += 1
            self._overlay_text_snapshot = None
            if self.writer is not None:
                self.writer.write_visual_feature(self._visual_feature_row(result))
            if result.is_blink_event and bool(self.config.visual_blink.auto_mark_blinks):
                self.visual_blink_auto_marker_count += 1
                self._write_marker(label="blink", key="v")

        self._draw_visual_blink_frame(cv2, frame, self.latest_visual_blink_result)
        return frame

    def _visual_feature_row(self, result: VisualBlinkResult) -> dict[str, object]:
        return {
            "time_s": f"{self.latest_time_s:.6f}",
            "timestamp_ms": int(result.timestamp_ms),
            "available": int(bool(result.available)),
            "face_found": int(bool(result.face_found)),
            "left_ear": f"{float(result.left_ear):.9f}",
            "right_ear": f"{float(result.right_ear):.9f}",
            "left_closed": int(bool(result.left_closed)),
            "right_closed": int(bool(result.right_closed)),
            "is_blink_event": int(bool(result.is_blink_event)),
            "blink_count": int(result.blink_count),
            "inference_ms": f"{float(result.inference_ms):.9f}",
            "error": result.error,
        }

    def _draw_visual_blink_frame(self, cv2, frame: np.ndarray, result: VisualBlinkResult) -> None:
        if not bool(self.config.visual_blink.enabled):
            return
        height, width = frame.shape[:2]
        x = 16
        y = max(140, min(height - 52, 140))
        if not result.available:
            if result.error:
                text = f"视觉不可用 / Visual unavailable: {result.error}"
            else:
                text = "视觉初始化中 / Visual initializing"
            self._draw_text(frame, text, (x, y), 18, (0, 210, 255))
            return

        left_color = (60, 60, 255) if result.left_closed else (70, 230, 80)
        right_color = (60, 60, 255) if result.right_closed else (70, 230, 80)
        if len(result.left_eye_points) >= 2:
            cv2.polylines(
                frame,
                [np.asarray(result.left_eye_points, dtype=np.int32)],
                True,
                left_color,
                2,
                cv2.LINE_AA,
            )
        if len(result.right_eye_points) >= 2:
            cv2.polylines(
                frame,
                [np.asarray(result.right_eye_points, dtype=np.int32)],
                True,
                right_color,
                2,
                cv2.LINE_AA,
            )
        status = "闭眼/Closed" if result.left_closed and result.right_closed else "睁眼/Open"
        if not result.face_found:
            status = "未见脸/No face"
        auto_text = "开/on" if bool(self.config.visual_blink.auto_mark_blinks) else "关/off"
        color = (60, 60, 255) if result.is_blink_event else (235, 235, 235)
        self._draw_text(
            frame,
            (
                f"视觉/Visual {status}  EAR L/R={result.left_ear:.3f}/{result.right_ear:.3f}  "
                f"blink={result.blink_count}  auto={auto_text}  {result.inference_ms:.1f}ms"
            ),
            (x, y),
            18,
            color,
        )

    def _overlay_plot_height(self) -> int:
        if self.config.mode == "fmcw":
            return 360
        return 160

    def _draw_marker_buttons(self, cv2, canvas: np.ndarray, width: int, height: int) -> None:
        layout = self._marker_button_layout(width, height)
        self._marker_button_regions = {key: region for key, region, _, _ in layout}
        for key, region, label, color in layout:
            left, top, right, bottom = region
            cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
            cv2.rectangle(canvas, (left, top), (right, bottom), (230, 230, 230), 1)
            self._draw_text(canvas, label, (left + 6, top + 7), 15, (255, 255, 255))

    def _marker_button_layout(
        self,
        width: int,
        height: int,
    ) -> tuple[tuple[str, tuple[int, int, int, int], str, tuple[int, int, int]], ...]:
        margin = 10
        gap = 6
        button_count = 3
        button_w = max(64, int((int(width) - margin * 2 - gap * (button_count - 1)) / button_count))
        button_h = 34
        top = max(126, int(height) - button_h - 10)
        labels = (
            ("b", "B blink", (40, 120, 40)),
            ("w", "W motion", (150, 90, 40)),
            ("m", "M mark", (80, 80, 80)),
        )
        regions = []
        left = margin
        for key, label, color in labels:
            right = min(int(width) - margin, left + button_w)
            regions.append((key, (left, top, right, top + button_h), label, color))
            left = right + gap
        return tuple(regions)

    def _handle_marker_click(self, x: int, y: int) -> bool:
        for key, (left, top, right, bottom) in self._marker_button_regions.items():
            if left <= int(x) <= right and top <= int(y) <= bottom:
                self._write_marker_for_key(ord(key))
                self._overlay_text_snapshot = None
                return True
        return False

    def _draw_fmcw_tracks(self, cv2, canvas, panel_y: int, width: int, height: int, plot_h: int):
        left = 20
        right = width - 20
        top = panel_y + 20
        bottom = height + plot_h - 20
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)

        series_list = [list(series) for series in self.fmcw_track_history if len(series) >= 2]
        if not series_list:
            return
        all_values = np.asarray([value for series in series_list for value in series], dtype=np.float64)
        center = float(np.median(all_values))
        span = float(np.percentile(np.abs(all_values - center), 95)) if all_values.size else 1.0
        span = max(span, 1e-3)
        colors = [
            (0, 255, 0),
            (0, 180, 255),
            (255, 160, 60),
            (220, 120, 255),
            (80, 220, 220),
        ]
        mid_y = int((top + bottom) / 2)
        cv2.line(canvas, (left, mid_y), (right, mid_y), (65, 65, 65), 1)
        for index, series in enumerate(series_list):
            points = []
            for idx, value in enumerate(series):
                x = int(left + idx * (right - left) / max(1, len(series) - 1))
                normalized = max(-1.0, min(1.0, (value - center) / span))
                y = int(mid_y - normalized * (bottom - top) * 0.45)
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[index % len(colors)], 2)

    def _draw_fmcw_tracks_strip(self, cv2, canvas, panel_y: int, width: int, height: int, plot_h: int):
        left = 20
        right = width - 20
        top = height + plot_h - 44
        bottom = height + plot_h - 20
        cv2.rectangle(canvas, (left, top), (right, bottom), (55, 55, 55), 1)
        self._draw_fmcw_tracks_panel(cv2, canvas, left, right, top, bottom)

    def _draw_fmcw_tracks_panel(
        self,
        cv2,
        canvas,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> None:
        series_list = [list(series) for series in self.fmcw_track_history if len(series) >= 2]
        if not series_list:
            return
        all_values = np.asarray([value for series in series_list for value in series], dtype=np.float64)
        center = float(np.median(all_values))
        span = float(np.percentile(np.abs(all_values - center), 95)) if all_values.size else 1.0
        span = max(span, 1e-3)
        colors = [
            (70, 150, 70),
            (70, 140, 190),
            (180, 130, 70),
            (160, 90, 180),
            (60, 170, 170),
        ]
        mid_y = int((top + bottom) / 2)
        cv2.line(canvas, (left, mid_y), (right, mid_y), (45, 45, 45), 1)
        for index, series in enumerate(series_list):
            points = []
            for idx, value in enumerate(series):
                x = int(left + idx * (right - left) / max(1, len(series) - 1))
                normalized = max(-1.0, min(1.0, (float(value) - center) / span))
                y = int(mid_y - normalized * (bottom - top) * 0.42)
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[index % len(colors)], 1)

    def _draw_fmcw_score_view(self, cv2, canvas, panel_y: int, width: int, height: int, plot_h: int):
        candidate_values = list(self.energy_history)
        candidate_thresholds = list(self.threshold_history)
        blink_values = list(self.fmcw_primary_blink_score_history)
        blink_thresholds = list(self.fmcw_primary_blink_threshold_history)
        delta_values = list(self.fmcw_track_delta_history)
        vote_evidence_values = list(self.fmcw_blink_vote_evidence_history)
        trajectory_values = list(self.fmcw_blink_trajectory_history)
        fixed_trajectory_values = list(self.fmcw_fixed_trajectory_history)
        if (
            len(candidate_values) < 2
            and len(blink_values) < 2
            and len(vote_evidence_values) < 2
            and len(trajectory_values) < 2
            and len(fixed_trajectory_values) < 2
        ):
            return

        left = 20
        right = width - 20
        panel_top = panel_y + 8
        panel_bottom = height + plot_h - 12
        gap = 7
        row_count = 4
        row_h = max(42, int((panel_bottom - panel_top - gap * (row_count - 1)) / row_count))
        rows = []
        top = panel_top
        for _ in range(row_count):
            rows.append((top, min(panel_bottom, top + row_h)))
            top += row_h + gap

        blink_top, blink_bottom = self._draw_fmcw_plot_panel(
            cv2,
            canvas,
            left,
            right,
            rows[0][0],
            rows[0][1],
            "单频18.5kHz眨眼分数 / 18.5 kHz tone blink score    绿=score  黄绿=threshold",
        )
        blink_scale = _positive_series_scale(
            blink_values,
            blink_thresholds,
            (self.config.blink.min_score,),
        )
        self._draw_scaled_series(
            cv2,
            canvas,
            blink_values,
            left,
            right,
            blink_top,
            blink_bottom,
            blink_scale,
            (0, 255, 0),
            2,
        )
        self._draw_scaled_series(
            cv2,
            canvas,
            blink_thresholds,
            left,
            right,
            blink_top,
            blink_bottom,
            blink_scale,
            (0, 190, 120),
            1,
        )

        candidate_top, candidate_bottom = self._draw_fmcw_plot_panel(
            cv2,
            canvas,
            left,
            right,
            rows[1][0],
            rows[1][1],
            "FMCW候选/大动作 / Candidate & motion    白=candidate  橙=threshold  紫=motion RMS",
        )
        candidate_scale = _positive_series_scale(
            candidate_values,
            candidate_thresholds,
            delta_values,
            (self.config.fmcw.candidate_min_score,),
        )
        self._draw_scaled_series(
            cv2,
            canvas,
            candidate_values,
            left,
            right,
            candidate_top,
            candidate_bottom,
            candidate_scale,
            (255, 255, 255),
            2,
        )
        self._draw_scaled_series(
            cv2,
            canvas,
            candidate_thresholds,
            left,
            right,
            candidate_top,
            candidate_bottom,
            candidate_scale,
            (0, 140, 255),
            1,
        )
        self._draw_scaled_series(
            cv2,
            canvas,
            delta_values,
            left,
            right,
            candidate_top,
            candidate_bottom,
            candidate_scale,
            (180, 160, 255),
            1,
        )

        trajectory_label = (
            "FMCW轨迹/投票 / Trajectory & vote    洋红=fixed mm  蓝=vote trajectory  青=vote evidence"
        )
        trajectory_top, trajectory_bottom = self._draw_fmcw_plot_panel(
            cv2,
            canvas,
            left,
            right,
            rows[2][0],
            rows[2][1],
            trajectory_label,
        )
        cv2.line(
            canvas,
            (left, (trajectory_top + trajectory_bottom) // 2),
            (right, (trajectory_top + trajectory_bottom) // 2),
            (50, 50, 70),
            1,
        )
        if any(abs(float(value)) > 1e-6 for value in fixed_trajectory_values):
            self._draw_centered_series(
                cv2,
                canvas,
                fixed_trajectory_values,
                left,
                right,
                trajectory_top,
                trajectory_bottom,
                _centered_series_scale(fixed_trajectory_values, floor=0.5),
                (255, 0, 255),
                2,
            )
        if any(abs(float(value)) > 1e-6 for value in trajectory_values):
            self._draw_centered_series(
                cv2,
                canvas,
                trajectory_values,
                left,
                right,
                trajectory_top,
                trajectory_bottom,
                _centered_series_scale(trajectory_values, floor=0.05),
                (255, 80, 80),
                1,
            )
        self._draw_scaled_series(
            cv2,
            canvas,
            vote_evidence_values,
            left,
            right,
            trajectory_top,
            trajectory_bottom,
            1.0,
            (255, 255, 0),
            2,
        )

        tracks_started = time.monotonic()
        tracks_top, tracks_bottom = self._draw_fmcw_plot_panel(
            cv2,
            canvas,
            left,
            right,
            rows[3][0],
            rows[3][1],
            "FMCW原始多轨 / Raw phase-difference tracks",
        )
        self._draw_fmcw_tracks_panel(cv2, canvas, left, right, tracks_top, tracks_bottom)
        self._profile_add("draw_tracks", time.monotonic() - tracks_started)

    def _draw_fmcw_plot_panel(
        self,
        cv2,
        canvas,
        left: int,
        right: int,
        top: int,
        bottom: int,
        label: str,
    ) -> tuple[int, int]:
        cv2.rectangle(canvas, (left, top), (right, bottom), (70, 70, 70), 1)
        self._draw_text(canvas, label, (left + 6, top + 3), 15, (210, 230, 255))
        plot_top = min(bottom - 8, top + 23)
        plot_bottom = max(plot_top + 2, bottom - 7)
        cv2.line(canvas, (left, plot_bottom), (right, plot_bottom), (48, 48, 48), 1)
        return plot_top, plot_bottom

    def _draw_scaled_series(
        self,
        cv2,
        canvas,
        series,
        left: int,
        right: int,
        top: int,
        bottom: int,
        max_value: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(series) < 2:
            return
        points = []
        for index, value in enumerate(series):
            x = int(left + index * (right - left) / max(1, len(series) - 1))
            y = int(bottom - min(max(float(value), 0.0) / max_value, 1.0) * (bottom - top))
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness)

    def _draw_centered_series(
        self,
        cv2,
        canvas,
        series,
        left: int,
        right: int,
        top: int,
        bottom: int,
        max_abs: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(series) < 2:
            return
        mid_y = int((top + bottom) / 2)
        half_height = max(1.0, float(bottom - top) * 0.45)
        max_abs = max(float(max_abs), 1e-6)
        points = []
        for index, value in enumerate(series):
            x = int(left + index * (right - left) / max(1, len(series) - 1))
            normalized = max(-1.0, min(1.0, float(value) / max_abs))
            y = int(mid_y - normalized * half_height)
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness)

    def _draw_fmcw_candidate_score(self, cv2, canvas, panel_y: int, width: int, height: int, plot_h: int):
        values = list(self.energy_history)
        thresholds = list(self.threshold_history)
        if len(values) < 2:
            return

        left = 20
        right = width - 20
        top = panel_y + 20
        bottom = height + plot_h - 20
        max_value = max(max(values), max(thresholds) if thresholds else 0.0, self.config.fmcw.candidate_min_score)
        max_value = max(max_value, 1e-6)
        for series, color, thickness in (
            (values, (255, 255, 255), 1),
            (thresholds, (0, 120, 255), 1),
        ):
            if len(series) < 2:
                continue
            points = []
            for index, value in enumerate(series):
                x = int(left + index * (right - left) / max(1, len(series) - 1))
                y = int(bottom - min(float(value) / max_value, 1.0) * (bottom - top))
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness)

    def _detected_label(self) -> str:
        if self.config.mode == "fmcw":
            if self.last_detection_method == "fmcw_primary_blink":
                return "眨眼候选 / Blink candidate"
            if self.last_detection_method == "fmcw_vote_candidate":
                return "投票候选 / Vote candidate"
            if self.last_detection_method == "fmcw_confirmed_blink":
                return "确认眨眼 / Blink confirmed"
            if self.last_detection_method == "fmcw_suppressed_motion":
                return "大动作已压制 / Motion suppressed"
            return "眨眼候选 / Blink candidate"
        if self.config.mode == "blink":
            return "眨眼候选 / Blink candidate"
        return "检测到挥手 / Wave detected"

    def _write_marker_for_key(self, key: int):
        marker_key = chr(key)
        if marker_key == "b":
            label = "blink"
        elif marker_key == "w":
            label = "large_motion"
        else:
            label = "manual"
        self._write_marker(label=label, key=marker_key)

    def _write_marker(self, label: str, key: str) -> None:
        self.manual_marker_count += 1
        if label == "blink":
            self.manual_blink_marker_count += 1
        elif label == "large_motion":
            self.manual_large_motion_marker_count += 1
        self._overlay_text_snapshot = None
        if self.writer is not None:
            self.writer.write_manual_marker(
                self.latest_time_s,
                label=label,
                key=key,
                feature_snapshot=self._latest_feature_snapshot(),
                event_id=self.latest_event_id,
            )

    def _enqueue_terminal_key(self, key: str) -> None:
        normalized = (key or "").lower()
        if normalized == "\x1b":
            normalized = "q"
        if normalized not in ("m", "b", "w", "q"):
            return
        try:
            self.terminal_key_queue.put_nowait(normalized)
        except queue.Full:
            pass

    def _handle_control_key(self, key: int | str) -> Optional[str]:
        if isinstance(key, str):
            if not key:
                return None
            code = ord(key[0].lower())
        else:
            code = int(key)
        if code in (ord("q"), 27):
            self.running = False
            return "user_quit"
        if code in (ord("m"), ord("b"), ord("w")):
            self._write_marker_for_key(code)
        return None

    def _process_terminal_keys(self) -> Optional[str]:
        shutdown_reason = None
        while True:
            try:
                key = self.terminal_key_queue.get_nowait()
            except queue.Empty:
                break
            reason = self._handle_control_key(key)
            if reason is not None:
                shutdown_reason = reason
        return shutdown_reason

    def _latest_feature_snapshot(self):
        if self.config.mode == "fmcw" and self.latest_fmcw_feature is not None:
            tracks = list(self.latest_fmcw_feature.track_values)
            snapshot = {
                "motion_energy": self.latest_fmcw_feature.track_delta_rms,
                "fmcw_track_0": _track_value(tracks, 0),
                "fmcw_track_1": _track_value(tracks, 1),
                "fmcw_track_2": _track_value(tracks, 2),
                "fmcw_track_3": _track_value(tracks, 3),
                "fmcw_track_4": _track_value(tracks, 4),
                "fmcw_track_delta_rms": self.latest_fmcw_feature.track_delta_rms,
                "fmcw_phase_std": self.latest_fmcw_feature.phase_std,
                "fmcw_phase_points": self.latest_fmcw_feature.phase_points,
                "fmcw_sync_lag_samples": "" if self.fmcw_sync_lag_samples is None else self.fmcw_sync_lag_samples,
                "fmcw_sync_confidence": self.fmcw_sync_confidence,
                "fmcw_pairs": ";".join(f"{a}:{b}" for a, b in self.latest_fmcw_feature.pairs),
                "fmcw_pattern": self.latest_fmcw_pattern,
                "fmcw_vote_confidence": self.latest_fmcw_vote_confidence,
                "fmcw_vote_score": self.latest_fmcw_vote_score,
                "fmcw_blink_vote_evidence": self.latest_fmcw_blink_vote_evidence,
                "fmcw_blink_trajectory_value": self.latest_fmcw_blink_trajectory_value,
                "fmcw_blink_trajectory_pattern": self.latest_fmcw_blink_trajectory_pattern,
                "fmcw_blink_trajectory_pair": self.latest_fmcw_blink_trajectory_pair,
                "fmcw_blink_trajectory_criterion": self.latest_fmcw_blink_trajectory_criterion,
                "fmcw_fixed_trajectory_value": self.latest_fmcw_fixed_trajectory_value,
                "fmcw_fixed_trajectory_phase_rad": self.latest_fmcw_fixed_trajectory_phase_rad,
                "fmcw_fixed_trajectory_distance_mm": self.latest_fmcw_fixed_trajectory_distance_mm,
                "fmcw_fixed_trajectory_pair": self.latest_fmcw_fixed_trajectory_pair,
                "fmcw_group_winners": "|".join(self.latest_fmcw_group_winners),
                "fmcw_candidate_count": self.latest_fmcw_candidate_count,
                "fmcw_candidate_score": self.latest_fmcw_candidate_score,
                "fmcw_candidate_threshold": self.latest_fmcw_candidate_threshold,
                "fmcw_candidate_is_event": int(self.latest_fmcw_candidate_is_event),
                "fmcw_candidate_event_id": self.latest_fmcw_candidate_event_id,
                "fmcw_confirm_state": self.latest_fmcw_confirm_state,
                "fmcw_confirm_is_event": int(self.latest_fmcw_confirm_is_event),
                "fmcw_confirm_event_id": self.latest_fmcw_confirm_event_id,
                "fmcw_confirm_max_delta_rms": self.latest_fmcw_confirm_max_delta_rms,
                "fmcw_confirm_high_delta_duration_s": self.latest_fmcw_confirm_high_delta_duration_s,
                "fmcw_confirm_pattern": self.latest_fmcw_confirm_pattern,
                "fmcw_confirm_confidence": self.latest_fmcw_confirm_confidence,
                "fmcw_confirm_vote_score": self.latest_fmcw_confirm_vote_score,
                "fmcw_confirm_pattern_rows": self.latest_fmcw_confirm_pattern_rows,
                "fmcw_confirm_pattern_stability": self.latest_fmcw_confirm_pattern_stability,
                "fmcw_confirm_window_pattern": self.latest_fmcw_confirm_window_pattern,
                "fmcw_confirm_window_confidence": self.latest_fmcw_confirm_window_confidence,
                "fmcw_confirm_window_vote_score": self.latest_fmcw_confirm_window_vote_score,
                "fmcw_confirm_window_candidate_count": self.latest_fmcw_confirm_window_candidate_count,
                "fmcw_confirm_window_group_winners": "|".join(self.latest_fmcw_confirm_window_group_winners),
                "fmcw_final_pattern": self.latest_fmcw_final_pattern,
                "blink_score": self.latest_fmcw_primary_blink_score,
                "blink_threshold": self.latest_fmcw_primary_blink_threshold,
                "blink_baseline": self.latest_fmcw_primary_blink_baseline,
                "blink_mad": self.latest_fmcw_primary_blink_mad,
                "blink_detector_event_id": self.latest_fmcw_primary_detector_event_id,
                "blink_detector_is_event": int(self.latest_fmcw_primary_detector_is_event),
                "blink_detector_method": self.latest_fmcw_primary_detector_method,
                "blink_peak_event_id": self.latest_fmcw_primary_peak_event_id,
                "blink_peak_is_event": int(self.latest_fmcw_primary_peak_is_event),
                "blink_peak_score": self.latest_fmcw_primary_peak_score,
                "blink_peak_threshold": self.latest_fmcw_primary_peak_threshold,
                "blink_peak_ratio": self.latest_fmcw_primary_peak_ratio,
                "blink_event_id": self.latest_fmcw_primary_blink_event_id,
                "blink_is_event": int(self.latest_fmcw_primary_blink_is_event),
                "blink_method": self.latest_fmcw_primary_blink_method,
            }
            snapshot.update(self._visual_blink_snapshot())
            return snapshot
        if self.previous_feature is None:
            snapshot = {}
        else:
            snapshot = {
                "amplitude": self.previous_feature.amplitude,
                "phase": self.previous_feature.phase,
                "motion_energy": self.previous_feature.motion_energy,
            }
        snapshot.update(self._visual_blink_snapshot())
        return snapshot

    def _visual_blink_snapshot(self) -> dict[str, object]:
        result = self.latest_visual_blink_result
        return {
            "visual_enabled": int(bool(self.config.visual_blink.enabled)),
            "visual_available": int(bool(result.available)),
            "visual_face_found": int(bool(result.face_found)),
            "visual_left_ear": result.left_ear,
            "visual_right_ear": result.right_ear,
            "visual_left_closed": int(bool(result.left_closed)),
            "visual_right_closed": int(bool(result.right_closed)),
            "visual_is_blink_event": int(bool(result.is_blink_event)),
            "visual_blink_count": int(result.blink_count),
            "visual_auto_marker_count": int(self.visual_blink_auto_marker_count),
            "visual_inference_ms": result.inference_ms,
            "visual_error": result.error,
        }

    def _write_metadata(self, shutdown_reason: str):
        if self.writer is None or self.session_dir is None:
            return
        payload = {
            "config": self.config.to_metadata(),
            "session_dir": str(self.session_dir),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "audio_devices": self.audio_device_selection,
            "camera_enabled": self.camera_enabled,
            "camera_open_seconds": self.camera_open_seconds,
            "video_saved": self.video_writer is not None,
            "auto_event_count": self.latest_event_id,
            "manual_marker_count": self.manual_marker_count,
            "manual_blink_marker_count": self.manual_blink_marker_count,
            "manual_large_motion_marker_count": self.manual_large_motion_marker_count,
            "visual_blink_available": self.latest_visual_blink_result.available,
            "visual_blink_count": self.latest_visual_blink_result.blink_count,
            "visual_blink_auto_marker_count": self.visual_blink_auto_marker_count,
            "visual_blink_error": self.latest_visual_blink_result.error,
            "detector_mode": self.config.mode,
            "blink_method": self.config.blink.method if self.config.mode == "blink" else None,
            "shutdown_reason": shutdown_reason,
        }
        self.writer.write_metadata(payload)

    def run(self) -> Path:
        cv2 = None
        try:
            import sounddevice as sd
            if not self.config.headless:
                import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency. Install with: python -m pip install -r requirements_hp_acoustic_wave.txt"
            ) from exc

        if self.config.mode == "blink":
            session_prefix = "hp_blink"
        elif self.config.mode == "fmcw":
            session_prefix = "hp_fmcw"
        else:
            session_prefix = "hp_wave"
        self.session_dir = create_session_dir(self.config.session_root, prefix=session_prefix)
        self.writer = SessionWriter(self.session_dir, self.config.audio.sample_rate)
        self.writer.open()
        self.audio_device_selection = resolve_audio_device_selection(
            requested_input_device=self.config.audio.input_device,
            requested_output_device=self.config.audio.output_device,
            default_device=sd.default.device,
            devices=sd.query_devices(),
            hostapis=sd.query_hostapis(),
        )
        print(format_audio_device_selection(self.audio_device_selection), flush=True)
        if cv2 is not None:
            self._open_camera(cv2)
            self._open_video_writer(cv2)
        self.terminal_key_reader = _TerminalKeyReader(self._enqueue_terminal_key)
        if self.terminal_key_reader.start():
            print("Terminal keys enabled: b=blink, w=large_motion, q=quit", flush=True)

        self.running = True
        shutdown_reason = "completed"
        device = None
        if self.config.audio.input_device is not None or self.config.audio.output_device is not None:
            device = (self.config.audio.input_device, self.config.audio.output_device)
        try:
            stream = sd.Stream(
                samplerate=self.config.audio.sample_rate,
                blocksize=self.config.audio.chunk_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                device=device,
            )
            with stream:
                if cv2 is not None:
                    cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(
                        self.config.window_name,
                        lambda event, x, y, flags, param: (
                            self._handle_marker_click(x, y)
                            if event == cv2.EVENT_LBUTTONDOWN
                            else False
                        ),
                    )
                next_ui_frame_time = 0.0
                ui_frame_interval_s = 1.0 / max(1.0, float(self.config.ui_fps))
                while self.running:
                    self._profile_count_add("loops", 1)
                    terminal_shutdown_reason = self._process_terminal_keys()
                    if terminal_shutdown_reason is not None:
                        shutdown_reason = terminal_shutdown_reason
                        break
                    self._process_audio_queue()
                    now = time.monotonic()
                    if cv2 is not None and now >= next_ui_frame_time:
                        next_ui_frame_time = now + ui_frame_interval_s
                        self._profile_count_add("ui_frames", 1)
                        camera_started = time.monotonic()
                        if self.camera_enabled:
                            if self.camera_reader is not None:
                                ok, frame = self.camera_reader.latest()
                            else:
                                ok, frame = self.camera.read()
                            if not ok:
                                frame = self._blank_frame(cv2)
                        else:
                            frame = self._blank_frame(cv2)
                        self._profile_add("camera_read", time.monotonic() - camera_started)
                        visual_started = time.monotonic()
                        frame = self._process_visual_blink_frame(cv2, frame)
                        self._profile_add("visual_blink", time.monotonic() - visual_started)
                        draw_started = time.monotonic()
                        canvas = self._draw_overlay(cv2, frame)
                        self._profile_add("draw", time.monotonic() - draw_started)
                        if self.video_writer is not None:
                            video_started = time.monotonic()
                            self.video_writer.write(canvas)
                            self._profile_add("video_write", time.monotonic() - video_started)
                        show_started = time.monotonic()
                        cv2.imshow(self.config.window_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                        self._profile_add("imshow_wait", time.monotonic() - show_started)
                        key_shutdown_reason = self._handle_control_key(key)
                        if key_shutdown_reason is not None:
                            shutdown_reason = key_shutdown_reason
                    if (
                        self.config.max_duration_s is not None
                        and self.first_audio_time_monotonic is not None
                        and time.monotonic() - self.first_audio_time_monotonic >= self.config.max_duration_s
                    ):
                        shutdown_reason = "duration_elapsed"
                        self.running = False
                    self._profile_maybe_print()
                    if cv2 is None:
                        time.sleep(0.005)
                    else:
                        sleep_s = min(0.005, max(0.0, next_ui_frame_time - time.monotonic()))
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
        except KeyboardInterrupt:
            shutdown_reason = "keyboard_interrupt"
        except Exception as exc:
            shutdown_reason = f"error: {exc}"
            raise
        finally:
            self._process_audio_queue()
            self._write_metadata(shutdown_reason)
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if self.camera_reader is not None:
                self.camera_reader.stop()
                self.camera_reader = None
            if self.terminal_key_reader is not None:
                self.terminal_key_reader.stop()
                self.terminal_key_reader = None
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            if self.visual_blink_detector is not None:
                self.visual_blink_detector.close()
                self.visual_blink_detector = None
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if self.writer is not None:
                self.writer.close()
        return self.session_dir


def _format_metric(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.9f}"


def _blink_metric_fields(metrics: dict) -> dict[str, str]:
    return {
        "blinklistener_viewing_amplitude": _format_metric(metrics.get("viewing_amplitude")),
        "blinklistener_viewing_range": _format_metric(metrics.get("viewing_range")),
        "blinklistener_raw_viewing_score": _format_metric(metrics.get("raw_viewing_score")),
        "blinklistener_relative_viewing_score": _format_metric(metrics.get("relative_viewing_score")),
        "blinklistener_center_i": _format_metric(metrics.get("center_i")),
        "blinklistener_center_q": _format_metric(metrics.get("center_q")),
        "twinkle_phase_pair_delta": _format_metric(metrics.get("phase_pair_delta")),
        "twinkle_trajectory_span": _format_metric(metrics.get("trajectory_span")),
        "twinkle_trajectory_rms": _format_metric(
            metrics.get("acceleration_rms", metrics.get("trajectory_rms"))
        ),
        "twinkle_peak_score": _format_metric(metrics.get("twinkle_peak_score")),
        "twinkle_peak_threshold": _format_metric(metrics.get("twinkle_peak_threshold")),
        "twinkle_peak_motion_energy": _format_metric(metrics.get("twinkle_peak_motion_energy")),
        "twinkle_peak_sign_changes": _format_metric(metrics.get("twinkle_peak_sign_changes")),
        "twinkle_candidate_peak": _format_metric(metrics.get("twinkle_candidate_peak")),
        "twinkle_candidate_accepted": _format_metric(metrics.get("twinkle_candidate_accepted")),
        "twinkle_candidate_local_peak": _format_metric(metrics.get("twinkle_candidate_local_peak")),
        "twinkle_candidate_rising_edge": _format_metric(metrics.get("twinkle_candidate_rising_edge")),
        "twinkle_candidate_event_level": _format_metric(metrics.get("twinkle_candidate_event_level")),
        "twinkle_candidate_active": _format_metric(metrics.get("twinkle_candidate_active")),
        "twinkle_reject_low_score": _format_metric(metrics.get("twinkle_reject_low_score")),
        "twinkle_reject_high_score": _format_metric(metrics.get("twinkle_reject_high_score")),
        "twinkle_reject_low_motion": _format_metric(metrics.get("twinkle_reject_low_motion")),
        "twinkle_reject_large_motion": _format_metric(metrics.get("twinkle_reject_large_motion")),
        "twinkle_reject_few_reversals": _format_metric(metrics.get("twinkle_reject_few_reversals")),
        "twinkle_reject_many_reversals": _format_metric(metrics.get("twinkle_reject_many_reversals")),
        "twinkle_reject_suppressed": _format_metric(metrics.get("twinkle_reject_suppressed")),
        "twinkle_reject_refractory": _format_metric(metrics.get("twinkle_reject_refractory")),
        "twinkle_reject_active": _format_metric(metrics.get("twinkle_reject_active")),
        "twinkle_large_motion_suppressed": _format_metric(metrics.get("twinkle_large_motion_suppressed")),
    }


def _format_sequence(values) -> str:
    if values is None:
        return ""
    return ";".join(f"{float(value):.9f}" for value in values)


def _format_track(values: list[float], index: int) -> str:
    value = _track_value(values, index)
    if value is None:
        return ""
    return f"{float(value):.9f}"


def _track_value(values: list[float], index: int) -> Optional[float]:
    if index >= len(values):
        return None
    return float(values[index])


def _duration_above(times: list[float], values: list[float], threshold: float) -> float:
    if len(times) < 2 or len(values) < 2:
        return 0.0
    duration = 0.0
    for index in range(min(len(times), len(values)) - 1):
        if values[index] >= threshold:
            duration += max(0.0, float(times[index + 1]) - float(times[index]))
    return float(duration)


def _positive_series_scale(*series_groups) -> float:
    values: list[float] = []
    for series in series_groups:
        for value in series:
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(max(0.0, numeric))
    return max(max(values, default=0.0), 1e-6)


def _centered_series_scale(series, *, floor: float = 1e-6) -> float:
    values = [abs(float(value)) for value in series if np.isfinite(float(value))]
    if not values:
        return max(float(floor), 1e-6)
    return max(float(np.percentile(np.asarray(values, dtype=np.float64), 95)), float(floor), 1e-6)


@lru_cache(maxsize=16)
def _ui_font(size: int):
    for path in _CJK_FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), int(size))
    return ImageFont.load_default()


def _render_text_image(text: str, size: int, color_bgr: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    font = _ui_font(int(size))
    probe = Image.new("RGBA", (1, 1))
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    width = max(1, int(bbox[2] - bbox[0]) + 6)
    height = max(1, int(bbox[3] - bbox[1]) + 6)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((0, 0), text, font=font, fill=color_rgb + (255,))
    rgba = np.asarray(image, dtype=np.uint8)
    return rgba[:, :, 2::-1].copy(), rgba[:, :, 3].copy()


def _draw_text(canvas: np.ndarray, text: str, xy: tuple[int, int], size: int, color_bgr: tuple[int, int, int]) -> None:
    if not text:
        return
    image, alpha = _render_text_image(str(text), int(size), tuple(int(value) for value in color_bgr))
    x = max(0, min(int(xy[0]), canvas.shape[1] - 1))
    y = max(0, min(int(xy[1]), canvas.shape[0] - 1))
    right = min(canvas.shape[1], x + image.shape[1])
    bottom = min(canvas.shape[0], y + image.shape[0])
    if right <= x or bottom <= y:
        return
    patch = image[: bottom - y, : right - x]
    patch_alpha = alpha[: bottom - y, : right - x, None].astype(np.float32) / 255.0
    region = canvas[y:bottom, x:right]
    region[:] = (patch.astype(np.float32) * patch_alpha + region.astype(np.float32) * (1.0 - patch_alpha)).astype(
        np.uint8
    )
