from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


LEFT_EYE_EAR = (263, 387, 386, 362, 380, 374)
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)

LEFT_EYE_CONTOUR = (
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)
RIGHT_EYE_CONTOUR = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)


@dataclass
class VisualBlinkResult:
    enabled: bool
    available: bool
    face_found: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
    left_closed: bool = False
    right_closed: bool = False
    is_blink_event: bool = False
    blink_count: int = 0
    inference_ms: float = 0.0
    timestamp_ms: int = 0
    left_eye_points: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    right_eye_points: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    error: str = ""


class VisualBlinkDetector:
    def __init__(self, config):
        self.config = config
        self.blink_count = 0
        self._was_closed = False
        self._last_blink_time_s = -1e9
        self._last_timestamp_ms = -1
        self._detector = None
        self._mp = None
        self._vision = None
        self._python = None
        self._error = ""
        self._initialization_attempted = False

    @property
    def error(self) -> str:
        return self._error

    def close(self) -> None:
        detector = self._detector
        self._detector = None
        if detector is not None:
            detector.close()

    def is_available(self) -> bool:
        if not bool(getattr(self.config, "enabled", False)):
            return False
        return self._ensure_detector()

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> VisualBlinkResult:
        if not bool(getattr(self.config, "enabled", False)):
            return VisualBlinkResult(enabled=False, available=False)
        if not self._ensure_detector():
            return VisualBlinkResult(enabled=True, available=False, error=self._error)
        if frame_bgr is None or frame_bgr.size == 0:
            return VisualBlinkResult(enabled=True, available=True, error="empty frame")

        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000.0)
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        inference_start = time.perf_counter()
        try:
            # OpenCV gives BGR. The channel-reversed view has negative strides,
            # while MediaPipe Image requires a contiguous ndarray.
            rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:  # pragma: no cover - depends on native mediapipe runtime.
            self._error = f"visual inference failed: {exc}"
            return VisualBlinkResult(enabled=True, available=False, error=self._error)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        face_found = bool(getattr(result, "face_landmarks", None))
        if not face_found:
            self._was_closed = False
            return VisualBlinkResult(
                enabled=True,
                available=True,
                face_found=False,
                blink_count=self.blink_count,
                inference_ms=inference_ms,
                timestamp_ms=timestamp_ms,
            )

        landmarks = result.face_landmarks[0]
        left_ear = compute_ear(landmarks, LEFT_EYE_EAR)
        right_ear = compute_ear(landmarks, RIGHT_EYE_EAR)
        threshold = float(getattr(self.config, "threshold", 0.22))
        left_closed = left_ear < threshold
        right_closed = right_ear < threshold
        both_closed = left_closed and right_closed
        timestamp_s = timestamp_ms / 1000.0
        refractory_s = float(getattr(self.config, "refractory_s", 0.25))
        is_event = False
        if both_closed and not self._was_closed:
            if timestamp_s - self._last_blink_time_s >= refractory_s:
                self.blink_count += 1
                self._last_blink_time_s = timestamp_s
                is_event = True
            self._was_closed = True
        elif not both_closed:
            self._was_closed = False

        height, width = frame_bgr.shape[:2]
        return VisualBlinkResult(
            enabled=True,
            available=True,
            face_found=True,
            left_ear=left_ear,
            right_ear=right_ear,
            left_closed=left_closed,
            right_closed=right_closed,
            is_blink_event=is_event,
            blink_count=self.blink_count,
            inference_ms=inference_ms,
            timestamp_ms=timestamp_ms,
            left_eye_points=landmarks_to_pixels(landmarks, LEFT_EYE_CONTOUR, width, height),
            right_eye_points=landmarks_to_pixels(landmarks, RIGHT_EYE_CONTOUR, width, height),
        )

    def _ensure_detector(self) -> bool:
        if self._detector is not None:
            return True
        if self._initialization_attempted:
            return False
        self._initialization_attempted = True
        _ensure_writable_native_cache()
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            self._error = "mediapipe is not installed"
            return False

        model_path = resolve_visual_model_path(str(getattr(self.config, "model_path", "")))
        if not model_path.exists() or model_path.stat().st_size <= 0:
            self._error = f"model not found: {model_path}"
            return False

        try:
            base_options = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=float(
                    getattr(self.config, "min_face_detection_confidence", 0.5)
                ),
                min_face_presence_confidence=float(
                    getattr(self.config, "min_face_presence_confidence", 0.5)
                ),
                min_tracking_confidence=float(getattr(self.config, "min_tracking_confidence", 0.5)),
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            self._error = f"visual initialization failed: {exc}"
            return False
        self._mp = mp
        self._python = python
        self._vision = vision
        self._error = ""
        return True


def default_visual_model_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "face_landmarker.task"


def _ensure_writable_native_cache() -> None:
    # MediaPipe imports Matplotlib on desktop. In sandboxed/macOS app contexts the
    # default user cache can be unwritable, which makes first startup visibly stall.
    root = Path(tempfile.gettempdir()) / "hp_acoustic_wave_cache"
    mpl = root / "matplotlib"
    xdg = root / "xdg"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))


def resolve_visual_model_path(model_path: str | Path | None) -> Path:
    if not model_path:
        return default_visual_model_path()
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def landmark_value(landmark: object, name: str) -> float:
    value = getattr(landmark, name)
    if callable(value):
        value = value()
    return float(value)


def distance(a: object, b: object) -> float:
    dx = landmark_value(a, "x") - landmark_value(b, "x")
    dy = landmark_value(a, "y") - landmark_value(b, "y")
    return math.sqrt(dx * dx + dy * dy)


def compute_ear(landmarks: Sequence[object], indices: Sequence[int]) -> float:
    if any(index >= len(landmarks) or index < 0 for index in indices):
        return 0.0

    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def landmark_to_pixel(landmark: object, width: int, height: int) -> tuple[int, int]:
    x = int(landmark_value(landmark, "x") * float(width))
    y = int(landmark_value(landmark, "y") * float(height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def landmarks_to_pixels(
    landmarks: Sequence[object],
    indices: Sequence[int],
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    points = []
    for index in indices:
        if 0 <= index < len(landmarks):
            points.append(landmark_to_pixel(landmarks[index], width, height))
    return tuple(points)
