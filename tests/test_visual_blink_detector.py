import numpy as np

from hp_acoustic_wave.visual_blink_detector import (
    LEFT_EYE_EAR,
    VisualBlinkDetector,
    compute_ear,
    default_visual_model_path,
    resolve_visual_model_path,
)


class Landmark:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


def test_compute_ear_uses_two_vertical_distances_over_horizontal_width():
    landmarks = [Landmark() for _ in range(468)]
    p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR
    landmarks[p1] = Landmark(0.0, 0.0)
    landmarks[p4] = Landmark(1.0, 0.0)
    landmarks[p2] = Landmark(0.25, 0.20)
    landmarks[p6] = Landmark(0.25, -0.20)
    landmarks[p3] = Landmark(0.75, 0.20)
    landmarks[p5] = Landmark(0.75, -0.20)

    assert abs(compute_ear(landmarks, LEFT_EYE_EAR) - 0.4) < 1e-9


def test_visual_model_default_path_points_to_current_project_assets():
    path = default_visual_model_path()

    assert path.name == "face_landmarker.task"
    assert path.parent.name == "assets"
    assert resolve_visual_model_path("assets/face_landmarker.task") == path


def test_visual_detector_passes_contiguous_rgb_image_to_mediapipe():
    captured = {}

    class Config:
        enabled = True
        threshold = 0.22
        refractory_s = 0.25

    class FakeImageFormat:
        SRGB = object()

    class FakeMP:
        ImageFormat = FakeImageFormat

        @staticmethod
        def Image(*, image_format, data):
            captured["data"] = data
            return object()

    class FakeDetector:
        def detect_for_video(self, image, timestamp_ms):
            return type("Result", (), {"face_landmarks": []})()

    detector = VisualBlinkDetector(Config())
    detector._detector = FakeDetector()
    detector._mp = FakeMP()

    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    result = detector.process_frame(frame, timestamp_ms=1)

    assert result.available
    assert captured["data"].flags["C_CONTIGUOUS"]
