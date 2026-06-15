import numpy as np

from hp_acoustic_wave.app import _draw_text


def test_draw_text_renders_chinese_pixels():
    canvas = np.zeros((80, 240, 3), dtype=np.uint8)

    _draw_text(canvas, "确认眨眼", (10, 10), 28, (255, 255, 255))

    assert int(np.count_nonzero(canvas)) > 0
