"""Guards the WS1b 'Clean Light Systems' theme tokens in visual.py."""
import numpy as np
from src.utils import visual


def test_overlay_uses_forest_green_for_authorized():
    # #15803d -> BGR (61, 128, 21)
    assert visual._AUTHORIZED_BGR == (61, 128, 21)


def test_overlay_uses_alert_red():
    # #b91c1c -> BGR (28, 28, 185)
    assert visual._ALERT_BGR == (28, 28, 185)


def test_draw_overlay_returns_same_shape_without_mutating():
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    dets = [{"bbox": [10, 10, 80, 50], "plate_text": "30F-12345"}]
    out = visual.draw_detection_overlay(img, dets, {"status": "AUTHORIZED"})
    assert out.shape == img.shape
    assert out is not img
    assert int(img.sum()) == 0
