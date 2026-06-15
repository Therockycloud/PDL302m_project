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


def test_build_theme_css_exists_and_is_string():
    css = visual.build_theme_css()
    assert isinstance(css, str) and len(css) > 500


def test_theme_uses_accent_token_not_neon():
    css = visual.build_theme_css()
    assert "--accent: #15803d" in css
    assert "--neon-green" not in css


def test_theme_has_no_violet_gradient_or_glass_blur():
    css = visual.build_theme_css()
    assert "#5b31df" not in css
    assert "backdrop-filter" not in css
    assert "glass-card" not in css


def test_status_css_authorized_is_borderless_green():
    css = visual.get_status_css("AUTHORIZED")
    assert "#15803d" in css
    assert "border: none" in css or "border:none" in css


def test_alarm_html_has_no_emoji_or_border():
    html = visual.get_alarm_html("MISMATCH")
    assert "⚠" not in html and "🚨" not in html
    assert "border: none" in html or "border:none" in html
    assert "#b91c1c" in html or "#de350b" in html


def test_overlay_draws_status_coloured_corner_brackets():
    import cv2
    img = np.zeros((200, 320, 3), dtype=np.uint8)
    out = visual.draw_detection_overlay(img, [], {"status": "AUTHORIZED"})
    # top-left corner region should contain the forest-green bracket
    corner = out[0:40, 0:40]
    assert corner.sum() > 0, "no bracket drawn in top-left corner"
    # the drawn colour should be the authorized BGR (green channel dominant)
    b, g, r = visual._AUTHORIZED_BGR
    mask = np.all(corner == (b, g, r), axis=-1)
    assert mask.any(), "corner bracket is not the authorized colour"
