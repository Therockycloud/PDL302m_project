"""WS-3/WS-4 Task 5: API and dashboard image-path must produce identical
verdicts for the same image.

Both surfaces delegate to ``infer_single_image`` (see
``src.engine.pipeline_factory``):

- The dashboard's Upload-Image path calls ``infer_single_image`` directly.
- The API's ``/verify`` endpoint calls it internally after decoding the
  uploaded file.

This test drives both call paths with the SAME fake pipeline and the SAME
source image, then asserts the user-visible verdict fields match exactly.
It reuses the fake-pipeline pattern from ``test_api.py`` (TestClient +
monkeypatched ``app_module._models``, entered via ``with`` so lifespan runs
but short-circuits the real ``build_pipeline`` call).
"""

import cv2
import numpy as np
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.engine.pipeline_factory import infer_single_image

from test_api import _make_fake_pipeline, _make_test_image_files


def test_api_and_dashboard_paths_agree_on_verdict(monkeypatch):
    """Same image + same fake pipeline -> same verdict via both surfaces."""
    fake_pipeline = _make_fake_pipeline(brand_clf=None)
    files = _make_test_image_files()
    image_bytes = files["file"][1]

    # ---- Dashboard path: infer_single_image() called directly -----------
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    dashboard_result = infer_single_image(decoded_image, fake_pipeline, cfg={})

    # ---- API path: same fake pipeline injected, POST /verify ------------
    monkeypatch.setattr(app_module, "_models", {"pipeline": fake_pipeline})
    with TestClient(app_module.app) as client:
        response = client.post("/verify", files=_make_test_image_files())

    assert response.status_code == 200
    api_result = response.json()

    compared_keys = ("status", "action", "plate_text", "color", "color_warning")
    for key in compared_keys:
        assert key in dashboard_result, f"dashboard result missing key: {key}"
        assert key in api_result, f"API result missing key: {key}"
        assert dashboard_result[key] == api_result[key], (
            f"verdict mismatch on '{key}': dashboard={dashboard_result[key]!r} "
            f"vs api={api_result[key]!r}"
        )

    # Sanity: both surfaces actually reached the AUTHORIZED branch (the fake
    # matcher always authorizes), not e.g. NO_PLATE for either.
    assert dashboard_result["status"] == "AUTHORIZED"
    assert dashboard_result["plate_text"] == "30M71854"
