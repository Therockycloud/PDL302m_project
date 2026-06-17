"""Tests for the out-of-process TF/Keras colour classifier.

The actual inference runs in a separate ``dpl-train`` interpreter, so the
end-to-end test is skipped unless that interpreter and the Keras model are
present. The unit tests (interpreter resolution, no-TF-in-caller) always run.
"""

import os
import sys

import numpy as np
import pytest

MODEL = os.path.join("data", "models", "color_classifier.keras")


def _worker_python_available() -> bool:
    from src.models.keras_color import _resolve_worker_python
    try:
        _resolve_worker_python(None)
        return True
    except FileNotFoundError:
        return False


def test_importing_wrapper_does_not_import_tensorflow():
    """The dashboard process must never pull TensorFlow into its import graph."""
    import src.models.keras_color  # noqa: F401
    assert "tensorflow" not in sys.modules


def test_resolve_worker_python_prefers_env(monkeypatch):
    from src.models import keras_color
    monkeypatch.setenv("DPL_TRAIN_PYTHON", sys.executable)
    assert keras_color._resolve_worker_python(None) == sys.executable


def test_resolve_worker_python_missing_raises(monkeypatch):
    from src.models import keras_color
    monkeypatch.delenv("DPL_TRAIN_PYTHON", raising=False)
    monkeypatch.setattr(keras_color, "_DEFAULT_PY", "/no/such/python")
    with pytest.raises(FileNotFoundError):
        keras_color._resolve_worker_python(None)


@pytest.mark.skipif(not os.path.exists(MODEL), reason="keras colour model not present")
@pytest.mark.skipif(not _worker_python_available(), reason="dpl-train interpreter not found")
def test_predict_returns_valid_class_and_confidence():
    from src.models.keras_color import KerasColorClassifier

    clf = KerasColorClassifier(MODEL)
    try:
        img = (np.random.rand(120, 200, 3) * 255).astype("uint8")
        name, conf = clf.predict(img)
        assert name in KerasColorClassifier.CLASS_NAMES
        assert 0.0 <= conf <= 1.0
    finally:
        clf.close()
    # TF still must not have leaked into the test (caller) process.
    assert "tensorflow" not in sys.modules
