"""Smoke test for the PyTorch colour classifier (runtime replacement for TF)."""

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

MODEL = os.path.join("data", "models", "color_MobileNetV3Small.pt")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="torch colour model not present")
def test_predict_returns_valid_class_and_confidence():
    from src.models.torch_color import TorchColorClassifier

    clf = TorchColorClassifier(MODEL)
    img = (np.random.rand(120, 200, 3) * 255).astype("uint8")
    name, conf = clf.predict(img)
    assert name in TorchColorClassifier.CLASS_NAMES
    assert 0.0 <= conf <= 1.0
