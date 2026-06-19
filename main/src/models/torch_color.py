"""PyTorch MobileNetV3-Small vehicle-colour classifier.

Replaces the TensorFlow ``ColorClassifier`` in the runtime: TF and PaddleOCR
(the OCR engine chosen in Benchmark C) crash together in one process
(``mutex lock failed``) on this macOS build, whereas PyTorch coexists with
both PaddleOCR and ONNX-Runtime. Weights come from the Benchmark-A run
(``scripts/benchmark_color.py``), whose class order is the sorted folder
names below.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# Sorted folder order used when the model was trained (Benchmark A).
_CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
_MEAN = np.array([0.485, 0.456, 0.406], "float32")
_STD = np.array([0.229, 0.224, 0.225], "float32")


class TorchColorClassifier:
    """Predict vehicle colour from a BGR crop. Mirrors ``ColorClassifier.predict``."""

    CLASS_NAMES = _CLASSES

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.device = device
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(_CLASSES))
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model = model.eval().to(device)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        # Body-crop to match training-time preprocessing (colab_train_color.py):
        # drop the top 20% (windshield/sky) and bottom 15% (tyres/road), keeping
        # the central body band where colour is most legible. Guard against
        # degenerate crops (empty / < 2px tall) by falling back to the full image.
        h = image.shape[0]
        cropped = image[int(h * 0.20):int(h * 0.85), :]
        if cropped.size > 0 and cropped.shape[0] >= 2:
            image = cropped
        im = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        im = (im - _MEAN) / _STD
        t = torch.tensor(im.transpose(2, 0, 1)[None]).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return _CLASSES[idx], float(probs[idx])
