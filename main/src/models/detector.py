"""YOLOv8 license plate detector wrapper.

Wraps the Ultralytics YOLOv8 model for license plate detection on CPU
(GPU/MPS is deliberately disabled, see ``_select_device``) with
configurable confidence thresholds. Parameters are read from
``config.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required for PlateDetector. "
        "Install it with: pip install ultralytics"
    ) from exc

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load project configuration from ``config.yaml``.

    Returns:
        A dictionary with the parsed YAML contents.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _select_device() -> str:
    """Return the inference device (always ``'cpu'``).

    No auto-detection happens: GPU/MPS inference is deliberately disabled
    because YOLO inference hung on macOS MPS drivers.

    Returns:
        The literal string ``'cpu'``.
    """
    return "cpu"


# ---------------------------------------------------------------------------
# PlateDetector
# ---------------------------------------------------------------------------
class PlateDetector:
    """Ultralytics YOLOv8 wrapper for license plate detection.

    Attributes:
        model: The loaded YOLOv8 model instance.
        conf_threshold: Minimum confidence for a detection to be kept.
        device: Inference device (always ``'cpu'``, see ``_select_device``).
        crop_padding: Fractional padding applied around each cropped plate.

    Example::

        detector = PlateDetector()
        detections = detector.detect(frame)
        for det in detections:
            print(det["confidence"], det["bbox"])
    """

    def __init__(
        self,
        model_path: str = "yolov8n.onnx",
        conf_threshold: float = 0.25,
    ) -> None:
        """Initialise the plate detector.

        Args:
            model_path: Path to the YOLOv8 weights file.  When a bare
                filename is given the model directory from ``config.yaml``
                is prepended automatically.
            conf_threshold: Confidence threshold for detections.
        """
        cfg = _load_config()
        det_cfg = cfg.get("detector", {})

        self.conf_threshold: float = conf_threshold or det_cfg.get(
            "conf_threshold", 0.25
        )
        self.crop_padding: float = det_cfg.get("crop_padding", 0.05)
        self.device: str = _select_device()
        self.model: YOLO | None = None

        # Resolve model path -------------------------------------------------
        if model_path.endswith(".pt"):
            onnx_path = model_path[:-3] + ".onnx"
            model_dir = cfg.get("paths", {}).get("model_save_dir", "")
            if Path(onnx_path).exists() or (Path(model_dir) / onnx_path).exists():
                model_path = onnx_path

        resolved_path = model_path
        if not os.path.isabs(model_path) and not Path(model_path).exists():
            model_dir = cfg.get("paths", {}).get("model_save_dir", "")
            candidate = Path(model_dir) / model_path
            if candidate.exists():
                resolved_path = str(candidate)

        # Load model ----------------------------------------------------------
        try:
            from ultralytics import settings
            settings.update({"sync": False})
            self.model = YOLO(resolved_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[PlateDetector] WARNING: Could not load model at "
                f"'{resolved_path}': {exc}.  detect() will return an "
                f"empty list."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run plate detection on a single image.

        Args:
            image: BGR image as a NumPy array of shape ``(H, W, 3)``.
            conf_threshold: Optional confidence threshold override.  When
                provided this value is used instead of ``self.conf_threshold``
                for this call only.

        Returns:
            A list of detection dictionaries, each containing:

            * **bbox** (*tuple[int, int, int, int]*) – ``(x1, y1, x2, y2)``
              pixel coordinates.
            * **confidence** (*float*) – Detection confidence in ``[0, 1]``.
            * **cropped_plate** (*np.ndarray*) – The plate region cropped
              from *image* with 5 % padding on every side.
        """
        if self.model is None:
            print(
                "[PlateDetector] WARNING: No model loaded. "
                "Returning empty list."
            )
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            results = self.model.predict(
                source=image,
                conf=conf,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[PlateDetector] Inference error: {exc}")
            return []

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cropped = self._crop_with_padding(image, x1, y1, x2, y2)
                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "cropped_plate": cropped,
                    }
                )

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _crop_with_padding(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """Crop a region from *image* with extra fractional padding.

        Args:
            image: Source image array.
            x1: Left edge of the bounding box.
            y1: Top edge of the bounding box.
            x2: Right edge of the bounding box.
            y2: Bottom edge of the bounding box.

        Returns:
            The cropped sub-image with ``crop_padding`` applied on each
            side, clipped to image boundaries.
        """
        h, w = image.shape[:2]
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        return image[cy1:cy2, cx1:cx2].copy()
