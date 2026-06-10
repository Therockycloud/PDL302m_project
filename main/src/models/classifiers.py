"""Keras/TensorFlow transfer-learning classifiers for vehicle features.

Contains :class:`BrandClassifier` (EfficientNetB0-based) and
:class:`ColorClassifier` (MobileNetV3Small-based).  Both follow the
same freeze-base → GAP → Dropout → Dense(softmax) pattern and read
their class names from ``config.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:
    raise ImportError(
        "tensorflow is required for the classifier wrappers. "
        "Install it with: pip install tensorflow"
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


_CFG = _load_config()

# ---------------------------------------------------------------------------
# BrandClassifier
# ---------------------------------------------------------------------------


class BrandClassifier:
    """EfficientNetB0-based vehicle brand classifier.

    The base EfficientNetB0 is frozen and topped with
    ``GlobalAveragePooling2D → Dropout(0.5) → Dense(num_classes, softmax)``.

    Attributes:
        CLASS_NAMES: Ordered list of brand labels from ``config.yaml``.
        model: The compiled :class:`keras.Model` instance (``None``
            until :meth:`build_model` is called).
        input_shape: Expected input image dimensions.

    Example::

        clf = BrandClassifier()
        clf.build_model()
        label, confidence = clf.predict(car_image)
    """

    CLASS_NAMES: list[str] = _CFG.get("brand_classifier", {}).get(
        "classes",
        [
            "Toyota",
            "Hyundai",
            "Kia",
            "Mazda",
            "Honda",
            "VinFast",
            "Ford",
            "Mitsubishi",
        ],
    )

    def __init__(self) -> None:
        """Initialise the brand classifier (model not built yet)."""
        brand_cfg = _CFG.get("brand_classifier", {})
        self.input_shape: tuple[int, int, int] = tuple(
            brand_cfg.get("input_shape", [224, 224, 3])
        )  # type: ignore[assignment]
        self.dropout_rate: float = brand_cfg.get("dropout_rate", 0.5)
        self.learning_rate: float = brand_cfg.get("learning_rate", 1e-4)
        self.model: keras.Model | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_model(self, num_classes: int = 8) -> keras.Model:
        """Build and compile the EfficientNetB0-based classifier.

        Args:
            num_classes: Number of output classes.  Defaults to the
                value in ``config.yaml``.

        Returns:
            The compiled Keras model.
        """
        num_classes = _CFG.get("brand_classifier", {}).get(
            "num_classes", num_classes
        )

        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        base.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Predict the vehicle brand from an image.

        Args:
            image: A BGR/RGB image as a NumPy array.  It will be
                resized to ``input_shape`` automatically.

        Returns:
            A tuple of ``(class_name, confidence)`` for the top
            prediction.

        Raises:
            RuntimeError: If :meth:`build_model` has not been called.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

        preprocessed = self._preprocess(image)
        preds = self.model.predict(preprocessed, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        class_name = (
            self.CLASS_NAMES[class_idx]
            if class_idx < len(self.CLASS_NAMES)
            else f"class_{class_idx}"
        )
        return class_name, confidence

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights from a file.

        If the weights file does not exist, a warning is printed and
        the model keeps its randomly initialised weights.

        Args:
            path: Filesystem path to a ``.weights.h5`` or SavedModel
                checkpoint.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

        if not os.path.exists(path):
            print(
                f"[BrandClassifier] WARNING: Weights file '{path}' not "
                f"found. Using random initialisation."
            )
            return

        try:
            self.model.load_weights(path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[BrandClassifier] WARNING: Failed to load weights "
                f"from '{path}': {exc}"
            )

    def summary(self) -> None:
        """Print the Keras model summary.

        Raises:
            RuntimeError: If :meth:`build_model` has not been called.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )
        self.model.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalise an image for inference.

        Args:
            image: Raw input image.

        Returns:
            A batch-dimensioned ``(1, H, W, 3)`` float array ready for
            the model.
        """
        h, w = self.input_shape[:2]
        img = tf.image.resize(image, (h, w))
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        return img.numpy()


# ---------------------------------------------------------------------------
# ColorClassifier
# ---------------------------------------------------------------------------


class ColorClassifier:
    """MobileNetV3Small-based vehicle colour classifier.

    The base MobileNetV3Small is frozen and topped with
    ``GlobalAveragePooling2D → Dropout(0.3) → Dense(num_classes, softmax)``.

    Attributes:
        CLASS_NAMES: Ordered list of colour labels from ``config.yaml``.
        model: The compiled :class:`keras.Model` instance (``None``
            until :meth:`build_model` is called).
        input_shape: Expected input image dimensions.

    Example::

        clf = ColorClassifier()
        clf.build_model()
        label, confidence = clf.predict(car_image)
    """

    CLASS_NAMES: list[str] = _CFG.get("color_classifier", {}).get(
        "classes",
        [
            "White",
            "Black",
            "Grey",
            "Silver",
            "Red",
            "Blue",
            "Brown",
            "Yellow",
        ],
    )

    def __init__(self) -> None:
        """Initialise the colour classifier (model not built yet)."""
        color_cfg = _CFG.get("color_classifier", {})
        self.input_shape: tuple[int, int, int] = tuple(
            color_cfg.get("input_shape", [224, 224, 3])
        )  # type: ignore[assignment]
        self.dropout_rate: float = 0.3
        self.learning_rate: float = color_cfg.get("learning_rate", 1e-4)
        self.model: keras.Model | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_model(self, num_classes: int = 8) -> keras.Model:
        """Build and compile the MobileNetV3Small-based classifier.

        Args:
            num_classes: Number of output classes.  Defaults to the
                value in ``config.yaml``.

        Returns:
            The compiled Keras model.
        """
        num_classes = _CFG.get("color_classifier", {}).get(
            "num_classes", num_classes
        )

        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        base.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        # MobileNetV3 expects inputs scaled to [-1, 1].
        x = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Predict the vehicle colour from an image.

        Args:
            image: A BGR/RGB image as a NumPy array.  It will be
                resized to ``input_shape`` automatically.

        Returns:
            A tuple of ``(class_name, confidence)`` for the top
            prediction.

        Raises:
            RuntimeError: If :meth:`build_model` has not been called.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

        preprocessed = self._preprocess(image)
        preds = self.model.predict(preprocessed, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        class_name = (
            self.CLASS_NAMES[class_idx]
            if class_idx < len(self.CLASS_NAMES)
            else f"class_{class_idx}"
        )
        return class_name, confidence

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights from a file.

        If the weights file does not exist, a warning is printed and
        the model keeps its randomly initialised weights.

        Args:
            path: Filesystem path to a ``.weights.h5`` or SavedModel
                checkpoint.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

        if not os.path.exists(path):
            print(
                f"[ColorClassifier] WARNING: Weights file '{path}' not "
                f"found. Using random initialisation."
            )
            return

        try:
            self.model.load_weights(path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ColorClassifier] WARNING: Failed to load weights "
                f"from '{path}': {exc}"
            )

    def summary(self) -> None:
        """Print the Keras model summary.

        Raises:
            RuntimeError: If :meth:`build_model` has not been called.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )
        self.model.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalise an image for inference.

        Args:
            image: Raw input image.

        Returns:
            A batch-dimensioned ``(1, H, W, 3)`` float array ready for
            the model.
        """
        h, w = self.input_shape[:2]
        img = tf.image.resize(image, (h, w))
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        return img.numpy()
