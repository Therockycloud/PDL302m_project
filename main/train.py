#!/usr/bin/env python3
"""CLI entrypoint for training vehicle anti-theft classifiers.

Usage examples::

    python train.py brand --data_dir main/data/raw/car_brands --epochs 10
    python train.py color --data_dir main/data/raw/car_colors --epochs 15 --batch_size 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Project root (repo-level) assumed to be CWD ─────────────────────
CONFIG_PATH = os.path.join("main", "configs", "config.yaml")
DEFAULT_SAVE_DIR = os.path.join("main", "data", "models")


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config(path: str = CONFIG_PATH) -> dict:
    """Loads the project YAML configuration file.

    Args:
        path: Relative or absolute path to ``config.yaml``.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_brand_model(cfg: dict) -> tf.keras.Model:
    """Builds and compiles the EfficientNet-B0 brand classifier.

    Falls back to a simple transfer-learning architecture when the
    dedicated ``src.models.classifiers`` module is unavailable.

    Args:
        cfg: The ``brand_classifier`` section of ``config.yaml``.

    Returns:
        A compiled ``tf.keras.Model``.
    """
    try:
        from src.models.classifiers import build_brand_classifier
        model = build_brand_classifier(cfg)
        logger.info("Brand model loaded from src.models.classifiers.")
        return model
    except ImportError:
        logger.warning(
            "src.models.classifiers not found — building default "
            "EfficientNetB0 brand classifier."
        )

    input_shape = tuple(cfg.get("input_shape", [224, 224, 3]))
    num_classes = cfg.get("num_classes", 8)
    lr = cfg.get("learning_rate", 1e-4)
    dropout = cfg.get("dropout_rate", 0.5)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(255.0),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_color_model(cfg: dict) -> tf.keras.Model:
    """Builds and compiles the MobileNetV3-Small color classifier.

    Falls back to a simple transfer-learning architecture when the
    dedicated ``src.models.classifiers`` module is unavailable.

    Args:
        cfg: The ``color_classifier`` section of ``config.yaml``.

    Returns:
        A compiled ``tf.keras.Model``.
    """
    try:
        from src.models.classifiers import build_color_classifier
        model = build_color_classifier(cfg)
        logger.info("Color model loaded from src.models.classifiers.")
        return model
    except ImportError:
        logger.warning(
            "src.models.classifiers not found — building default "
            "MobileNetV3Small color classifier."
        )

    input_shape = tuple(cfg.get("input_shape", [224, 224, 3]))
    num_classes = cfg.get("num_classes", 8)
    lr = cfg.get("learning_rate", 1e-4)

    base = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Subcommand handlers ─────────────────────────────────────────────

def _train_brand(args: argparse.Namespace) -> None:
    """Trains the brand classifier end-to-end.

    Args:
        args: Parsed CLI arguments for the ``brand`` subcommand.
    """
    config = _load_config()
    brand_cfg = config.get("brand_classifier", {})

    epochs = args.epochs or brand_cfg.get("initial_epochs", 10)
    batch_size = args.batch_size or brand_cfg.get("batch_size", 32)
    data_dir = args.data_dir
    save_dir = args.save_dir

    logger.info("═══ Brand Classifier Training ═══")
    logger.info("  data_dir   : %s", data_dir)
    logger.info("  epochs     : %d", epochs)
    logger.info("  batch_size : %d", batch_size)
    logger.info("  save_dir   : %s", save_dir)

    # ── Data ─────────────────────────────────────────────────────
    try:
        from src.datasets.vehicle_dataset import load_classification_dataset
    except ImportError as exc:
        logger.error("Cannot import dataset loader: %s", exc)
        sys.exit(1)

    train_ds, val_ds = load_classification_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        img_height=brand_cfg.get("input_shape", [224])[0],
        img_width=brand_cfg.get("input_shape", [224, 224])[1],
        subset="both",
    )

    # ── Model ────────────────────────────────────────────────────
    model = _build_brand_model(brand_cfg)
    model.summary()

    # ── Training ─────────────────────────────────────────────────
    try:
        from src.engine.trainer import ModelTrainer
    except ImportError as exc:
        logger.error("Cannot import ModelTrainer: %s", exc)
        sys.exit(1)

    trainer = ModelTrainer(model, model_name="brand_classifier")
    history = trainer.train(
        train_ds, val_ds, epochs=epochs, save_dir=save_dir
    )

    # ── Post-training ────────────────────────────────────────────
    trainer.plot_history(history)
    metrics = trainer.evaluate(val_ds)
    logger.info("Final validation metrics: %s", metrics)


def _train_color(args: argparse.Namespace) -> None:
    """Trains the color classifier end-to-end.

    Args:
        args: Parsed CLI arguments for the ``color`` subcommand.
    """
    config = _load_config()
    color_cfg = config.get("color_classifier", {})

    epochs = args.epochs or color_cfg.get("epochs", 10)
    batch_size = args.batch_size or color_cfg.get("batch_size", 32)
    data_dir = args.data_dir
    save_dir = args.save_dir

    logger.info("═══ Color Classifier Training ═══")
    logger.info("  data_dir   : %s", data_dir)
    logger.info("  epochs     : %d", epochs)
    logger.info("  batch_size : %d", batch_size)
    logger.info("  save_dir   : %s", save_dir)

    # ── Data ─────────────────────────────────────────────────────
    try:
        from src.datasets.vehicle_dataset import load_classification_dataset
    except ImportError as exc:
        logger.error("Cannot import dataset loader: %s", exc)
        sys.exit(1)

    train_ds, val_ds = load_classification_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        img_height=color_cfg.get("input_shape", [224])[0],
        img_width=color_cfg.get("input_shape", [224, 224])[1],
        subset="both",
    )

    # ── Model ────────────────────────────────────────────────────
    model = _build_color_model(color_cfg)
    model.summary()

    # ── Training ─────────────────────────────────────────────────
    try:
        from src.engine.trainer import ModelTrainer
    except ImportError as exc:
        logger.error("Cannot import ModelTrainer: %s", exc)
        sys.exit(1)

    trainer = ModelTrainer(model, model_name="color_classifier")
    history = trainer.train(
        train_ds, val_ds, epochs=epochs, save_dir=save_dir
    )

    # ── Post-training ────────────────────────────────────────────
    trainer.plot_history(history)
    metrics = trainer.evaluate(val_ds)
    logger.info("Final validation metrics: %s", metrics)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    """Parses CLI arguments and dispatches to the correct training handler."""
    parser = argparse.ArgumentParser(
        description="Train vehicle anti-theft classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py brand --data_dir main/data/raw/car_brands\n"
            "  python train.py color --data_dir main/data/raw/car_colors "
            "--epochs 15 --batch_size 64\n"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Classifier to train."
    )

    # ── brand ────────────────────────────────────────────────────
    brand_parser = subparsers.add_parser(
        "brand", help="Train the EfficientNet-B0 brand classifier."
    )
    brand_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with class sub-folders for brands.",
    )
    brand_parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs."
    )
    brand_parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size."
    )
    brand_parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Where to save model checkpoints (default: %(default)s).",
    )
    brand_parser.set_defaults(func=_train_brand)

    # ── color ────────────────────────────────────────────────────
    color_parser = subparsers.add_parser(
        "color", help="Train the MobileNetV3-Small color classifier."
    )
    color_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with class sub-folders for colors.",
    )
    color_parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs."
    )
    color_parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size."
    )
    color_parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Where to save model checkpoints (default: %(default)s).",
    )
    color_parser.set_defaults(func=_train_color)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
