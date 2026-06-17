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

    # Functional API + explicit training=False on the frozen base keeps its
    # BatchNorm layers in inference mode (see the colour model for why a
    # Sequential breaks frozen transfer learning). EfficientNetB0 bundles its own
    # normalisation and expects [0,255]; the dataset delivers [0,1], hence
    # Rescaling(255.0).
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

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

    # Functional API with an explicit ``training=False`` on the frozen base so
    # its BatchNorm layers run in INFERENCE mode (moving averages). Inside a
    # Sequential the base would run with training=True and BN would use per-batch
    # statistics, destabilising the frozen features so badly the head never
    # learns (accuracy collapses to chance). MobileNetV3Small has
    # include_preprocessing=True (expects [0,255]); the dataset delivers [0,1],
    # hence Rescaling(255.0) — matching the brand model.
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Fine-tuning ─────────────────────────────────────────────────────

def _fine_tune(model, train_ds, val_ds, model_name, save_dir, epochs, lr,
               unfreeze_from=0.5):
    """Unfreeze the top of the frozen backbone and continue training at low LR.

    Finds the nested backbone (a ``tf.keras.Model`` layer), unfreezes its top
    ``(1 - unfreeze_from)`` fraction of layers, and recompiles at ``lr``. The
    backbone is still called with ``training=False`` (set at build time), so its
    BatchNorm layers stay in inference mode — the recommended fine-tuning recipe
    that avoids wrecking the pretrained running stats on small batches.
    """
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is None:
        logger.warning("No backbone sub-model found; skipping fine-tune.")
        return None
    base.trainable = True
    cut = int(len(base.layers) * unfreeze_from)
    for layer in base.layers[:cut]:
        layer.trainable = False
    # Keep all BatchNorm frozen regardless of depth.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    logger.info("Fine-tuning: unfroze %d/%d backbone layers at lr=%g",
                trainable, len(base.layers), lr)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    from src.engine.trainer import ModelTrainer
    trainer = ModelTrainer(model, model_name=model_name)
    return trainer.train(train_ds, val_ds, epochs=epochs, save_dir=save_dir)


# ── Test-set evaluation ─────────────────────────────────────────────

def _evaluate_on_test(model, test_ds, class_names, model_name, save_dir):
    """Evaluate on the held-out test split and write a metrics report.

    Computes overall accuracy + macro-F1 and a per-class classification
    report, persisting both to ``<save_dir>/<model_name>_test_report.json``.
    """
    import json
    import numpy as np

    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(yb.numpy(), axis=1).tolist())

    try:
        from sklearn.metrics import classification_report, accuracy_score, f1_score
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
    except ImportError:
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        macro_f1, report = None, None

    out = {
        "model": model_name,
        "n_test": len(y_true),
        "class_names": class_names,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "per_class": report,
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_test_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("TEST  accuracy=%.4f  macro_f1=%s  (n=%d) -> %s",
                acc, f"{macro_f1:.4f}" if macro_f1 is not None else "n/a",
                len(y_true), path)
    return out


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
        from src.datasets.vehicle_dataset import load_split_dataset
    except ImportError as exc:
        logger.error("Cannot import dataset loader: %s", exc)
        sys.exit(1)

    train_ds, val_ds, test_ds, class_names = load_split_dataset(
        base_dir=data_dir,
        batch_size=batch_size,
        img_height=brand_cfg.get("input_shape", [224])[0],
        img_width=brand_cfg.get("input_shape", [224, 224])[1],
    )
    logger.info("  classes    : %s", class_names)

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

    # ── Optional fine-tuning (unfreeze backbone top at low LR) ────
    if getattr(args, "fine_tune", False):
        ft_epochs = args.fine_tune_epochs or brand_cfg.get("fine_tune_epochs", 10)
        ft_lr = args.fine_tune_lr or brand_cfg.get("fine_tune_learning_rate", 1e-5)
        ft_hist = _fine_tune(model, train_ds, val_ds, "brand_classifier",
                             save_dir, ft_epochs, ft_lr)
        if ft_hist is not None:
            history = ft_hist

    # ── Post-training ────────────────────────────────────────────
    trainer.plot_history(history)
    metrics = trainer.evaluate(val_ds)
    logger.info("Final validation metrics: %s", metrics)
    _evaluate_on_test(model, test_ds, class_names, "brand_classifier", save_dir)


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
        from src.datasets.vehicle_dataset import load_split_dataset
    except ImportError as exc:
        logger.error("Cannot import dataset loader: %s", exc)
        sys.exit(1)

    train_ds, val_ds, test_ds, class_names = load_split_dataset(
        base_dir=data_dir,
        batch_size=batch_size,
        img_height=color_cfg.get("input_shape", [224])[0],
        img_width=color_cfg.get("input_shape", [224, 224])[1],
    )
    logger.info("  classes    : %s", class_names)

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

    # ── Optional fine-tuning (unfreeze backbone top at low LR) ────
    if getattr(args, "fine_tune", False):
        ft_epochs = args.fine_tune_epochs or color_cfg.get("fine_tune_epochs", 10)
        ft_lr = args.fine_tune_lr or color_cfg.get("fine_tune_learning_rate", 1e-5)
        ft_hist = _fine_tune(model, train_ds, val_ds, "color_classifier",
                             save_dir, ft_epochs, ft_lr)
        if ft_hist is not None:
            history = ft_hist

    # ── Post-training ────────────────────────────────────────────
    trainer.plot_history(history)
    metrics = trainer.evaluate(val_ds)
    logger.info("Final validation metrics: %s", metrics)
    _evaluate_on_test(model, test_ds, class_names, "color_classifier", save_dir)


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
    brand_parser.add_argument(
        "--fine_tune", action="store_true",
        help="After frozen-head training, unfreeze the backbone top and train at low LR.",
    )
    brand_parser.add_argument(
        "--fine_tune_epochs", type=int, default=None, help="Fine-tune epochs.",
    )
    brand_parser.add_argument(
        "--fine_tune_lr", type=float, default=None, help="Fine-tune learning rate.",
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
    color_parser.add_argument(
        "--fine_tune", action="store_true",
        help="After frozen-head training, unfreeze the backbone top and train at low LR.",
    )
    color_parser.add_argument(
        "--fine_tune_epochs", type=int, default=None, help="Fine-tune epochs.",
    )
    color_parser.add_argument(
        "--fine_tune_lr", type=float, default=None, help="Fine-tune learning rate.",
    )
    color_parser.set_defaults(func=_train_color)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
