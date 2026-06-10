"""Keras model training pipeline for vehicle anti-theft classifiers.

Provides a reusable training wrapper with EarlyStopping, ModelCheckpoint,
and TensorBoard callbacks. Supports evaluation and training curve plotting.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates compiled Keras classification models.

    Wraps ``model.fit()`` with standard callbacks (EarlyStopping,
    ModelCheckpoint, TensorBoard) and provides evaluation and
    training-curve plotting utilities.

    Attributes:
        model: A compiled ``tf.keras.Model`` ready for training.
        model_name: Human-readable name used for checkpoint filenames
            and log directories.
    """

    def __init__(self, model: tf.keras.Model, model_name: str = "model") -> None:
        """Initializes ModelTrainer with a compiled Keras model.

        Args:
            model: A compiled ``tf.keras.Model`` instance.
            model_name: Descriptive name for file-system artefacts
                (checkpoints, logs, plots).

        Raises:
            ValueError: If *model* is ``None`` or has not been compiled.
        """
        if model is None:
            raise ValueError("Model cannot be None.")
        if not hasattr(model, "optimizer") or model.optimizer is None:
            raise ValueError(
                "Model must be compiled before passing to ModelTrainer. "
                "Call model.compile() first."
            )
        self.model = model
        self.model_name = model_name
        self._save_dir: str | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 10,
        save_dir: str = "main/data/models",
    ) -> tf.keras.callbacks.History:
        """Runs the training loop with standard callbacks.

        Callbacks configured:
            * **EarlyStopping** – monitors ``val_loss``, patience 5,
              restores best weights.
            * **ModelCheckpoint** – saves the best-only model to
              ``<save_dir>/<model_name>.keras``.
            * **TensorBoard** – writes logs to
              ``<save_dir>/logs/<model_name>``.

        Args:
            train_ds: Training ``tf.data.Dataset`` (batched and
                preprocessed).
            val_ds: Validation ``tf.data.Dataset`` (batched and
                preprocessed).
            epochs: Maximum number of training epochs.
            save_dir: Root directory for model checkpoints, logs,
                and plots.

        Returns:
            A ``tf.keras.callbacks.History`` object containing
            per-epoch metrics.

        Raises:
            RuntimeError: If ``model.fit()`` fails unexpectedly.
        """
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{self.model_name}.keras")
        log_dir = os.path.join(save_dir, "logs", self.model_name)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            ),
        ]

        logger.info(
            "Starting training for '%s' | epochs=%d | save_dir=%s",
            self.model_name,
            epochs,
            save_dir,
        )

        try:
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
            )
        except Exception as exc:
            logger.exception("Training failed for '%s'.", self.model_name)
            raise RuntimeError(
                f"Training loop failed for '{self.model_name}'."
            ) from exc

        logger.info(
            "Training complete for '%s'. Best weights saved to %s",
            self.model_name,
            save_path,
        )
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_ds: tf.data.Dataset) -> dict[str, float]:
        """Evaluates the model on a test dataset.

        Args:
            test_ds: Test ``tf.data.Dataset`` (batched and preprocessed).

        Returns:
            Dictionary mapping metric names (``loss``, ``accuracy``, …)
            to their scalar values.

        Raises:
            RuntimeError: If ``model.evaluate()`` fails.
        """
        try:
            results = self.model.evaluate(test_ds, return_dict=True)
        except Exception as exc:
            logger.exception("Evaluation failed for '%s'.", self.model_name)
            raise RuntimeError(
                f"Evaluation failed for '{self.model_name}'."
            ) from exc

        logger.info(
            "Evaluation results for '%s': %s",
            self.model_name,
            results,
        )
        return dict(results)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_history(self, history: tf.keras.callbacks.History) -> str:
        """Saves training/validation loss and accuracy curves as a PNG.

        The figure is saved to
        ``<save_dir>/<model_name>_training_curves.png``.

        Args:
            history: ``History`` object returned by ``train()``.

        Returns:
            Absolute path to the saved PNG file.

        Raises:
            ValueError: If *history* does not contain plottable metrics.
        """
        hist = history.history
        if not hist:
            raise ValueError("History object is empty — nothing to plot.")

        save_dir = self._save_dir or "main/data/models"
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir, f"{self.model_name}_training_curves.png"
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss subplot ---
        axes[0].plot(hist.get("loss", []), label="Train Loss")
        axes[0].plot(hist.get("val_loss", []), label="Val Loss")
        axes[0].set_title(f"{self.model_name} — Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Accuracy subplot ---
        acc_key = "accuracy" if "accuracy" in hist else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
        axes[1].plot(hist.get(acc_key, []), label="Train Accuracy")
        axes[1].plot(hist.get(val_acc_key, []), label="Val Accuracy")
        axes[1].set_title(f"{self.model_name} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        logger.info("Training curves saved to %s", plot_path)
        return plot_path
