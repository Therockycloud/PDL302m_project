#!/usr/bin/env python3
"""Audit training-split labels for a classifier by flagging high-confidence errors.

Usage::

    python scripts/audit_labels.py color --thr 0.8
    python scripts/audit_labels.py brand --thr 0.8

The script loads the deployed ``.keras`` model, runs it over the **train**
split with ``shuffle=False``, and reports every image where:

* the model's top prediction **differs** from the folder label, AND
* the prediction confidence is **≥ thr** (default 0.80).

High-confidence wrong predictions are strong indicators of a mislabelled image.

Output
------
* Console: table of suspect images sorted by confidence (descending).
* CSV: ``data/models/<task>_label_suspects.csv`` with columns
  ``path,true_label,pred_label,confidence``.

Notes
-----
* Only the **train** split is inspected.  ``val`` and **test** are left alone.
* The frozen test set (``data/processed/classifiers/*/test``) is never touched.
* Run with ``KMP_DUPLICATE_LIB_OK=TRUE`` to avoid OpenMP conflicts on macOS.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Silence TF INFO/WARNING noise before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "data", "models")
CLASSIFIERS_DIR = os.path.join(REPO_ROOT, "data", "processed", "classifiers")

TASK_MAP = {
    "color": {
        "model_path": os.path.join(MODELS_DIR, "color_classifier.keras"),
        "data_dir": os.path.join(CLASSIFIERS_DIR, "colors"),
        "img_size": (224, 224),
    },
    "brand": {
        "model_path": os.path.join(MODELS_DIR, "brand_classifier.keras"),
        "data_dir": os.path.join(CLASSIFIERS_DIR, "brands"),
        "img_size": (224, 224),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_path: str) -> tf.keras.Model:
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading model: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def _build_dataset(
    train_dir: str,
    img_size: tuple[int, int],
    batch_size: int = 32,
) -> tuple[tf.data.Dataset, list[str]]:
    """Return a normalised (un-augmented) dataset over the train split."""
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,  # preserve path order for filename lookup
    )
    class_names: list[str] = ds.class_names
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), class_names


def _collect_file_paths(train_dir: str, class_names: list[str]) -> list[str]:
    """Walk the train directory in the same deterministic order TF uses."""
    paths: list[str] = []
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                paths.append(os.path.join(cls_dir, fname))
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def audit(task: str, thr: float) -> list[dict]:
    """Run label audit for *task* and return list of suspect dicts."""
    cfg = TASK_MAP[task]
    train_dir = os.path.join(cfg["data_dir"], "train")

    if not os.path.isdir(train_dir):
        print(f"ERROR: train directory not found: {train_dir}", file=sys.stderr)
        sys.exit(1)

    model = _load_model(cfg["model_path"])
    ds, class_names = _build_dataset(train_dir, cfg["img_size"])
    file_paths = _collect_file_paths(train_dir, class_names)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train images found: {len(file_paths)}")

    # Collect predictions
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    for x_batch, y_batch in ds:
        probs = model(x_batch, training=False).numpy()  # (B, C)
        all_probs.append(probs)
        all_labels.extend(y_batch.numpy().tolist())

    all_probs_np = np.concatenate(all_probs, axis=0)  # (N, C)

    suspects: list[dict] = []
    for i, (true_idx, probs) in enumerate(zip(all_labels, all_probs_np)):
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        if pred_idx != true_idx and conf >= thr:
            path = file_paths[i] if i < len(file_paths) else f"<unknown_{i}>"
            suspects.append(
                {
                    "path": path,
                    "true_label": class_names[true_idx],
                    "pred_label": class_names[pred_idx],
                    "confidence": conf,
                }
            )

    # Sort by confidence descending
    suspects.sort(key=lambda d: d["confidence"], reverse=True)
    return suspects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit train-split labels for a vehicle classifier."
    )
    parser.add_argument(
        "task",
        choices=list(TASK_MAP.keys()),
        help="Which classifier to audit.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.80,
        help="Confidence threshold for flagging a misclassification (default 0.80).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default 32).",
    )
    args = parser.parse_args()

    suspects = audit(args.task, args.thr)

    # --- Print summary ---
    print(f"\nSuspect images (model confident ≥ {args.thr:.0%} but label differs):")
    print(f"  Count: {len(suspects)}")
    if suspects:
        print(f"\n{'PATH':<80}  {'TRUE':<14}  {'PRED':<14}  CONF")
        print("-" * 120)
        for row in suspects[:50]:  # show at most 50 in console
            print(
                f"{row['path']:<80}  {row['true_label']:<14}  "
                f"{row['pred_label']:<14}  {row['confidence']:.3f}"
            )
        if len(suspects) > 50:
            print(f"  ... ({len(suspects) - 50} more — see CSV)")

    # --- Write CSV ---
    csv_path = os.path.join(MODELS_DIR, f"{args.task}_label_suspects.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "true_label", "pred_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(suspects)
    print(f"\nCSV written: {csv_path}  ({len(suspects)} rows)")


if __name__ == "__main__":
    main()
