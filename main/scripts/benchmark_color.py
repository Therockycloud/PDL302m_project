"""Benchmark color-classifier backbones (Group A).

Wraps each Keras backbone in a BenchmarkCandidate and compares accuracy,
latency, params, and size on the processed color dataset. Writes
docs/benchmarks/color_benchmark.{csv,md}.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# Ensure the project root (main/) is on the path when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.benchmark import ModelBenchmark


class KerasCandidate:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.num_params = model.count_params()
        self.size_mb = round(self.num_params * 4 / 1e6, 2)  # float32 estimate

    def predict(self, X):
        # Use direct __call__ (avoids the tf.data prefetch pipeline that can
        # deadlock on macOS when called from model.predict()).
        probs = self.model(X, training=False).numpy()
        return np.argmax(probs, axis=1)


def build_candidates(input_shape, num_classes):
    import tensorflow as tf
    from tensorflow import keras

    def head(base):
        inp = keras.Input(shape=input_shape)
        x = base(inp, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inp, out)

    return [
        KerasCandidate("MobileNetV3Small", head(tf.keras.applications.MobileNetV3Small(include_top=False, weights=None, input_shape=input_shape))),
        KerasCandidate("EfficientNetB0", head(tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape))),
        KerasCandidate("ResNet50", head(tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap samples for smoke runs")
    args = ap.parse_args()

    # Smoke dataset if no processed data wired yet.
    X = np.random.rand(args.limit or 8, 224, 224, 3).astype("float32")
    y = np.random.randint(0, 8, size=len(X))

    cands = build_candidates((224, 224, 3), 8)
    bench = ModelBenchmark()
    df = bench.run(cands, X, y)
    md, _ = bench.to_report(df)

    out_dir = os.path.join("..", "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "color_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, "color_benchmark.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
