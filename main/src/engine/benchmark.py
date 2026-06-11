"""Generic model-comparison harness.

Measures top-1 accuracy, mean CPU latency, parameter count, and on-disk
size for a set of candidates over a labelled dataset, and renders a
Markdown comparison table. Model-agnostic: candidates just implement the
``BenchmarkCandidate`` protocol.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd


class BenchmarkCandidate(Protocol):
    name: str
    num_params: int
    size_mb: float

    def predict(self, X: np.ndarray) -> np.ndarray:  # class indices
        ...


class ModelBenchmark:
    """Run accuracy/latency/size comparison across candidates."""

    def run(self, candidates, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        rows = []
        y = np.asarray(y)
        for c in candidates:
            t0 = time.perf_counter()
            preds = np.asarray(c.predict(X))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            accuracy = float((preds == y).mean()) if len(y) else 0.0
            latency_ms = elapsed_ms / max(1, len(X))
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            rows.append(
                {
                    "name": c.name,
                    "accuracy": round(accuracy, 4),
                    "latency_ms": round(latency_ms, 3),
                    "fps": round(fps, 1),
                    "num_params": int(c.num_params),
                    "size_mb": round(float(c.size_mb), 2),
                }
            )
        return pd.DataFrame(rows)

    def to_report(self, df: pd.DataFrame) -> tuple[str, list[str]]:
        md = df.to_markdown(index=False)
        return md, []
