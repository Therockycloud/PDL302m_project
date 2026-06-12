"""Benchmark plate detectors (Group B): pretrained-finetune vs trained.

Compares two YOLO plate models on a labelled validation set using
Ultralytics' built-in mAP, plus latency/size. Writes
docs/benchmarks/plate_benchmark.{csv,md}.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Ensure the project root (main/) is on the path when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def measure(model_path: str, data_yaml: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device="cpu", verbose=False)
    size_mb = round(os.path.getsize(model_path) / 1e6, 2) if os.path.exists(model_path) else 0.0
    speed = getattr(metrics, "speed", {}) or {}
    latency_ms = round(float(speed.get("inference", 0.0)), 3)
    # Label by the run directory name (".../plate_finetune/weights/best.pt"
    # -> "plate_finetune") so the two candidates are distinguishable.
    run_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    return {
        "name": run_name or os.path.basename(model_path),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "latency_ms": latency_ms,
        "size_mb": size_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to plate data.yaml")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--trained", required=True)
    args = ap.parse_args()

    rows = [measure(args.pretrained, args.data), measure(args.trained, args.data)]
    df = pd.DataFrame(rows)
    out_dir = os.path.join("..", "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "plate_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, "plate_benchmark.md"), "w") as fh:
        fh.write(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
