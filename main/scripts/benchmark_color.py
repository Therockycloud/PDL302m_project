"""Benchmark colour-classifier backbones on the REAL car-colour dataset (Group A).

Uses PyTorch + torchvision on Apple MPS. TensorFlow's ``model.fit`` deadlocks
at 0% CPU on this macOS build, whereas torch/MPS trains reliably (same path
that trained the YOLO detector). Compares MobileNetV3-Small, EfficientNet-B0
and ResNet50: top-1 accuracy, macro-F1, CPU latency, params, size.

Results are written incrementally to ``docs/benchmarks/color_benchmark.{csv,md}``
and finished backbones are skipped on re-run (resume-safe).

Run from ``main/``::

    python scripts/benchmark_color.py --epochs 15
    python scripts/benchmark_color.py --epochs 15 --only ResNet50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

cv2.setNumThreads(0)
_MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAIN))

DATA_DIR = _MAIN / "data" / "raw" / "car_colors"
OUT_DIR = _MAIN.parent / "docs" / "benchmarks"
IMG = 224
ALL_BACKBONES = ["MobileNetV3Small", "EfficientNetB0", "ResNet50"]
_MEAN = np.array([0.485, 0.456, 0.406], "float32")
_STD = np.array([0.229, 0.224, 0.225], "float32")


def _load_data(seed: int = 42):
    """Load all images into normalized CHW tensors with a per-class 80/20 split."""
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    rng = np.random.default_rng(seed)
    tr, ytr, va, yva = [], [], [], []
    for ci, c in enumerate(classes):
        imgs = []
        for f in sorted((DATA_DIR / c).glob("*")):
            im = cv2.imread(str(f))
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            im = (im - _MEAN) / _STD
            imgs.append(im.transpose(2, 0, 1))  # HWC -> CHW
        if not imgs:
            continue
        idx = rng.permutation(len(imgs))
        n_val = max(1, int(0.2 * len(imgs)))
        for j, k in enumerate(idx):
            (va if j < n_val else tr).append(imgs[k])
            (yva if j < n_val else ytr).append(ci)
    return (torch.tensor(np.asarray(tr)), torch.tensor(ytr),
            torch.tensor(np.asarray(va)), torch.tensor(yva), classes)


def _build(name: str, num_classes: int):
    if name == "MobileNetV3Small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        head = m.classifier[3]
    elif name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head = m.classifier[1]
    else:
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    for p in m.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    return m


def _batches(X, y, bs, shuffle, device):
    idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j].to(device), (y[j].to(device) if y is not None else None)


@torch.no_grad()
def _predict(model, X, bs, device):
    model.eval()
    out = []
    for xb, _ in _batches(X, None, bs, False, device):
        out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only", choices=ALL_BACKBONES)
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    X_tr, y_tr, X_va, y_va, classes = _load_data()
    print(f"device={device} classes={classes} train={len(y_tr)} val={len(y_va)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "color_benchmark.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["name"] for r in rows}
    targets = [args.only] if args.only else ALL_BACKBONES

    for name in targets:
        if name in done:
            print(f"=== skip {name} ===", flush=True)
            continue
        print(f"\n=== {name}: training {args.epochs} epochs on {device} ===", flush=True)
        model = _build(name, len(classes)).to(device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            model.train()
            t0 = time.perf_counter()
            tot = 0.0
            for xb, yb in _batches(X_tr, y_tr, args.batch, True, device):
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            va_acc = accuracy_score(y_va.numpy(), _predict(model, X_va, args.batch, device))
            print(f"  epoch {ep + 1}/{args.epochs}  loss={tot / len(X_tr):.3f}  "
                  f"val_acc={va_acc:.3f}  ({time.perf_counter() - t0:.1f}s)", flush=True)

        preds = _predict(model, X_va, args.batch, device)
        acc = round(float(accuracy_score(y_va.numpy(), preds)), 4)
        f1 = round(float(f1_score(y_va.numpy(), preds, average="macro")), 4)

        # CPU latency (edge-relevance) on a single image
        model_cpu = model.to("cpu").eval()
        x1 = X_va[:1]
        for _ in range(3):
            model_cpu(x1)  # warmup
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                model_cpu(x1)
        lat = round((time.perf_counter() - t0) / 20 * 1000, 2)

        n_params = sum(p.numel() for p in model.parameters())
        ckpt = _MAIN / "data" / "models" / f"color_{name}.pt"
        torch.save(model.state_dict(), ckpt)
        size_mb = round(os.path.getsize(ckpt) / 1e6, 2)

        rows.append({"name": name, "accuracy": acc, "macro_f1": f1, "latency_ms": lat,
                     "num_params": n_params, "size_mb": size_mb})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"{name}: acc={acc} macroF1={f1} cpu_lat={lat}ms params={n_params} size={size_mb}MB", flush=True)
        del model, model_cpu
        if device == "mps":
            torch.mps.empty_cache()

    df = pd.DataFrame(rows)[["name", "accuracy", "macro_f1", "latency_ms", "num_params", "size_mb"]]
    (OUT_DIR / "color_benchmark.md").write_text(
        "# Benchmark A — Vehicle Colour CNN (PyTorch / MPS)\n\n"
        f"Dataset: `main/data/raw/car_colors` ({len(classes)} classes, {len(y_va)} val images). "
        f"Transfer learning (frozen ImageNet backbone, trained head), {args.epochs} epochs. "
        "Latency = single-image CPU inference.\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8")
    print("\n" + df.to_markdown(index=False), flush=True)


if __name__ == "__main__":
    main()
