"""Create a reproducible train/val/test split for the classifier datasets.

Reads the balanced canonical trees (``raw/car_colors``, ``raw/car_brands``,
100 img/class) and writes a physical, seeded split into
``processed/classifiers/<task>/{train,val,test}/<class>``. A physical split
gives a STABLE held-out test set (the previous pipeline had no test set at all —
``image_dataset_from_directory`` only produced train/val via validation_split).

Default ratios 70/15/15 -> per class: 70 train, 15 val, 15 test.
"""
import argparse
import random
import shutil
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
RAW = MAIN / "data" / "raw"
OUT = MAIN / "data" / "processed" / "classifiers"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TASKS = {"colors": "car_colors", "brands": "car_brands"}


def split_task(task: str, src_name: str, ratios, seed: int):
    src = RAW / src_name
    rng = random.Random(seed)
    summary = {}
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS)
        rng.shuffle(files)
        n = len(files)
        n_test = int(round(n * ratios[2]))
        n_val = int(round(n * ratios[1]))
        n_train = n - n_val - n_test
        parts = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in parts.items():
            dst = OUT / task / split / cls_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dst / f.name)
        summary[cls_dir.name] = {k: len(v) for k, v in parts.items()}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ratios = (args.train, args.val, args.test)
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    for task, src_name in TASKS.items():
        print(f"== {task} ({src_name}) ==")
        summary = split_task(task, src_name, ratios, args.seed)
        tot = {"train": 0, "val": 0, "test": 0}
        for cls, c in summary.items():
            for k in tot:
                tot[k] += c[k]
        print(f"  per-class: train={summary[next(iter(summary))]['train']} "
              f"val={summary[next(iter(summary))]['val']} "
              f"test={summary[next(iter(summary))]['test']}")
        print(f"  TOTAL: train={tot['train']} val={tot['val']} test={tot['test']}")


if __name__ == "__main__":
    main()
