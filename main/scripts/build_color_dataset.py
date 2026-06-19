#!/usr/bin/env python3
"""Merge the VCoR vehicle-colour dataset with the project's existing colour data.

Sources (READ-ONLY — never modified):
  * VCoR (newly downloaded), one folder per split, lowercase colour subfolders::

        /Users/konalyn/Downloads/archive/{train,val,test}/<lowercolor>/*.jpg

  * Existing project data (already curated, ~100 images/class)::

        main/data/raw/car_colors/<Class>/*

Output (created/overwritten by this script — safe to re-run)::

    main/data/raw/car_colors_vcor/<Class>/

Only 8 of VCoR's 15 colours are mapped; the other 7 (beige, gold, green,
orange, pink, purple, tan) are intentionally ignored because the project's
label set doesn't include them.

Usage::

    cd main && python scripts/build_color_dataset.py

Notes
-----
* This script is READ-ONLY on every source directory. It only writes under
  ``main/data/raw/car_colors_vcor/``.
* Re-running clears each destination class folder first, so counts never
  double up across runs.
* Every image is validated with ``cv2.imread`` before being counted; anything
  that fails to decode is skipped and tallied separately.
"""

from __future__ import annotations

import os
import shutil
import sys

import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VCOR_ROOT = "/Users/konalyn/Downloads/archive"
VCOR_SPLITS = ("train", "val", "test")

EXISTING_DIR = os.path.join(REPO_ROOT, "data", "raw", "car_colors")
DEST_DIR = os.path.join(REPO_ROOT, "data", "raw", "car_colors_vcor")

# VCoR lowercase folder -> our Class folder. VCoR's other 7 colours
# (beige, gold, green, orange, pink, purple, tan) are intentionally ignored.
CLASS_MAP = {
    "black": "Black",
    "blue": "Blue",
    "brown": "Brown",
    "grey": "Grey",
    "red": "Red",
    "silver": "Silver",
    "white": "White",
    "yellow": "Yellow",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def validate_image(path: str) -> bool:
    """Return True if cv2 can decode this image."""
    img = cv2.imread(path)
    return img is not None


def copy_validated(src_path: str, dest_path: str, stats: dict) -> None:
    """Validate src_path with cv2 and copy to dest_path if it decodes."""
    if not validate_image(src_path):
        stats["skipped"] += 1
        stats["skipped_paths"].append(src_path)
        return
    shutil.copy2(src_path, dest_path)
    stats["copied"] += 1


def build_class(vcor_color: str, class_name: str) -> dict:
    """Build one merged class folder; returns per-class stats dict."""
    dest_class_dir = os.path.join(DEST_DIR, class_name)

    # Re-runnable: clear destination class dir first so re-runs don't
    # double-count. This only touches our own output tree, never the sources.
    if os.path.isdir(dest_class_dir):
        shutil.rmtree(dest_class_dir)
    os.makedirs(dest_class_dir, exist_ok=True)

    stats = {"copied": 0, "skipped": 0, "skipped_paths": []}

    # 1) VCoR train/val/test for this colour
    for split in VCOR_SPLITS:
        src_dir = os.path.join(VCOR_ROOT, split, vcor_color)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            src_path = os.path.join(src_dir, fname)
            if not os.path.isfile(src_path) or not is_image_file(fname):
                continue
            dest_name = f"vcor_{split}_{fname}"
            dest_path = os.path.join(dest_class_dir, dest_name)
            copy_validated(src_path, dest_path, stats)

    # 2) Existing project data for this class
    src_existing_dir = os.path.join(EXISTING_DIR, class_name)
    if os.path.isdir(src_existing_dir):
        for fname in sorted(os.listdir(src_existing_dir)):
            src_path = os.path.join(src_existing_dir, fname)
            if not os.path.isfile(src_path) or not is_image_file(fname):
                continue
            dest_name = f"orig_{fname}"
            dest_path = os.path.join(dest_class_dir, dest_name)
            copy_validated(src_path, dest_path, stats)

    return stats


def main() -> int:
    if not os.path.isdir(VCOR_ROOT):
        print(f"ERROR: VCoR source root not found: {VCOR_ROOT}", file=sys.stderr)
        return 1
    if not os.path.isdir(EXISTING_DIR):
        print(f"ERROR: existing data dir not found: {EXISTING_DIR}", file=sys.stderr)
        return 1

    os.makedirs(DEST_DIR, exist_ok=True)

    per_class_counts: dict[str, int] = {}
    total_skipped = 0
    all_skipped_paths: list[str] = []

    for vcor_color, class_name in CLASS_MAP.items():
        stats = build_class(vcor_color, class_name)
        per_class_counts[class_name] = stats["copied"]
        total_skipped += stats["skipped"]
        all_skipped_paths.extend(stats["skipped_paths"])
        print(
            f"[{class_name:6s}] copied={stats['copied']:5d}  "
            f"skipped(corrupt)={stats['skipped']:3d}"
        )

    grand_total = sum(per_class_counts.values())

    print()
    print("=" * 40)
    print("Per-class final counts")
    print("=" * 40)
    for class_name in CLASS_MAP.values():
        print(f"  {class_name:8s}: {per_class_counts[class_name]}")
    print("-" * 40)
    print(f"  {'TOTAL':8s}: {grand_total}")
    print("=" * 40)
    print(f"Skipped/corrupt images: {total_skipped}")
    if all_skipped_paths:
        print("Skipped paths:")
        for p in all_skipped_paths:
            print(f"  - {p}")
    print()
    print(f"Output written to: {DEST_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
