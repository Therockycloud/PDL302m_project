"""Download + prepare the license-plate detection dataset for YOLO training.

Pulls the Roboflow YOLOv8 export zips from the public HuggingFace dataset
``keremberke/license-plate-object-detection`` (no auth), unzips them under
``main/data/raw/plate_det/``, and writes a YOLO ``data.yaml``.

Run from the repo root or from ``main/`` — paths are resolved relative to this
file, so CWD does not matter.
"""

from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

_THIS = Path(__file__).resolve()
_MAIN = _THIS.parents[1]                      # .../main
_DEST = _MAIN / "data" / "raw" / "plate_det"

_BASE = "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data"
_ZIPS = {"train": "train.zip", "valid": "valid.zip", "test": "test.zip"}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"[done] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


def _unzip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"[unzip] {zip_path.name} -> {out_dir}")


def _coco_to_yolo(split_dir: Path) -> int:
    """Convert a Roboflow COCO export split into YOLO ``images/`` + ``labels/``.

    The export ships images flat in *split_dir* alongside a single
    ``_annotations.coco.json`` (bbox = absolute ``[x, y, w, h]``, top-left
    origin, single class id 0). Returns the number of images placed under
    ``images/``. Idempotent: re-running skips work already done.
    """
    import json
    import shutil

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[ERROR] no _annotations.coco.json in {split_dir}")
        return 0
    coco = json.loads(ann_path.read_text())

    by_id = {img["id"]: img for img in coco["images"]}
    boxes: dict[int, list[str]] = {img_id: [] for img_id in by_id}
    for a in coco["annotations"]:
        img = by_id.get(a["image_id"])
        if img is None:
            continue
        W, H = float(img["width"]), float(img["height"])
        x, y, w, h = a["bbox"]
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        boxes[a["image_id"]].append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")

    n = 0
    for img_id, img in by_id.items():
        fname = img["file_name"]
        src = split_dir / fname
        dst = images_dir / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        if dst.exists():
            n += 1
        # write label (empty file for background images is valid for YOLO)
        (labels_dir / (Path(fname).stem + ".txt")).write_text(
            "\n".join(boxes.get(img_id, [])), encoding="utf-8"
        )
    return n


def main() -> int:
    _DEST.mkdir(parents=True, exist_ok=True)
    for split, fname in _ZIPS.items():
        zp = _DEST / fname
        _download(f"{_BASE}/{fname}", zp)
        out = _DEST / split
        if not out.exists():
            _unzip(zp, out)

    n_tr = _coco_to_yolo(_DEST / "train")
    n_va = _coco_to_yolo(_DEST / "valid")
    print(f"\ntrain images: {n_tr}   valid images: {n_va}")

    if n_tr < 100 or n_va < 10:
        print("\n[ERROR] not enough images after conversion — inspect the tree above.")
        return 1

    data_yaml = _DEST / "data.yaml"
    yaml_text = (
        f"path: {_DEST}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 1\n"
        f"names: ['license_plate']\n"
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    print(f"\n[ok] wrote {data_yaml}\n{yaml_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
