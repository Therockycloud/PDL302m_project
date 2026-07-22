#!/usr/bin/env python3
"""Download external Vietnamese car license-plate image corpora."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

DEFAULT_OUTPUT = _MAIN / "data/raw/license_plates_kaggle"
KAGGLE_DATASET = "datnguyen1111/vietnamese-car-license-plate-detection"
WINTER2897_DRIVE_ID = "1KLK-DWgT3VoQH4fcTxAt2eB3sm7DGWAf"
WINTER2897_URL = f"https://drive.google.com/uc?id={WINTER2897_DRIVE_ID}"
MRZAIZAI2K_API = (
    "https://api.github.com/repos/mrzaizai2k/"
    "License-Plate-Recognition-YOLOv7-and-CNN/git/trees/main?recursive=1"
)
MRZAIZAI2K_RAW = (
    "https://raw.githubusercontent.com/mrzaizai2k/"
    "License-Plate-Recognition-YOLOv7-and-CNN/main"
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class DownloadAttempt:
    source: str
    success: bool
    detail: str
    image_count: int = 0
    label_count: int = 0


def _kaggle_available() -> tuple[bool, str]:
    kaggle_bin = shutil.which("kaggle")
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_bin is None:
        return False, "kaggle CLI not found on PATH"
    if not creds.is_file():
        return False, f"missing credentials at {creds}"
    return True, kaggle_bin


def _download_kaggle(dataset: str, output_dir: Path) -> DownloadAttempt:
    ok, detail = _kaggle_available()
    if not ok:
        return DownloadAttempt("kaggle", False, detail)
    archive = output_dir / "_downloads" / "kaggle_vn_car_plates.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    command = [
        detail,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(archive.parent),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return DownloadAttempt("kaggle", False, exc.stderr.strip() or str(exc))
    return DownloadAttempt(
        "kaggle",
        True,
        f"downloaded via Kaggle CLI to {archive.parent}",
    )


def _download_gdrive(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required for Google Drive fallback") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(destination), quiet=False)


def _extract_yolo_zip(zip_path: Path, train_root: Path) -> tuple[int, int]:
    images_dir = train_root / "images"
    labels_dir = train_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    label_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not name or name.startswith("."):
                continue
            lower = name.lower()
            if lower.endswith(IMAGE_SUFFIXES):
                target = images_dir / name
                if target.exists():
                    continue
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                image_count += 1
            elif lower.endswith(".txt"):
                target = labels_dir / name
                if target.exists():
                    continue
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                label_count += 1
    return image_count, label_count


def _github_list_paths() -> list[str]:
    request = urllib.request.Request(
        MRZAIZAI2K_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "PDL302m-project"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return [item["path"] for item in payload.get("tree", []) if item.get("type") == "blob"]


def _download_mrzaizai2k(train_root: Path) -> tuple[int, int]:
    images_dir = train_root / "images"
    labels_dir = train_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    paths = _github_list_paths()
    image_paths = [
        path for path in paths
        if path.startswith("data/test/images/") and path.lower().endswith(IMAGE_SUFFIXES)
    ]
    image_count = 0
    label_count = 0
    for rel_path in image_paths:
        stem = Path(rel_path).stem
        suffix = Path(rel_path).suffix
        image_target = images_dir / f"mrzaizai2k_{stem}{suffix}"
        label_target = labels_dir / f"mrzaizai2k_{stem}.txt"
        for rel, target, counter in (
            (rel_path, image_target, "image"),
            (f"data/test/labels/{stem}.txt", label_target, "label"),
        ):
            if target.exists():
                continue
            url = f"{MRZAIZAI2K_RAW}/{rel}"
            request = urllib.request.Request(url, headers={"User-Agent": "PDL302m-project"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    target.write_bytes(response.read())
            except urllib.error.HTTPError:
                continue
            if counter == "image":
                image_count += 1
            else:
                label_count += 1
    return image_count, label_count


def _inventory_yolo_root(train_root: Path) -> dict[str, object]:
    images = sorted((train_root / "images").glob("*"))
    labels = sorted((train_root / "labels").glob("*.txt"))
    resolutions: Counter[str] = Counter()
    try:
        import cv2
    except ImportError:
        cv2 = None
    sample_sizes: list[list[int]] = []
    for image_path in images[:200]:
        if cv2 is None:
            break
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        resolutions[f"{width}x{height}"] += 1
        if len(sample_sizes) < 5:
            sample_sizes.append([width, height])
    return {
        "image_count": len(images),
        "label_count": len(labels),
        "paired_estimate": len({path.stem for path in images} & {path.stem for path in labels}),
        "sample_resolutions": sample_sizes,
        "resolution_histogram_sample": dict(resolutions.most_common(10)),
    }


def write_provenance(
    output_dir: Path,
    *,
    attempts: list[DownloadAttempt],
    winter_inventory: dict[str, object] | None,
    mrzaizai2k_inventory: dict[str, object] | None,
) -> Path:
    provenance = output_dir / "PROVENANCE.md"
    lines = [
        "# External Vietnamese car plate corpus",
        "",
        "## Download attempts",
        "",
    ]
    for attempt in attempts:
        status = "success" if attempt.success else "failed"
        lines.append(f"- **{attempt.source}** ({status}): {attempt.detail}")
        if attempt.image_count:
            lines.append(f"  - images: {attempt.image_count}, labels: {attempt.label_count}")
    lines.extend([
        "",
        "## Primary corpus",
        "",
        "- **Preferred Kaggle target:** "
        f"[{KAGGLE_DATASET}](https://www.kaggle.com/datasets/{KAGGLE_DATASET})",
        "- **Used fallback:** Mi AI / winter2897 Vietnamese License Plate Detection (YOLO)",
        f"  - Google Drive file id: `{WINTER2897_DRIVE_ID}`",
        f"  - URL: {WINTER2897_URL}",
        "- **License:** publicly shared research corpus via Mi AI / winter2897 "
        "(see [dataset doc](https://github.com/winter2897/"
        "Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md)); "
        "no separate commercial license stated.",
        "",
        "## Supplement",
        "",
        "- **mrzaizai2k YOLOv7 LPR test frames**",
        f"  - GitHub raw base: {MRZAIZAI2K_RAW}",
        "- Stored under `mrzaizai2k/train/` without touching `main/data/raw/license_plates/clip3_new_*`.",
        "",
        "## Inventory",
        "",
        "```json",
        json.dumps(
            {
                "winter2897": winter_inventory,
                "mrzaizai2k": mrzaizai2k_inventory,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Kaggle credentials (if retrying preferred source)",
        "",
        "Place API credentials at `~/.kaggle/kaggle.json`:",
        "",
        "```json",
        '{"username":"<kaggle-username>","key":"<kaggle-api-key>"}',
        "```",
        "",
        "Then install the CLI (`pip install kaggle`) and rerun this script.",
        "",
    ])
    provenance.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return provenance


def run_download(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    skip_kaggle: bool = False,
    skip_mrzaizai2k: bool = False,
) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    downloads = output / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    attempts: list[DownloadAttempt] = []

    if not skip_kaggle:
        attempts.append(_download_kaggle(KAGGLE_DATASET, output))
    else:
        ok, detail = _kaggle_available()
        attempts.append(
            DownloadAttempt(
                "kaggle",
                False,
                detail if not ok else "skipped via --skip-kaggle",
            )
        )

    winter_zip = downloads / "winter2897_plate_yolo.zip"
    winter_train = output / "train"
    winter_ok = False
    winter_images = 0
    winter_labels = 0
    if not any(item.source == "kaggle" and item.success for item in attempts):
        try:
            if not winter_zip.is_file():
                _download_gdrive(WINTER2897_DRIVE_ID, winter_zip)
            winter_images, winter_labels = _extract_yolo_zip(winter_zip, winter_train)
            winter_inventory_probe = _inventory_yolo_root(winter_train)
            winter_ok = winter_inventory_probe["image_count"] > 0
            attempts.append(
                DownloadAttempt(
                    "winter2897-gdrive",
                    winter_ok,
                    f"extracted YOLO corpus from {WINTER2897_URL}",
                    image_count=int(winter_inventory_probe["image_count"]),
                    label_count=int(winter_inventory_probe["label_count"]),
                )
            )
        except Exception as exc:
            attempts.append(DownloadAttempt("winter2897-gdrive", False, str(exc)))

    mrzaizai2k_inventory = None
    if not skip_mrzaizai2k:
        mr_train = output / "mrzaizai2k" / "train"
        try:
            mr_images, mr_labels = _download_mrzaizai2k(mr_train)
            mrzaizai2k_inventory = _inventory_yolo_root(mr_train)
            attempts.append(
                DownloadAttempt(
                    "mrzaizai2k-github",
                    mrzaizai2k_inventory["image_count"] > 0,
                    f"mirrored frames from {MRZAIZAI2K_RAW}",
                    image_count=int(mrzaizai2k_inventory["image_count"]),
                    label_count=int(mrzaizai2k_inventory["label_count"]),
                )
            )
        except Exception as exc:
            attempts.append(DownloadAttempt("mrzaizai2k-github", False, str(exc)))

    winter_inventory = _inventory_yolo_root(winter_train) if winter_train.is_dir() else None
    provenance = write_provenance(
        output,
        attempts=attempts,
        winter_inventory=winter_inventory,
        mrzaizai2k_inventory=mrzaizai2k_inventory,
    )
    return {
        "output_dir": str(output),
        "provenance": str(provenance),
        "attempts": [asdict(attempt) for attempt in attempts],
        "winter_inventory": winter_inventory,
        "mrzaizai2k_inventory": mrzaizai2k_inventory,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument("--skip-mrzaizai2k", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_download(
        output_dir=args.output,
        skip_kaggle=args.skip_kaggle,
        skip_mrzaizai2k=args.skip_mrzaizai2k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
