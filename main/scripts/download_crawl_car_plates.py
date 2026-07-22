#!/usr/bin/env python3
"""Extract and resume-download crawl corpora for Vietnamese car plate review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

HF_ROOT = _MAIN / "data/raw/license_plates_crawl_hf"
TRUNGDINH_ROOT = _MAIN / "data/raw/license_plates_crawl_trungdinh"
HF_ZIP = HF_ROOT / "_downloads/dataset.zip"
TRUNGDINH_DRIVE_ID = "1xchPXf7a1r466ngow_W_9bittRqQEf_T"
TRUNGDINH_URL = f"https://drive.google.com/uc?id={TRUNGDINH_DRIVE_ID}"
TRUNGDINH_REPO = "https://github.com/trungdinh22/License-Plate-Recognition"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
YOLO_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class ExtractResult:
    split: str
    image_count: int
    label_count: int
    paired_estimate: int


@dataclass(frozen=True, slots=True)
class CrawlAttempt:
    source: str
    success: bool
    detail: str
    image_count: int = 0
    label_count: int = 0


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _class_histogram(labels_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for label_path in labels_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if values:
                counts[values[0]] += 1
    return dict(sorted(counts.items()))


def _inventory_split(split_root: Path) -> ExtractResult:
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    images = sorted(images_dir.glob("*")) if images_dir.is_dir() else []
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    return ExtractResult(
        split=split_root.name,
        image_count=len(images),
        label_count=len(labels),
        paired_estimate=len(image_stems & label_stems),
    )


def inventory_yolo_splits(root: Path) -> dict[str, object]:
    splits = [_inventory_split(root / split) for split in YOLO_SPLITS if (root / split).is_dir()]
    class_hist: dict[str, int] = Counter()
    sample_resolutions: list[list[int]] = []
    resolution_hist: Counter[str] = Counter()
    try:
        import cv2
    except ImportError:
        cv2 = None
    for split in YOLO_SPLITS:
        labels_dir = root / split / "labels"
        if labels_dir.is_dir():
            for class_id, count in _class_histogram(labels_dir).items():
                class_hist[class_id] += count
        images_dir = root / split / "images"
        if cv2 is None or not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.glob("*"))[:80]:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            resolution_hist[f"{width}x{height}"] += 1
            if len(sample_resolutions) < 5:
                sample_resolutions.append([width, height])
    return {
        "splits": [asdict(item) for item in splits],
        "total_images": sum(item.image_count for item in splits),
        "total_labels": sum(item.label_count for item in splits),
        "class_histogram": dict(sorted(class_hist.items())),
        "sample_resolutions": sample_resolutions,
        "resolution_histogram_sample": dict(resolution_hist.most_common(10)),
    }


def extract_hf_yolo_zip(
    zip_path: Path,
    output_root: Path,
    *,
    splits: tuple[str, ...] = YOLO_SPLITS,
) -> list[ExtractResult]:
    """Extract HF-style ``images/{split}`` + ``labels/{split}`` into per-split YOLO roots."""

    if not zip_path.is_file():
        raise FileNotFoundError(f"missing HF zip: {zip_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        for split in splits:
            split_root = output_root / split
            images_dir = split_root / "images"
            labels_dir = split_root / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            image_count = 0
            label_count = 0
            for member in members:
                normalized = member.replace("\\", "/")
                if normalized.startswith(f"images/{split}/"):
                    name = Path(member).name
                    if not name or not name.lower().endswith(IMAGE_SUFFIXES):
                        continue
                    target = images_dir / name
                    if target.exists():
                        continue
                    with archive.open(member) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    image_count += 1
                elif normalized.startswith(f"labels/{split}/"):
                    name = Path(member).name
                    if not name or not name.lower().endswith(".txt"):
                        continue
                    target = labels_dir / name
                    if target.exists():
                        continue
                    with archive.open(member) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    label_count += 1
            inventory = _inventory_split(split_root)
            results.append(
                ExtractResult(
                    split=split,
                    image_count=max(image_count, inventory.image_count),
                    label_count=max(label_count, inventory.label_count),
                    paired_estimate=inventory.paired_estimate,
                )
            )
    return results


def _download_gdrive(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required for Google Drive downloads") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(destination), quiet=False, resume=True)


def _cleanup_partial_downloads(downloads_dir: Path, *, final_name: str) -> None:
    for path in downloads_dir.glob("*.part"):
        path.unlink(missing_ok=True)
    for path in downloads_dir.glob(f"{final_name}.*.part"):
        path.unlink(missing_ok=True)


def extract_flat_yolo_zip(zip_path: Path, train_root: Path) -> tuple[int, int]:
    """Extract a flat YOLO zip into ``train_root/images`` and ``train_root/labels``."""

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


def write_hf_provenance(
    output_root: Path,
    *,
    zip_path: Path,
    zip_sha256: str | None,
    extract_results: list[ExtractResult],
    inventory: dict[str, object],
) -> Path:
    provenance = output_root / "PROVENANCE.md"
    try:
        zip_rel = zip_path.relative_to(_MAIN).as_posix()
    except ValueError:
        zip_rel = zip_path.as_posix()
    lines = [
        "# Hugging Face crawl corpus (TongHop-style VN plate detection)",
        "",
        "## Origin",
        "",
        "- **Confirmed:** zip extracted from local Hugging Face cache artifact "
        f"at `{zip_rel}`.",
        "- **dataset.yaml inside zip:** classes `BSD`, `BSV` (Vietnamese long vs square "
        "plate boxes); Windows paths reference `TongHop\\YOLODataset/...`.",
        "- **NOT verified as** `keremberke/license-plate-object-detection` "
        "(that dataset is a generic international plate detector; different layout/classes).",
        "- **Public HF dataset URL:** unknown / not recorded in local cache metadata "
        "(only commit + sha256 sidecar present under `_downloads/.cache/huggingface/`).",
        "- **License:** unknown / not stated in the archive.",
        "",
        "## Archive",
        "",
        f"- zip path: `{zip_path}`",
        f"- zip sha256: `{zip_sha256 or 'not computed'}`",
        f"- zip size bytes: {zip_path.stat().st_size if zip_path.is_file() else 0}",
        "",
        "## Extract layout",
        "",
        "Per-split YOLO folders under this directory:",
        "",
        "```",
        "train/images  train/labels",
        "val/images    val/labels",
        "test/images   test/labels",
        "```",
        "",
        "## Split inventory",
        "",
        "```json",
        json.dumps([asdict(item) for item in extract_results], indent=2, sort_keys=True),
        "```",
        "",
        "## Corpus inventory",
        "",
        "```json",
        json.dumps(inventory, indent=2, sort_keys=True),
        "```",
        "",
    ]
    provenance.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return provenance


def write_trungdinh_provenance(
    output_root: Path,
    *,
    attempt: CrawlAttempt,
    inventory: dict[str, object] | None,
) -> Path:
    provenance = output_root / "PROVENANCE.md"
    lines = [
        "# trungdinh22 License Plate Detection crawl",
        "",
        "## Origin",
        "",
        f"- GitHub repo: [{TRUNGDINH_REPO}]({TRUNGDINH_REPO})",
        f"- Google Drive file id: `{TRUNGDINH_DRIVE_ID}`",
        f"- URL: {TRUNGDINH_URL}",
        "- **Overlap risk:** same Drive corpus is cited by Mi AI / winter2897 Vietnamese "
        "plate-det docs and prior `license_plates_kaggle` import (~29 unique car labels after reserved).",
        "- **License:** publicly shared research corpus; no separate commercial license stated.",
        "",
        "## Download attempt",
        "",
        f"- status: {'success' if attempt.success else 'failed'}",
        f"- detail: {attempt.detail}",
        "",
    ]
    if inventory is not None:
        lines.extend([
            "## Inventory",
            "",
            "```json",
            json.dumps(inventory, indent=2, sort_keys=True),
            "```",
            "",
        ])
    provenance.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return provenance


def run_hf_extract(
    *,
    output_root: str | Path = HF_ROOT,
    zip_path: str | Path = HF_ZIP,
    compute_sha256: bool = True,
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    archive = Path(zip_path).expanduser().resolve()
    zip_sha256 = sha256_file(archive) if compute_sha256 and archive.is_file() else None
    extract_results = extract_hf_yolo_zip(archive, output)
    inventory = inventory_yolo_splits(output)
    provenance = write_hf_provenance(
        output,
        zip_path=archive,
        zip_sha256=zip_sha256,
        extract_results=extract_results,
        inventory=inventory,
    )
    return {
        "output_dir": str(output),
        "zip_path": str(archive),
        "zip_sha256": zip_sha256,
        "extract_results": [asdict(item) for item in extract_results],
        "inventory": inventory,
        "provenance": str(provenance),
    }


def run_trungdinh_download(
    *,
    output_root: str | Path = TRUNGDINH_ROOT,
    drive_id: str = TRUNGDINH_DRIVE_ID,
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    downloads = output / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / "trungdinh_plate_det.zip"
    attempt: CrawlAttempt
    inventory: dict[str, object] | None = None
    try:
        _cleanup_partial_downloads(downloads, final_name=archive.name)
        if not archive.is_file() or archive.stat().st_size < 1_000_000:
            _download_gdrive(drive_id, archive)
        train_root = output / "train"
        images, labels = extract_flat_yolo_zip(archive, train_root)
        inventory = inventory_yolo_splits(output) if any((output / split).is_dir() for split in YOLO_SPLITS) else {
            "train": {
                "image_count": images,
                "label_count": labels,
            }
        }
        ok = images > 0 or int(inventory.get("total_images", 0)) > 0
        attempt = CrawlAttempt(
            "trungdinh-gdrive",
            ok,
            f"downloaded from {TRUNGDINH_URL}",
            image_count=int(inventory.get("total_images", images) if isinstance(inventory, dict) else images),
            label_count=int(inventory.get("total_labels", labels) if isinstance(inventory, dict) else labels),
        )
    except Exception as exc:
        attempt = CrawlAttempt("trungdinh-gdrive", False, str(exc))
    provenance = write_trungdinh_provenance(output, attempt=attempt, inventory=inventory)
    return {
        "output_dir": str(output),
        "attempt": asdict(attempt),
        "inventory": inventory,
        "provenance": str(provenance),
    }


def run_all(
    *,
    skip_hf: bool = False,
    skip_trungdinh: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if not skip_hf:
        payload["hf"] = run_hf_extract()
    if not skip_trungdinh:
        payload["trungdinh"] = run_trungdinh_download()
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-only", action="store_true")
    parser.add_argument("--trungdinh-only", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-trungdinh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.hf_only:
        result = {"hf": run_hf_extract()}
    elif args.trungdinh_only:
        result = {"trungdinh": run_trungdinh_download()}
    else:
        result = run_all(
            skip_hf=args.skip_hf,
            skip_trungdinh=args.skip_trungdinh,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
