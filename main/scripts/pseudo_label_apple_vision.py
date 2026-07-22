#!/usr/bin/env python3
"""Pseudo-label Vietnamese plate crops with Apple Vision text recognition."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from scripts.build_plate_ocr_dataset import (  # noqa: E402
    MANIFEST_FIELDS,
    _iter_yolo_pairs,
    _safe_write,
    _sha256,
    _write_manifest,
    parse_yolo_box,
)
from src.datasets.plate_ocr_dataset import load_plate_manifest  # noqa: E402
from src.models.vn_plate_text import normalize_plate_text, validate_vietnamese_plate  # noqa: E402

DEFAULT_SOURCES = (
    _MAIN / "data/raw/plate_det/train",
    _MAIN / "data/raw/plate_det/valid",
)
DEFAULT_RESERVED = (
    _MAIN / "data/plate_ocr/real_validation.csv",
    _MAIN / "data/plate_ocr/expanded_real_test.csv",
    _MAIN / "data/plate_ocr/frozen_regression.csv",
)
DEFAULT_OUTPUT = _MAIN / "data/plate_ocr/generated/pseudo_vision"
DEFAULT_AUDIT_DIR = _MAIN / "data/plate_ocr/review/pseudo_vision_audit"
DEFAULT_EXCLUDE_PATTERNS = ("*Gen*",)
AUDIT_SEED = 20260713
AUDIT_COLUMNS = 6
AUDIT_ROWS = 5
AUDIT_PER_SHEET = AUDIT_COLUMNS * AUDIT_ROWS
AUDIT_SAMPLE_PER_BAND = 30
MIN_MERGED_BAND_ROWS = 10
UPSCALE_MIN_HEIGHT = 128


@dataclass
class DropStats:
    total_boxes: int = 0
    excluded_source: int = 0
    reserved_source_hash: int = 0
    too_small: int = 0
    unreadable: int = 0
    invalid_format: int = 0
    reserved_label: int = 0
    reserved_crop_hash: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_boxes": self.total_boxes,
            "dropped": {
                "excluded-source": self.excluded_source,
                "reserved-source-hash": self.reserved_source_hash,
                "too-small": self.too_small,
                "unreadable": self.unreadable,
                "invalid-format": self.invalid_format,
                "reserved-label": self.reserved_label,
                "reserved-crop-hash": self.reserved_crop_hash,
            },
            "kept": self.kept,
        }


@dataclass(frozen=True, slots=True)
class VisionObservation:
    text: str
    confidence: float
    y_center: float


@dataclass(frozen=True, slots=True)
class PlateRecognition:
    label: str
    confidence: float
    parts: int
    upscaled_height: int


RecognizeFn = Callable[[np.ndarray], list[tuple[str, float, float]]]


def build_reserved_identities(
    reserved_manifests: Sequence[str | Path],
) -> tuple[set[str], set[str], set[str]]:
    """Return reserved labels, source hashes, and crop hashes."""

    labels: set[str] = set()
    source_hashes: set[str] = set()
    crop_hashes: set[str] = set()
    for manifest in reserved_manifests:
        for row in load_plate_manifest(manifest):
            labels.add(row.label)
            for field_name, target in (
                ("source_sha256", source_hashes),
                ("crop_sha256", crop_hashes),
            ):
                value = str(row.metadata.get(field_name, "")).strip().lower()
                if value:
                    target.add(value)
    return labels, source_hashes, crop_hashes


def _levenshtein_distance(left: str, right: str) -> int:
    """Return edit distance between two strings (small inputs only)."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for col, right_char in enumerate(right, start=1):
            insert_cost = current[col - 1] + 1
            delete_cost = previous[col] + 1
            replace_cost = previous[col - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def label_conflicts_with_reserved(label: str, reserved_labels: set[str]) -> bool:
    """True when *label* exactly or near-matches any reserved plate identity."""

    candidate = normalize_plate_text(label)
    if not candidate:
        return False
    for reserved in reserved_labels:
        if candidate == reserved:
            return True
        if candidate.startswith(reserved) or reserved.startswith(candidate):
            return True
        if _levenshtein_distance(candidate, reserved) <= 1:
            return True
    return False


def upscale_crop(crop: np.ndarray, min_height: int = UPSCALE_MIN_HEIGHT) -> np.ndarray:
    height, width = crop.shape[:2]
    if height >= min_height:
        return crop
    scale = min_height / height
    new_width = max(1, round(width * scale))
    return cv2.resize(crop, (new_width, min_height), interpolation=cv2.INTER_CUBIC)


def merge_observations(
    observations: Sequence[VisionObservation],
) -> PlateRecognition | None:
    if not observations:
        return None

    ordered = sorted(observations, key=lambda item: item.y_center, reverse=True)
    kept = [
        item
        for item in ordered
        if len(normalize_plate_text(item.text)) >= 2
    ]
    if kept:
        label = "".join(normalize_plate_text(item.text) for item in kept)
        confidence = min(item.confidence for item in kept)
        if validate_vietnamese_plate(label):
            return PlateRecognition(label=label, confidence=confidence, parts=len(kept), upscaled_height=0)

    best = max(observations, key=lambda item: item.confidence)
    label = normalize_plate_text(best.text)
    if validate_vietnamese_plate(label):
        return PlateRecognition(label=label, confidence=best.confidence, parts=1, upscaled_height=0)
    return None


def create_apple_vision_recognizer() -> RecognizeFn:
    """Build a recognizer backed by VNRecognizeTextRequest."""

    def recognize(crop_bgr: np.ndarray) -> list[tuple[str, float, float]]:
        from Foundation import NSURL
        import Vision

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            if not cv2.imwrite(str(tmp_path), crop_bgr):
                raise RuntimeError(f"could not write temporary OCR image: {tmp_path}")
            request_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                NSURL.fileURLWithPath_(str(tmp_path)), None
            )
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(False)
            ok = request_handler.performRequests_error_([request], None)
            if not ok:
                return []
            results: list[tuple[str, float, float]] = []
            for observation in request.results() or []:
                candidate = observation.topCandidates_(1)[0]
                text = str(candidate.string())
                confidence = float(candidate.confidence())
                box = observation.boundingBox()
                y_center = float(box.origin.y + box.size.height / 2.0)
                results.append((text, confidence, y_center))
            return results
        finally:
            tmp_path.unlink(missing_ok=True)

    return recognize


def _recognize_crop(
    crop: np.ndarray,
    recognize: RecognizeFn,
    *,
    min_height: int,
) -> tuple[PlateRecognition | None, str | None]:
    """Return ``(reading, drop_reason)`` where drop_reason is unreadable/invalid-format."""

    upscaled = upscale_crop(crop, min_height=min_height)
    raw = recognize(upscaled)
    if not raw:
        return None, "unreadable"
    observations = [
        VisionObservation(text=text, confidence=confidence, y_center=y_center)
        for text, confidence, y_center in raw
    ]
    result = merge_observations(observations)
    if result is None:
        return None, "invalid-format"
    return PlateRecognition(
        label=result.label,
        confidence=result.confidence,
        parts=result.parts,
        upscaled_height=int(upscaled.shape[0]),
    ), None


def _deterministic_name(
    source_hash: str,
    image_path: Path,
    source_root: Path,
    box_index: int,
) -> str:
    relative_id = hashlib.sha256(
        image_path.relative_to(source_root).as_posix().encode("utf-8")
    ).hexdigest()[:12]
    return f"pseudo_{source_hash[:16]}_{relative_id}_{box_index:02d}.jpg"


def pseudo_label_apple_vision(
    sources: Sequence[str | Path],
    output_dir: str | Path,
    *,
    reserved_manifests: Sequence[str | Path] = DEFAULT_RESERVED,
    exclude_source_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    min_height: int = 24,
    limit: int | None = None,
    progress_every: int = 200,
    recognize: RecognizeFn | None = None,
) -> tuple[Path, DropStats, float]:
    if recognize is None:
        recognize = create_apple_vision_recognizer()

    reserved_labels, reserved_source_hashes, reserved_crop_hashes = build_reserved_identities(
        reserved_manifests
    )
    output = Path(output_dir).expanduser().resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stats = DropStats()
    rows: list[dict[str, object]] = []
    started = time.monotonic()

    for source in sources:
        source_root = Path(source).expanduser().resolve()
        for image_path, yolo_path in _iter_yolo_pairs(source_root):
            label_lines = [
                line for line in yolo_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            valid_box_lines = [line for line in label_lines if len(line.split()) >= 5]

            if any(
                fnmatch.fnmatchcase(image_path.name, pattern)
                for pattern in exclude_source_patterns
            ):
                stats.total_boxes += len(valid_box_lines)
                stats.excluded_source += len(valid_box_lines)
                continue

            source_hash = _sha256(image_path)
            if source_hash in reserved_source_hashes:
                stats.total_boxes += len(valid_box_lines)
                stats.reserved_source_hash += len(valid_box_lines)
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                stats.total_boxes += len(valid_box_lines)
                stats.unreadable += len(valid_box_lines)
                continue
            height, width = image.shape[:2]
            source_group = image_path.stem.split(".rf.", 1)[0]
            source_group_hash = hashlib.sha256(source_group.encode("utf-8")).hexdigest()[:16]

            for box_index, line in enumerate(label_lines):
                values = line.split()
                if len(values) < 5:
                    continue
                stats.total_boxes += 1
                cx, cy, bw, bh = map(float, values[1:5])
                try:
                    x1, y1, x2, y2 = parse_yolo_box(
                        (cx, cy, bw, bh),
                        image_width=width,
                        image_height=height,
                        context=f"{yolo_path}:{box_index + 1}",
                    )
                except ValueError:
                    stats.too_small += 1
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < min_height:
                    stats.too_small += 1
                    continue

                reading, drop_reason = _recognize_crop(
                    crop, recognize, min_height=UPSCALE_MIN_HEIGHT
                )
                if reading is None:
                    if drop_reason == "unreadable":
                        stats.unreadable += 1
                    else:
                        stats.invalid_format += 1
                    continue

                label = reading.label
                if label_conflicts_with_reserved(label, reserved_labels):
                    stats.reserved_label += 1
                    continue

                name = _deterministic_name(source_hash, image_path, source_root, box_index)
                target = images_dir / name
                _safe_write(target, crop)
                crop_hash = _sha256(target)
                if crop_hash in reserved_crop_hashes:
                    target.unlink(missing_ok=True)
                    stats.reserved_crop_hash += 1
                    continue

                rows.append({
                    "image_path": target.relative_to(output).as_posix(),
                    "label": label,
                    "source_type": "pseudo",
                    "group_id": f"source:{source_group_hash}",
                    "split": "train",
                    "verified": "false",
                    "source_ref": str(image_path),
                    "source_sha256": source_hash,
                    "crop_sha256": crop_hash,
                    "ocr_confidence": f"{reading.confidence:.8f}",
                    "parameters_json": json.dumps(
                        {
                            "teacher": "apple-vision-VNRecognizeTextRequest",
                            "recognition_level": "accurate",
                            "yolo_box": [cx, cy, bw, bh],
                            "parts": reading.parts,
                            "upscaled_height": reading.upscaled_height,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                })
                stats.kept += 1

                if limit is not None and stats.kept >= limit:
                    manifest = _write_manifest(output / "manifest.csv", rows)
                    elapsed = time.monotonic() - started
                    return manifest, stats, elapsed

                if stats.total_boxes % progress_every == 0:
                    print(
                        f"progress boxes={stats.total_boxes} kept={stats.kept}",
                        flush=True,
                    )

    manifest = _write_manifest(output / "manifest.csv", rows)
    elapsed = time.monotonic() - started
    return manifest, stats, elapsed


def confidence_band(confidence: float) -> str:
    if confidence >= 1.0:
        return "1.0"
    index = min(int(confidence * 10), 9)
    lower = index / 10.0
    upper = (index + 1) / 10.0
    return f"{lower:.1f}-{upper:.1f}"


def _ordered_bands() -> list[str]:
    return [f"{index / 10:.1f}-{(index + 1) / 10:.1f}" for index in range(10)] + ["1.0"]


def merge_small_bands(band_counts: dict[str, int]) -> list[list[str]]:
    bands = _ordered_bands()
    merged: list[list[str]] = []
    index = 0
    while index < len(bands):
        group = [bands[index]]
        total = band_counts.get(bands[index], 0)
        while total < MIN_MERGED_BAND_ROWS and index + 1 < len(bands):
            index += 1
            group.append(bands[index])
            total += band_counts.get(bands[index], 0)
        merged.append(group)
        index += 1
    return merged


def compute_band_stats(manifest_path: str | Path) -> dict[str, object]:
    rows = load_plate_manifest(manifest_path)
    band_rows: dict[str, list[PlateRecognition | object]] = defaultdict(list)
    for row in rows:
        confidence = float(row.metadata.get("ocr_confidence", "0") or 0.0)
        band_rows[confidence_band(confidence)].append(row)

    bands = _ordered_bands()
    total = len(rows)
    histogram: list[dict[str, object]] = []
    for band in bands:
        items = band_rows.get(band, [])
        count = len(items)
        unique_labels = len({item.label for item in items})
        histogram.append({
            "band": band,
            "count": count,
            "share": (count / total) if total else 0.0,
            "unique_labels": unique_labels,
        })
    return {"rows": total, "bands": histogram}


def write_band_stats(manifest_path: str | Path, output_path: str | Path) -> dict[str, object]:
    summary = compute_band_stats(manifest_path)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def generate_audit_sheets(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = AUDIT_SEED,
    sample_per_band: int = AUDIT_SAMPLE_PER_BAND,
) -> tuple[Path, list[Path], int]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = load_plate_manifest(manifest_path)
    by_band: dict[str, list] = defaultdict(list)
    for row in rows:
        confidence = float(row.metadata.get("ocr_confidence", "0") or 0.0)
        by_band[confidence_band(confidence)].append(row)

    band_counts = {band: len(by_band.get(band, [])) for band in _ordered_bands()}
    merged_groups = merge_small_bands(band_counts)
    rng = random.Random(seed)

    selected: list[tuple[object, str, float]] = []
    for group in merged_groups:
        group_rows: list = []
        merged_band = "+".join(group)
        for band in group:
            group_rows.extend(by_band.get(band, []))
        if len(group_rows) <= sample_per_band:
            chosen = list(group_rows)
        else:
            chosen = rng.sample(group_rows, sample_per_band)
        for row in chosen:
            confidence = float(row.metadata.get("ocr_confidence", "0") or 0.0)
            selected.append((row, merged_band, confidence))

    selected.sort(key=lambda item: (_ordered_bands().index(confidence_band(item[2])) if confidence_band(item[2]) in _ordered_bands() else 99, item[2], str(item[0].image_path)))

    sheet_paths: list[Path] = []
    map_rows: list[dict[str, object]] = []
    cell_width, cell_height = 260, 130

    for sheet_index in range((len(selected) + AUDIT_PER_SHEET - 1) // AUDIT_PER_SHEET):
        subset = selected[sheet_index * AUDIT_PER_SHEET:(sheet_index + 1) * AUDIT_PER_SHEET]
        canvas = np.full(
            (AUDIT_ROWS * cell_height, AUDIT_COLUMNS * cell_width, 3),
            245,
            np.uint8,
        )
        sheet_name = f"contact_sheet_{sheet_index + 1:03d}.jpg"
        sheet_path = output / sheet_name
        sheet_path.parent.mkdir(parents=True, exist_ok=True)

        for offset, (row, merged_band, confidence) in enumerate(subset):
            crop = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            display = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            display_h, display_w = display.shape[:2]
            col = offset % AUDIT_COLUMNS
            row_index = offset // AUDIT_COLUMNS
            x0 = col * cell_width + (cell_width - display_w) // 2
            y0 = row_index * cell_height + 24
            y0 = max(20, min(y0, row_index * cell_height + cell_height - display_h - 8))
            x0 = max(4, min(x0, col * cell_width + cell_width - display_w - 4))
            canvas[y0:y0 + display_h, x0:x0 + display_w] = display
            cell_index = sheet_index * AUDIT_PER_SHEET + offset + 1
            label_y = max(12, y0 - 6)
            cv2.putText(
                canvas,
                str(cell_index),
                (x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            map_rows.append({
                "sheet_file": sheet_name,
                "cell_index": cell_index,
                "image_path": row.image_path.relative_to(manifest_path.parent).as_posix(),
                "ocr_confidence": f"{confidence:.8f}",
                "band": merged_band,
            })

        if not cv2.imwrite(str(sheet_path), canvas):
            raise RuntimeError(f"could not write audit sheet: {sheet_path}")
        sheet_paths.append(sheet_path)

    map_path = output / "sheet_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sheet_file", "cell_index", "image_path", "ocr_confidence", "band"],
        )
        writer.writeheader()
        writer.writerows(map_rows)
    return map_path, sheet_paths, len(sheet_paths)


def run_audit_pipeline(
    manifest_path: str | Path,
    *,
    band_stats_path: str | Path | None = None,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if band_stats_path is None:
        band_stats_path = manifest_path.parent / "band_stats.json"
    band_summary = write_band_stats(manifest_path, band_stats_path)
    map_path, sheet_paths, sheet_count = generate_audit_sheets(manifest_path, audit_dir)
    return {
        "band_stats_path": str(band_stats_path),
        "band_summary": band_summary,
        "audit_dir": str(Path(audit_dir).resolve()),
        "sheet_map": str(map_path),
        "sheet_count": sheet_count,
        "sheet_paths": [str(path) for path in sheet_paths],
    }


def _default_sources(values: list[Path] | None) -> list[Path]:
    if values:
        return values
    return [path.resolve() for path in DEFAULT_SOURCES]


def _default_reserved(values: list[Path] | None) -> list[Path]:
    if values:
        return values
    return [path.resolve() for path in DEFAULT_RESERVED]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="pseudo-label crops with Apple Vision")
    run.add_argument("--source", type=Path, action="append", default=None)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--reserved-manifest", type=Path, action="append", default=None)
    run.add_argument("--exclude-source-pattern", action="append", default=None)
    run.add_argument("--min-height", type=int, default=24)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--progress-every", type=int, default=200)

    audit = sub.add_parser("audit", help="band stats and label-free audit sheets")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--band-stats", type=Path, default=None)
    audit.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    audit.add_argument("--seed", type=int, default=AUDIT_SEED)

    all_in_one = sub.add_parser("all", help="run pseudo-labeling then audit")
    all_in_one.add_argument("--source", type=Path, action="append", default=None)
    all_in_one.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    all_in_one.add_argument("--reserved-manifest", type=Path, action="append", default=None)
    all_in_one.add_argument("--exclude-source-pattern", action="append", default=None)
    all_in_one.add_argument("--min-height", type=int, default=24)
    all_in_one.add_argument("--limit", type=int, default=None)
    all_in_one.add_argument("--progress-every", type=int, default=200)
    all_in_one.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "audit":
        result = run_audit_pipeline(
            args.manifest,
            band_stats_path=args.band_stats or (args.manifest.parent / "band_stats.json"),
            audit_dir=args.audit_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    sources = _default_sources(args.source)
    reserved = _default_reserved(args.reserved_manifest)
    exclude_patterns = (
        tuple(args.exclude_source_pattern)
        if args.exclude_source_pattern is not None
        else DEFAULT_EXCLUDE_PATTERNS
    )

    manifest, stats, elapsed = pseudo_label_apple_vision(
        sources,
        args.output,
        reserved_manifests=reserved,
        exclude_source_patterns=exclude_patterns,
        min_height=args.min_height,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    summary = stats.to_dict()
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["manifest"] = str(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.command == "all":
        audit_summary = run_audit_pipeline(manifest, audit_dir=args.audit_dir)
        print(json.dumps(audit_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
