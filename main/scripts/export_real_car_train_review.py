#!/usr/bin/env python3
"""Export reserved-disjoint car-plate review crops for human transcription."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from scripts.build_plate_ocr_dataset import (  # noqa: E402
    _iter_yolo_pairs,
    _safe_write,
    _sha256,
    parse_yolo_box,
)
from scripts.export_real_train_review import (  # noqa: E402
    AUDIT_CELL_HEIGHT,
    AUDIT_CELL_WIDTH,
    AUDIT_COLUMNS,
    AUDIT_PER_SHEET,
    AUDIT_ROWS,
)
from scripts.near_reserved_filter import (  # noqa: E402
    classify_reserved_label_drop,
    is_near_reserved_label,
)
from scripts.pseudo_label_apple_vision import (  # noqa: E402
    UPSCALE_MIN_HEIGHT,
    RecognizeFn,
    _recognize_crop,
    build_reserved_identities,
    confidence_band,
    create_apple_vision_recognizer,
)
from src.datasets.plate_ocr_dataset import load_plate_manifest  # noqa: E402

DEFAULT_SOURCES = (
    _MAIN / "data/raw/plate_det/train",
    _MAIN / "data/raw/plate_det/valid",
)
DEFAULT_RESERVED = (
    _MAIN / "data/plate_ocr/real_validation.csv",
    _MAIN / "data/plate_ocr/expanded_real_test.csv",
    _MAIN / "data/plate_ocr/frozen_regression.csv",
)
DEFAULT_OUTPUT = _MAIN / "data/plate_ocr/review/real_car_train_audit"
DEFAULT_EXCLUDE_PATTERNS = ("*Gen*",)
MIN_CAR_ASPECT_RATIO = 2.2
MIN_CAR_SIDE_PX = 20
MAX_CROPS_PER_LABEL = 3
SHEET_LABEL_CAP = 250

CANDIDATE_FIELDS = (
    "image_path",
    "label_draft",
    "ocr_confidence",
    "source_ref",
    "source_sha256",
    "crop_sha256",
    "box_json",
    "aspect_ratio",
)


@dataclass
class CarReviewStats:
    total_boxes: int = 0
    excluded_source: int = 0
    reserved_source_hash: int = 0
    low_ar: int = 0
    car_boxes_seen: int = 0
    unreadable: int = 0
    invalid_format: int = 0
    reserved_label: int = 0
    near_reserved_label: int = 0
    reserved_crop_hash: int = 0
    kept_before_dedup: int = 0
    kept: int = 0
    unique_labels: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_boxes": self.total_boxes,
            "car_boxes_seen": self.car_boxes_seen,
            "dropped": {
                "excluded-source": self.excluded_source,
                "reserved-source-hash": self.reserved_source_hash,
                "low-ar": self.low_ar,
                "unreadable": self.unreadable,
                "invalid-format": self.invalid_format,
                "reserved-label": self.reserved_label,
                "near-reserved-label": self.near_reserved_label,
                "reserved-crop-hash": self.reserved_crop_hash,
            },
            "kept_before_dedup": self.kept_before_dedup,
            "kept": self.kept,
            "unique_labels": self.unique_labels,
        }


@dataclass
class _Candidate:
    crop: np.ndarray
    label: str
    ocr_confidence: float
    source_ref: Path
    source_sha256: str
    crop_sha256: str
    box: list[int]
    aspect_ratio: float
    crop_name: str


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_car_like(crop: np.ndarray) -> tuple[bool, float]:
    height, width = crop.shape[:2]
    if min(width, height) < MIN_CAR_SIDE_PX:
        return False, 0.0
    aspect_ratio = width / height
    return aspect_ratio >= MIN_CAR_ASPECT_RATIO, aspect_ratio


def _deterministic_crop_name(source_hash: str, image_path: Path, source_root: Path, box_index: int) -> str:
    relative_id = hashlib.sha256(
        image_path.relative_to(source_root).as_posix().encode("utf-8")
    ).hexdigest()[:12]
    return f"car_{source_hash[:16]}_{relative_id}_{box_index:02d}.jpg"


def select_diverse_per_label(candidates: Sequence[_Candidate], *, max_per_label: int) -> list[_Candidate]:
    """Keep up to *max_per_label* crops per label, preferring diverse source hashes."""

    by_confidence = sorted(candidates, key=lambda item: (-item.ocr_confidence, item.source_sha256))
    selected: list[_Candidate] = []
    selected_keys: set[tuple[str, str]] = set()
    used_sources: set[str] = set()
    for candidate in by_confidence:
        if candidate.source_sha256 not in used_sources:
            selected.append(candidate)
            selected_keys.add((candidate.crop_sha256, candidate.source_sha256))
            used_sources.add(candidate.source_sha256)
            if len(selected) >= max_per_label:
                return selected
    for candidate in by_confidence:
        key = (candidate.crop_sha256, candidate.source_sha256)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        if len(selected) >= max_per_label:
            break
    return selected


def confidence_histogram(candidates: Sequence[_Candidate]) -> list[dict[str, object]]:
    counts = Counter(confidence_band(item.ocr_confidence) for item in candidates)
    bands = [f"{index / 10:.1f}-{(index + 1) / 10:.1f}" for index in range(10)] + ["1.0"]
    total = len(candidates)
    return [
        {
            "band": band,
            "count": counts.get(band, 0),
            "share": (counts.get(band, 0) / total) if total else 0.0,
        }
        for band in bands
    ]


def assert_no_leakage(
    candidates: Sequence[_Candidate],
    *,
    reserved_labels: set[str],
    reserved_source_hashes: set[str],
    reserved_crop_hashes: set[str],
) -> dict[str, int]:
    candidate_labels = {item.label for item in candidates}
    candidate_source = {item.source_sha256.lower() for item in candidates}
    candidate_crop = {item.crop_sha256.lower() for item in candidates}
    report = {
        "label_intersection": len(candidate_labels & reserved_labels),
        "source_sha256_intersection": len(candidate_source & reserved_source_hashes),
        "crop_sha256_intersection": len(candidate_crop & reserved_crop_hashes),
    }
    assert report["label_intersection"] == 0, report
    assert report["source_sha256_intersection"] == 0, report
    assert report["crop_sha256_intersection"] == 0, report
    return report


def assert_no_near_reserved_leakage(
    candidates: Sequence[_Candidate],
    reserved_labels: set[str],
) -> int:
    leaked = [item.label for item in candidates if is_near_reserved_label(item.label, reserved_labels)]
    assert not leaked, {"near_reserved_leak_count": len(leaked), "examples": leaked[:5]}
    return 0


def collect_car_candidates(
    sources: Sequence[str | Path],
    *,
    reserved_manifests: Sequence[str | Path],
    exclude_source_patterns: Sequence[str],
    recognize: RecognizeFn,
    progress_every: int = 200,
) -> tuple[list[_Candidate], CarReviewStats]:
    reserved_labels, reserved_source_hashes, reserved_crop_hashes = build_reserved_identities(
        reserved_manifests
    )
    stats = CarReviewStats()
    raw_by_label: dict[str, list[_Candidate]] = defaultdict(list)

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
                    stats.low_ar += 1
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    stats.low_ar += 1
                    continue

                car_like, aspect_ratio = _is_car_like(crop)
                if not car_like:
                    stats.low_ar += 1
                    continue
                stats.car_boxes_seen += 1

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
                reserved_drop = classify_reserved_label_drop(label, reserved_labels)
                if reserved_drop == "reserved-label":
                    stats.reserved_label += 1
                    continue
                if reserved_drop == "near-reserved-label":
                    stats.near_reserved_label += 1
                    continue

                crop_name = _deterministic_crop_name(source_hash, image_path, source_root, box_index)
                crop_hash = hashlib.sha256(cv2.imencode(".jpg", crop)[1].tobytes()).hexdigest()
                if crop_hash in reserved_crop_hashes:
                    stats.reserved_crop_hash += 1
                    continue

                raw_by_label[label].append(
                    _Candidate(
                        crop=crop,
                        label=label,
                        ocr_confidence=reading.confidence,
                        source_ref=image_path,
                        source_sha256=source_hash,
                        crop_sha256=crop_hash,
                        box=[x1, y1, x2, y2],
                        aspect_ratio=aspect_ratio,
                        crop_name=crop_name,
                    )
                )
                stats.kept_before_dedup += 1

                if stats.total_boxes % progress_every == 0:
                    print(
                        f"progress boxes={stats.total_boxes} car={stats.car_boxes_seen} "
                        f"kept_raw={stats.kept_before_dedup}",
                        flush=True,
                    )

    kept: list[_Candidate] = []
    for label, group in raw_by_label.items():
        kept.extend(select_diverse_per_label(group, max_per_label=MAX_CROPS_PER_LABEL))
    stats.kept = len(kept)
    stats.unique_labels = len({item.label for item in kept})
    return kept, stats


def write_candidates_csv(output_dir: Path, candidates: Sequence[_Candidate]) -> Path:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidates.csv"
    rows: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: (-item.ocr_confidence, item.label, item.crop_name)):
        crop_path = crops_dir / candidate.crop_name
        _safe_write(crop_path, candidate.crop)
        rows.append({
            "image_path": crop_path.relative_to(output_dir).as_posix(),
            "label_draft": candidate.label,
            "ocr_confidence": f"{candidate.ocr_confidence:.8f}",
            "source_ref": str(candidate.source_ref),
            "source_sha256": candidate.source_sha256,
            "crop_sha256": candidate.crop_sha256,
            "box_json": json.dumps(candidate.box, separators=(",", ":")),
            "aspect_ratio": f"{candidate.aspect_ratio:.4f}",
        })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _render_contact_sheets(
    items: Sequence[dict[str, object]],
    output_dir: Path,
    *,
    image_key: str,
    map_fieldnames: Sequence[str],
    map_row_builder,
) -> tuple[Path, int, int]:
    """Render 6x5 label-free contact sheets without wiping existing outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    map_rows: list[dict[str, object]] = []
    sheet_count = 0
    cell_count = 0

    for sheet_index in range((len(items) + AUDIT_PER_SHEET - 1) // AUDIT_PER_SHEET):
        subset = items[sheet_index * AUDIT_PER_SHEET:(sheet_index + 1) * AUDIT_PER_SHEET]
        canvas = np.full(
            (AUDIT_ROWS * AUDIT_CELL_HEIGHT, AUDIT_COLUMNS * AUDIT_CELL_WIDTH, 3),
            245,
            np.uint8,
        )
        sheet_name = f"contact_sheet_{sheet_index + 1:03d}.jpg"
        sheet_path = output_dir / sheet_name

        for offset, item in enumerate(subset):
            image_path = Path(str(item[image_key]))
            crop = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            display = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            display_h, display_w = display.shape[:2]
            col = offset % AUDIT_COLUMNS
            row_index = offset // AUDIT_COLUMNS
            x0 = col * AUDIT_CELL_WIDTH + (AUDIT_CELL_WIDTH - display_w) // 2
            y0 = row_index * AUDIT_CELL_HEIGHT + 24
            y0 = max(20, min(y0, row_index * AUDIT_CELL_HEIGHT + AUDIT_CELL_HEIGHT - display_h - 8))
            x0 = max(4, min(x0, col * AUDIT_CELL_WIDTH + AUDIT_CELL_WIDTH - display_w - 4))
            canvas[y0:y0 + display_h, x0:x0 + display_w] = display
            cell_index = sheet_index * AUDIT_PER_SHEET + offset + 1
            cv2.putText(
                canvas,
                str(cell_index),
                (x0, max(12, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            map_rows.append(map_row_builder(item, sheet_name=sheet_name, cell_index=cell_index))
            cell_count += 1

        if not cv2.imwrite(str(sheet_path), canvas):
            raise RuntimeError(f"could not write audit sheet: {sheet_path}")
        sheet_count += 1

    map_path = output_dir / "sheet_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_fieldnames))
        writer.writeheader()
        writer.writerows(map_rows)
    return map_path, sheet_count, cell_count


def render_transcription_sheets(
    output_dir: Path,
    candidates: Sequence[_Candidate],
    *,
    label_cap: int = SHEET_LABEL_CAP,
) -> dict[str, object]:
    best_by_label: dict[str, _Candidate] = {}
    for candidate in candidates:
        previous = best_by_label.get(candidate.label)
        if previous is None or candidate.ocr_confidence > previous.ocr_confidence:
            best_by_label[candidate.label] = candidate

    sheet_items = sorted(best_by_label.values(), key=lambda item: (-item.ocr_confidence, item.label))[:label_cap]
    items = [
        {
            "image_path": output_dir / "crops" / candidate.crop_name,
            "label_draft": candidate.label,
            "ocr_confidence": candidate.ocr_confidence,
            "source_ref": candidate.source_ref,
            "source_sha256": candidate.source_sha256,
            "crop_sha256": candidate.crop_sha256,
            "box_json": json.dumps(candidate.box, separators=(",", ":")),
        }
        for candidate in sheet_items
    ]

    def map_row_builder(item: dict[str, object], *, sheet_name: str, cell_index: int) -> dict[str, object]:
        image_path = Path(str(item["image_path"])).resolve()
        try:
            image_value = image_path.relative_to(_MAIN).as_posix()
        except ValueError:
            image_value = image_path.as_posix()
        source_ref = str(item["source_ref"])
        try:
            source_value = Path(source_ref).resolve().relative_to(_MAIN).as_posix()
        except ValueError:
            source_value = source_ref
        return {
            "sheet_file": sheet_name,
            "cell_index": cell_index,
            "image_path": image_value,
            "source_ref": source_value,
            "source_sha256": item["source_sha256"],
            "crop_sha256": item["crop_sha256"],
            "box_json": item["box_json"],
            "label_draft": item["label_draft"],
            "ocr_confidence": f"{float(item['ocr_confidence']):.8f}",
        }

    map_path, sheet_count, cell_count = _render_contact_sheets(
        items,
        output_dir,
        image_key="image_path",
        map_fieldnames=(
            "sheet_file", "cell_index", "image_path", "source_ref",
            "source_sha256", "crop_sha256", "box_json", "label_draft", "ocr_confidence",
        ),
        map_row_builder=map_row_builder,
    )
    template_path = output_dir / "transcription_template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_index", "label", "status", "notes"])
        writer.writeheader()
        for row in csv.DictReader(map_path.open(encoding="utf-8")):
            writer.writerow({
                "cell_index": row["cell_index"],
                "label": "",
                "status": "",
                "notes": "",
            })
    return {
        "sheet_map": str(map_path),
        "transcription_template": str(template_path),
        "sheet_count": sheet_count,
        "cell_count": cell_count,
        "sheet_label_cap": label_cap,
        "unique_labels_available": len(best_by_label),
    }


def run_pipeline(
    *,
    sources: Sequence[str | Path] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT,
    reserved_manifests: Sequence[str | Path] | None = None,
    exclude_source_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    recognize: RecognizeFn | None = None,
    progress_every: int = 200,
    sheet_label_cap: int = SHEET_LABEL_CAP,
) -> dict[str, object]:
    started = time.monotonic()
    source_paths = [Path(path).resolve() for path in (sources or DEFAULT_SOURCES)]
    reserved = list(reserved_manifests or DEFAULT_RESERVED)
    if recognize is None:
        recognize = create_apple_vision_recognizer()

    output = _prepare_output_dir(Path(output_dir))
    candidates, stats = collect_car_candidates(
        source_paths,
        reserved_manifests=reserved,
        exclude_source_patterns=exclude_source_patterns,
        recognize=recognize,
        progress_every=progress_every,
    )

    reserved_labels, reserved_source_hashes, reserved_crop_hashes = build_reserved_identities(reserved)
    leakage = assert_no_leakage(
        candidates,
        reserved_labels=reserved_labels,
        reserved_source_hashes=reserved_source_hashes,
        reserved_crop_hashes=reserved_crop_hashes,
    )
    near_reserved_leakage = assert_no_near_reserved_leakage(candidates, reserved_labels)

    candidates_csv = write_candidates_csv(output, candidates)
    sheets = render_transcription_sheets(output, candidates, label_cap=sheet_label_cap)

    stats_payload = stats.to_dict()
    stats_payload["confidence_histogram"] = confidence_histogram(candidates)
    stats_payload["leakage_check"] = leakage
    stats_payload["near_reserved_leakage"] = near_reserved_leakage
    stats_payload["sheet_cells"] = sheets["cell_count"]
    stats_path = output / "stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elapsed = time.monotonic() - started
    return {
        "output_dir": str(output),
        "candidates_csv": str(candidates_csv),
        "candidate_rows": len(candidates),
        "unique_labels": stats.unique_labels,
        "stats_path": str(stats_path),
        "sheets": sheets,
        "leakage_check": leakage,
        "near_reserved_leakage": near_reserved_leakage,
        "elapsed_seconds": round(elapsed, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=None)
    parser.add_argument("--exclude-source-pattern", action="append", default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--sheet-label-cap", type=int, default=SHEET_LABEL_CAP)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = args.source or list(DEFAULT_SOURCES)
    reserved = args.reserved_manifest or list(DEFAULT_RESERVED)
    exclude_patterns = (
        tuple(args.exclude_source_pattern)
        if args.exclude_source_pattern is not None
        else DEFAULT_EXCLUDE_PATTERNS
    )
    result = run_pipeline(
        sources=sources,
        output_dir=args.output,
        reserved_manifests=reserved,
        exclude_source_patterns=exclude_patterns,
        progress_every=args.progress_every,
        sheet_label_cap=args.sheet_label_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "leakage_check:",
        json.dumps(result["leakage_check"], sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
