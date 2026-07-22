#!/usr/bin/env python3
"""Export ranked real-train review candidates and label-free audit contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from scripts.build_plate_ocr_dataset import build_review_candidates  # noqa: E402
from src.datasets.plate_ocr_dataset import load_plate_manifest  # noqa: E402

DEFAULT_SOURCE = _MAIN / "data/raw/plate_det"
DEFAULT_RESERVED = (
    _MAIN / "data/plate_ocr/real_validation.csv",
    _MAIN / "data/plate_ocr/expanded_real_test.csv",
    _MAIN / "data/plate_ocr/frozen_regression.csv",
)
DEFAULT_PSEUDO_MANIFEST = _MAIN / "data/plate_ocr/generated/pseudo_vision/manifest_conf_ge_0.5.csv"
DEFAULT_CANDIDATES_DIR = _MAIN / "data/plate_ocr/review/real_train_candidates"
DEFAULT_REAL_AUDIT_DIR = _MAIN / "data/plate_ocr/review/real_train_audit"
DEFAULT_PSEUDO_VERIFY_DIR = _MAIN / "data/plate_ocr/review/pseudo_conf1_verify"
DEFAULT_EXCLUDE_PATTERNS = ("*Gen*",)

AUDIT_COLUMNS = 6
AUDIT_ROWS = 5
AUDIT_PER_SHEET = AUDIT_COLUMNS * AUDIT_ROWS
AUDIT_CELL_WIDTH = 260
AUDIT_CELL_HEIGHT = 130


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def collect_reserved_hashes(manifests: Sequence[str | Path]) -> tuple[set[str], set[str]]:
    """Return lower-case source_sha256 and crop_sha256 sets from plate manifests."""

    source_hashes: set[str] = set()
    crop_hashes: set[str] = set()
    for manifest in manifests:
        for row in load_plate_manifest(manifest):
            for field_name, target in (
                ("source_sha256", source_hashes),
                ("crop_sha256", crop_hashes),
            ):
                value = str(row.metadata.get(field_name, "")).strip().lower()
                if value:
                    target.add(value)
    return source_hashes, crop_hashes


def hash_collision_report(
    candidates_csv: str | Path,
    *,
    reserved_manifests: Sequence[str | Path],
) -> dict[str, int]:
    """Count overlaps between candidate hashes and reserved identities."""

    reserved_source, reserved_crop = collect_reserved_hashes(reserved_manifests)
    candidate_source: set[str] = set()
    candidate_crop: set[str] = set()
    with Path(candidates_csv).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = str(row.get("source_sha256", "")).strip().lower()
            crop = str(row.get("crop_sha256", "")).strip().lower()
            if source:
                candidate_source.add(source)
            if crop:
                candidate_crop.add(crop)
    candidate_rows = sum(1 for _ in csv.DictReader(Path(candidates_csv).open(encoding="utf-8")))
    return {
        "candidate_rows": candidate_rows,
        "source_sha256_intersection": len(candidate_source & reserved_source),
        "crop_sha256_intersection": len(candidate_crop & reserved_crop),
    }


def export_review_candidates(
    *,
    source: str | Path,
    output_dir: str | Path,
    count: int,
    reserved_manifests: Sequence[str | Path],
    exclude_source_patterns: Sequence[str],
) -> Path:
    """Rank unique-source crops from a YOLO detection tree."""

    output = _prepare_output_dir(output_dir)
    return build_review_candidates(
        source,
        output,
        count=count,
        reserved_manifests=reserved_manifests,
        exclude_source_patterns=exclude_source_patterns,
    )


def render_label_free_audit_sheets(
    items: Sequence[dict[str, object]],
    output_dir: str | Path,
    *,
    image_key: str,
    map_fieldnames: Sequence[str],
    map_row_builder,
) -> tuple[Path, int, int]:
    """Render 6x5 label-free contact sheets and a sheet_map.csv."""

    output = _prepare_output_dir(output_dir)
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
        sheet_path = output / sheet_name

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

    map_path = output / "sheet_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_fieldnames))
        writer.writeheader()
        writer.writerows(map_rows)
    return map_path, sheet_count, cell_count


def render_real_train_audit(
    candidates_csv: str | Path,
    candidates_dir: str | Path,
    output_dir: str | Path,
    *,
    sheet_item_count: int = 300,
) -> dict[str, object]:
    """Render top-N real-train candidates into label-free audit sheets."""

    candidates_csv = Path(candidates_csv).expanduser().resolve()
    candidates_dir = Path(candidates_dir).expanduser().resolve()
    rows = list(csv.DictReader(candidates_csv.open(encoding="utf-8")))[:sheet_item_count]
    items = [
        {
            "image_path": candidates_dir / row["crop_path"],
            "source_ref": row["source_path"],
            "source_sha256": row["source_sha256"],
            "crop_sha256": row["crop_sha256"],
            "box_json": row["box_json"],
        }
        for row in rows
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
        }

    map_path, sheet_count, cell_count = render_label_free_audit_sheets(
        items,
        output_dir,
        image_key="image_path",
        map_fieldnames=(
            "sheet_file", "cell_index", "image_path", "source_ref",
            "source_sha256", "crop_sha256", "box_json",
        ),
        map_row_builder=map_row_builder,
    )
    template_path = Path(output_dir).expanduser().resolve() / "transcription_template.csv"
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
    }


def render_pseudo_conf1_verify(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    min_confidence: float = 0.999,
) -> dict[str, object]:
    """Render label-free sheets for near-certain pseudo-vision rows."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    rows = [
        row for row in load_plate_manifest(manifest_path)
        if float(row.metadata.get("ocr_confidence", "0") or 0.0) >= min_confidence
    ]
    rows.sort(key=lambda row: (-float(row.metadata.get("ocr_confidence", "0") or 0.0), str(row.image_path)))
    items = [
        {
            "image_path": row.image_path,
            "teacher_label": row.label,
            "ocr_confidence": row.metadata.get("ocr_confidence", ""),
            "source_ref": row.metadata.get("source_ref", ""),
            "source_sha256": row.metadata.get("source_sha256", ""),
            "crop_sha256": row.metadata.get("crop_sha256", ""),
        }
        for row in rows
    ]

    def map_row_builder(item: dict[str, object], *, sheet_name: str, cell_index: int) -> dict[str, object]:
        source_ref = str(item["source_ref"])
        try:
            source_value = Path(source_ref).resolve().relative_to(_MAIN).as_posix()
        except ValueError:
            source_value = source_ref
        return {
            "sheet_file": sheet_name,
            "cell_index": cell_index,
            "image_path": Path(str(item["image_path"])).relative_to(manifest_path.parent).as_posix(),
            "teacher_label": item["teacher_label"],
            "ocr_confidence": item["ocr_confidence"],
            "source_ref": source_value,
            "source_sha256": item["source_sha256"],
            "crop_sha256": item["crop_sha256"],
        }

    map_path, sheet_count, cell_count = render_label_free_audit_sheets(
        items,
        output_dir,
        image_key="image_path",
        map_fieldnames=(
            "sheet_file", "cell_index", "image_path", "teacher_label", "ocr_confidence",
            "source_ref", "source_sha256", "crop_sha256",
        ),
        map_row_builder=map_row_builder,
    )
    return {
        "manifest_rows": len(rows),
        "sheet_map": str(map_path),
        "sheet_count": sheet_count,
        "cell_count": cell_count,
    }


def run_pipeline(
    *,
    count: int = 500,
    sheet_item_count: int = 300,
    candidates_dir: str | Path = DEFAULT_CANDIDATES_DIR,
    real_audit_dir: str | Path = DEFAULT_REAL_AUDIT_DIR,
    pseudo_verify_dir: str | Path = DEFAULT_PSEUDO_VERIFY_DIR,
    pseudo_manifest: str | Path = DEFAULT_PSEUDO_MANIFEST,
    source: str | Path = DEFAULT_SOURCE,
    reserved_manifests: Sequence[str | Path] | None = None,
    exclude_source_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> dict[str, object]:
    reserved = list(reserved_manifests or (*DEFAULT_RESERVED, pseudo_manifest))
    review_csv = export_review_candidates(
        source=source,
        output_dir=candidates_dir,
        count=count,
        reserved_manifests=reserved,
        exclude_source_patterns=exclude_source_patterns,
    )
    candidate_rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    audit = render_real_train_audit(
        review_csv,
        candidates_dir,
        real_audit_dir,
        sheet_item_count=sheet_item_count,
    )
    pseudo = render_pseudo_conf1_verify(pseudo_manifest, pseudo_verify_dir)
    collisions = hash_collision_report(review_csv, reserved_manifests=reserved)
    return {
        "candidates_csv": str(review_csv),
        "candidates_dir": str(candidates_dir),
        "candidate_count_requested": count,
        "candidate_count_produced": len(candidate_rows),
        "real_audit_dir": str(Path(real_audit_dir).resolve()),
        "real_audit": audit,
        "pseudo_verify_dir": str(Path(pseudo_verify_dir).resolve()),
        "pseudo_verify": pseudo,
        "hash_collisions": collisions,
        "reserved_manifests": [str(Path(path).resolve()) for path in reserved],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="rank candidates and render audit sheets")
    export.add_argument("--count", type=int, default=500)
    export.add_argument("--sheet-count", type=int, default=300)
    export.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    export.add_argument("--real-audit-dir", type=Path, default=DEFAULT_REAL_AUDIT_DIR)
    export.add_argument("--pseudo-verify-dir", type=Path, default=DEFAULT_PSEUDO_VERIFY_DIR)
    export.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_PSEUDO_MANIFEST)
    export.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    export.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    export.add_argument("--exclude-source-pattern", action="append", default=[])

    check = sub.add_parser("check-collisions", help="verify candidate hashes do not overlap reserved sets")
    check.add_argument("--candidates-csv", type=Path, required=True)
    check.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-collisions":
        reserved = args.reserved_manifest or [*DEFAULT_RESERVED, DEFAULT_PSEUDO_MANIFEST]
        report = hash_collision_report(args.candidates_csv, reserved_manifests=reserved)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["source_sha256_intersection"] == 0 and report["crop_sha256_intersection"] == 0 else 2

    reserved = args.reserved_manifest or [*DEFAULT_RESERVED, args.pseudo_manifest]
    patterns = args.exclude_source_pattern or list(DEFAULT_EXCLUDE_PATTERNS)
    result = run_pipeline(
        count=args.count,
        sheet_item_count=args.sheet_count,
        candidates_dir=args.candidates_dir,
        real_audit_dir=args.real_audit_dir,
        pseudo_verify_dir=args.pseudo_verify_dir,
        pseudo_manifest=args.pseudo_manifest,
        source=args.source,
        reserved_manifests=reserved,
        exclude_source_patterns=patterns,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    collisions = result["hash_collisions"]
    if collisions["source_sha256_intersection"] or collisions["crop_sha256_intersection"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
