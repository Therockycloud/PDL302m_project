"""Smoke tests for real-train review export helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.export_real_train_review import (
    collect_reserved_hashes,
    hash_collision_report,
    render_real_train_audit,
)
from src.datasets.plate_ocr_dataset import CORE_FIELDS


def _row(
    image_path: str,
    label: str = "30M71854",
    source_type: str = "real",
    group_id: str = "vehicle-1",
    split: str = "train",
    verified: bool = True,
) -> dict[str, object]:
    return {
        "image_path": image_path,
        "label": label,
        "source_type": source_type,
        "group_id": group_id,
        "split": split,
        "verified": verified,
    }


def _write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    for item in rows:
        path = tmp_path / str(item["image_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((24, 80, 3), 127, np.uint8))
    manifest = tmp_path / "manifest.csv"
    fields = list(rows[0]) if rows else list(CORE_FIELDS)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _make_yolo_source(tmp_path: Path, name: str, pixel: int) -> None:
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True, exist_ok=True)
    (split / "labels").mkdir(parents=True, exist_ok=True)
    image_path = split / f"images/{name}.jpg"
    cv2.imwrite(str(image_path), np.full((80, 160, 3), pixel, np.uint8))
    (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_hash_collision_report_is_empty_for_disjoint_candidates(tmp_path):
    _make_yolo_source(tmp_path, "available_jpg.rf.aaaa", 120)
    reserved_source = tmp_path / "source/train/images/reserved_jpg.rf.aaaa.jpg"
    cv2.imwrite(str(reserved_source), np.full((80, 160, 3), 40, np.uint8))
    reserved = _write_manifest(tmp_path / "reserved", [{
        **_row("held.jpg", split="test"),
        "source_sha256": hashlib.sha256(reserved_source.read_bytes()).hexdigest(),
    }])

    from scripts.build_plate_ocr_dataset import build_review_candidates

    review_csv = build_review_candidates(
        tmp_path / "source",
        tmp_path / "candidates",
        count=5,
        reserved_manifests=[reserved],
    )
    report = hash_collision_report(review_csv, reserved_manifests=[reserved])
    assert report["source_sha256_intersection"] == 0
    assert report["crop_sha256_intersection"] == 0


def test_render_real_train_audit_writes_sheet_map_and_template(tmp_path):
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    crop_path = candidates_dir / "candidate-0000.jpg"
    cv2.imwrite(str(crop_path), np.full((48, 120, 3), 200, np.uint8))
    review_csv = candidates_dir / "review_candidates.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "crop_path", "group_id", "source_path", "source_sha256",
            "crop_sha256", "box_json", "crop_width", "crop_height", "focus_score",
            "review_label", "review_status", "review_notes",
        ])
        writer.writeheader()
        writer.writerow({
            "candidate_id": "candidate-0000",
            "crop_path": crop_path.name,
            "group_id": "det:vehicle_jpg",
            "source_path": "/tmp/source.jpg",
            "source_sha256": "a" * 64,
            "crop_sha256": "b" * 64,
            "box_json": json.dumps([1, 2, 3, 4]),
            "crop_width": "120",
            "crop_height": "48",
            "focus_score": "12.5",
            "review_label": "",
            "review_status": "",
            "review_notes": "",
        })

    audit = render_real_train_audit(review_csv, candidates_dir, tmp_path / "audit", sheet_item_count=1)
    sheet_map = Path(audit["sheet_map"])
    template = Path(audit["transcription_template"])
    rows = list(csv.DictReader(sheet_map.open(encoding="utf-8")))
    assert rows[0]["cell_index"] == "1"
    assert "label" not in rows[0]
    template_rows = list(csv.DictReader(template.open(encoding="utf-8")))
    assert template_rows == [{"cell_index": "1", "label": "", "status": "", "notes": ""}]
    assert (tmp_path / "audit/contact_sheet_001.jpg").is_file()


def test_collect_reserved_hashes_reads_manifest_metadata(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest", [{
        **_row("sample.jpg"),
        "source_sha256": "c" * 64,
        "crop_sha256": "d" * 64,
    }])
    source_hashes, crop_hashes = collect_reserved_hashes([manifest])
    assert "c" * 64 in source_hashes
    assert "d" * 64 in crop_hashes
