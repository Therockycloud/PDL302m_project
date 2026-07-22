"""Tests for Apple Vision plate pseudo-labeling."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.pseudo_label_apple_vision import (
    _deterministic_name,
    build_reserved_identities,
    merge_observations,
    pseudo_label_apple_vision,
    VisionObservation,
    run_audit_pipeline,
)
from src.datasets.plate_ocr_dataset import CORE_FIELDS, compose_plate_manifests, load_plate_manifest


def _write_reserved_manifest(tmp_path: Path, *, label: str, source_hash: str, crop_hash: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "held.jpg"
    cv2.imwrite(str(image_path), np.full((24, 80, 3), 90, np.uint8))
    manifest = tmp_path / "reserved.csv"
    fields = list(CORE_FIELDS) + ["source_sha256", "crop_sha256"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "image_path": image_path.name,
            "label": label,
            "source_type": "real",
            "group_id": "held",
            "split": "test",
            "verified": "true",
            "source_sha256": source_hash,
            "crop_sha256": crop_hash,
        })
    return manifest


def _make_source(tmp_path: Path, name: str = "plate_jpg.rf.aaaa") -> tuple[Path, Path]:
    source = tmp_path / "source" / "train"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    image_path = source / "images" / f"{name}.jpg"
    cv2.imwrite(str(image_path), np.full((120, 240, 3), 180, np.uint8))
    (source / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.6 0.35\n", encoding="utf-8")
    return source, image_path


def test_merge_orders_two_line_plates_top_first():
    observations = [
        VisionObservation(text="71854", confidence=0.9, y_center=0.25),
        VisionObservation(text="30M", confidence=0.95, y_center=0.75),
    ]
    result = merge_observations(observations)
    assert result is not None
    assert result.label == "30M71854"
    assert result.parts == 2
    assert result.confidence == pytest.approx(0.9)


def test_merge_drops_invalid_format():
    observations = [VisionObservation(text="NOTAPLATE", confidence=0.99, y_center=0.5)]
    assert merge_observations(observations) is None


def test_pseudo_label_drops_invalid_format(tmp_path):
    source, _ = _make_source(tmp_path)

    def recognize(_crop):
        return [("BADTEXT", 0.99, 0.5)]

    manifest, stats, _elapsed = pseudo_label_apple_vision(
        [source], tmp_path / "out", reserved_manifests=[], recognize=recognize
    )
    assert stats.invalid_format == 1
    assert stats.kept == 0
    assert load_plate_manifest(manifest) == []


def test_pseudo_label_excludes_reserved_label_and_crop_hash(tmp_path):
    source, image_path = _make_source(tmp_path)

    def recognize_reserved_label(_crop):
        return [("30M71854", 0.99, 0.5)]

    reserved_label_manifest = _write_reserved_manifest(
        tmp_path / "reserved_label",
        label="30M71854",
        source_hash="1" * 64,
        crop_hash="2" * 64,
    )
    _, stats_label, _ = pseudo_label_apple_vision(
        [source],
        tmp_path / "out_label",
        reserved_manifests=[reserved_label_manifest],
        recognize=recognize_reserved_label,
    )
    assert stats_label.reserved_label == 1
    assert stats_label.kept == 0

    def recognize_other_label(_crop):
        return [("51G10096", 0.99, 0.5)]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    crop = image[39:81, 48:192]
    ok, encoded = cv2.imencode(".jpg", crop)
    assert ok
    actual_crop_hash = hashlib.sha256(encoded.tobytes()).hexdigest()

    reserved_crop_manifest = _write_reserved_manifest(
        tmp_path / "reserved_crop",
        label="99Z99999",
        source_hash="1" * 64,
        crop_hash=actual_crop_hash,
    )
    manifest, stats_crop, _ = pseudo_label_apple_vision(
        [source],
        tmp_path / "out_crop",
        reserved_manifests=[reserved_crop_manifest],
        recognize=recognize_other_label,
    )
    assert stats_crop.reserved_crop_hash == 1
    assert stats_crop.kept == 0
    assert load_plate_manifest(manifest) == []


def test_manifest_rows_carry_expected_metadata(tmp_path):
    source, _ = _make_source(tmp_path)

    def recognize(_crop):
        return [("30M71854", 0.87654321, 0.5)]

    manifest, stats, _ = pseudo_label_apple_vision(
        [source], tmp_path / "out", reserved_manifests=[], recognize=recognize
    )
    loaded = load_plate_manifest(manifest)
    assert stats.kept == 1
    row = loaded[0]
    assert row.source_type == "pseudo"
    assert row.verified is False
    assert row.split == "train"
    assert row.metadata["ocr_confidence"] == "0.87654321"
    assert row.metadata["crop_sha256"]
    assert "apple-vision-VNRecognizeTextRequest" in row.metadata["parameters_json"]


def test_deterministic_naming_matches_source_hash_scheme(tmp_path):
    source, image_path = _make_source(tmp_path, name="vehicle7_jpg.rf.bbbb")
    source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    expected = _deterministic_name(source_hash, image_path, source, 0)

    def recognize(_crop):
        return [("30M71854", 0.99, 0.5)]

    manifest, _, _ = pseudo_label_apple_vision(
        [source], tmp_path / "out", reserved_manifests=[], recognize=recognize
    )
    row = load_plate_manifest(manifest)[0]
    assert row.image_path.name == expected
    assert row.image_path.name.startswith(f"pseudo_{source_hash[:16]}_")


def test_build_reserved_identities_collects_hashes_without_using_images(tmp_path):
    manifest = _write_reserved_manifest(
        tmp_path,
        label="30M71854",
        source_hash="a" * 64,
        crop_hash="b" * 64,
    )
    labels, source_hashes, crop_hashes = build_reserved_identities([manifest])
    assert "30M71854" in labels
    assert "a" * 64 in source_hashes
    assert "b" * 64 in crop_hashes


def test_audit_pipeline_writes_band_stats_and_sheet_map(tmp_path):
    source, _ = _make_source(tmp_path)

    def recognize(_crop):
        return [("30M71854", 0.55, 0.5)]

    output = tmp_path / "generated"
    manifest, _, _ = pseudo_label_apple_vision(
        [source], output, reserved_manifests=[], recognize=recognize
    )
    audit_dir = tmp_path / "audit"
    result = run_audit_pipeline(manifest, audit_dir=audit_dir)
    assert Path(result["band_stats_path"]).is_file()
    assert Path(result["sheet_map"]).is_file()
    assert result["sheet_count"] >= 1
    rows = list(csv.DictReader(Path(result["sheet_map"]).open(encoding="utf-8")))
    assert rows[0]["cell_index"] == "1"
    assert "label" not in rows[0]
