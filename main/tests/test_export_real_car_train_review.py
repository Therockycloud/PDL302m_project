"""Tests for reserved-disjoint car plate review export."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.export_real_car_train_review import (
    _is_car_like,
    assert_no_leakage,
    collect_car_candidates,
    run_pipeline,
    select_diverse_per_label,
    _Candidate,
)
from src.datasets.plate_ocr_dataset import CORE_FIELDS


def _write_reserved_manifest(
    tmp_path: Path,
    *,
    label: str = "30M71854",
    source_hash: str = "1" * 64,
    crop_hash: str = "2" * 64,
) -> Path:
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


def _make_car_source(tmp_path: Path, name: str = "plate_jpg.rf.aaaa", pixel: int = 180) -> tuple[Path, Path]:
    source = tmp_path / "source" / "train"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    image_path = source / "images" / f"{name}.jpg"
    cv2.imwrite(str(image_path), np.full((80, 240, 3), pixel, np.uint8))
    (source / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.6 0.25\n", encoding="utf-8")
    return source, image_path


def test_is_car_like_filters_motorcycle_aspect_ratio():
    moto = np.zeros((80, 120, 3), np.uint8)
    car = np.zeros((60, 180, 3), np.uint8)
    assert _is_car_like(moto) == (False, 1.5)
    ok, aspect = _is_car_like(car)
    assert ok is True
    assert aspect == pytest.approx(3.0)


def test_select_diverse_per_label_prefers_distinct_sources():
    candidates = [
        _Candidate(
            crop=np.zeros((1, 1, 3), np.uint8),
            label="30M71854",
            ocr_confidence=0.99,
            source_ref=Path("a.jpg"),
            source_sha256="a" * 64,
            crop_sha256="1" * 64,
            box=[0, 0, 1, 1],
            aspect_ratio=3.0,
            crop_name="a.jpg",
        ),
        _Candidate(
            crop=np.zeros((1, 1, 3), np.uint8),
            label="30M71854",
            ocr_confidence=0.95,
            source_ref=Path("b.jpg"),
            source_sha256="b" * 64,
            crop_sha256="2" * 64,
            box=[0, 0, 1, 1],
            aspect_ratio=3.0,
            crop_name="b.jpg",
        ),
        _Candidate(
            crop=np.zeros((1, 1, 3), np.uint8),
            label="30M71854",
            ocr_confidence=0.90,
            source_ref=Path("a2.jpg"),
            source_sha256="a" * 64,
            crop_sha256="3" * 64,
            box=[0, 0, 1, 1],
            aspect_ratio=3.0,
            crop_name="a2.jpg",
        ),
    ]
    selected = select_diverse_per_label(candidates, max_per_label=2)
    assert len(selected) == 2
    assert {item.source_sha256 for item in selected} == {"a" * 64, "b" * 64}


def test_collect_car_candidates_drops_near_reserved_label(tmp_path):
    source, _ = _make_car_source(tmp_path, name="near_reserved")

    def recognize(_crop):
        return [("30M7185", 0.99, 0.5)]

    reserved = _write_reserved_manifest(tmp_path / "reserved_near", label="30M71854")
    candidates, stats = collect_car_candidates(
        [source],
        reserved_manifests=[reserved],
        exclude_source_patterns=(),
        recognize=recognize,
    )
    assert candidates == []
    assert stats.near_reserved_label == 1


def test_collect_car_candidates_drops_reserved_label_and_low_ar(tmp_path):
    source, _ = _make_car_source(tmp_path)

    def recognize(_crop):
        return [("30M71854", 0.99, 0.5)]

    reserved = _write_reserved_manifest(tmp_path / "reserved", label="30M71854")
    candidates, stats = collect_car_candidates(
        [source],
        reserved_manifests=[reserved],
        exclude_source_patterns=("*Gen*",),
        recognize=recognize,
    )
    assert candidates == []
    assert stats.reserved_label == 1
    assert stats.low_ar == 0

    moto_source = tmp_path / "source2" / "train"
    (moto_source / "images").mkdir(parents=True)
    (moto_source / "labels").mkdir()
    moto_image = moto_source / "images" / "moto_jpg.rf.bbbb.jpg"
    cv2.imwrite(str(moto_image), np.full((80, 120, 3), 100, np.uint8))
    (moto_source / "labels" / "moto_jpg.rf.bbbb.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    _, stats2 = collect_car_candidates(
        [moto_source],
        reserved_manifests=[],
        exclude_source_patterns=("*Gen*",),
        recognize=recognize,
    )
    assert stats2.low_ar == 1


def test_run_pipeline_writes_outputs_without_vision(tmp_path):
    source, image_path = _make_car_source(tmp_path)

    def recognize(_crop):
        return [("51F12345", 0.88, 0.5)]

    result = run_pipeline(
        sources=[source],
        output_dir=tmp_path / "review",
        reserved_manifests=[],
        recognize=recognize,
        sheet_label_cap=1,
    )
    output = Path(result["output_dir"])
    assert (output / "candidates.csv").is_file()
    assert (output / "stats.json").is_file()
    assert (output / "sheet_map.csv").is_file()
    assert (output / "transcription_template.csv").is_file()
    assert list((output / "crops").glob("*.jpg"))
    assert result["leakage_check"]["label_intersection"] == 0

    rows = list(csv.DictReader((output / "candidates.csv").open(encoding="utf-8")))
    assert rows[0]["label_draft"] == "51F12345"
    assert float(rows[0]["aspect_ratio"]) >= 2.2
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == rows[0]["source_sha256"]

    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["kept"] == 1
    assert stats["unique_labels"] == 1


def test_assert_no_leakage_detects_label_collision():
    candidate = _Candidate(
        crop=np.zeros((1, 1, 3), np.uint8),
        label="30M71854",
        ocr_confidence=0.9,
        source_ref=Path("a.jpg"),
        source_sha256="c" * 64,
        crop_sha256="d" * 64,
        box=[0, 0, 1, 1],
        aspect_ratio=3.0,
        crop_name="a.jpg",
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(
            [candidate],
            reserved_labels={"30M71854"},
            reserved_source_hashes=set(),
            reserved_crop_hashes=set(),
        )
