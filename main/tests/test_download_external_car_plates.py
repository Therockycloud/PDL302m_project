"""Tests for external Vietnamese car plate corpus download helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_external_car_plates import (
    DownloadAttempt,
    _extract_yolo_zip,
    _inventory_yolo_root,
    _kaggle_available,
    write_provenance,
)
from scripts.pseudo_label_apple_vision import label_conflicts_with_reserved as reserved_helper


def test_kaggle_available_reports_missing_cli_or_credentials(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    ok, detail = _kaggle_available()
    assert ok is False
    assert "kaggle CLI" in detail


def test_extract_yolo_zip_splits_images_and_labels(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("yolo_plate_dataset/frame0.jpg", b"jpg")
        archive.writestr("yolo_plate_dataset/frame0.txt", "0 0.5 0.5 0.2 0.1\n")
        archive.writestr("yolo_plate_dataset/frame1.jpg", b"jpg2")
    train_root = tmp_path / "train"
    images, labels = _extract_yolo_zip(archive_path, train_root)
    assert images == 2
    assert labels == 1
    assert (train_root / "images/frame0.jpg").is_file()
    assert (train_root / "labels/frame0.txt").is_file()


def test_inventory_yolo_root_counts_pairs(tmp_path: Path, monkeypatch):
  pytest.importorskip("cv2")
  import numpy as np
  import cv2

  train = tmp_path / "train"
  (train / "images").mkdir(parents=True)
  (train / "labels").mkdir()
  image_path = train / "images" / "car_a.jpg"
  cv2.imwrite(str(image_path), np.full((60, 180, 3), 120, np.uint8))
  (train / "labels" / "car_a.txt").write_text("0 0.5 0.5 0.5 0.2\n", encoding="utf-8")
  report = _inventory_yolo_root(train)
  assert report["image_count"] == 1
  assert report["label_count"] == 1
  assert report["paired_estimate"] == 1


def test_write_provenance_includes_attempts(tmp_path: Path):
    path = write_provenance(
        tmp_path,
        attempts=[DownloadAttempt("kaggle", False, "missing credentials")],
        winter_inventory={"image_count": 10},
        mrzaizai2k_inventory=None,
    )
    text = path.read_text(encoding="utf-8")
    assert "kaggle" in text
    assert "winter2897" in text
    assert "10" in text


def test_label_conflicts_with_reserved_near_matches():
    reserved = {"66P189575", "30M71854"}
    assert reserved_helper("66P189575", reserved) is True
    assert reserved_helper("66P18957", reserved) is True
    assert reserved_helper("66P1895755", reserved) is True
    assert reserved_helper("30M71855", reserved) is True
    assert reserved_helper("51G01177", reserved) is False
