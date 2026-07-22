"""Tests for crawl corpus extract/download helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.download_crawl_car_plates import (
    extract_flat_yolo_zip,
    extract_hf_yolo_zip,
    inventory_yolo_splits,
    write_hf_provenance,
)


def test_extract_hf_yolo_zip_splits_images_and_labels(tmp_path: Path):
    archive_path = tmp_path / "hf.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset.yaml", "names: ['BSD', 'BSV']\n")
        archive.writestr("images/train/frame0.png", b"png")
        archive.writestr("labels/train/frame0.txt", "0 0.5 0.5 0.2 0.1\n")
        archive.writestr("images/val/frame1.jpg", b"jpg")
        archive.writestr("labels/val/frame1.txt", "1 0.5 0.5 0.2 0.1\n")

    output_root = tmp_path / "hf_root"
    results = extract_hf_yolo_zip(archive_path, output_root)
    assert len(results) == 3
    assert (output_root / "train/images/frame0.png").is_file()
    assert (output_root / "train/labels/frame0.txt").is_file()
    assert (output_root / "val/images/frame1.jpg").is_file()
    assert (output_root / "val/labels/frame1.txt").is_file()


def test_extract_flat_yolo_zip_splits_images_and_labels(tmp_path: Path):
    archive_path = tmp_path / "flat.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/frame0.jpg", b"jpg")
        archive.writestr("nested/frame0.txt", "0 0.5 0.5 0.2 0.1\n")
    train_root = tmp_path / "train"
    images, labels = extract_flat_yolo_zip(archive_path, train_root)
    assert images == 1
    assert labels == 1
    assert (train_root / "images/frame0.jpg").is_file()
    assert (train_root / "labels/frame0.txt").is_file()


def test_inventory_yolo_splits_counts_pairs(tmp_path: Path, monkeypatch):
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    root = tmp_path / "corpus"
    split = root / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    image_path = split / "images" / "car_a.jpg"
    cv2.imwrite(str(image_path), np.full((60, 180, 3), 120, np.uint8))
    (split / "labels" / "car_a.txt").write_text("0 0.5 0.5 0.5 0.2\n", encoding="utf-8")
    report = inventory_yolo_splits(root)
    assert report["total_images"] == 1
    assert report["total_labels"] == 1
    assert report["class_histogram"]["0"] == 1


def test_write_hf_provenance_documents_unknown_url(tmp_path: Path):
    zip_path = tmp_path / "dataset.zip"
    zip_path.write_bytes(b"zip")
    from scripts.download_crawl_car_plates import ExtractResult

    path = write_hf_provenance(
        tmp_path,
        zip_path=zip_path,
        zip_sha256="abc",
        extract_results=[ExtractResult("train", 1, 1, 1)],
        inventory={"total_images": 1},
    )
    text = path.read_text(encoding="utf-8")
    assert "BSD" in text
    assert "unknown" in text.lower()
    assert "keremberke" in text
