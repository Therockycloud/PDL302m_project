"""Tests for registry CSV + photo store."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from src.utils.registry_store import (  # noqa: E402
    DuplicatePlateError,
    add_vehicle,
    delete_vehicle,
    list_vehicles,
    normalize_plate,
)


def _jpeg_bytes() -> bytes:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_normalize_plate_strips_symbols_and_uppercases() -> None:
    assert normalize_plate(" 30f-123.45 ") == "30F12345"


def test_list_vehicles_empty_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"
    assert list_vehicles(db_path, photos_dir) == []


def test_add_and_list_vehicle_without_photo(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"

    added = add_vehicle("51A-123.45", "Toyota Vios", "White", db_path=db_path, photos_dir=photos_dir)
    vehicles = list_vehicles(db_path, photos_dir)

    assert added["plate_display"] == "51A-123.45"
    assert added["photo_path"] is None
    assert len(vehicles) == 1
    assert vehicles[0]["plate_key"] == "51A12345"
    assert vehicles[0]["brand"] == "Toyota Vios"
    assert vehicles[0]["color"] == "White"


def test_add_vehicle_rejects_duplicate_normalized_plate(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"
    add_vehicle("30F-12345", "Toyota", "White", db_path=db_path, photos_dir=photos_dir)

    with pytest.raises(DuplicatePlateError):
        add_vehicle("30f 123.45", "Honda", "Black", db_path=db_path, photos_dir=photos_dir)


def test_add_vehicle_saves_photo_as_normalized_jpg(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"

    added = add_vehicle(
        "43H-321.65",
        "Ford Ranger",
        "White",
        image_bytes=_jpeg_bytes(),
        db_path=db_path,
        photos_dir=photos_dir,
    )

    assert added["photo_path"] is not None
    assert Path(added["photo_path"]).is_file()
    vehicles = list_vehicles(db_path, photos_dir)
    assert vehicles[0]["photo_path"] == added["photo_path"]


def test_delete_vehicle_removes_row_and_photo(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"
    add_vehicle(
        "29D-88888",
        "Mazda CX5",
        "Grey",
        image_bytes=_jpeg_bytes(),
        db_path=db_path,
        photos_dir=photos_dir,
    )

    removed = delete_vehicle("29d 888.88", db_path=db_path, photos_dir=photos_dir)

    assert removed is True
    assert list_vehicles(db_path, photos_dir) == []
    assert not (photos_dir / "29D88888.jpg").exists()


def test_delete_vehicle_returns_false_when_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "database.csv"
    photos_dir = tmp_path / "photos"
    assert delete_vehicle("99Z-00000", db_path=db_path, photos_dir=photos_dir) is False
