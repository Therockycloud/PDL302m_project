"""CSV + photo persistence for the dashboard Registry mode."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _PROJECT_ROOT / "main" / "data" / "database.csv"
DEFAULT_PHOTOS_DIR = _PROJECT_ROOT / "main" / "data" / "registry" / "photos"

CSV_COLUMNS = ("license_plate", "car_brand", "car_color")


class DuplicatePlateError(ValueError):
    """Raised when adding a plate that already exists (normalized match)."""


def normalize_plate(plate: str) -> str:
    """Strip spaces, dashes, and dots; return upper-case plate key."""
    return re.sub(r"[\s\-\.]", "", str(plate)).upper()


def _photo_path(photos_dir: Path, plate_key: str) -> Path:
    return photos_dir / f"{plate_key}.jpg"


def _read_csv_rows(db_path: Path) -> list[dict[str, str]]:
    if not db_path.is_file():
        return []
    with db_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {col: str(row.get(col, "")).strip() for col in CSV_COLUMNS}
            for row in reader
            if any(str(row.get(col, "")).strip() for col in CSV_COLUMNS)
        ]


def _write_csv_rows(db_path: Path, rows: list[dict[str, str]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def _save_photo(photos_dir: Path, plate_key: str, image_bytes: bytes) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    cv2.imwrite(str(_photo_path(photos_dir, plate_key)), image)


def list_vehicles(
    db_path: Path | str | None = None,
    photos_dir: Path | str | None = None,
) -> list[dict[str, str | None]]:
    """Return registered vehicles with optional photo paths."""
    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    photos = Path(photos_dir) if photos_dir is not None else DEFAULT_PHOTOS_DIR
    vehicles: list[dict[str, str | None]] = []
    for row in _read_csv_rows(db):
        plate_display = row["license_plate"]
        plate_key = normalize_plate(plate_display)
        if not plate_key:
            continue
        photo = _photo_path(photos, plate_key)
        vehicles.append(
            {
                "plate_display": plate_display,
                "plate_key": plate_key,
                "brand": row["car_brand"],
                "color": row["car_color"],
                "photo_path": str(photo) if photo.is_file() else None,
            }
        )
    return vehicles


def add_vehicle(
    plate: str,
    brand: str,
    color: str,
    image_bytes: bytes | None = None,
    *,
    db_path: Path | str | None = None,
    photos_dir: Path | str | None = None,
) -> dict[str, str | None]:
    """Append a vehicle row and optionally save a reference photo."""
    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    photos = Path(photos_dir) if photos_dir is not None else DEFAULT_PHOTOS_DIR

    plate_display = str(plate).strip()
    plate_key = normalize_plate(plate_display)
    if not plate_key:
        raise ValueError("License plate is required.")

    rows = _read_csv_rows(db)
    for row in rows:
        if normalize_plate(row["license_plate"]) == plate_key:
            raise DuplicatePlateError(f"Plate {plate_display} is already registered.")

    brand_value = str(brand).strip()
    color_value = str(color).strip()
    rows.append(
        {
            "license_plate": plate_display,
            "car_brand": brand_value,
            "car_color": color_value,
        }
    )
    _write_csv_rows(db, rows)

    if image_bytes:
        _save_photo(photos, plate_key, image_bytes)

    photo = _photo_path(photos, plate_key)
    return {
        "plate_display": plate_display,
        "plate_key": plate_key,
        "brand": brand_value,
        "color": color_value,
        "photo_path": str(photo) if photo.is_file() else None,
    }


def delete_vehicle(
    plate: str,
    *,
    db_path: Path | str | None = None,
    photos_dir: Path | str | None = None,
) -> bool:
    """Remove a vehicle by normalized plate. Returns True if a row was removed."""
    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    photos = Path(photos_dir) if photos_dir is not None else DEFAULT_PHOTOS_DIR
    plate_key = normalize_plate(plate)
    if not plate_key:
        return False

    rows = _read_csv_rows(db)
    kept = [row for row in rows if normalize_plate(row["license_plate"]) != plate_key]
    if len(kept) == len(rows):
        return False

    _write_csv_rows(db, kept)
    photo = _photo_path(photos, plate_key)
    if photo.is_file():
        photo.unlink()
    return True
