"""Reproducible manifest loader for Vietnamese plate OCR data.

Evaluation rows are deliberately stricter than training rows: validation and
test data must be manually verified real images. Synthetic and pseudo-labelled
data can therefore never silently inflate a real-world accuracy claim.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.models.vn_plate_text import (
    normalize_plate_crop,
    normalize_plate_text,
    validate_vietnamese_plate,
)


CORE_FIELDS = (
    "image_path",
    "label",
    "source_type",
    "group_id",
    "split",
    "verified",
)
VALID_SPLITS = frozenset({"train", "val", "test"})
VALID_SOURCE_TYPES = frozenset({"real", "synthetic", "pseudo"})


def plate_identity_group(label: object) -> str:
    """Return a stable pseudonymous (not anonymous) identifier for plate text."""

    normalized = normalize_plate_text(label)
    if not validate_vietnamese_plate(normalized):
        raise ValueError(f"invalid label {label!r}")
    return f"vehicle:{hashlib.sha256(normalized.encode('ascii')).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class PlateManifestRow:
    image_path: Path
    label: str
    source_type: str
    group_id: str
    split: str
    verified: bool
    metadata: Mapping[str, str]

    @property
    def counts_toward_real_accuracy(self) -> bool:
        return self.source_type == "real" and self.verified and self.split in {"val", "test"}


def _parse_verified(value: object, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"row {row_number}: invalid verified value {value!r}")


def load_plate_manifest(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> list[PlateManifestRow]:
    """Load and validate a CSV manifest, resolving images beside the CSV."""

    manifest_path = Path(path).expanduser().resolve()
    trusted_root = Path(allowed_root).expanduser().resolve() if allowed_root is not None else None
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(CORE_FIELDS) - columns
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)

    result: list[PlateManifestRow] = []
    seen_images: set[Path] = set()
    group_splits: dict[str, str] = {}
    label_groups: dict[str, str] = {}
    identity_splits: dict[tuple[str, str], str] = {}
    content_splits: dict[str, tuple[str, str]] = {}
    metadata_fields = columns - set(CORE_FIELDS)

    for row_number, raw in enumerate(raw_rows, start=2):
        image_value = str(raw["image_path"] or "").strip()
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if trusted_root is not None and not image_path.is_relative_to(trusted_root):
            raise ValueError(f"row {row_number}: image_path escapes allowed_root {trusted_root}")
        if image_path in seen_images:
            raise ValueError(f"row {row_number}: duplicate image_path {image_value!r}")
        seen_images.add(image_path)
        if not image_path.is_file():
            raise ValueError(f"row {row_number}: missing image file {image_value!r}")

        label = normalize_plate_text(raw["label"])
        if not validate_vietnamese_plate(label):
            raise ValueError(f"row {row_number}: invalid label {raw['label']!r}")

        source_type = str(raw["source_type"] or "").strip().lower()
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"row {row_number}: invalid source_type {source_type!r}")
        split = str(raw["split"] or "").strip().lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"row {row_number}: invalid split {split!r}")
        verified = _parse_verified(raw["verified"], row_number)
        group_id = str(raw["group_id"] or "").strip()
        if not group_id:
            raise ValueError(f"row {row_number}: group_id must not be empty")

        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise ValueError(
                f"group_id {group_id!r} appears in multiple splits: "
                f"{previous_split!r} and {split!r}"
            )
        label_key = ("normalized label", label)
        previous_label_split = identity_splits.setdefault(label_key, split)
        if previous_label_split != split:
            raise ValueError(f"normalized label {label!r} crosses splits")
        if source_type == "real" and verified:
            previous_group = label_groups.setdefault(label, group_id)
            if previous_group != group_id:
                raise ValueError(
                    f"label {label!r} appears with multiple group_id values: "
                    f"{previous_group!r} and {group_id!r}"
                )
        for field in ("source_sha256", "crop_sha256"):
            value = str(raw.get(field) or "").strip().lower()
            if value:
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"row {row_number}: invalid {field}")
                previous = content_splits.setdefault(value, (split, field))
                if previous[0] != split:
                    raise ValueError(
                        f"content hash {value!r} crosses splits via {previous[1]} and {field}"
                    )
        if split in {"val", "test"} and not (source_type == "real" and verified):
            raise ValueError(
                f"row {row_number}: {split} claims require a manually verified real label"
            )

        result.append(
            PlateManifestRow(
                image_path=image_path,
                label=label,
                source_type=source_type,
                group_id=group_id,
                split=split,
                verified=verified,
                metadata={field: str(raw.get(field) or "") for field in sorted(metadata_fields)},
            )
        )
    return result


def _row_identities(row: PlateManifestRow) -> list[tuple[str, str, str]]:
    identities = [
        ("content hash", str(row.metadata.get(field, "")).strip().lower(), field)
        for field in ("source_sha256", "crop_sha256")
        if str(row.metadata.get(field, "")).strip()
    ]
    identities.append(("plate label", row.label, "label"))
    identities.append(("group_id", row.group_id, "group_id"))
    return identities


def compose_plate_manifests(
    manifests: Sequence[str | Path],
    *,
    split: str,
    reserved_manifests: Sequence[str | Path] = (),
    allowed_root: str | Path | None = None,
) -> list[PlateManifestRow]:
    """Compose one split while enforcing connected held-out identity isolation."""

    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split {split!r}")
    rows = [
        row for manifest in manifests
        for row in load_plate_manifest(manifest, allowed_root=allowed_root) if row.split == split
    ]
    reserved = [
        row for manifest in reserved_manifests
        for row in load_plate_manifest(manifest, allowed_root=allowed_root)
    ]
    reserved_identities = {
        (kind, value) for row in reserved for kind, value, _field in _row_identities(row)
    }
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        for kind, value, field in _row_identities(row):
            identity = (kind, value)
            if identity in reserved_identities:
                raise ValueError(f"{split} row matches reserved {kind} via {field}: {value}")
            previous = seen.setdefault(identity, row.split)
            if previous != row.split:
                raise ValueError(f"{kind} crosses splits")
    return rows


def grouped_split(
    group_ids: Sequence[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[str]:
    """Assign rows to splits deterministically, atomically by ``group_id``."""

    if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values with a positive sum")
    unique_groups = sorted({str(group) for group in group_ids})
    if any(not group for group in unique_groups):
        raise ValueError("group_id must not be empty")
    random.Random(seed).shuffle(unique_groups)
    total = len(unique_groups)
    normalized = [value / sum(ratios) for value in ratios]
    train_end = round(total * normalized[0])
    val_end = train_end + round(total * normalized[1])
    assignments = {
        group: "train" if index < train_end else "val" if index < val_end else "test"
        for index, group in enumerate(unique_groups)
    }
    return [assignments[str(group)] for group in group_ids]


class PlateOCRDataset:
    """Minimal indexable dataset compatible with PyTorch data loaders.

    Samples are CHW float32 NumPy arrays. PyTorch's default collator converts
    these arrays to tensors without importing torch in this data-only module.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str = "train",
        *,
        reserved_manifests: Sequence[str | Path] = (),
        allowed_root: str | Path | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split {split!r}")
        self.rows = compose_plate_manifests(
            [manifest], split=split, reserved_manifests=reserved_manifests,
            allowed_root=allowed_root,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        item = self.rows[index]
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode plate image: {item.image_path}")
        strip = normalize_plate_crop(image, output_size=(192, 64))
        chw = np.ascontiguousarray(strip.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw, item.label


__all__ = [
    "CORE_FIELDS",
    "PlateManifestRow",
    "PlateOCRDataset",
    "compose_plate_manifests",
    "grouped_split",
    "load_plate_manifest",
    "plate_identity_group",
]
