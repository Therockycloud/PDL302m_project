"""Tests for reproducible plate OCR manifests and datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pytest
from PIL import ImageFont

from scripts.build_plate_ocr_dataset import (
    audit_manifest_provenance,
    build_review_candidates,
    build_review_sheet,
    extract_video_candidates,
    generate_synthetic,
    pseudo_label_yolo,
    main as dataset_builder_main,
    validate_manifest,
    parse_pixel_box,
    parse_yolo_box,
    replay_synthetic_manifest,
)
from src.datasets.plate_ocr_dataset import (
    CORE_FIELDS,
    PlateOCRDataset,
    grouped_split,
    load_plate_manifest,
    compose_plate_manifests,
)


def row(
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


def write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
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


def test_manifest_normalizes_labels_and_resolves_paths(tmp_path):
    manifest = write_manifest(tmp_path, [row("images/a.png", " 30m-718.54 ")])
    loaded = load_plate_manifest(manifest)
    assert loaded[0].label == "30M71854"
    assert loaded[0].image_path == (tmp_path / "images/a.png").resolve()


def test_manifest_rejects_missing_columns(tmp_path):
    item = row("a.png")
    del item["verified"]
    manifest = write_manifest(tmp_path, [item])
    with pytest.raises(ValueError, match="missing required columns.*verified"):
        load_plate_manifest(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(row("a.png", group_id="vehicle-2")), "duplicate image_path"),
        (lambda rows: rows[0].update(image_path="missing.png"), "missing image file"),
        (lambda rows: rows[0].update(label="NOT-A-PLATE"), "invalid label"),
        (lambda rows: rows[0].update(split="holdout"), "invalid split"),
        (lambda rows: rows[0].update(source_type="downloaded"), "invalid source_type"),
        (lambda rows: rows[0].update(verified="maybe"), "invalid verified"),
    ],
)
def test_manifest_rejects_invalid_rows(tmp_path, mutate, message):
    rows = [row("a.png")]
    manifest = write_manifest(tmp_path, rows)
    mutate(rows)
    # Rewrite after mutation without creating newly referenced missing files.
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CORE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        load_plate_manifest(manifest)


def test_manifest_rejects_group_leakage(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [
            row("a.png", "30M71854", "real", "vehicle-1", "train", True),
            row("b.png", "30M71854", "real", "vehicle-1", "test", True),
        ],
    )
    with pytest.raises(ValueError, match="group_id"):
        load_plate_manifest(manifest)


def test_manifest_rejects_one_plate_identity_in_multiple_groups(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [
            row("a.png", "30M71854", "real", "vehicle:a", "test", True),
            row("b.png", "30M71854", "real", "vehicle:b", "test", True),
        ],
    )
    with pytest.raises(ValueError, match="label.*group_id"):
        load_plate_manifest(manifest)


def test_manifest_rejects_pseudo_train_and_real_test_same_label(tmp_path):
    manifest = write_manifest(tmp_path, [
        row("pseudo.png", "30M71854", "pseudo", "source:a", "train", False),
        row("real.png", "30M71854", "real", "vehicle:a", "test", True),
    ])
    with pytest.raises(ValueError, match="label.*crosses splits"):
        load_plate_manifest(manifest)


@pytest.mark.parametrize("identity_field", ["source_sha256", "crop_sha256"])
def test_manifest_rejects_hash_identity_crossing_splits(tmp_path, identity_field):
    manifest = write_manifest(tmp_path, [
        {**row("a.png", "30M71854", group_id="v1", split="train"), identity_field: "a" * 64},
        {**row("b.png", "51G10096", group_id="v2", split="test"), identity_field: "a" * 64},
    ])
    with pytest.raises(ValueError, match=identity_field):
        load_plate_manifest(manifest)


def test_reserved_manifest_rejects_train_identity_and_allows_disjoint(tmp_path):
    reserved = write_manifest(tmp_path / "reserved", [{
        **row("held.jpg", "30M71854", group_id="held", split="test"),
        "source_sha256": "1" * 64, "crop_sha256": "2" * 64,
    }])
    conflicting = write_manifest(tmp_path / "conflict", [{
        **row("train.jpg", "51G10096", group_id="train"), "source_sha256": "1" * 64,
    }])
    with pytest.raises(ValueError, match="reserved"):
        compose_plate_manifests([conflicting], split="train", reserved_manifests=[reserved])
    clean = write_manifest(tmp_path / "clean", [{
        **row("train.jpg", "51G10096", group_id="train"),
        "source_sha256": "3" * 64, "crop_sha256": "4" * 64,
    }])
    assert len(compose_plate_manifests([clean], split="train", reserved_manifests=[reserved])) == 1


def test_reserved_manifest_rejects_train_group_used_by_test(tmp_path):
    reserved = write_manifest(tmp_path / "reserved", [
        row("held.jpg", "30M71854", group_id="shared", split="test"),
    ])
    train = write_manifest(tmp_path / "train", [
        row("train.jpg", "51G10096", group_id="shared", split="train"),
    ])
    with pytest.raises(ValueError, match="train row matches reserved group_id"):
        compose_plate_manifests([train], split="train", reserved_manifests=[reserved])


def test_reserved_manifest_rejects_val_group_used_by_test(tmp_path):
    reserved = write_manifest(tmp_path / "reserved", [
        row("held.jpg", "30M71854", group_id="shared", split="test"),
    ])
    validation = write_manifest(tmp_path / "validation", [
        row("val.jpg", "51G10096", group_id="shared", split="val"),
    ])
    with pytest.raises(ValueError, match="val row matches reserved group_id"):
        compose_plate_manifests([validation], split="val", reserved_manifests=[reserved])


def test_reserved_manifest_allows_disjoint_groups(tmp_path):
    reserved = write_manifest(tmp_path / "reserved", [
        row("held.jpg", "30M71854", group_id="held", split="test"),
    ])
    validation = write_manifest(tmp_path / "validation", [
        row("val.jpg", "51G10096", group_id="validation", split="val"),
    ])
    assert compose_plate_manifests(
        [validation], split="val", reserved_manifests=[reserved]
    )[0].group_id == "validation"


def test_reserved_manifest_rejects_train_label_and_crop(tmp_path):
    reserved = write_manifest(tmp_path / "reserved", [{
        **row("held.jpg", "30M71854", group_id="held", split="test"),
        "source_sha256": "1" * 64, "crop_sha256": "2" * 64,
    }])
    for directory, label, crop_hash, source_type, verified in (
        ("label", "30M71854", "4" * 64, "pseudo", False),
        ("crop", "51G10096", "2" * 64, "real", True),
    ):
        train = write_manifest(tmp_path / directory, [{
            **row("train.jpg", label, source_type=source_type, group_id="train", verified=verified),
            "crop_sha256": crop_hash,
        }])
        with pytest.raises(ValueError, match="reserved"):
            compose_plate_manifests([train], split="train", reserved_manifests=[reserved])


@pytest.mark.parametrize(("reserved_field", "train_field"), [
    ("crop_sha256", "source_sha256"), ("source_sha256", "crop_sha256")
])
def test_reserved_hash_namespace_is_shared(tmp_path, reserved_field, train_field):
    digest = "a" * 64
    reserved = write_manifest(tmp_path / "reserved", [{
        **row("held.jpg", "30M71854", group_id="held", split="test"), reserved_field: digest,
    }])
    train = write_manifest(tmp_path / "train", [{
        **row("train.jpg", "51G10096", group_id="train"), train_field: digest,
    }])
    with pytest.raises(ValueError, match="reserved.*hash"):
        compose_plate_manifests([train], split="train", reserved_manifests=[reserved])


def test_manifest_allowed_root_rejects_escaping_image(tmp_path):
    allowed = tmp_path / "allowed"; allowed.mkdir()
    outside = tmp_path / "outside.png"; cv2.imwrite(str(outside), np.zeros((5, 5, 3), np.uint8))
    manifest = allowed / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS); writer.writeheader(); writer.writerow(row("../outside.png"))
    with pytest.raises(ValueError, match="allowed_root"):
        load_plate_manifest(manifest, allowed_root=allowed)


def test_allowed_root_threads_through_composition_and_dataset(tmp_path):
    allowed = tmp_path / "allowed"; allowed.mkdir()
    outside = tmp_path / "outside.png"; cv2.imwrite(str(outside), np.zeros((5, 5, 3), np.uint8))
    manifest = allowed / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS); writer.writeheader(); writer.writerow(row("../outside.png"))
    with pytest.raises(ValueError, match="allowed_root"):
        compose_plate_manifests([manifest], split="train", allowed_root=allowed)
    with pytest.raises(ValueError, match="allowed_root"):
        PlateOCRDataset(manifest, allowed_root=allowed)


@pytest.mark.parametrize("split", ["val", "test"])
def test_held_out_split_requires_manually_verified_real_labels(tmp_path, split):
    manifest = write_manifest(
        tmp_path, [row("a.png", "30M71854", "pseudo", "vehicle-1", split, False)]
    )
    with pytest.raises(ValueError, match="verified real"):
        load_plate_manifest(manifest)


def test_synthetic_and_pseudo_rows_are_allowed_only_in_train(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [
            row("a.png", source_type="synthetic", group_id="syn-1", verified=False),
            row("b.png", source_type="pseudo", group_id="syn-1", verified=False),
        ],
    )
    loaded = load_plate_manifest(manifest)
    assert all(not item.counts_toward_real_accuracy for item in loaded)


def test_grouped_split_is_deterministic_and_keeps_groups_together():
    groups = [f"vehicle-{index // 3}" for index in range(30)]
    first = grouped_split(groups, seed=42)
    second = grouped_split(groups, seed=42)
    assert first == second
    assert all(len({first[i] for i, group in enumerate(groups) if group == target}) == 1 for target in set(groups))
    assert {"train", "val", "test"}.issubset(set(first))


def test_torch_compatible_dataset_loads_normalized_crop(tmp_path):
    manifest = write_manifest(tmp_path, [row("a.png")])
    dataset = PlateOCRDataset(manifest, split="train")
    image, label = dataset[0]
    assert image.shape == (3, 64, 192)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert label == "30M71854"


def test_synthetic_requires_an_existing_font(tmp_path):
    with pytest.raises(FileNotFoundError, match="font"):
        generate_synthetic(tmp_path, count=1, seed=42, font_path=tmp_path / "missing.ttf")


def test_synthetic_records_every_random_input_for_exact_replay(tmp_path, monkeypatch):
    font_path = tmp_path / "font.ttf"
    font_path.write_bytes(b"test-font-for-mocked-renderer")
    default_font = ImageFont.load_default()
    monkeypatch.setattr("scripts.build_plate_ocr_dataset.ImageFont.truetype", lambda *_a, **_k: default_font)
    first = generate_synthetic(tmp_path / "first", count=1, seed=42, font_path=font_path)
    second = generate_synthetic(tmp_path / "second", count=1, seed=42, font_path=font_path)
    first_row = next(csv.DictReader(first.open(encoding="utf-8")))
    second_row = next(csv.DictReader(second.open(encoding="utf-8")))
    params = json.loads(first_row["parameters_json"])
    assert {
        "perspective_offsets", "glare_opacity", "noise_seed", "occlusion_x",
        "occlusion_y", "occlusion_height", "occlusion_color",
    }.issubset(params)
    assert len(params["perspective_offsets"]) == 4
    assert re.search(r'"glare_opacity":0\.\d{7,}', first_row["parameters_json"])
    assert re.search(r'"noise_sigma":\d+\.\d{7,}', first_row["parameters_json"])
    assert first_row["parameters_json"] == second_row["parameters_json"]
    assert first_row["source_sha256"] == second_row["source_sha256"]
    assert not Path(first_row["font_path"]).is_absolute()
    assert (first.parent / first_row["font_path"]).is_file()
    assert replay_synthetic_manifest(first)["verified"] == 1
    assert all(first_row[field] for field in ("renderer_schema", "pillow_version", "opencv_version", "numpy_version"))


def test_synthetic_real_font_replays_when_font_available(tmp_path):
    import matplotlib
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    assert font.is_file(), f"expected packaged test font is absent: {font}"
    manifest = generate_synthetic(tmp_path / "real-font", count=1, seed=42, font_path=font)
    manifest_row = next(csv.DictReader(manifest.open()))
    assert manifest_row["crop_sha256"]
    assert replay_synthetic_manifest(manifest) == {"verified": 1}


@pytest.mark.parametrize(("field", "value", "message"), [
    ("renderer_schema", "future-v99", "renderer_schema"),
    ("pillow_version", "0.0", "Pillow version"),
    ("opencv_version", "0.0", "OpenCV version"),
    ("numpy_version", "0.0", "NumPy version"),
])
def test_synthetic_replay_rejects_schema_and_version_drift(tmp_path, monkeypatch, field, value, message):
    font = tmp_path / "font.ttf"; font.write_bytes(b"mock-font")
    default_font = ImageFont.load_default()
    monkeypatch.setattr("scripts.build_plate_ocr_dataset.ImageFont.truetype", lambda *_a, **_k: default_font)
    manifest = generate_synthetic(tmp_path / "out", count=1, seed=42, font_path=font)
    rows = list(csv.DictReader(manifest.open())); fields = list(rows[0]); rows[0][field] = value
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        replay_synthetic_manifest(manifest)


def test_pseudo_label_crops_yolo_boxes_and_keeps_candidates_unverified(tmp_path):
    source = tmp_path / "source" / "train"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    cv2.imwrite(str(source / "images/a.jpg"), np.full((100, 200, 3), 200, np.uint8))
    (source / "labels/a.txt").write_text("0 0.5 0.5 0.5 0.4\n", encoding="utf-8")

    class Reader:
        def read_plate(self, image):
            assert image.shape[:2] == (40, 100)
            return {"text": "30M-718.54", "ocr_conf": 0.99}

    manifest = pseudo_label_yolo(tmp_path / "source", tmp_path / "out", 0.95, Reader())
    loaded = load_plate_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0].source_type == "pseudo"
    assert loaded[0].verified is False
    assert loaded[0].split == "train"


def test_pseudo_label_groups_roboflow_variants_of_one_source(tmp_path):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    for digest in ("aaaa", "bbbb"):
        name = f"vehicle7_jpg.rf.{digest}"
        cv2.imwrite(str(split / f"images/{name}.jpg"), np.full((40, 80, 3), 200, np.uint8))
        (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    class Reader:
        def read_plate(self, _image):
            return {"text": "30M71854", "ocr_conf": 0.99}

    manifest = pseudo_label_yolo(tmp_path / "source", tmp_path / "out", 0.95, Reader())
    loaded = load_plate_manifest(manifest)
    assert len({item.group_id for item in loaded}) == 1


def test_pseudo_source_group_survives_differing_ocr_labels(tmp_path):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True); (split / "labels").mkdir()
    for digest in ("aaaa", "bbbb"):
        name = f"vehicle7_jpg.rf.{digest}"
        cv2.imwrite(str(split / f"images/{name}.jpg"), np.full((40, 80, 3), 200, np.uint8))
        (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    labels = iter(("30M71854", "51G10096"))
    class Reader:
        def read_plate(self, _image): return {"text": next(labels), "ocr_conf": 0.99}
    manifest = pseudo_label_yolo(tmp_path / "source", tmp_path / "out", 0.95, Reader())
    assert len({item.group_id for item in load_plate_manifest(manifest)}) == 1


def test_pseudo_duplicate_stems_in_different_dirs_get_unique_outputs(tmp_path):
    for split_name, value in (("train", 100), ("valid", 200)):
        split = tmp_path / f"source/{split_name}"
        (split / "images").mkdir(parents=True); (split / "labels").mkdir()
        cv2.imwrite(str(split / "images/same.jpg"), np.full((40, 80, 3), value, np.uint8))
        (split / "labels/same.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    class Reader:
        def read_plate(self, _image): return {"text": "30M71854", "ocr_conf": 0.99}
    manifest = pseudo_label_yolo(tmp_path / "source", tmp_path / "out", 0.95, Reader())
    assert len({item.image_path.name for item in load_plate_manifest(manifest)}) == 2


def test_pseudo_write_failure_and_collision_fail_loudly(tmp_path, monkeypatch):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True); (split / "labels").mkdir()
    cv2.imwrite(str(split / "images/a.jpg"), np.full((40, 80, 3), 100, np.uint8))
    (split / "labels/a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    class Reader:
        def read_plate(self, _image): return {"text": "30M71854", "ocr_conf": 0.99}
    monkeypatch.setattr("scripts.build_plate_ocr_dataset.cv2.imwrite", lambda *_a, **_k: False)
    with pytest.raises(RuntimeError, match="write image"):
        pseudo_label_yolo(tmp_path / "source", tmp_path / "failed", 0.95, Reader())
    monkeypatch.undo()
    pseudo_label_yolo(tmp_path / "source", tmp_path / "collision", 0.95, Reader())
    with pytest.raises(FileExistsError, match="overwrite"):
        pseudo_label_yolo(tmp_path / "source", tmp_path / "collision", 0.95, Reader())


@pytest.mark.parametrize("values", [(0.5, 0.5, -0.1, 0.2), (float("nan"), 0.5, 0.2, 0.2)])
def test_yolo_box_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="box"):
        parse_yolo_box(values, image_width=100, image_height=50, context="test box")


def test_pixel_box_clamps_and_rejects_empty():
    assert parse_pixel_box((-5, -2, 120, 60), image_width=100, image_height=50, context="box") == (0, 0, 100, 50)
    with pytest.raises(ValueError, match="box"):
        parse_pixel_box((120, 1, 130, 2), image_width=100, image_height=50, context="box")


def test_review_candidates_are_unique_ranked_and_unlabeled(tmp_path):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    candidates = [
        ("vehicle1_jpg.rf.aaaa", np.full((80, 160, 3), 127, np.uint8)),
        ("vehicle1_jpg.rf.bbbb", np.full((80, 160, 3), 127, np.uint8)),
        ("vehicle2_jpg.rf.cccc", np.indices((100, 200))[1].astype(np.uint8)),
    ]
    for name, image in candidates:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(split / f"images/{name}.jpg"), image)
        (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    review_csv = build_review_candidates(tmp_path / "source", tmp_path / "review", count=10)
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    assert [item["candidate_id"] for item in rows] == ["candidate-0000", "candidate-0001"]
    assert len({item["group_id"] for item in rows}) == 2
    assert rows[0]["group_id"] == "det:vehicle2_jpg"
    assert "label" not in rows[0]
    assert rows[0]["source_sha256"]
    assert rows[0]["crop_sha256"]
    assert json.loads(rows[0]["box_json"])
    assert (tmp_path / "review/contact_sheet_001.jpg").is_file()


def test_review_candidates_exclude_reserved_source_hashes(tmp_path):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    for name, value in (("reserved_jpg.rf.aaaa", 80), ("available_jpg.rf.bbbb", 160)):
        image_path = split / f"images/{name}.jpg"
        cv2.imwrite(str(image_path), np.full((80, 160, 3), value, np.uint8))
        (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    reserved_source = split / "images/reserved_jpg.rf.aaaa.jpg"
    reserved = write_manifest(tmp_path / "reserved", [{
        **row("held.jpg", split="test"),
        "source_sha256": hashlib.sha256(reserved_source.read_bytes()).hexdigest(),
    }])
    review_csv = build_review_candidates(
        tmp_path / "source", tmp_path / "review", count=10,
        reserved_manifests=[reserved],
    )

    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["group_id"] == "det:available_jpg"
    assert "label" not in rows[0]


def test_review_candidates_exclude_matching_source_names(tmp_path):
    split = tmp_path / "source/train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    for name in ("CarLongPlateGen1_jpg.rf.aaaa", "CarLongPlate1_jpg.rf.bbbb"):
        image_path = split / f"images/{name}.jpg"
        cv2.imwrite(str(image_path), np.full((80, 160, 3), 127, np.uint8))
        (split / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    review_csv = build_review_candidates(
        tmp_path / "source", tmp_path / "review", count=10,
        exclude_source_patterns=["*Gen*"],
    )

    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    assert [item["group_id"] for item in rows] == ["det:CarLongPlate1_jpg"]


def test_extract_video_candidates_is_deterministic_and_unverified(tmp_path):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 40))
    for value in range(6):
        writer.write(np.full((40, 80, 3), value * 30, np.uint8))
    writer.release()
    plate = tmp_path / "plate.csv"
    plate.write_text("frame,label,x1,y1,x2,y2,vehicle_id\n0,30M71854,10,5,70,35,car-a\n2,30M71854,10,5,70,35,car-a\n4,51G10096,10,5,70,35,car-b\n", encoding="utf-8")
    manifest = extract_video_candidates(video, "30M71854", tmp_path / "out", every_n=2, annotations=plate)
    loaded = load_plate_manifest(manifest)
    assert len({item.group_id for item in loaded}) == 1
    assert all(item.source_type == "real" and not item.verified for item in loaded)


def test_extract_video_accepts_literal_plate_without_annotations(tmp_path):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 40))
    for value in range(5):
        frame = np.full((40, 80, 3), value * 30, np.uint8)
        frame[5:20, 10:30] = (10, 120, 240)
        writer.write(frame)
    writer.release()

    class Detector:
        def detect(self, _frame):
            return [
                {"bbox": (40, 10, 70, 30), "confidence": 0.4},
                {"bbox": (20, 5, 40, 20), "confidence": 0.9},
                {"bbox": (10, 5, 30, 20), "confidence": 0.9},
            ]

    manifest = extract_video_candidates(
        video, "30M71854", tmp_path / "out", every_n=2, detector=Detector()
    )
    loaded = load_plate_manifest(manifest)
    assert len(loaded) == 3
    assert {item.label for item in loaded} == {"30M71854"}
    assert len({item.group_id for item in loaded}) == 1
    crop = cv2.imread(str(loaded[0].image_path))
    assert crop.shape[:2] == (15, 20)
    capture = cv2.VideoCapture(str(video))
    ok, decoded_frame = capture.read()
    capture.release()
    assert ok
    expected_pixels = decoded_frame[5:20, 10:30]
    assert float(np.mean(np.abs(crop.astype(np.int16) - expected_pixels.astype(np.int16)))) < 8.0


def test_extract_video_skips_frames_without_plate_detections(tmp_path):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 40))
    writer.write(np.zeros((40, 80, 3), np.uint8))
    writer.release()

    class Detector:
        def detect(self, _frame):
            return []

    manifest = extract_video_candidates(video, "30M71854", tmp_path / "out", detector=Detector())
    assert load_plate_manifest(manifest) == []


def test_extract_video_hashes_source_once(tmp_path, monkeypatch):
    import scripts.build_plate_ocr_dataset as builder
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 40))
    for _ in range(3): writer.write(np.zeros((40, 80, 3), np.uint8))
    writer.release()
    annotations = tmp_path / "boxes.csv"
    annotations.write_text("frame,x1,y1,x2,y2,vehicle_id\n0,1,1,20,20,v\n1,1,1,20,20,v\n2,1,1,20,20,v\n")
    original = builder._sha256; calls = 0
    def spy(path):
        nonlocal calls
        if Path(path).resolve() == video.resolve(): calls += 1
        return original(path)
    monkeypatch.setattr(builder, "_sha256", spy)
    extract_video_candidates(video, "30M71854", tmp_path / "out", every_n=1, annotations=annotations)
    assert calls == 1


def test_review_sheet_emits_csv_and_contact_sheet(tmp_path):
    manifest = write_manifest(tmp_path, [row("a.png"), row("b.png", label="51G10096", group_id="v2")])
    review_csv = build_review_sheet(manifest, tmp_path / "review", count=2)
    assert review_csv.is_file()
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    assert {"review_label", "review_verified", "review_notes"}.issubset(rows[0])
    assert (tmp_path / "review" / "contact_sheet_001.jpg").is_file()


def test_committed_frozen_regression_manifest_is_separate_and_loadable():
    manifest = Path(__file__).parents[1] / "data/plate_ocr/frozen_regression.csv"
    loaded = load_plate_manifest(manifest)
    assert len(loaded) == 16
    assert len({item.label for item in loaded}) == 16
    assert all(item.split == "test" and item.counts_toward_real_accuracy for item in loaded)


def test_validate_expanded_gate_counts_test_only_and_exits_nonzero(tmp_path, capsys):
    manifest = write_manifest(tmp_path, [row("val.png", split="val"), row("test.png", label="51G10096", group_id="v2", split="test")])
    summary = validate_manifest(manifest)
    assert summary["verified_real_test_rows"] == 1
    assert dataset_builder_main(["validate", "--manifest", str(manifest)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["expanded_real_gate_passed"] is False


def test_validate_frozen_regression_mode_exits_zero(capsys):
    manifest = Path(__file__).parents[1] / "data/plate_ocr/frozen_regression.csv"
    assert dataset_builder_main(["validate", "--manifest", str(manifest), "--frozen-regression"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["validation_mode"] == "frozen-regression"
    assert output["verified_real_test_rows"] == 16


def test_frozen_regression_flag_rejects_copied_manifest(tmp_path):
    canonical = Path(__file__).parents[1] / "data/plate_ocr/frozen_regression.csv"
    copied = tmp_path / "frozen.csv"; copied.write_bytes(canonical.read_bytes())
    with pytest.raises(ValueError, match="canonical"):
        dataset_builder_main(["validate", "--manifest", str(copied), "--frozen-regression"])


def test_frozen_regression_validates_referenced_image_hashes(monkeypatch):
    import scripts.build_plate_ocr_dataset as builder
    original = builder._sha256
    monkeypatch.setattr(builder, "_sha256", lambda path: "0" * 64 if Path(path).name == "00.png" else original(path))
    canonical = Path(__file__).parents[1] / "data/plate_ocr/frozen_regression.csv"
    with pytest.raises(ValueError, match="image hash"):
        builder.validate_canonical_frozen_regression(canonical)


def test_committed_expanded_real_manifest_passes_gate_and_provenance_hashes():
    data_dir = Path(__file__).parents[1] / "data/plate_ocr"
    manifest = data_dir / "expanded_real_test.csv"
    loaded = load_plate_manifest(manifest)
    frozen_labels = {item.label for item in load_plate_manifest(data_dir / "frozen_regression.csv")}
    assert len(loaded) == 102
    assert len({item.group_id for item in loaded}) == 60
    assert len({item.label for item in loaded}) == 60
    assert not ({item.label for item in loaded} & frozen_labels)
    for item in loaded:
        assert item.counts_toward_real_accuracy
        assert item.group_id == f"vehicle:{hashlib.sha256(item.label.encode('ascii')).hexdigest()[:16]}"
        assert item.metadata["reviewer"] == "visual-manual"
        assert "Gen" not in item.metadata["source_ref"]
        assert hashlib.sha256(item.image_path.read_bytes()).hexdigest() == item.metadata["crop_sha256"]
    provenance = audit_manifest_provenance(manifest)
    assert provenance["crop_hashes_verified"] == 102
    assert provenance["sources_verified"] + provenance["sources_unavailable"] == 102
    assert provenance["sources_reconstructed"] == provenance["sources_verified"]
    print(
        "external provenance sources unavailable: "
        f"{provenance['sources_unavailable']}/102"
    )


def test_committed_real_validation_is_integral_and_disjoint_from_both_test_sets():
    data_dir = Path(__file__).parents[1] / "data/plate_ocr"
    manifest = data_dir / "real_validation.csv"
    validation = load_plate_manifest(manifest)
    reserved = [
        item
        for name in ("expanded_real_test.csv", "frozen_regression.csv")
        for item in load_plate_manifest(data_dir / name)
    ]

    assert len(validation) == 64
    assert len({item.label for item in validation}) == 64
    assert len({item.group_id for item in validation}) == 64
    assert len({item.metadata["source_sha256"] for item in validation}) == 64
    assert len({item.metadata["crop_sha256"] for item in validation}) == 64
    assert all(
        item.split == "val"
        and item.source_type == "real"
        and item.verified
        and item.metadata["reviewer"] == "visual-manual"
        and "Gen" not in item.metadata["source_ref"]
        and item.group_id
        == f"vehicle:{hashlib.sha256(item.label.encode('ascii')).hexdigest()[:16]}"
        for item in validation
    )
    for item in validation:
        assert item.metadata["source_ref"]
        assert json.loads(item.metadata["box_json"])
        assert "1000 reserved-disjoint non-Gen candidates" in item.metadata["notes"]
        assert hashlib.sha256(item.image_path.read_bytes()).hexdigest() == item.metadata["crop_sha256"]

    for identity in (
        lambda item: item.label,
        lambda item: item.group_id,
        lambda item: item.metadata.get("source_sha256", ""),
        lambda item: item.metadata.get("crop_sha256", ""),
    ):
        assert not ({identity(item) for item in validation} & {identity(item) for item in reserved})

    provenance = audit_manifest_provenance(manifest)
    assert provenance["rows"] == 64
    assert provenance["crop_hashes_verified"] == 64
    assert provenance["sources_verified"] + provenance["sources_unavailable"] == 64
    assert provenance["sources_reconstructed"] == provenance["sources_verified"]

    corrected = next(item for item in validation if item.image_path.name == "candidate-0430.jpg")
    assert corrected.label == "51G42861"
    assert corrected.group_id == "vehicle:0001c7ed5daa8616"
    with (data_dir / "real_validation_review.csv").open(newline="", encoding="utf-8") as handle:
        review = next(
            item for item in csv.DictReader(handle)
            if item["candidate_id"] == "candidate-0430"
        )
    assert review["label"] == "51G42861"


def test_real_validation_is_reserved_when_composing_training_data(tmp_path):
    data_dir = Path(__file__).parents[1] / "data/plate_ocr"
    validation = load_plate_manifest(data_dir / "real_validation.csv")[0]
    train = write_manifest(tmp_path, [{
        **row(
            "train.jpg", validation.label, source_type="pseudo",
            group_id="pseudo-source", split="train", verified=False,
        ),
        "source_sha256": "f" * 64,
        "crop_sha256": "e" * 64,
    }])

    with pytest.raises(ValueError, match="train row matches reserved plate label"):
        compose_plate_manifests(
            [train], split="train",
            reserved_manifests=[
                data_dir / "real_validation.csv",
                data_dir / "expanded_real_test.csv",
                data_dir / "frozen_regression.csv",
            ],
        )


def test_provenance_audit_reports_missing_external_sources_without_failing(tmp_path):
    crop = tmp_path / "crop.jpg"
    cv2.imwrite(str(crop), np.full((10, 20, 3), 100, np.uint8))
    crop_hash = hashlib.sha256(crop.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    fields = list(CORE_FIELDS) + ["source_ref", "source_sha256", "crop_sha256", "box_json"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            **row("crop.jpg", group_id="vehicle:abc", split="test"),
            "source_ref": "missing/source.jpg", "source_sha256": "0" * 64,
            "crop_sha256": crop_hash, "box_json": "[0,0,20,10]",
        })
    report = audit_manifest_provenance(manifest)
    assert report["crop_hashes_verified"] == 1
    assert report["sources_verified"] == 0
    assert report["sources_unavailable"] == 1
