#!/usr/bin/env python3
"""Build and validate reproducible Vietnamese plate OCR manifests."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Iterable

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from src.datasets.plate_ocr_dataset import (  # noqa: E402
    CORE_FIELDS,
    load_plate_manifest,
    plate_identity_group,
)
from src.models.vn_plate_text import normalize_plate_text, validate_vietnamese_plate  # noqa: E402


METADATA_FIELDS = (
    "seed",
    "parameters_json",
    "font_path",
    "font_sha256",
    "source_ref",
    "source_sha256",
    "crop_sha256",
    "ocr_confidence",
    "renderer_schema",
    "pillow_version",
    "opencv_version",
    "numpy_version",
)
MANIFEST_FIELDS = CORE_FIELDS + METADATA_FIELDS
_CANONICAL_FROZEN = (_MAIN / "data/plate_ocr/frozen_regression.csv").resolve()
_CANONICAL_FROZEN_SHA256 = "21bf99be329c836abf4d2fd0e9fad16e5945fc826176621f7e657e4438a0f9a9"


def parse_pixel_box(values, *, image_width: int, image_height: int, context: str) -> tuple[int, int, int, int]:
    """Clamp a finite pixel box; malformed/empty boxes raise with context."""

    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid box values") from exc
    if len(numbers) != 4 or not all(np.isfinite(numbers)):
        raise ValueError(f"{context}: invalid box values")
    x1, y1, x2, y2 = numbers
    x1, x2 = max(0, min(image_width, round(x1))), max(0, min(image_width, round(x2)))
    y1, y2 = max(0, min(image_height, round(y1))), max(0, min(image_height, round(y2)))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{context}: box is empty after clamping")
    return x1, y1, x2, y2


def parse_yolo_box(values, *, image_width: int, image_height: int, context: str) -> tuple[int, int, int, int]:
    """Validate normalized YOLO ``cx,cy,w,h`` and return a safe pixel box."""

    try:
        cx, cy, width, height = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid YOLO box") from exc
    if not all(np.isfinite((cx, cy, width, height))) or width <= 0 or height <= 0:
        raise ValueError(f"{context}: invalid YOLO box")
    return parse_pixel_box(
        ((cx - width / 2) * image_width, (cy - height / 2) * image_height,
         (cx + width / 2) * image_width, (cy + height / 2) * image_height),
        image_width=image_width, image_height=image_height, context=context,
    )


def _safe_write(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, parameters or []):
        raise RuntimeError(f"could not write image: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            writer.writerow({field: source.get(field, "") for field in MANIFEST_FIELDS})
    return path


def _valid_random_plate(rng: random.Random) -> str:
    province = rng.randint(10, 99)
    letters = rng.choice("ABCDEFGHKLMNPRSTUVXYZ")
    if rng.random() < 0.15:
        letters += rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(str(rng.randrange(10)) for _ in range(rng.choice((4, 5, 5))))
    return f"{province:02d}{letters}{digits}"


def _apply_degradation(image: np.ndarray, parameters: dict[str, object]) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = source + np.array(parameters["perspective_offsets"], np.float32)
    image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (width, height), borderValue=(30, 30, 30))
    blur_kind, blur_size = str(parameters["blur"]), int(parameters["blur_size"])
    if blur_kind == "motion":
        kernel = np.zeros((blur_size, blur_size), np.float32)
        kernel[blur_size // 2, :] = 1.0 / blur_size
        image = cv2.filter2D(image, -1, kernel)
    else:
        image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    brightness = float(parameters["brightness"])
    image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    glare_center = tuple(int(value) for value in parameters["glare_center"])
    glare_radius = int(parameters["glare_radius"])
    overlay = image.copy()
    cv2.circle(overlay, glare_center, glare_radius, (255, 255, 255), -1)
    glare_opacity = float(parameters["glare_opacity"])
    image = cv2.addWeighted(overlay, glare_opacity, image, 0.8, 0)
    noise_sigma, noise_seed = float(parameters["noise_sigma"]), int(parameters["noise_seed"])
    noise_rng = np.random.default_rng(noise_seed)
    image = np.clip(image.astype(np.float32) + noise_rng.normal(0, noise_sigma, image.shape), 0, 255).astype(np.uint8)
    downscale = float(parameters["downscale"])
    small = cv2.resize(image, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    occ_width = int(parameters["occlusion_width"])
    occ_x, occ_y = int(parameters["occlusion_x"]), int(parameters["occlusion_y"])
    occ_height, occ_color = int(parameters["occlusion_height"]), int(parameters["occlusion_color"])
    if occ_width:
        cv2.rectangle(image, (occ_x, occ_y), (occ_x + occ_width, min(height - 1, occ_y + occ_height)), (occ_color,) * 3, -1)
    quality = int(parameters["jpeg_quality"])
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not JPEG-encode synthetic plate")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _degrade(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    jitter = max(1, round(min(width, height) * rng.uniform(0.01, 0.05)))
    parameters = {
        "perspective_jitter_px": jitter,
        "perspective_offsets": [[rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)] for _ in range(4)],
        "blur": rng.choice(("motion", "defocus")),
        "blur_size": rng.choice((3, 5)),
        "brightness": rng.uniform(0.35, 0.95),
        "glare_center": [rng.randrange(width), rng.randrange(height)],
        "glare_radius": rng.randint(max(2, height // 12), max(3, height // 3)),
        "glare_opacity": rng.uniform(0.08, 0.28),
        "noise_sigma": rng.uniform(2.0, 14.0),
        "noise_seed": rng.randrange(2**32),
        "downscale": rng.uniform(0.45, 0.9),
    }
    parameters["occlusion_width"] = rng.randint(0, max(1, width // 12))
    if parameters["occlusion_width"]:
        parameters.update({
            "occlusion_x": rng.randrange(max(1, width - int(parameters["occlusion_width"]))),
            "occlusion_y": rng.randrange(height),
            "occlusion_height": rng.randint(1, max(2, height // 7)),
            "occlusion_color": rng.randrange(80),
        })
    else:
        parameters.update({"occlusion_x": 0, "occlusion_y": 0, "occlusion_height": 0, "occlusion_color": 0})
    parameters["jpeg_quality"] = rng.randint(35, 90)
    return _apply_degradation(image, parameters), parameters


def _render_plate(label: str, layout: str, font_path: Path) -> np.ndarray:
    two_line = layout == "two-line"
    width, height = ((192, 96) if two_line else (256, 72))
    canvas = Image.new("RGB", (width, height), (235, 235, 220))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=4, outline=(20, 20, 20), width=3)
    if two_line:
        font = ImageFont.truetype(str(font_path), 34)
        for line_index, text in enumerate((label[: len(label) // 2], label[len(label) // 2 :])):
            box = draw.textbbox((0, 0), text, font=font)
            draw.text(((width - (box[2] - box[0])) // 2, 4 + line_index * 42), text, font=font, fill=(10, 10, 10))
    else:
        font = ImageFont.truetype(str(font_path), 42)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(((width - (box[2] - box[0])) // 2, 9), label, font=font, fill=(10, 10, 10))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def generate_synthetic(
    output_dir: str | Path,
    *,
    count: int,
    seed: int = 42,
    font_path: str | Path,
) -> Path:
    """Render deterministic one/two-line plates and record all random inputs."""

    font_path = Path(font_path).expanduser().resolve()
    if not font_path.is_file():
        raise FileNotFoundError(f"font file does not exist: {font_path}")
    if count < 0:
        raise ValueError("count must be non-negative")
    output = Path(output_dir).expanduser().resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    font_hash = _sha256(font_path)
    asset = output / "assets" / f"font-{font_hash}{font_path.suffix.lower()}"
    if asset.exists():
        raise FileExistsError(f"refusing to overwrite existing font asset: {asset}")
    asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(font_path, asset)
    rows: list[dict[str, object]] = []
    for index in range(count):
        label = _valid_random_plate(rng)
        two_line = rng.random() < 0.45
        layout = "two-line" if two_line else "one-line"
        degraded, parameters = _degrade(_render_plate(label, layout, asset), rng)
        name = f"synthetic_{seed}_{index:06d}.jpg"
        target = images / name
        _safe_write(target, degraded)
        image_hash = _sha256(target)
        rows.append({
            "image_path": target.relative_to(output).as_posix(), "label": label,
            "source_type": "synthetic", "group_id": plate_identity_group(label),
            "split": "train", "verified": "false", "seed": seed,
            "parameters_json": json.dumps({"layout": layout, **parameters}, sort_keys=True, separators=(",", ":")),
            "font_path": asset.relative_to(output).as_posix(), "font_sha256": font_hash,
            "crop_sha256": image_hash,
            "renderer_schema": "plate-synthetic-v1", "pillow_version": PIL.__version__,
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        })
    return _write_manifest(output / "manifest.csv", rows)


def replay_synthetic_manifest(manifest: str | Path) -> dict[str, int]:
    """Replay recorded synthetic rows without RNG and verify exact JPEG hashes."""

    manifest_path = Path(manifest).expanduser().resolve()
    verified = 0
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["source_type"] != "synthetic":
            continue
        if row["renderer_schema"] != "plate-synthetic-v1":
            raise ValueError(f"unsupported renderer_schema: {row['renderer_schema']}")
        for field, display_name, current_version in (
            ("pillow_version", "Pillow", PIL.__version__),
            ("opencv_version", "OpenCV", cv2.__version__),
            ("numpy_version", "NumPy", np.__version__),
        ):
            if row[field] != current_version:
                raise ValueError(
                    f"synthetic renderer {display_name} version drift: "
                    f"recorded {row[field]}, current {current_version}"
                )
        font = (manifest_path.parent / row["font_path"]).resolve()
        if _sha256(font) != row["font_sha256"]:
            raise ValueError("synthetic font hash mismatch")
        parameters = json.loads(row["parameters_json"])
        layout = parameters.pop("layout")
        image = _apply_degradation(_render_plate(row["label"], layout, font), parameters)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok or hashlib.sha256(encoded.tobytes()).hexdigest() != row["crop_sha256"]:
            raise ValueError(f"synthetic replay hash mismatch: {row['image_path']}")
        verified += 1
    return {"verified": verified}


def _iter_yolo_pairs(source: Path):
    for label_path in sorted(source.glob("**/labels/*.txt")):
        image_dir = label_path.parent.parent / "images"
        matches = [image_dir / f"{label_path.stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
        image_path = next((path for path in matches if path.is_file()), None)
        if image_path is not None:
            yield image_path, label_path


def pseudo_label_yolo(source: str | Path, output_dir: str | Path, min_conf: float, reader=None) -> Path:
    """Crop YOLO boxes and retain high-confidence PaddleOCR candidates."""

    if reader is None:
        from src.models.ppocr_reader import PaddleOCRReader
        reader = PaddleOCRReader(lang="en")
    source = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for image_path, yolo_path in _iter_yolo_pairs(source):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        for box_index, line in enumerate(yolo_path.read_text(encoding="utf-8").splitlines()):
            values = line.split()
            if len(values) < 5:
                continue
            cx, cy, bw, bh = map(float, values[1:5])
            x1, y1, x2, y2 = parse_yolo_box(
                (cx, cy, bw, bh), image_width=width, image_height=height,
                context=f"{yolo_path}:{box_index + 1}",
            )
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            reading = reader.read_plate(crop)
            label = normalize_plate_text(reading.get("text", ""))
            confidence = float(reading.get("ocr_conf", 0.0))
            if confidence < min_conf or not validate_vietnamese_plate(label):
                continue
            source_hash = _sha256(image_path)
            relative_id = hashlib.sha256(
                image_path.relative_to(source).as_posix().encode("utf-8")
            ).hexdigest()[:12]
            name = f"pseudo_{source_hash[:16]}_{relative_id}_{box_index:02d}.jpg"
            target = images_dir / name
            _safe_write(target, crop)
            source_group = image_path.stem.split(".rf.", 1)[0]
            source_group_hash = hashlib.sha256(source_group.encode("utf-8")).hexdigest()[:16]
            rows.append({
                "image_path": target.relative_to(output).as_posix(), "label": label,
                "source_type": "pseudo", "group_id": f"source:{source_group_hash}",
                "split": "train", "verified": "false", "source_ref": str(image_path),
                "source_sha256": source_hash, "ocr_confidence": f"{confidence:.8f}",
                "parameters_json": json.dumps({"yolo_box": [cx, cy, bw, bh]}, separators=(",", ":")),
            })
    return _write_manifest(output / "manifest.csv", rows)


def build_review_candidates(
    source: str | Path,
    output_dir: str | Path,
    *,
    count: int = 400,
    reserved_manifests: Iterable[str | Path] = (),
    exclude_source_patterns: Iterable[str] = (),
) -> Path:
    """Rank unique YOLO plate crops for label-free human transcription."""

    source = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reserved_hashes = {
        str(row.metadata.get(field, "")).strip().lower()
        for manifest in reserved_manifests
        for row in load_plate_manifest(manifest)
        for field in ("source_sha256", "crop_sha256")
        if str(row.metadata.get(field, "")).strip()
    }
    best_by_group: dict[str, dict[str, object]] = {}
    for image_path, yolo_path in _iter_yolo_pairs(source):
        if any(fnmatch.fnmatchcase(image_path.name, pattern) for pattern in exclude_source_patterns):
            continue
        source_hash = _sha256(image_path)
        if source_hash in reserved_hashes:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        source_group = image_path.stem.split(".rf.", 1)[0]
        for box_index, line in enumerate(yolo_path.read_text(encoding="utf-8").splitlines()):
            values = line.split()
            if len(values) < 5:
                continue
            x1, y1, x2, y2 = parse_yolo_box(
                values[1:5], image_width=width, image_height=height,
                context=f"{yolo_path}:{box_index + 1}",
            )
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_height, crop_width = crop.shape[:2]
            focus = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            adequate = int(crop_width >= 80 and crop_height >= 24)
            rank = (adequate, focus, crop_width * crop_height, str(image_path), -box_index)
            candidate = {
                "rank": rank,
                "crop": crop,
                "group_id": f"det:{source_group}",
                "source_path": image_path,
                "source_sha256": source_hash,
                "box": [x1, y1, x2, y2],
                "box_index": box_index,
                "width": crop_width,
                "height": crop_height,
                "focus": focus,
            }
            previous = best_by_group.get(source_group)
            if previous is None or candidate["rank"] > previous["rank"]:
                best_by_group[source_group] = candidate

    selected: list[dict[str, object]] = []
    for item in sorted(best_by_group.values(), key=lambda item: item["rank"], reverse=True):
        ok, encoded = cv2.imencode(".jpg", item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"could not encode candidate from {item['source_path']}")
        if hashlib.sha256(encoded.tobytes()).hexdigest() in reserved_hashes:
            continue
        selected.append(item)
        if len(selected) == count:
            break
    fields = (
        "candidate_id", "crop_path", "group_id", "source_path", "source_sha256",
        "crop_sha256", "box_json", "crop_width", "crop_height", "focus_score",
        "review_label", "review_status", "review_notes",
    )
    records: list[dict[str, str]] = []
    for index, item in enumerate(selected):
        candidate_id = f"candidate-{index:04d}"
        target = output / f"{candidate_id}.jpg"
        _safe_write(target, item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({
            "candidate_id": candidate_id,
            "crop_path": target.name,
            "group_id": str(item["group_id"]),
            "source_path": str(item["source_path"]),
            "source_sha256": str(item["source_sha256"]),
            "crop_sha256": _sha256(target),
            "box_json": json.dumps(item["box"], separators=(",", ":")),
            "crop_width": str(item["width"]),
            "crop_height": str(item["height"]),
            "focus_score": f"{float(item['focus']):.8f}",
            "review_label": "", "review_status": "", "review_notes": "",
        })
    review_csv = output / "review_candidates.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    cell_width, cell_height, columns, per_sheet = 240, 105, 4, 40
    for sheet_index in range((len(records) + per_sheet - 1) // per_sheet):
        canvas = np.full((10 * cell_height, columns * cell_width, 3), 245, np.uint8)
        for offset, record in enumerate(records[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]):
            crop = cv2.imread(str(output / record["crop_path"]), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            crop = cv2.resize(crop, (220, 72), interpolation=cv2.INTER_AREA)
            x, y = (offset % columns) * cell_width + 10, (offset // columns) * cell_height + 5
            canvas[y:y + 72, x:x + 220] = crop
            cv2.putText(canvas, record["candidate_id"], (x, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        _safe_write(output / f"contact_sheet_{sheet_index + 1:03d}.jpg", canvas)
    return review_csv


def extract_video_candidates(
    video: str | Path,
    plate: str,
    output_dir: str | Path,
    *,
    every_n: int = 6,
    annotations: str | Path | None = None,
    detector=None,
) -> Path:
    """Extract deterministic labelled frames, optionally using box annotations."""

    if every_n <= 0:
        raise ValueError("every_n must be positive")
    video = Path(video).expanduser().resolve()
    label = normalize_plate_text(plate)
    if not validate_vietnamese_plate(label):
        raise ValueError(f"invalid literal plate label: {plate!r}")
    output = Path(output_dir).expanduser().resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_rows: list[dict[str, str]] = []
    if annotations is not None:
        annotation_path = Path(annotations).expanduser().resolve()
        with annotation_path.open(newline="", encoding="utf-8") as handle:
            annotation_rows = [row for row in csv.DictReader(handle) if int(row["frame"]) % every_n == 0]
    elif detector is None:
        from src.models.detector import PlateDetector

        detector = PlateDetector(
            model_path=str(_MAIN / "data/models/plate_yolov8n.onnx"),
            conf_threshold=0.25,
        )
    by_frame: dict[int, list[dict[str, str]]] = {}
    for annotation in annotation_rows:
        by_frame.setdefault(int(annotation["frame"]), []).append(annotation)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    video_hash = _sha256(video)
    rows: list[dict[str, object]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_annotations = by_frame.get(frame_index, [])
        if annotations is None and frame_index % every_n == 0:
            detections = detector.detect(frame)
            if detections:
                def detection_key(item):
                    confidence = float(item.get("confidence", item.get("conf", 0.0)))
                    bbox = tuple(int(value) for value in item["bbox"])
                    return (-confidence, bbox)

                best = sorted(detections, key=detection_key)[0]
                x1, y1, x2, y2 = (int(value) for value in best["bbox"])
                frame_annotations = [{
                    "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
                    "vehicle_id": label,
                    "detection_confidence": str(best.get("confidence", best.get("conf", 0.0))),
                }]
        for box_index, annotation in enumerate(frame_annotations):
            x1, y1, x2, y2 = parse_pixel_box(
                (annotation[key] for key in ("x1", "y1", "x2", "y2")),
                image_width=frame.shape[1], image_height=frame.shape[0],
                context=f"{video}:frame={frame_index}:box={box_index}",
            )
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            target = images_dir / f"video_{video_hash[:16]}_f{frame_index:08d}_{box_index:02d}.jpg"
            _safe_write(target, crop)
            rows.append({
                "image_path": target.relative_to(output).as_posix(), "label": label,
                "source_type": "real", "group_id": plate_identity_group(label),
                "split": "train", "verified": "false", "source_ref": f"{video}#frame={frame_index}",
                "source_sha256": video_hash,
                "parameters_json": json.dumps({"box": [x1, y1, x2, y2], "every_n": every_n}, separators=(",", ":")),
            })
        frame_index += 1
    capture.release()
    return _write_manifest(output / "manifest.csv", rows)


def audit_manifest_provenance(manifest: str | Path) -> dict[str, int]:
    """Verify committed crops and, when present, their ignored source corpus."""

    manifest_path = Path(manifest).expanduser().resolve()
    rows = load_plate_manifest(manifest_path)
    report = {
        "rows": len(rows),
        "crop_hashes_verified": 0,
        "sources_verified": 0,
        "sources_reconstructed": 0,
        "sources_unavailable": 0,
    }
    for row in rows:
        expected_crop_hash = row.metadata.get("crop_sha256", "")
        if not expected_crop_hash:
            raise ValueError(f"missing crop_sha256 for {row.image_path}")
        actual_crop_hash = _sha256(row.image_path)
        if actual_crop_hash != expected_crop_hash:
            raise ValueError(f"crop hash mismatch for {row.image_path}")
        report["crop_hashes_verified"] += 1

        source_ref = row.metadata.get("source_ref", "")
        source = Path(source_ref).expanduser()
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        if not source.is_file():
            report["sources_unavailable"] += 1
            continue
        expected_source_hash = row.metadata.get("source_sha256", "")
        if _sha256(source) != expected_source_hash:
            raise ValueError(f"source hash mismatch for {source}")
        report["sources_verified"] += 1

        box = json.loads(row.metadata.get("box_json", ""))
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid box_json for {row.image_path}")
        source_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f"could not decode provenance source {source}")
        x1, y1, x2, y2 = (int(value) for value in box)
        reconstructed = source_image[y1:y2, x1:x2]
        ok, encoded = cv2.imencode(".jpg", reconstructed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok or hashlib.sha256(encoded.tobytes()).hexdigest() != expected_crop_hash:
            raise ValueError(f"crop reconstruction mismatch for {row.image_path}")
        report["sources_reconstructed"] += 1
    return report


def build_review_sheet(manifest: str | Path, output_dir: str | Path, *, count: int = 120) -> Path:
    rows = load_plate_manifest(manifest)[:count]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_path = output / "review.csv"
    fields = list(CORE_FIELDS) + ["review_label", "review_verified", "review_notes"]
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({
                "image_path": str(item.image_path), "label": item.label,
                "source_type": item.source_type, "group_id": item.group_id,
                "split": item.split, "verified": str(item.verified).lower(),
                "review_label": "", "review_verified": "", "review_notes": "",
            })
    cell_width, cell_height, columns, per_sheet = 240, 110, 4, 40
    for sheet_index in range((len(rows) + per_sheet - 1) // per_sheet):
        subset = rows[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]
        canvas = np.full((10 * cell_height, columns * cell_width, 3), 245, np.uint8)
        for index, item in enumerate(subset):
            crop = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            crop = cv2.resize(crop, (220, 72), interpolation=cv2.INTER_AREA)
            x, y = (index % columns) * cell_width + 10, (index // columns) * cell_height + 5
            canvas[y:y + 72, x:x + 220] = crop
            cv2.putText(canvas, f"{sheet_index * per_sheet + index:03d} {item.label}", (x, y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        _safe_write(output / f"contact_sheet_{sheet_index + 1:03d}.jpg", canvas)
    return review_path


def validate_manifest(manifest: str | Path) -> dict[str, object]:
    rows = load_plate_manifest(manifest)
    def counts(attribute: str):
        values: dict[str, int] = {}
        for item in rows:
            key = str(getattr(item, attribute)).lower()
            values[key] = values.get(key, 0) + 1
        return values
    return {
        "rows": len(rows), "unique_labels": len({item.label for item in rows}),
        "unique_group_ids": len({item.group_id for item in rows}),
        "split_counts": counts("split"), "source_counts": counts("source_type"),
        "verified_counts": counts("verified"),
        "verified_real_test_rows": sum(
            item.split == "test" and item.source_type == "real" and item.verified for item in rows
        ),
    }


def validate_canonical_frozen_regression(manifest: str | Path) -> dict[str, object]:
    path = Path(manifest).expanduser().resolve()
    if path != _CANONICAL_FROZEN:
        raise ValueError(f"frozen-regression mode is bound to canonical manifest {_CANONICAL_FROZEN}")
    if _sha256(path) != _CANONICAL_FROZEN_SHA256:
        raise ValueError("canonical frozen regression checksum mismatch")
    summary = validate_manifest(path)
    if summary["rows"] != 16 or summary["unique_labels"] != 16 or summary["verified_real_test_rows"] != 16:
        raise ValueError("canonical frozen regression inventory mismatch")
    for row in load_plate_manifest(path):
        expected_hash = row.metadata.get("source_sha256", "")
        if not expected_hash or _sha256(row.image_path) != expected_hash:
            raise ValueError(f"canonical frozen regression image hash mismatch: {row.image_path}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--count", type=int, required=True)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--font", type=Path, required=True, help="local .ttf/.otf font; no downloads are performed")
    synthetic.add_argument("--output", type=Path, default=_MAIN / "data/plate_ocr/generated/synthetic")
    pseudo = sub.add_parser("pseudo-label")
    pseudo.add_argument("--source", type=Path, default=_MAIN / "data/raw/plate_det")
    pseudo.add_argument("--min-conf", type=float, default=0.95)
    pseudo.add_argument("--output", type=Path, default=_MAIN / "data/plate_ocr/generated/pseudo")
    extract = sub.add_parser("extract-video")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--plate", required=True, help="literal normalized plate label")
    extract.add_argument("--annotations", type=Path, help="optional CSV with frame,x1,y1,x2,y2,vehicle_id")
    extract.add_argument("--every-n", type=int, default=6)
    extract.add_argument("--output", type=Path, default=_MAIN / "data/plate_ocr/generated/video")
    review = sub.add_parser("review-sheet")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--count", type=int, default=120)
    review.add_argument("--output", type=Path, default=_MAIN / "data/plate_ocr/review")
    candidates = sub.add_parser("review-candidates")
    candidates.add_argument("--source", type=Path, default=_MAIN / "data/raw/plate_det")
    candidates.add_argument("--count", type=int, default=400)
    candidates.add_argument("--output", type=Path, default=_MAIN / "data/plate_ocr/review/candidates")
    candidates.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    candidates.add_argument("--exclude-source-pattern", action="append", default=[])
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument(
        "--frozen-regression",
        action="store_true",
        help="strictly validate the canonical 16-image frozen regression set and every stored image hash",
    )
    replay = sub.add_parser("replay-synthetic")
    replay.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "synthetic":
        result = generate_synthetic(args.output, count=args.count, seed=args.seed, font_path=args.font)
    elif args.command == "pseudo-label":
        result = pseudo_label_yolo(args.source, args.output, args.min_conf)
    elif args.command == "extract-video":
        result = extract_video_candidates(args.video, args.plate, args.output, every_n=args.every_n, annotations=args.annotations)
    elif args.command == "review-sheet":
        result = build_review_sheet(args.manifest, args.output, count=args.count)
    elif args.command == "review-candidates":
        result = build_review_candidates(
            args.source, args.output, count=args.count,
            reserved_manifests=args.reserved_manifest,
            exclude_source_patterns=args.exclude_source_pattern,
        )
    elif args.command == "replay-synthetic":
        print(json.dumps(replay_synthetic_manifest(args.manifest), indent=2, sort_keys=True))
        return 0
    else:
        summary = (
            validate_canonical_frozen_regression(args.manifest)
            if args.frozen_regression else validate_manifest(args.manifest)
        )
        summary["validation_mode"] = "frozen-regression" if args.frozen_regression else "expanded-real"
        summary["expanded_real_gate_passed"] = summary["verified_real_test_rows"] >= 100
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if args.frozen_regression or summary["expanded_real_gate_passed"] else 2
    if args.command in {"synthetic", "pseudo-label", "extract-video"}:
        load_plate_manifest(result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
