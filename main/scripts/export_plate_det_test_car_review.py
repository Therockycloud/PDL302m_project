#!/usr/bin/env python3
"""Export reserved-disjoint car plates from plate_det test frames and clip3_new."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

_MAIN = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MAIN.parent
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from scripts.build_plate_ocr_dataset import (  # noqa: E402
    _safe_write,
    _sha256,
    parse_yolo_box,
)
from scripts.export_real_car_train_review import (  # noqa: E402
    CANDIDATE_FIELDS as _BASE_CANDIDATE_FIELDS,
    CarReviewStats,
    _Candidate,
    _is_car_like,
    _prepare_output_dir,
    assert_no_leakage,
    confidence_histogram,
    render_transcription_sheets,
    select_diverse_per_label,
)
from scripts.near_reserved_filter import (  # noqa: E402
    classify_reserved_label_drop,
    is_near_reserved_label,
)
from scripts.pseudo_label_apple_vision import (  # noqa: E402
    UPSCALE_MIN_HEIGHT,
    RecognizeFn,
    _recognize_crop,
    build_reserved_identities,
    create_apple_vision_recognizer,
)

DEFAULT_TEST_DIR = _MAIN / "data/raw/plate_det/test"
DEFAULT_CLIP3_DIR = _MAIN / "data/raw/license_plates"
DEFAULT_DETECTOR_MODEL = _MAIN / "data/models/plate_yolov8n.onnx"
DEFAULT_RESERVED = (
    _MAIN / "data/plate_ocr/real_validation.csv",
    _MAIN / "data/plate_ocr/expanded_real_test.csv",
    _MAIN / "data/plate_ocr/frozen_regression.csv",
)
DEFAULT_OUTPUT = _MAIN / "data/plate_ocr/review/external_car_test_audit"
MAX_CROPS_PER_LABEL = 3
SHEET_LABEL_CAP = 250

CANDIDATE_FIELDS = _BASE_CANDIDATE_FIELDS + ("filter_notes",)

DetectionsByImage = dict[str, list[dict[str, object]]]
DetectFn = Callable[[Sequence[Path]], DetectionsByImage]


@dataclass
class ExternalCarReviewStats(CarReviewStats):
    frames_processed: int = 0
    near_reserved_label: int = 0
    detector_boxes: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["frames_processed"] = self.frames_processed
        payload["dropped"]["near-reserved-label"] = self.near_reserved_label
        payload["detector_boxes"] = self.detector_boxes
        return payload


@dataclass
class _RawBox:
    image_path: Path
    source_root: Path
    box_index: int
    box: list[int]
    crop: np.ndarray
    aspect_ratio: float
    filter_notes: str = ""


def _deterministic_crop_name(
    source_hash: str,
    image_path: Path,
    source_root: Path,
    box_index: int,
) -> str:
    try:
        relative = image_path.relative_to(source_root).as_posix()
    except ValueError:
        relative = image_path.name
    relative_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"car_{source_hash[:16]}_{relative_id}_{box_index:02d}.jpg"


def _iter_test_frames(test_dir: Path) -> Iterable[Path]:
    for image_path in sorted(test_dir.glob("*.jpg")):
        if image_path.is_file():
            yield image_path


def _iter_clip3_pairs(clip_dir: Path) -> Iterable[tuple[Path, Path]]:
    for image_path in sorted(clip_dir.glob("clip3_new_*.jpg")):
        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            yield image_path, label_path


def _host_to_docker_path(path: Path) -> str:
    return "/app/" + path.resolve().relative_to(_PROJECT_ROOT).as_posix()


def _docker_to_host_path(path: str) -> Path:
    if path.startswith("/app/"):
        return (_PROJECT_ROOT / path.removeprefix("/app/")).resolve()
    return Path(path).resolve()


def detect_frames_local(
    image_paths: Sequence[Path],
    *,
    model_path: Path,
    progress_every: int = 50,
) -> DetectionsByImage:
    from src.models.detector import PlateDetector

    detector = PlateDetector(model_path=str(model_path))
    results: DetectionsByImage = {}
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            results[str(image_path.resolve())] = []
            continue
        detections = detector.detect(image)
        results[str(image_path.resolve())] = [
            {
                "bbox": [int(value) for value in det["bbox"]],
                "confidence": float(det["confidence"]),
            }
            for det in detections
        ]
        if index % progress_every == 0:
            print(f"detect progress frames={index}/{len(image_paths)}", flush=True)
    return results


def run_internal_detect_batch(input_json: Path, output_json: Path) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    model_path = _docker_to_host_path(str(payload["model_path"]))
    image_paths = [_docker_to_host_path(str(path)) for path in payload["images"]]
    results = detect_frames_local(image_paths, model_path=model_path)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def detect_frames_docker(
    image_paths: Sequence[Path],
    *,
    model_path: Path,
    progress_every: int = 50,
) -> DetectionsByImage:
    tmp_dir = _MAIN / "data/tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256(str(time.time_ns()).encode("utf-8")).hexdigest()[:12]
    input_json = tmp_dir / f"plate_det_detect_in_{stamp}.json"
    output_json = tmp_dir / f"plate_det_detect_out_{stamp}.json"
    payload = {
        "model_path": _host_to_docker_path(model_path),
        "images": [_host_to_docker_path(path) for path in image_paths],
    }
    input_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-w",
        "/app/main",
        "backend",
        "python",
        "scripts/export_plate_det_test_car_review.py",
        "--internal-detect",
        "--input-json",
        _host_to_docker_path(input_json),
        "--output-json",
        _host_to_docker_path(output_json),
    ]
    subprocess.run(cmd, cwd=_PROJECT_ROOT, check=True)
    raw = json.loads(output_json.read_text(encoding="utf-8"))
    remapped: DetectionsByImage = {}
    for key, value in raw.items():
        remapped[str(_docker_to_host_path(key))] = value
    input_json.unlink(missing_ok=True)
    output_json.unlink(missing_ok=True)
    print(f"detect progress frames={len(image_paths)}/{len(image_paths)}", flush=True)
    return remapped


def _boxes_from_yolo(image_path: Path, label_path: Path) -> list[_RawBox]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    source_root = image_path.parent
    boxes: list[_RawBox] = []
    for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        values = line.split()
        if len(values) < 5:
            continue
        cx, cy, bw, bh = map(float, values[1:5])
        try:
            x1, y1, x2, y2 = parse_yolo_box(
                (cx, cy, bw, bh),
                image_width=width,
                image_height=height,
                context=f"{label_path}:{box_index + 1}",
            )
        except ValueError:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        car_like, aspect_ratio = _is_car_like(crop)
        if not car_like:
            continue
        boxes.append(
            _RawBox(
                image_path=image_path,
                source_root=source_root,
                box_index=box_index,
                box=[x1, y1, x2, y2],
                crop=crop,
                aspect_ratio=aspect_ratio,
                filter_notes="yolo-label",
            )
        )
    return boxes


def _boxes_from_detector(
    image_path: Path,
    detections: Sequence[dict[str, object]],
    *,
    source_root: Path,
) -> list[_RawBox]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    boxes: list[_RawBox] = []
    for box_index, detection in enumerate(detections):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        car_like, aspect_ratio = _is_car_like(crop)
        if not car_like:
            continue
        confidence = float(detection.get("confidence", 0.0))
        boxes.append(
            _RawBox(
                image_path=image_path,
                source_root=source_root,
                box_index=box_index,
                box=[x1, y1, x2, y2],
                crop=crop,
                aspect_ratio=aspect_ratio,
                filter_notes=f"detector:{confidence:.4f}",
            )
        )
    return boxes


def _finalize_candidate(
    raw: _RawBox,
    *,
    label: str,
    ocr_confidence: float,
    source_hash: str,
) -> _Candidate:
    crop_hash = hashlib.sha256(cv2.imencode(".jpg", raw.crop)[1].tobytes()).hexdigest()
    return _Candidate(
        crop=raw.crop,
        label=label,
        ocr_confidence=ocr_confidence,
        source_ref=raw.image_path,
        source_sha256=source_hash,
        crop_sha256=crop_hash,
        box=raw.box,
        aspect_ratio=raw.aspect_ratio,
        crop_name=_deterministic_crop_name(source_hash, raw.image_path, raw.source_root, raw.box_index),
    )


def collect_external_car_candidates(
    *,
    test_dir: Path,
    clip_dir: Path,
    reserved_manifests: Sequence[str | Path],
    recognize: RecognizeFn,
    detect_frames: DetectFn,
    progress_every: int = 200,
) -> tuple[list[_Candidate], dict[str, str], ExternalCarReviewStats]:
    reserved_labels, reserved_source_hashes, reserved_crop_hashes = build_reserved_identities(
        reserved_manifests
    )
    stats = ExternalCarReviewStats()
    raw_by_label: dict[str, list[_Candidate]] = defaultdict(list)
    filter_notes_by_crop: dict[str, str] = {}

    test_frames = list(_iter_test_frames(test_dir))
    stats.frames_processed += len(test_frames)
    detections = detect_frames(test_frames)

    for image_path in test_frames:
        source_hash = _sha256(image_path)
        if source_hash in reserved_source_hashes:
            stats.reserved_source_hash += 1
            continue

        frame_detections = detections.get(str(image_path.resolve()), [])
        stats.detector_boxes += len(frame_detections)
        raw_boxes = _boxes_from_detector(
            image_path,
            frame_detections,
            source_root=test_dir,
        )
        stats.total_boxes += len(frame_detections)
        stats.low_ar += len(frame_detections) - len(raw_boxes)
        stats.car_boxes_seen += len(raw_boxes)
        _process_raw_boxes(
            raw_boxes,
            source_hash=source_hash,
            reserved_labels=reserved_labels,
            reserved_crop_hashes=reserved_crop_hashes,
            recognize=recognize,
            stats=stats,
            raw_by_label=raw_by_label,
            filter_notes_by_crop=filter_notes_by_crop,
            progress_every=progress_every,
        )

    for image_path, label_path in _iter_clip3_pairs(clip_dir):
        stats.frames_processed += 1
        source_hash = _sha256(image_path)
        if source_hash in reserved_source_hashes:
            stats.reserved_source_hash += 1
            continue

        label_lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        valid_box_lines = [line for line in label_lines if len(line.split()) >= 5]
        stats.total_boxes += len(valid_box_lines)
        raw_boxes = _boxes_from_yolo(image_path, label_path)
        stats.low_ar += len(valid_box_lines) - len(raw_boxes)
        stats.car_boxes_seen += len(raw_boxes)
        _process_raw_boxes(
            raw_boxes,
            source_hash=source_hash,
            reserved_labels=reserved_labels,
            reserved_crop_hashes=reserved_crop_hashes,
            recognize=recognize,
            stats=stats,
            raw_by_label=raw_by_label,
            filter_notes_by_crop=filter_notes_by_crop,
            progress_every=progress_every,
        )

    kept: list[_Candidate] = []
    kept_notes: dict[str, str] = {}
    for label, group in raw_by_label.items():
        for candidate in select_diverse_per_label(group, max_per_label=MAX_CROPS_PER_LABEL):
            kept.append(candidate)
            kept_notes[candidate.crop_name] = filter_notes_by_crop.get(candidate.crop_name, "")
    stats.kept = len(kept)
    stats.unique_labels = len({item.label for item in kept})
    return kept, kept_notes, stats


def _process_raw_boxes(
    raw_boxes: Sequence[_RawBox],
    *,
    source_hash: str,
    reserved_labels: set[str],
    reserved_crop_hashes: set[str],
    recognize: RecognizeFn,
    stats: ExternalCarReviewStats,
    raw_by_label: dict[str, list[_Candidate]],
    filter_notes_by_crop: dict[str, str],
    progress_every: int,
) -> None:
    for raw in raw_boxes:
        reading, drop_reason = _recognize_crop(raw.crop, recognize, min_height=UPSCALE_MIN_HEIGHT)
        if reading is None:
            if drop_reason == "unreadable":
                stats.unreadable += 1
            else:
                stats.invalid_format += 1
            continue

        label = reading.label
        reserved_drop = classify_reserved_label_drop(label, reserved_labels)
        if reserved_drop == "reserved-label":
            stats.reserved_label += 1
            continue
        if reserved_drop == "near-reserved-label":
            stats.near_reserved_label += 1
            continue

        crop_hash = hashlib.sha256(cv2.imencode(".jpg", raw.crop)[1].tobytes()).hexdigest()
        if crop_hash in reserved_crop_hashes:
            stats.reserved_crop_hash += 1
            continue

        candidate = _finalize_candidate(
            raw,
            label=label,
            ocr_confidence=reading.confidence,
            source_hash=source_hash,
        )
        filter_notes_by_crop[candidate.crop_name] = raw.filter_notes
        raw_by_label[label].append(candidate)
        stats.kept_before_dedup += 1

        if stats.total_boxes % progress_every == 0:
            print(
                f"progress boxes={stats.total_boxes} car={stats.car_boxes_seen} "
                f"kept_raw={stats.kept_before_dedup}",
                flush=True,
            )


def assert_no_near_reserved_leakage(
    candidates: Sequence[_Candidate],
    reserved_labels: set[str],
) -> int:
    leaked = [item.label for item in candidates if is_near_reserved_label(item.label, reserved_labels)]
    assert not leaked, {"near_reserved_leak_count": len(leaked), "examples": leaked[:5]}
    return 0


def write_candidates_csv(
    output_dir: Path,
    candidates: Sequence[_Candidate],
    *,
    filter_notes_by_crop: dict[str, str] | None = None,
) -> Path:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidates.csv"
    notes = filter_notes_by_crop or {}
    rows: list[dict[str, object]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.ocr_confidence, item.label, item.crop_name),
    ):
        crop_path = crops_dir / candidate.crop_name
        _safe_write(crop_path, candidate.crop)
        rows.append({
            "image_path": crop_path.relative_to(output_dir).as_posix(),
            "label_draft": candidate.label,
            "ocr_confidence": f"{candidate.ocr_confidence:.8f}",
            "source_ref": str(candidate.source_ref),
            "source_sha256": candidate.source_sha256,
            "crop_sha256": candidate.crop_sha256,
            "box_json": json.dumps(candidate.box, separators=(",", ":")),
            "aspect_ratio": f"{candidate.aspect_ratio:.4f}",
            "filter_notes": notes.get(candidate.crop_name, ""),
        })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def run_pipeline(
    *,
    test_dir: str | Path = DEFAULT_TEST_DIR,
    clip_dir: str | Path = DEFAULT_CLIP3_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT,
    reserved_manifests: Sequence[str | Path] | None = None,
    recognize: RecognizeFn | None = None,
    detect_frames: DetectFn | None = None,
    progress_every: int = 200,
    sheet_label_cap: int = SHEET_LABEL_CAP,
) -> dict[str, object]:
    started = time.monotonic()
    reserved = list(reserved_manifests or DEFAULT_RESERVED)
    if recognize is None:
        recognize = create_apple_vision_recognizer()
    if detect_frames is None:
        detect_frames = lambda paths: detect_frames_docker(paths, model_path=DEFAULT_DETECTOR_MODEL)

    output = _prepare_output_dir(Path(output_dir))
    candidates, filter_notes_by_crop, stats = collect_external_car_candidates(
        test_dir=Path(test_dir).resolve(),
        clip_dir=Path(clip_dir).resolve(),
        reserved_manifests=reserved,
        recognize=recognize,
        detect_frames=detect_frames,
        progress_every=progress_every,
    )

    reserved_labels, reserved_source_hashes, reserved_crop_hashes = build_reserved_identities(reserved)
    leakage = assert_no_leakage(
        candidates,
        reserved_labels=reserved_labels,
        reserved_source_hashes=reserved_source_hashes,
        reserved_crop_hashes=reserved_crop_hashes,
    )
    near_leakage = assert_no_near_reserved_leakage(candidates, reserved_labels)

    candidates_csv = write_candidates_csv(
        output,
        candidates,
        filter_notes_by_crop=filter_notes_by_crop,
    )
    sheets = render_transcription_sheets(output, candidates, label_cap=sheet_label_cap)

    stats_payload = stats.to_dict()
    stats_payload["confidence_histogram"] = confidence_histogram(candidates)
    stats_payload["leakage_check"] = leakage
    stats_payload["near_reserved_leakage"] = near_leakage
    stats_payload["sheet_cells"] = sheets["cell_count"]
    stats_payload["drop_histogram"] = {
        key: int(value)
        for key, value in stats_payload["dropped"].items()  # type: ignore[union-attr]
        if value
    }
    stats_path = output / "stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elapsed = time.monotonic() - started
    return {
        "output_dir": str(output),
        "candidates_csv": str(candidates_csv),
        "candidate_rows": len(candidates),
        "unique_labels": stats.unique_labels,
        "frames_processed": stats.frames_processed,
        "stats_path": str(stats_path),
        "sheets": sheets,
        "leakage_check": leakage,
        "elapsed_seconds": round(elapsed, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP3_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--sheet-label-cap", type=int, default=SHEET_LABEL_CAP)
    parser.add_argument(
        "--detector",
        choices=("docker", "local"),
        default="docker",
        help="Run plate_yolov8n.onnx via Docker backend or locally.",
    )
    parser.add_argument("--internal-detect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_detect:
        if args.input_json is None or args.output_json is None:
            raise SystemExit("--internal-detect requires --input-json and --output-json")
        run_internal_detect_batch(args.input_json, args.output_json)
        return 0

    reserved = args.reserved_manifest or list(DEFAULT_RESERVED)
    if args.detector == "local":
        detect_frames: DetectFn = lambda paths: detect_frames_local(
            paths,
            model_path=DEFAULT_DETECTOR_MODEL,
        )
    else:
        detect_frames = lambda paths: detect_frames_docker(paths, model_path=DEFAULT_DETECTOR_MODEL)

    result = run_pipeline(
        test_dir=args.test_dir,
        clip_dir=args.clip_dir,
        output_dir=args.output,
        reserved_manifests=reserved,
        detect_frames=detect_frames,
        progress_every=args.progress_every,
        sheet_label_cap=args.sheet_label_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("leakage_check:", json.dumps(result["leakage_check"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
