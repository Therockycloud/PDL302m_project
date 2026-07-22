"""Per-stage latency measurement for the real runtime pipeline.

Measures wall-clock time for each stage of the deployed vehicle-verification
pipeline (vehicle detection -> plate detection -> OCR -> colour
classification -> database matching) on a single test image, using the SAME
components built by ``src.engine.pipeline_factory.build_pipeline`` that the
FastAPI ``/verify`` endpoint and the Streamlit dashboard use. This is a
measurement script only — it does not modify any runtime code, model
weights, or the database.

Protocol: 1 warmup pass (discarded, absorbs PaddleOCR/model cold-start),
then >=5 timed passes; reports the MEDIAN per-stage time in milliseconds.

Usage (run from the ``main/`` directory so the config's relative paths
resolve, matching the convention of ``main/src/api/app.py``):

    cd main && KMP_DUPLICATE_LIB_OK=TRUE \\
        /opt/homebrew/Caskroom/miniforge/base/bin/python \\
        scripts/measure_stage_latency.py

Writes: nothing (prints a markdown table to stdout). This is a diagnostic
script, not a report generator — the results are meant to be pasted by hand
into Report_4_Final_Report.md §5.3, same spirit as
``main/scripts/eval_color_deployed.py``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
import yaml  # noqa: E402

# Repo root = two levels up from this file (main/scripts/measure_stage_latency.py).
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_MAIN_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _MAIN_DIR.parent
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine.pipeline_factory import build_pipeline  # noqa: E402

_CONFIG_PATH = _MAIN_DIR / "configs" / "config.yaml"
_DEFAULT_IMAGE = _MAIN_DIR / "data" / "test" / "test_authorized.jpg"

STAGES = [
    "vehicle_detection",
    "plate_detection",
    "ocr_read",
    "color_classification",
    "db_match",
]


def _load_config(config_path: pathlib.Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_one_pass(image, pipeline: dict) -> dict[str, float]:
    """Run the pipeline stage-by-stage (mirroring
    ``pipeline_factory.infer_single_image``) but with a separate timer per
    stage. Returns a dict of stage_name -> elapsed_ms."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    dets = pipeline["vehicle_detector"].detect(image)
    timings["vehicle_detection"] = (time.perf_counter() - t0) * 1000.0
    if not dets:
        vehicle_crop = image
    else:
        chosen = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        vehicle_crop = chosen["crop"]

    plate_reader = pipeline["plate_reader"]

    t0 = time.perf_counter()
    plate_dets = plate_reader.plate_detector.detect(vehicle_crop)
    timings["plate_detection"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if plate_dets:
        best = max(plate_dets, key=lambda d: d["conf"])
        plate_text = plate_reader.ocr_reader.read_plate(best["crop"])
    else:
        plate_text = ""
    timings["ocr_read"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    color_clf = pipeline.get("color_clf")
    if color_clf is not None:
        color, color_conf = color_clf.predict(vehicle_crop)
    else:
        color, color_conf = "UNKNOWN", 0.0
    timings["color_classification"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if plate_text.strip():
        pipeline["matcher"].verify_vehicle(plate_text, color, color_conf)
    timings["db_match"] = (time.perf_counter() - t0) * 1000.0

    timings["total"] = sum(timings[s] for s in STAGES)
    timings["_plate_text"] = plate_text
    timings["_color"] = f"{color} ({color_conf:.2f})"
    return timings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default=str(_DEFAULT_IMAGE), help="Path to the test image.")
    p.add_argument("--passes", type=int, default=5, help="Number of TIMED passes (after 1 warmup pass). Minimum 5.")
    args = p.parse_args()

    n_passes = max(5, args.passes)

    image_path = pathlib.Path(args.image)
    if not image_path.is_file():
        print(f"[FATAL] Test image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[FATAL] Could not read image (cv2.imread returned None): {image_path}")
        sys.exit(1)

    print(f"[config] Loading: {_CONFIG_PATH}")
    cfg = _load_config(_CONFIG_PATH)

    print("[pipeline] Building real runtime pipeline via build_pipeline(cfg) ...")
    pipeline = build_pipeline(cfg)

    print(f"[warmup] Running 1 warmup pass on {image_path.name} (discarded) ...")
    warmup_result = _run_one_pass(image, pipeline)
    print(
        f"[warmup] plate={warmup_result['_plate_text']!r} color={warmup_result['_color']} "
        f"total={warmup_result['total']:.1f} ms"
    )

    print(f"[measure] Running {n_passes} timed passes on {image_path.name} ...")
    all_timings: list[dict[str, float]] = []
    for i in range(n_passes):
        result = _run_one_pass(image, pipeline)
        all_timings.append(result)
        print(
            f"  pass {i + 1}/{n_passes}: total={result['total']:.1f} ms "
            f"(plate={result['_plate_text']!r}, color={result['_color']})"
        )

    print("\n[result] Median per-stage latency (ms), n=%d timed passes, after 1 warmup pass:\n" % n_passes)
    header = "| Tầng (stage) | Median (ms) | Min (ms) | Max (ms) |"
    sep = "| :--- | ---: | ---: | ---: |"
    print(header)
    print(sep)

    stage_labels = {
        "vehicle_detection": "1. Phát hiện xe (vehicle detection)",
        "plate_detection": "2. Phát hiện biển số trên crop xe (plate detection)",
        "ocr_read": "3. Đọc biển số (PaddleOCR)",
        "color_classification": "4. Phân loại màu xe (TorchColorClassifier)",
        "db_match": "5. Đối chiếu CSDL (DatabaseMatcher.verify_vehicle)",
    }
    medians: dict[str, float] = {}
    for stage in STAGES:
        values = [t[stage] for t in all_timings]
        med = statistics.median(values)
        medians[stage] = med
        print(f"| {stage_labels[stage]} | {med:.1f} | {min(values):.1f} | {max(values):.1f} |")

    total_values = [t["total"] for t in all_timings]
    total_median = statistics.median(total_values)
    print(f"| **Tổng (sum of stages, tham khảo)** | **{total_median:.1f}** | {min(total_values):.1f} | {max(total_values):.1f} |")

    print(f"\n[result] Median TOTAL (sum of per-stage medians may differ slightly from median-of-totals): {total_median:.1f} ms = {total_median / 1000.0:.3f} s")
    print(f"[result] Sum of per-stage medians: {sum(medians.values()):.1f} ms")
    print("\n[done] measure_stage_latency.py complete (no files written, no runtime code modified).")


if __name__ == "__main__":
    main()
