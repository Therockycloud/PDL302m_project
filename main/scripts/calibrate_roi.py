"""Visual ROI / trigger calibration helper.

Overlays the configured parking ROI and the live vehicle detections on a
frame so the ``pipeline.trigger`` parameters can be tuned to a real camera
angle (e.g. a corner-mounted camera looking out at a reverse-parking car)
WITHOUT touching code. For each detected vehicle it prints the normalized
centre, area ratio, and whether it would currently pass the size + ROI gate.

Usage (run from ``main/``)::

    python scripts/calibrate_roi.py --source data/test/sample_parking.mp4 --frame-frac 0.6
    python scripts/calibrate_roi.py --source some_photo.jpg
    # try an ROI without editing config.yaml first:
    python scripts/calibrate_roi.py --source clip.mp4 --roi 0.15 0.35 0.85 1.0

Writes an annotated image to ``--out`` (default ``/tmp/roi_calibration.jpg``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAIN))  # make ``src`` importable when run from anywhere
_CONFIG = _MAIN / "configs" / "config.yaml"
_DEFAULT_ROI = (0.2, 0.4, 0.8, 1.0)  # must match ParkingTrigger._DEFAULT_ROI


def _load_cfg() -> dict:
    with open(_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _read_frame(source: str, frac: float) -> np.ndarray:
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"could not read image: {source}")
        return img
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {source}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame from video")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video or image path")
    ap.add_argument("--frame-frac", type=float, default=0.6, help="position in video [0..1]")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="override ROI (normalized); else use config / default")
    ap.add_argument("--out", default="/tmp/roi_calibration.jpg")
    args = ap.parse_args()

    cfg = _load_cfg()
    pcfg = cfg.get("pipeline", {})
    tcfg = pcfg.get("trigger", {})
    min_area_ratio = float(tcfg.get("min_area_ratio", 0.15))
    roi = tuple(args.roi) if args.roi else (tuple(tcfg["roi"]) if tcfg.get("roi") else _DEFAULT_ROI)

    frame = _read_frame(args.source, args.frame_frac)
    h, w = frame.shape[:2]

    from src.models.vehicle_detector import VehicleDetector
    det_model = str(_MAIN / cfg["paths"]["model_save_dir"] / cfg["detector"]["model_name"])
    detector = VehicleDetector(model_path=det_model, conf=cfg["detector"].get("conf_threshold", 0.3))
    dets = detector.detect(frame)

    # draw ROI rectangle
    x0, y0, x1, y1 = roi
    p0 = (int(x0 * w), int(y0 * h))
    p1 = (int(x1 * w), int(y1 * h))
    cv2.rectangle(frame, p0, p1, (255, 0, 0), 2)
    cv2.putText(frame, f"ROI {roi}  min_area={min_area_ratio}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    print(f"frame {w}x{h}  ROI={roi}  min_area_ratio={min_area_ratio}  vehicles={len(dets)}")
    for i, d in enumerate(dets):
        bx1, by1, bx2, by2 = d["bbox"]
        area_ratio = max(0, bx2 - bx1) * max(0, by2 - by1) / float(w * h)
        cx = (bx1 + bx2) / 2.0 / w
        cy = (by1 + by2) / 2.0 / h
        in_roi = x0 <= cx <= x1 and y0 <= cy <= y1
        big = area_ratio >= min_area_ratio
        gate = in_roi and big
        color = (0, 200, 0) if gate else (0, 165, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, f"a={area_ratio:.2f} {'GATE' if gate else ''}", (bx1, max(0, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  veh{i}: center=({cx:.2f},{cy:.2f}) area_ratio={area_ratio:.3f} "
              f"in_roi={in_roi} big_enough={big} -> would_gate={gate}")

    cv2.imwrite(args.out, frame)
    print(f"\nannotated frame -> {args.out}")
    print("Tune pipeline.trigger.roi / min_area_ratio in config.yaml until the parked "
          "vehicle shows GATE (green).")


if __name__ == "__main__":
    main()
