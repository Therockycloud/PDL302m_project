"""Train a YOLOv8n license-plate detector.

Two modes for the Benchmark-B comparison:
  * ``finetune`` — transfer-learn from COCO ``yolov8n.pt``
  * ``scratch``  — random init from ``yolov8n.yaml``

Run from ``main/`` so the run artifacts land under ``data/models/plate_runs``.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    ap.add_argument("--data", help="path to data.yaml")
    ap.add_argument("--mode", choices=["finetune", "scratch"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction (smoke runs)")
    ap.add_argument("--project", default="data/models/plate_runs")
    ap.add_argument("--name")
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    # Resume an interrupted run: Ultralytics restores epochs/args from the
    # checkpoint's run directory, so no other flags are needed.
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    if not (args.data and args.mode and args.name):
        ap.error("--data, --mode and --name are required unless --resume is given")

    weights = "yolov8n.pt" if args.mode == "finetune" else "yolov8n.yaml"
    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
