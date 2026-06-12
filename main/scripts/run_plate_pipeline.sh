#!/bin/bash
# End-to-end plate-detector pipeline: finetune -> export ONNX -> scratch -> benchmark.
# Designed to run unattended in the background. Logs every stage with timestamps.
set -u

PY="/opt/homebrew/Caskroom/miniforge/base/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
ROOT="/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
cd "$ROOT/main" || exit 1

DATA="data/raw/plate_det/data.yaml"
PROJECT="$ROOT/main/data/models/plate_runs"
EPOCHS="${1:-80}"
BATCH="${2:-16}"

echo "=================================================================="
echo "[START] $(date)  epochs=$EPOCHS batch=$BATCH"
echo "=================================================================="

echo "### [1/4] FINETUNE  $(date)"
"$PY" scripts/train_plate.py --data "$DATA" --mode finetune \
  --epochs "$EPOCHS" --batch "$BATCH" --device mps \
  --project "$PROJECT" --name plate_finetune

FT="$PROJECT/plate_finetune/weights/best.pt"
if [ -f "$FT" ]; then
  echo "### export finetune -> ONNX  $(date)"
  "$PY" -c "from ultralytics import YOLO; YOLO('$FT').export(format='onnx', imgsz=640)"
  cp "$PROJECT/plate_finetune/weights/best.onnx" "data/models/plate_yolov8n.onnx" \
    && echo "### exported -> data/models/plate_yolov8n.onnx  $(date)"
else
  echo "### [WARN] finetune best.pt missing — skipping export"
fi

echo "### [2/4] SCRATCH  $(date)"
"$PY" scripts/train_plate.py --data "$DATA" --mode scratch \
  --epochs "$EPOCHS" --batch "$BATCH" --device mps \
  --project "$PROJECT" --name plate_scratch

SC="$PROJECT/plate_scratch/weights/best.pt"

echo "### [3/4] BENCHMARK B  $(date)"
if [ -f "$FT" ] && [ -f "$SC" ]; then
  "$PY" scripts/benchmark_plate.py --data "$DATA" --pretrained "$FT" --trained "$SC"
else
  echo "### [WARN] missing weights — skipping benchmark (FT=$FT SC=$SC)"
fi

echo "### [4/4] DONE  $(date)"
