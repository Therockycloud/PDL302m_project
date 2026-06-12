#!/bin/bash
# Resume-aware plate pipeline: resume finetune -> export ONNX -> scratch -> benchmark.
# Wrapped by caffeinate at launch to survive idle sleep. Re-runnable: each stage
# is skipped if its output already exists.
set -u

PY="/opt/homebrew/Caskroom/miniforge/base/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
ROOT="/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
cd "$ROOT/main" || exit 1

DATA="data/raw/plate_det/data.yaml"
PROJECT="$ROOT/main/data/models/plate_runs"
EPOCHS="${1:-80}"
BATCH="${2:-16}"

FT_DIR="$PROJECT/plate_finetune"
FT_LAST="$FT_DIR/weights/last.pt"
FT_BEST="$FT_DIR/weights/best.pt"
SC_DIR="$PROJECT/plate_scratch"
SC_BEST="$SC_DIR/weights/best.pt"

echo "=================================================================="
echo "[START] $(date)  epochs=$EPOCHS batch=$BATCH"
echo "=================================================================="

echo "### [1/4] FINETUNE (resume if possible)  $(date)"
if [ -f "$FT_LAST" ]; then
  echo "resuming from $FT_LAST"
  "$PY" scripts/train_plate.py --resume "$FT_LAST"
else
  echo "fresh finetune"
  "$PY" scripts/train_plate.py --data "$DATA" --mode finetune \
    --epochs "$EPOCHS" --batch "$BATCH" --device mps \
    --project "$PROJECT" --name plate_finetune
fi

if [ -f "$FT_BEST" ]; then
  echo "### export finetune -> ONNX  $(date)"
  "$PY" -c "from ultralytics import YOLO; YOLO('$FT_BEST').export(format='onnx', imgsz=640)"
  cp "$FT_DIR/weights/best.onnx" "data/models/plate_yolov8n.onnx" \
    && echo "### exported -> data/models/plate_yolov8n.onnx  $(date)"
else
  echo "### [WARN] finetune best.pt missing — skipping export"
fi

echo "### [2/4] SCRATCH  $(date)"
if [ -f "$SC_BEST" ]; then
  echo "scratch already trained — skipping"
elif [ -f "$SC_DIR/weights/last.pt" ]; then
  echo "resuming scratch from last.pt"
  "$PY" scripts/train_plate.py --resume "$SC_DIR/weights/last.pt"
else
  "$PY" scripts/train_plate.py --data "$DATA" --mode scratch \
    --epochs "$EPOCHS" --batch "$BATCH" --device mps \
    --project "$PROJECT" --name plate_scratch
fi

echo "### [3/4] BENCHMARK B  $(date)"
if [ -f "$FT_BEST" ] && [ -f "$SC_BEST" ]; then
  "$PY" scripts/benchmark_plate.py --data "$DATA" --pretrained "$FT_BEST" --trained "$SC_BEST"
else
  echo "### [WARN] missing weights — skipping benchmark"
fi

echo "### [4/4] DONE  $(date)"
