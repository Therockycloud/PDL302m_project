# Benchmark B — License-Plate Detector (YOLOv8n)

Validation set: 1,765 images (HuggingFace `keremberke/license-plate-object-detection`,
single class `license_plate`; contains Vietnamese plates — `xemay`/`CarLongPlate`).
Both models trained 80 epochs on Apple M1 Max (MPS), imgsz 640, batch 16.
Latency measured on CPU (Ultralytics `model.val`).

| Model | Init | Precision | Recall | mAP50 | mAP50-95 | latency (ms/img, CPU) | size (MB) |
|:------|:-----|----------:|-------:|------:|---------:|----------------------:|----------:|
| **plate_finetune** | transfer from COCO `yolov8n.pt` | **0.9823** | **0.9674** | **0.9896** | **0.7040** | 110.3 | 6.24 |
| plate_scratch | random init (`yolov8n.yaml`) | 0.9816 | 0.9560 | 0.9790 | 0.6972 | 110.4 | 6.24 |

Precision/Recall taken from the best-checkpoint row of each run's `results.csv`
(the row whose `metrics/mAP50(B)` matches the reported mAP50 above): epoch 60
for `plate_finetune` (`main/data/models/plate_runs/plate_finetune/results.csv`),
epoch 70 for `plate_scratch` (`main/data/models/plate_runs/plate_scratch/results.csv`).

## Conclusion

Both configurations converge to a strong single-class detector, but **transfer
learning from COCO wins on every accuracy metric** (mAP50 +1.06 pts, mAP50-95
+0.68 pts) at identical latency and size. Fine-tuning also converged far earlier
(mAP50 ≈ 0.97 by epoch 9 vs. the scratch run needing the full budget to catch up).

**Selected model:** `plate_finetune` → exported to `main/data/models/plate_yolov8n.onnx`
(12.3 MB ONNX) and wired into the pipeline as the stage-2 plate detector. The
nano backbone (6.24 MB `.pt`) keeps the model well within the edge/CPU budget
targeted for future hardware.
