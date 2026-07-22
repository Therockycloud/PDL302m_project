# Project: Vehicle Anti-Theft System Upgrade

## Architecture
- **Streamlit Frontend**: Dashboard UI implemented in `main/src/ui/dashboard.py` showing vehicle processing, live feed visualization, and database records.
- **Deep Learning Pipeline**:
  - Bounding box detection: YOLOv8-nano in `main/src/models/detector.py`.
  - License Plate OCR: PaddleOCR-only at runtime (won Benchmark C, 81% exact-match; `ocr.engine: "ppocr"` in `main/configs/config.yaml`), in `main/src/models/ocr.py`. No silent fallback — raises a hard `RuntimeError` if PaddleOCR fails to init. EasyOCR is train/eval/benchmark-only (`requirements-train.txt`).
  - **VN plate CTC experiment (not deployed):** MobileNetV3-Small + CTC → ONNX candidate trained on synthetic + pseudo-labels; `deployment_ready: false` (0/64 val exact-match, CER ~0.659 on `real_validation.csv`). Gate to replace Paddle (≥90% exact-match on held-out real) not met — Paddle remains runtime OCR. Artifacts: `main/data/models/vn_plate_run/`. Data policy: `main/data/plate_ocr/README.md`.
  - Attributes: Vehicle colour classifier (PyTorch MobileNetV3-Small) as a soft-warning layer in `main/src/models/torch_color.py`. Brand classification was dropped from the decision after weak results.
- **Database/Storage**: CSV database tracking authorized vehicles, logs of scanned vehicles, and authorization states in `main/data/database.csv`.

## Code Layout
- `main/src/ui/dashboard.py`: Streamlit application entrypoint.
- `main/src/models/detector.py`: YOLOv8-nano vehicle and license plate detector.
- `main/src/models/ocr.py`: PaddleOCR processor (runtime-only; hard error if unavailable, no EasyOCR fallback).
- `main/src/models/torch_color.py`: PyTorch MobileNetV3-Small colour classifier (runtime). `main/src/models/classifiers.py`: Keras brand/colour classifiers (training/eval only).
- `main/data/test/sample_parking.mp4`: Placeholder for downloaded test video.
- `presentations/`: Folder containing HTML slides and presentation resources.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration | Codebase and presentations exploration | None | DONE |
| 2 | UI Redesign | Light-Mode & Borderless UI Redesign | Exploration | DONE |
| 3 | Video Simulation | Find, download sample_parking.mp4 and add UI feature | Exploration | DONE |
| 4 | Pipeline Optimization | CPU optimizations for YOLOv8, PaddleOCR, colour classifier | Exploration | DONE |
| 5 | Presentation Audit & Polish | Audit and polish HTML slides in presentations/ | Exploration | DONE |
| 6 | E2E Verification | Run E2E pipeline and verify all aspects | 2, 3, 4, 5 | DONE |
| 7 | VN Plate CTC Experiment | Train MobileNetV3+CTC ONNX recognizer; evaluate on held-out real plates; gate ≥90% exact-match to replace Paddle | 4 | DONE (not deployed — Paddle remains OCR) |

## Known Limits
- Single camera environment; no multi-site deployment tested.
- Verified car-plate training data insufficient for CTC experiment (domain gap: val/test are car plates, pseudo-labels mostly motorcycles).

## Interface Contracts
- **Streamlit UI ↔ Pipeline**: Bounding box coordinates, crop image, OCR text result, brand prediction (diagnostic-only, not part of the verify decision), color prediction (soft-warning, ALLOW_WARN), latency metrics, FPS.
- **Database ↔ UI/Pipeline**: Read/write matching records from/to `main/data/database.csv`.
