# Project: Vehicle Anti-Theft System Upgrade

## Architecture
- **Streamlit Frontend**: Dashboard UI implemented in `main/src/ui/dashboard.py` showing vehicle processing, live feed visualization, and database records.
- **Deep Learning Pipeline**:
  - Bounding box detection: YOLOv8-nano in `main/src/models/detector.py`.
  - License Plate OCR: PaddleOCR (primary, won Benchmark C) with EasyOCR fallback, in `main/src/models/ocr.py`.
  - Attributes: Vehicle colour classifier (PyTorch MobileNetV3-Small) as a soft-warning layer in `main/src/models/torch_color.py`. Brand classification was dropped from the decision after weak results.
- **Database/Storage**: CSV database tracking authorized vehicles, logs of scanned vehicles, and authorization states in `main/data/database.csv`.

## Code Layout
- `main/src/ui/dashboard.py`: Streamlit application entrypoint.
- `main/src/models/detector.py`: YOLOv8-nano vehicle and license plate detector.
- `main/src/models/ocr.py`: PaddleOCR processor (EasyOCR fallback).
- `main/src/models/torch_color.py`: PyTorch MobileNetV3-Small colour classifier (runtime). `main/src/models/classifiers.py`: Keras brand/colour classifiers (training/eval only).
- `main/data/test/sample_parking.mp4`: Placeholder for downloaded test video.
- `presentations/`: Folder containing HTML slides and presentation resources.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration | Codebase and presentations exploration | None | PLANNED |
| 2 | UI Redesign | Light-Mode & Borderless UI Redesign | Exploration | PLANNED |
| 3 | Video Simulation | Find, download sample_parking.mp4 and add UI feature | Exploration | PLANNED |
| 4 | Pipeline Optimization | CPU optimizations for YOLOv8, PaddleOCR, colour classifier | Exploration | PLANNED |
| 5 | Presentation Audit & Polish | Audit and polish HTML slides in presentations/ | Exploration | PLANNED |
| 6 | E2E Verification | Run E2E pipeline and verify all aspects | 2, 3, 4, 5 | PLANNED |

## Interface Contracts
- **Streamlit UI ↔ Pipeline**: Bounding box coordinates, crop image, OCR text result, brand prediction, color prediction, latency metrics, FPS.
- **Database ↔ UI/Pipeline**: Read/write matching records from/to `main/data/database.csv`.
