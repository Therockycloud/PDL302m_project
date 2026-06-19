# Model Specifications & System Architecture

This document consolidates the configuration parameters, dataset loaders, neural network architectures, and helper utility specifications for the DPL302m project.

---

## 1. Directory Layout & Data Management

```
main/
├── configs/
│   └── config.yaml          # Global hyperparameters and system paths
├── data/
│   ├── database.csv         # Registered vehicles database
│   ├── raw/                 # Original unmodified source datasets
│   │   ├── license_plates/  # Vietnamese Car License Plate Detection dataset
│   │   ├── car_brands/      # Stanford Cars and VinFast brand images
│   │   └── car_colors/      # Car Color Recognition dataset
│   └── processed/           # Processed datasets, pre-split and normalized
└── src/
    ├── datasets/
    │   └── vehicle_dataset.py # Custom data loaders for classifiers
    └── models/
        ├── classifiers.py   # EfficientNet-B0 (Brand, TF/Keras) & MobileNetV3-Small (Color, TF/Keras training; PyTorch runtime) classifiers
        ├── torch_color.py   # PyTorch MobileNetV3-Small runtime colour classifier
        ├── detector.py      # Ultralytics YOLOv8 vehicle plate detection wrapper
        └── ocr.py           # EasyOCR character recognition engine wrapper
```

---

## 2. Dataset Links & Downloading

### A. License Plate Localization
* **Dataset Name:** Vietnamese Car License Plate Detection
* **Source:** [Kaggle Link](https://www.kaggle.com/datasets/datnguyen1111/vietnamese-car-license-plate-detection)
* **Format:** YOLO annotation text files (`.txt`) and images (`.jpg`).
* **Download:** Extract files into `main/data/raw/license_plates/`.

### B. Car Brand Classification
* **Dataset Name:** Stanford Cars Dataset & Scraped Local Brands
* **Source:** [Stanford Cars Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
* **VinFast Additions:** Images gathered via scraping Google Images.
* **Download:** Save under subdirectories: `main/data/raw/car_brands/<brand_name>/` for `Toyota`, `Hyundai`, `Kia`, `Mazda`, `Honda`, `VinFast`, `Ford`, and `Mitsubishi`.

### C. Car Color Classification
* **Dataset Name (original):** Car Color Recognition Dataset
* **Source:** [Car Color Classification Dataset on Kaggle](https://www.kaggle.com/datasets/landrykezebou/car-color-recognition-dataset)
* **Download:** Save under subdirectories: `main/data/raw/car_colors/<color_name>/` for `White`, `Black`, `Grey`, `Silver`, `Red`, `Blue`, `Brown`, and `Yellow`.
* **Dataset Name (deployed model, added later):** **VCoR — Vehicle Color Recognition** (Kaggle), merged with the original images to reach the **86% TTA test accuracy** deployed at runtime. Layout `{train,val,test}/<lowercolor>/*.jpg`; 8 of its colour folders map onto the project's 8 classes (others — beige/gold/green/orange/pink/purple/tan — are dropped). Merge script: `main/scripts/build_color_dataset.py` → `main/data/raw/car_colors_vcor/`. See `docs/model_specifications.md` §3.4/§4 below and `reports/documents/Report_3_Model_Results.md` §5.1 for full methodology + honest domain-gap caveat (VCoR is clean web photos, not garage CCTV).

---

## 3. Sub-Systems Specifications

### A. License Plate Detection (YOLOv8)
* **Model Class:** `VehicleDetector` (`main/src/models/detector.py`)
* **Base Architecture:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
* **Training Parameters:**
  * **Input Shape:** 640x640 pixels
  * **Epochs:** 50
  * **Batch Size:** 32
  * **Confidence Threshold:** 0.25
* **Processing:** Crops the detected bounding box of the license plate with a safety margin (padding of $5\%$) to avoid clipping plate characters.

### B. Character Recognition (PaddleOCR — primary; EasyOCR — fallback)
* **Model Class:** `OCRReader` (`main/src/models/ocr.py`)
* **Primary Backend:** PaddleOCR (PP-OCRv4, CRNN+CTC) — configured via `main/configs/config.yaml` (`ocr.engine: ppocr`). Benchmark C: 81% exact-match on real CCTV plates vs. EasyOCR 0%.
* **Fallback Backend:** EasyOCR — activated by setting `ocr.engine: easyocr` in config.
* **Post-processing:**
  * Alphanumeric character cleaning (removes punctuation, dashes, spaces, and dots).
  * 2-line plate correction: sorts the bounding boxes based on vertical and horizontal coordinates to read the top line first, then the bottom line.

### C. Vehicle Feature Classification (Keras/TensorFlow)

#### 1. Brand Classifier (`BrandClassifier` in `main/src/models/classifiers.py`)
* **Base Model:** `EfficientNetB0` (Pre-trained on ImageNet)
* **Top Layer:** `GlobalAveragePooling2D → Dropout(0.5) → Dense(num_classes=8, activation='softmax')`
* **Input Shape:** 224x224x3
* **Optimization:** Adam Optimizer, `categorical_crossentropy` loss.
* **Freezing:** Base model is frozen during transfer learning.

#### 2. Color Classifier — Training/Eval (`ColorClassifier` in `main/src/models/classifiers.py`, TF/Keras)
* **Base Model:** `MobileNetV3Small` (Pre-trained on ImageNet)
* **Top Layer:** `Rescaling(255.0) → GlobalAveragePooling2D → Dropout(0.3) → Dense(num_classes=8, activation='softmax')`
* **Input Shape:** 224x224x3
* **Preprocessing:** `Rescaling(255.0)` converts [0,1] input back to [0,255] — MobileNetV3's built-in `include_preprocessing=True` then normalises internally. *Note: an earlier version incorrectly used `Rescaling(1/127.5, -1)` causing double-preprocessing (fixed).*
* **Optimization:** Adam Optimizer, `categorical_crossentropy` loss.

#### 2b. Color Classifier — Runtime/Inference (`main/src/models/torch_color.py`, PyTorch) — DEPLOYED, ~86% TTA
* **Weights file:** `main/data/models/color_MobileNetV3Small.pt`
* **Base Model:** `MobileNetV3-Small` (PyTorch / torchvision), **full fine-tune** (backbone unfrozen, not just head).
* **Training:** Google Colab (GPU), script `main/scripts/colab_train_color.py`. Dataset = VCoR (5,881 usable images) + original project images. Recipe: discriminative LR (head 1e-3 / backbone 1e-4), class-weighted loss, label smoothing 0.1, test-time augmentation (TTA, hflip-averaged softmax) at eval, body-crop preprocessing (drop top 20%/bottom 15% of the image).
* **Held-out test accuracy (reproducible, re-measured on the deployed `.pt`):** 85.3% plain / **86.3% TTA**, macro-F1 0.84 — up from the original ~55% frozen-backbone baseline. Full numbers: `docs/benchmarks/color_finetune_report.md` (generated by `main/scripts/eval_color_deployed.py`).
* **Honest caveat:** measured on VCoR (clean web photos), not on real garage CCTV — expect lower accuracy under CCTV lighting/resolution domain shift; white-balance preprocessing + a small CCTV fine-tune set would help close the gap.
* **Note:** The runtime uses PyTorch to avoid OpenMP conflicts with PaddleOCR in the same process. TF/Keras was used for the original training/evaluation pipeline only; the deployed model is trained and evaluated entirely in PyTorch.

---

## 4. Helper and Utility Specifications

### A. Verification Database Matcher (`main/src/utils/matching.py`)
* **Database Schema (`database.csv`):** `license_plate,car_brand,car_color`
* **Matching Logic (`verify_vehicle`) — delivered (plate-primary):**
  1. Searches the database for the detected plate character string.
  2. If the plate is not present $\rightarrow$ return status `UNREGISTERED` and action `DENY`.
  3. If the plate is present $\rightarrow$ plate match alone yields `AUTHORIZED`.
  4. If colour deviates from registered record $\rightarrow$ `AUTHORIZED` with soft warning (`ALLOW_WARN`); barrier opens but alert is logged.
  5. Brand prediction is **diagnostic only — đã loại khỏi quyết định**: recorded in logs but does not affect AUTHORIZED/MISMATCH/UNREGISTERED outcome.

### B. Visual Overlay Overlay (`main/src/utils/visualization.py`)
* Draw bounding boxes and text around detected license plates.
* Overlay status overlays:
  * **GREEN** for `AUTHORIZED` cars.
  * **RED** for `MISMATCH` or `UNREGISTERED` cars.
