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
        ├── classifiers.py   # ResNet50 (Brand) & MobileNetV3 (Color) classifiers
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
* **Dataset Name:** Car Color Recognition Dataset
* **Source:** [Car Color Classification Dataset on Kaggle](https://www.kaggle.com/datasets/landrykezebou/car-color-recognition-dataset)
* **Download:** Save under subdirectories: `main/data/raw/car_colors/<color_name>/` for `White`, `Black`, `Grey`, `Silver`, `Red`, `Blue`, `Brown`, and `Yellow`.

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

### B. Character Recognition (EasyOCR)
* **Model Class:** `OCRReader` (`main/src/models/ocr.py`)
* **Backend:** EasyOCR
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

#### 2. Color Classifier (`ColorClassifier` in `main/src/models/classifiers.py`)
* **Base Model:** `MobileNetV3Small` (Pre-trained on ImageNet)
* **Top Layer:** `GlobalAveragePooling2D → Dropout(0.3) → Dense(num_classes=8, activation='softmax')`
* **Input Shape:** 224x224x3
* **Preprocessing:** Scaling layer rescales input pixels to `[-1, 1]` range.
* **Optimization:** Adam Optimizer, `categorical_crossentropy` loss.

---

## 4. Helper and Utility Specifications

### A. Verification Database Matcher (`main/src/utils/matching.py`)
* **Database Schema (`database.csv`):** `license_plate,car_brand,car_color`
* **Matching Logic (`verify_vehicle`):**
  1. Searches the database for the detected plate character string.
  2. If the plate is not present $\rightarrow$ return status `UNREGISTERED` and action `DENY`.
  3. If the plate is present $\rightarrow$ compares the detected brand and color with the registered records.
  4. If both brand and color match $\rightarrow$ return status `AUTHORIZED` and action `ALLOW`.
  5. If brand or color does not match $\rightarrow$ return status `MISMATCH` and action `DENY_ALERT`.

### B. Visual Overlay Overlay (`main/src/utils/visualization.py`)
* Draw bounding boxes and text around detected license plates.
* Overlay status overlays:
  * **GREEN** for `AUTHORIZED` cars.
  * **RED** for `MISMATCH` or `UNREGISTERED` cars.
