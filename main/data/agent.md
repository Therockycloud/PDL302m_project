# Dataset Specifications and Data Management

This directory contains raw datasets, processed subsets, and helper scripts for parsing data.

---

## 1. Directory Structure

Ensure the following subdirectories are maintained for development:
```
main/data/
├── agent.md                 # This file
├── database.csv             # Registered plates database
├── raw/                     # Original downloaded datasets (unmodified)
│   ├── license_plates/      # YOLO/COCO formatted plate images
│   ├── car_brands/          # Car brand images (categorized in subfolders)
│   └── car_colors/          # Car color images (categorized in subfolders)
└── processed/               # Data splits and cropped license plates for OCR training
    ├── plates_train/
    ├── plates_val/
    └── classifiers/         # Normalized 224x224 crop splits
```

---

## 2. Dataset Links & Download Instructions

### A. License Plate Localization
* **Dataset Name:** Vietnamese Car License Plate Detection
* **URL:** [Kaggle Link](https://www.kaggle.com/datasets/datnguyen1111/vietnamese-car-license-plate-detection)
* **Format:** YOLO annotation text files (`.txt`) and corresponding images (`.jpg`).
* **Download Instructions:** Download the ZIP, extract into `main/data/raw/license_plates/`.

### B. Car Brand Classification
* **Dataset Name:** Stanford Cars Dataset or Vehicle Brand Datasets
* **URL:** [Stanford Cars Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) (or similar custom brand datasets).
* **Target Classes for Vietnam:** Focus on top brands: `Toyota`, `Hyundai`, `Kia`, `Mazda`, `Honda`, `VinFast`, `Ford`, `Mitsubishi`.
* **Download Instructions:** Extract images into subfolders inside `main/data/raw/car_brands/<brand_name>/`.

### C. Car Color Classification
* **Dataset Name:** Car Color Recognition Dataset
* **URL:** [Car Color Classification Dataset on Kaggle](https://www.kaggle.com/datasets/landrykezebou/car-color-recognition-dataset)
* **Target Classes:** `White`, `Black`, `Grey`, `Silver`, `Red`, `Blue`, `Brown`, `Yellow`.
* **Download Instructions:** Extract images into subfolders inside `main/data/raw/car_colors/<color_name>/`.

---

## 3. Data Processing Pipeline (CLO6)

The data pipeline script should automate:
1. **License Plate Split:** Divide plate images into 80% train, 20% validation.
2. **License Plate Bbox Crop:** Write a helper script using annotations to crop license plate regions from car images. These crops are used to evaluate the OCR model.
3. **Classifiers Preprocessing:** Resize car brand and color images to 224x224, apply random horizontal flips, and normalize pixels to $[0, 1]$ range.
