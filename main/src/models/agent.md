# Model Architectures & Inference Wrappers

This directory contains the code definition and loading wrappers for the three deep learning sub-systems.

---

## 1. Sub-Systems Specifications

### A. License Plate Detection (`lpr_yolo.py`)
* **Model:** Ultralytics YOLOv8.
* **Wrapper Tasks:**
  * Load model weights (`yolov8n.pt` fine-tuned).
  * Run inference on raw frames.
  * Crop the detected bounding box of the license plate with a safety margin (padding of $5\%$) to avoid clipping characters.
  * Return cropped plate images as a numpy array.

### B. Character Recognition (`ocr_engine.py`)
* **Engine:** PaddleOCR (recommended for accuracy) or EasyOCR (recommended for simple installation).
* **Wrapper Tasks:**
  * Clean character string: convert to uppercase, strip whitespaces, dashes (`-`), and dots (`.`).
  * Format character sequence: handle 2-line plates by sorting characters based on their vertical and horizontal coordinates to read top-row then bottom-row.

### C. Vehicle Feature Classification (`classifier.py`)
* **Class: `BrandClassifier`**
  * Load a transfer-learned model (e.g., ResNet50 base).
  * Inputs: Car image cropped from bounding boxes.
  * Outputs: Softmax probabilities across classes (`Toyota`, `Hyundai`, `VinFast`, etc.).
* **Class: `ColorClassifier`**
  * Load a transfer-learned model (e.g., MobileNetV2 base).
  * Outputs: Softmax probabilities across classes (`White`, `Black`, `Red`, etc.).

---

## 2. Requirements & Dependencies
* If a GPU is available, all wrappers should automatically call `.to('cuda')` (PyTorch) or map to the GPU device (TensorFlow) to maintain a processing speed $\ge 30$ FPS.
