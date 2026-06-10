# Hyperparameter & Model Configuration Specifications

This directory contains the configuration files (YAML/JSON) defining parameters for model training, data loaders, and inference.

---

## 1. YOLOv8 Detection Configuration (`yolov8_config.yaml`)
* **Task:** Object Detection (License Plate localization).
* **Base Model:** `yolov8n.pt` (Nano model for high real-time FPS) or `yolov8s.pt` (Small model).
* **Epochs:** 50 - 100 epochs.
* **Batch Size:** 16 or 32.
* **Image Size:** 640x640 pixels.
* **Optimizer:** SGD or AdamW (learning rate `lr0=0.01`).
* **Augmentation Hyperparameters:**
  * Hue adjustment (`hsv_h=0.015`)
  * Saturation adjustment (`hsv_s=0.7`)
  * Brightness scaling (`hsv_v=0.4`)
  * Translate fraction (`translate=0.1`)
  * Scale fraction (`scale=0.5`)

---

## 2. Vehicle Brand Classifier Configuration (`brand_classifier.json`)
* **Architecture:** ResNet50 (pre-trained on ImageNet).
* **Input Resolution:** 224x224x3.
* **Freeze Strategy:** Freeze base layers (0-140), train top dense layers, then fine-tune with a very small learning rate.
* **Batch Size:** 32.
* **Learning Rate:** `1e-4` for transfer learning, `1e-5` for fine-tuning.
* **Regularization:**
  * Dropout rate: `0.5` before the final softmax layer.
  * L2 weight decay: `1e-4`.
  * Batch Normalization: Enabled on top dense layers.
* **Optimizer:** Adam.
* **Loss Function:** `categorical_crossentropy`.

---

## 3. Vehicle Color Classifier Configuration (`color_classifier.json`)
* **Architecture:** MobileNetV2 (pre-trained on ImageNet).
* **Input Resolution:** 224x224x3.
* **Batch Size:** 32.
* **Learning Rate:** `1e-4` with Adam.
* **Loss Function:** `categorical_crossentropy`.
* **Classes (8 common car colors in Vietnam):** `White`, `Black`, `Grey`, `Silver`, `Red`, `Blue`, `Brown`, `Yellow`.

---

## 4. Verification Database Configuration (`database.csv`)
* **Schema:** `license_plate,car_brand,car_color`
* **Example records:**
  ```csv
  30F-12345,Toyota Vios,White
  51G-67890,Hyundai Accent,Black
  43A-11111,VinFast Lux A,Red
  30H-99999,Honda Civic,Blue
  ```
