# Dataset Loader Modules

This directory contains scripts for loading raw/processed images into models during training and validation.

---

## 1. Class: `VehicleDataset` (`vehicle_dataset.py`)
* **Framework:** TensorFlow/Keras or PyTorch (depending on the target classifier model).
* **Purpose:** Yield batched, augmented inputs for Car Brand and Car Color classifiers.
* **Key Tasks:**
  * Parse folders `main/data/raw/car_brands/` and `main/data/raw/car_colors/`.
  * Create a label index mapping (e.g., `{'Toyota': 0, 'Hyundai': 1}`).
  * Perform train/val/test splitting dynamically or load from a pre-split structure.
  * Apply data augmentations (flips, contrast shifts).
  * Load images on-the-fly to avoid RAM overflows.

---

## 2. Best Practices
* **TensorFlow implementation:**
  * Use `tf.keras.utils.image_dataset_from_directory` to quickly build structured pipelines with smart caching.
  * Add `.prefetch(buffer_size=tf.data.AUTOTUNE)` to optimize training speed on cloud servers.
* **PyTorch implementation:**
  * Use `torchvision.datasets.ImageFolder` combined with custom `transforms.Compose`.
  * Ensure `num_workers > 2` and `pin_memory=True` are configured on the `DataLoader` for fast GPU transfers.
