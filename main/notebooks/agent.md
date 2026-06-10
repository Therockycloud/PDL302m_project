# Guidelines for Jupyter Notebooks

This directory contains Jupyter notebooks used for prototyping, exploratory data analysis (EDA), and initial experimental model training.

---

## 1. Notebook Roadmap

### `01_eda_and_data_prep.ipynb`
* **Purpose:** Explore the datasets, identify class imbalances, inspect image quality, and prepare data splits.
* **Sections:**
  1. Load sample images from License Plate, Car Brand, and Car Color datasets.
  2. Visualize bounding boxes of license plates.
  3. Plot bar charts showing class distributions of brand categories and color categories.
  4. Write clean images and labels to the `main/data/processed/` directory.

### `02_model_prototyping.ipynb`
* **Purpose:** Build draft architectures, verify forward passes, test transfer learning base models, and run simple training loops.
* **Sections:**
  1. Load processed datasets using PyTorch `DataLoader` or TensorFlow `tf.data.Dataset`.
  2. Load pre-trained models (e.g., MobileNetV2, ResNet50) from Keras/PyTorch model hubs.
  3. Run a quick 3-epoch training loop to verify loss curves are descending.
  4. Integrate the YOLOv8 model and run inference on 5 sample car images.
  5. Load EasyOCR/PaddleOCR and read characters of cropped plates.

---

## 2. Best Practices for Notebooks
* **Reproducibility:** Always set random seeds at the top of the notebook:
  ```python
  import random
  import numpy as np
  import tensorflow as tf
  import torch

  random.seed(42)
  np.random.seed(42)
  tf.random.set_seed(42)
  torch.manual_seed(42)
  ```
* **Readability:** Keep code cells short. Document observations after every visualization using markdown cells.
* **Cleanup:** Release GPU memory when switching tasks. In Keras, use `tf.keras.backend.clear_session()`. In PyTorch, use `torch.cuda.empty_cache()`.
