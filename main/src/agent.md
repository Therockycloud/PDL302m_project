# Project Source Code Overview (`main/src/`)

This directory contains the modular python packages of the project. Avoid writing long, monolithic scripts. Keep helper methods separate from core architectures.

---

## 1. Directory Structure

```
main/src/
├── agent.md                 # This file
├── datasets/                # Custom data loaders & pipeline utilities
│   ├── agent.md
│   └── vehicle_dataset.py   # Loader class for brands/colors
├── models/                  # Neural network model classes
│   ├── agent.md
│   ├── lpr_yolo.py          # YOLOv8 vehicle plate detection wrapper
│   ├── ocr_engine.py        # PaddleOCR/EasyOCR reader wrapper
│   └── classifier.py        # ResNet/MobileNet classifiers
├── engine/                  # Training pipelines and metrics
│   ├── agent.md
│   ├── trainer.py           # Classifier model trainer
│   └── evaluator.py         # Complete system pipeline evaluator
└── utils/                   # Helper functions (matching, postprocessing, UI)
    ├── agent.md
    ├── matching.py          # CSV matching verification logic
    └── visualization.py     # Streamlit overlays and warnings
```

---

## 2. Coding Guidelines
1. **Typing:** Use Python type hints where possible (e.g., `def crop_image(img: np.ndarray, bbox: list) -> np.ndarray:`).
2. **Exception Handling:** Always enclose API and model loading calls in try-except blocks. Handle fallback states (e.g., if YOLOv8 fails to detect a plate, proceed to log an "underectable plate" alarm).
3. **Docstrings:** Standardize docstrings (Google style or Sphinx style) for model wrappers.
4. **Config Files:** Keep paths and parameters inside `main/configs/` instead of hardcoding them inside `main/src/`.
