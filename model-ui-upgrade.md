# Step-by-Step Plan: Model & UI Upgrade

This document outlines the step-by-step roadmap to restructure the project repository, satisfy the `agent.md` guidelines for course requirements, replace the heavy ResNet50 model, and build a premium FastAPI + Streamlit system.

---

## 🛠️ Objectives & Syllabus Alignment

- **Syllabus compliance**: Transfer learning using **TensorFlow/Keras** for classifier training.
- **Model modernization**: Replace **ResNet50** (~98MB) with **EfficientNet-B0** (~16MB) to ensure light CPU usage for class demos.
- **Data preparedness**: Since data preparation isn't completed, a mock data generation script will be created to allow immediate prototyping and testing.
- **Real-world architecture**: Restructure the project to separate Backend API logic (`src/api/`) from Frontend UI logic (`src/ui/`).

---

## 📁 Repository Restructuring Map

We will align the repository structure exactly with the `agent.md` layout:

```plaintext
main/
├── configs/
│   ├── config.yaml            # Path configurations, thresholds, training hyperparameters
│   └── agent.md               # Folder specification
├── data/
│   ├── database.csv           # Registered plates CSV (Toyota Vios, Hyundai Accent, etc.)
│   ├── raw/                   # Raw images (YOLO annotations, Brand categories, Color categories)
│   ├── processed/             # Cropped and normalized split directories
│   └── agent.md               # Folder specification
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb  # EDA and dataset splitting
│   ├── 02_model_prototyping.ipynb  # Model loading and inference tests
│   └── agent.md               # Folder specification
├── src/
│   ├── agent.md               # Folder specification
│   ├── api/                   # FastAPI Backend
│   │   ├── __init__.py
│   │   └── app.py             # Server endpoints (/verify, /status)
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── vehicle_dataset.py # Keras ImageFolder / Dataset Loader with augmentations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py        # YOLOv8 plate detector wrapper
│   │   ├── ocr.py             # PaddleOCR/EasyOCR reader engine wrapper
│   │   └── classifiers.py     # EfficientNet-B0 (Brand) & MobileNetV3-Small (Color)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Keras training loops, callbacks, Keras Tuner
│   │   └── evaluator.py       # integrated system pipeline metrics
│   └── utils/
│       ├── __init__.py
│       ├── matching.py        # CSV matching verification logic
│       └── visual.py          # UI visual/audio components
│   └── ui/                    # Streamlit Frontend
│       └── dashboard.py       # Glassmorphism dark-mode Streamlit Dashboard
├── tests/
│   ├── test_matching.py       # Exists - DatabaseMatcher unit tests
│   ├── test_ocr.py            # OCR parsing unit tests
│   ├── test_dataset.py        # Dataset validation unit tests
│   └── agent.md               # Folder specification
├── requirements.txt           # Python dependencies
└── train.py                   # Model training CLI entrypoint
```

---

## 🚀 Step-by-Step Implementation Roadmap

### Step 1: Base Configuration & Directory Scaffolding
*Goal: Restructure directories and setup the core configuration files.*

1. **Create missing folders and modules**:
   - Create directories: `main/configs/`, `main/data/raw/`, `main/data/processed/`, `main/notebooks/`, `main/src/api/`, `main/src/ui/`.
   - Add empty `__init__.py` files inside python packages under `main/src/` to make them importable.
2. **Add unified `main/configs/config.yaml`**:
   - Set parameters for models (conf thresholds, crop padding of 5%), database paths, and training parameters (learning rate `1e-4`, Batch Size 32, input resolution 224x224).
3. **Verify base environment**:
   - Add required packages (`fastapi`, `uvicorn`, `efficientnet`, `paddleocr`, `pyyaml`) to [requirements.txt](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/requirements.txt).

---

### Step 2: Synthetic Prototyping Data & Processing Pipeline
*Goal: Build mock datasets and data loading pipelines to unblock model/pipeline development.*

1. **Write `main/src/utils/mock_generator.py`**:
   - Generate mock directories and a small set of colored vehicle images with text annotations under `main/data/raw/` so the pipeline can execute without downloading Kaggle.
2. **Implement `main/src/datasets/vehicle_dataset.py`**:
   - Write `VehicleDataset` loader using `tf.keras.utils.image_dataset_from_directory` with random horizontal flip and resizing to 224x224.
3. **Draft Jupyter Notebooks**:
   - Create `main/notebooks/01_eda_and_data_prep.ipynb` verifying raw folder exploration, image size distribution, and class balance plotting.

---

### Step 3: Model Wrappers (Detection, OCR, & Classification)
*Goal: Implement model loading wrappers according to the `agent.md` specifications.*

1. **YOLOv8 Detection wrapper (`main/src/models/detector.py`)**:
   - Load YOLOv8-nano model and crop bounding box of plates with a safety margin (5% padding).
2. **OCR wrapper (`main/src/models/ocr.py`)**:
   - Load OCR engine, strip spaces/dashes/dots, and sort coordinates for 2-line plates (top-row first, bottom-row second).
3. **Lightweight Classifiers (`main/src/models/classifiers.py`)**:
   - **EfficientNet-B0** base class for `BrandClassifier`.
   - **MobileNetV3-Small** base class for `ColorClassifier`.
   - Expose forward prediction returning softmax outputs.

---

### Step 4: Training & Evaluation Engine
*Goal: Satisfy course requirements with transfer learning trainers and callbacks.*

1. **Implement `main/src/engine/trainer.py`**:
   - Implement transfer learning: freeze base layers, compile with Adam optimizer, `categorical_crossentropy` loss.
   - Configure Callbacks: `EarlyStopping` (patience=5), `ModelCheckpoint` (save best weights), and `TensorBoard`.
2. **Implement `main/src/engine/evaluator.py`**:
   - Add `SystemEvaluator` to compute Fake Plate Detection Rate, False Alarm Rate, and End-to-End Latency.
3. **Wiring [train.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/train.py)**:
   - Create argparse CLI in `train.py` to trigger model training for Brand or Color classifiers.

---

### Step 5: FastAPI Backend Development
*Goal: Build a REST API encapsulating the pipeline.*

1. **Implement `main/src/api/app.py`**:
   - Create `/verify` endpoint accepting image uploads.
   - Run: YOLO Crop $\rightarrow$ OCR Read $\rightarrow$ EfficientNet Brand Classify $\rightarrow$ MobileNet Color Classify $\rightarrow$ `DatabaseMatcher.verify_vehicle`.
   - Return detailed JSON response showing matched/mismatched status and confidence scores.

---

### Step 6: Premium Streamlit Dashboard UI
*Goal: Modernize the presentation layer to look like a premium real-world product.*

1. **Implement `main/src/ui/dashboard.py`**:
   - Apply custom glassmorphic CSS styling via `st.markdown("<style>...</style>")`.
   - Create a clean layout with sidebar control, main webcam/video feed window, real-time FPS counter, and detection log list.
2. **Add audible alarms (`main/src/utils/visual.py`)**:
   - Inject HTML audio tags with warning siren sounds in Streamlit to trigger immediately on a `MISMATCH` or `UNREGISTERED` plate result.

---

### Step 7: Tests and Final Verification
*Goal: Run test suites and performance audits before final presentation.*

1. **Create `main/tests/test_ocr.py` & `main/tests/test_dataset.py`**:
   - Test text normalization and 2-line coordinates sorting.
   - Test dataset loader shape outputs `(BatchSize, 224, 224, 3)`.
2. **Execute validation scripts**:
   - Run `python .agents/scripts/checklist.py main` to verify security, linting, and performance indicators.

---

## 🏁 Phase X: Verification Checklist

- [ ] `pytest main/tests/` passes successfully.
- [ ] Brand classifier model is confirmed to be EfficientNet-B0 ($<20$MB size).
- [ ] End-to-end API response latency on CPU is $<500$ms.
- [ ] Visual UI is verified with glassmorphic elements and no default raw Streamlit styling.
- [ ] Warnings are tested: flashing red banner and siren alert sounds correctly on mismatch.
