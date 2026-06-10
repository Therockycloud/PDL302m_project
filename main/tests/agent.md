# Unit and Integration Testing Guidelines

This directory contains test cases to verify the stability of code modules, ensuring changes do not break core logic.

---

## 1. Test Modules Overview

### `test_matching.py`
* **Target:** `main/src/utils/matching.py`
* **Test Cases:**
  * Test matching when plate, brand, and color match exactly.
  * Test case-insensitivity and whitespace stripping.
  * Test mismatch warning when brand differs (e.g., Hyundai vs Toyota).
  * Test mismatch warning when color differs (e.g., White vs Red).
  * Test warning when plate is not present in database.

### `test_ocr.py`
* **Target:** `main/src/models/ocr_engine.py`
* **Test Cases:**
  * Test that punctuation, dashes, and periods are successfully stripped.
  * Test sorting of bounding boxes for 2-line license plates (verifying the letters on top-line are read before numbers on bottom-line).

### `test_dataset.py`
* **Target:** `main/src/datasets/vehicle_dataset.py`
* **Test Cases:**
  * Verify image shapes loaded by `DataLoader` are exactly `(BatchSize, 224, 224, 3)`.
  * Verify pixel values are correctly normalized to $[0.0, 1.0]$.
  * Verify data augmentation output structures.

---

## 2. Running Tests
Run all unit tests using python's built-in `unittest` or `pytest` library from the `main/` directory:
```bash
pytest tests/
```
