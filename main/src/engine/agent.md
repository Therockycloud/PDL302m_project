# Training and Evaluation Engine

This directory contains scripts responsible for running training loops, evaluating metrics, and managing model checkpoints.

---

## 1. Class: `ModelTrainer` (`trainer.py`)
* **Purpose:** Train and fine-tune Car Brand and Car Color classifiers.
* **Key Tasks:**
  * Configure training loops with custom callbacks:
    * `EarlyStopping` (patience of 5–10 epochs to prevent over-fitting).
    * `ModelCheckpoint` (save only the best weights based on validation loss).
    * `TensorBoard` (log loss and accuracy curves).
  * Run hyperparameter search trials (e.g., Keras Tuner RandomSearch).

---

## 2. Class: `SystemEvaluator` (`evaluator.py`)
* **Purpose:** Evaluate the performance of the **integrated** system on the final test set.
* **Key Tasks:**
  * Run end-to-end evaluation:
    1. Pass raw test images through YOLOv8.
    2. Pass cropped plate region through OCR.
    3. Pass car image through Brand and Color classifiers.
    4. Run matching logic against `database.csv`.
  * Calculate **system-level metrics**:
    * **Fake Plate Detection Rate:** Percentage of mismatched plates correctly flagged as warnings.
    * **False Alarm Rate:** Percentage of authorized plates incorrectly flagged as mismatches.
    * **End-to-End Latency (ms):** Average time taken to process a single vehicle.
