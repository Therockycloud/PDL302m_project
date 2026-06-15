# WS4 — Code Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the runtime dependency set free of the TensorFlow/PaddleOCR conflict, drop an orphaned uncommitted change, and lock in the good ONNX-preference change — keeping the test suite green.

**Architecture:** TensorFlow/Keras is only used by the *training/eval* path (`train.py`, `run_evaluation.py`, `classifiers.py`, `trainer.py`, `vehicle_dataset.py`), never by the live dashboard runtime (which uses `torch_color` + ONNX detectors + PaddleOCR). So we split requirements into a lean runtime set and a separate `requirements-train.txt`. We revert an orphaned uncommitted micro-change in `classifiers.py`, and commit the already-good `.onnx`-preference change in `detector.py`.

**Tech Stack:** Python 3.12 (miniforge), pytest, pip requirements files.

**Scope note:** CSS identifier renames / violet-gradient removal / dead-hover removal in `visual.py` are intentionally **excluded here** — they belong to the WS1b Streamlit redesign plan, which rewrites `visual.py` wholesale. The README "15 tests" fix belongs to WS3 (content). This plan is pure non-visual code hygiene.

**Run convention:** all pytest runs use `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest`.

---

### Task 1: Split TensorFlow out of the runtime requirements

**Files:**
- Create: `main/requirements-train.txt`
- Modify: `main/requirements.txt:4`
- Test: `main/tests/test_requirements_split.py`

- [ ] **Step 1: Write the failing test**

```python
# main/tests/test_requirements_split.py
"""Guards the runtime/training dependency split (WS4).

TensorFlow conflicts with PaddleOCR at runtime, so it must live only in the
training requirements, never the runtime set the dashboard installs.
"""
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]  # .../main


def _read(name: str) -> str:
    return (MAIN / name).read_text(encoding="utf-8").lower()


def test_tensorflow_absent_from_runtime_requirements():
    assert "tensorflow" not in _read("requirements.txt")


def test_tensorflow_present_in_train_requirements():
    assert "tensorflow" in _read("requirements-train.txt")


def test_dashboard_has_no_toplevel_tensorflow_import():
    src = (MAIN / "src" / "ui" / "dashboard.py").read_text(encoding="utf-8")
    assert "import tensorflow" not in src
    assert "tf.keras" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_requirements_split.py -v`
Expected: FAIL — `test_tensorflow_absent_from_runtime_requirements` fails (TF still on line 4) and `test_tensorflow_present_in_train_requirements` fails (file missing).

- [ ] **Step 3: Create the training requirements file**

```text
# main/requirements-train.txt
# Training & offline-evaluation only (train.py, src/engine/run_evaluation.py).
# Kept OUT of the runtime requirements because TensorFlow's OpenMP runtime
# conflicts with PaddleOCR when both load in the live dashboard process.
-r requirements.txt
tensorflow>=2.10.0      # bundles Keras; used by brand/colour classifiers + trainer
```

- [ ] **Step 4: Remove TensorFlow from the runtime requirements**

In `main/requirements.txt`, delete line 4:

```text
tensorflow>=2.10.0
```

Leave the surrounding lines (`ultralytics`, `onnxruntime`, `opencv-python`, ...) untouched.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_requirements_split.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add main/requirements.txt main/requirements-train.txt main/tests/test_requirements_split.py
git commit -m "build: split TensorFlow into requirements-train (out of runtime)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Revert the orphaned `classifiers.py` micro-change

**Context:** The working tree carries an uncommitted change swapping `self.model.predict(...)` → `self.model(preprocessed, training=False).numpy()` in `BrandClassifier.predict` and `ColorClassifier.predict`. These TF classes are not on the dashboard runtime path; the change is untested and orphaned. Restore the committed version.

**Files:**
- Restore: `main/src/models/classifiers.py`

- [ ] **Step 1: Confirm the only diff is the two predict→__call__ lines**

Run: `git diff main/src/models/classifiers.py`
Expected: exactly two changed hunks, both `-  preds = self.model.predict(preprocessed, verbose=0)` → `+  preds = self.model(preprocessed, training=False).numpy()`. If anything else appears, STOP and report.

- [ ] **Step 2: Restore the committed version**

Run: `git checkout -- main/src/models/classifiers.py`

- [ ] **Step 3: Verify the working tree is clean for that file**

Run: `git diff --stat main/src/models/classifiers.py`
Expected: no output (no diff).

- [ ] **Step 4: Run the suite to confirm nothing broke**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q`
Expected: `28 passed, 5 skipped` (or current baseline + the 3 new requirements tests = `31 passed, 5 skipped`).

No commit — this only discards an uncommitted change.

---

### Task 3: Lock in the `detector.py` ONNX-preference change

**Context:** The working tree carries a good uncommitted change in `PlateDetector.__init__`: when given a `.pt` path, prefer a sibling `.onnx` if present (faster, dependency-light). The default model path also changed `"yolov8n.pt"` → `"yolov8n.onnx"`. This is already exercised by the existing detector/E2E tests. Commit it.

**Files:**
- Commit: `main/src/models/detector.py`

- [ ] **Step 1: Review the diff is only the ONNX-preference block + default path**

Run: `git diff main/src/models/detector.py`
Expected: the default `model_path` change to `"yolov8n.onnx"` and the added `if model_path.endswith(".pt"): ...` resolution block. If unrelated changes appear, STOP and report.

- [ ] **Step 2: Run the detector + E2E tests to confirm they pass with this change**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_vehicle_detector.py tests/test_plate_pipeline_e2e.py -v`
Expected: PASS (or SKIP for any test gated on optional weights) — no failures.

- [ ] **Step 3: Commit**

```bash
git add main/src/models/detector.py
git commit -m "perf(detector): prefer sibling .onnx weights over .pt at load

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire suite**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q`
Expected: `31 passed, 5 skipped` (28 baseline + 3 new requirements tests), 0 failed.

- [ ] **Step 2: Confirm the working tree only has expected leftovers**

Run: `git status --short`
Expected: `main/src/utils/visual.py` may still show as modified (left for the WS1b plan); no other unexpected modified tracked files. `classifiers.py` and `detector.py` should NOT appear (one reverted, one committed).

---

## Self-Review

**Spec coverage (spec §7 WS4):**
- "requirements.txt: remove tensorflow; add requirements-train.txt" → Task 1 ✓
- "uncommitted classifiers.py diff: revert" → Task 2 ✓
- "detector.py uncommitted diff: keep" → Task 3 (commit, with verification) ✓
- "visual.py uncommitted diff: superseded by WS1" → explicitly deferred (scope note + Task 4 Step 2) ✓
- "Identifier renames / dead hover / violet gradient" → deferred to WS1b (scope note) ✓ — these live in `visual.py` which WS1b rewrites; doing them here would collide.
- "Run pytest after to confirm green" → Task 4 ✓

**Placeholder scan:** none — every step has exact files, commands, expected output, and full file contents.

**Type/name consistency:** test file path `main/tests/test_requirements_split.py` and function names are consistent across Task 1 and Task 4. Requirements filenames (`requirements.txt`, `requirements-train.txt`) consistent throughout.

**Note for executor:** the env interpreter is miniforge base python (`/opt/homebrew/Caskroom/miniforge/base/bin/python`); plain `python` after `cd main` resolves to it in this project's shell. `KMP_DUPLICATE_LIB_OK=TRUE` is mandatory or imports hang.
