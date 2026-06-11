# Sub-project B — Parking Pipeline Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken single-stage pipeline with a 2-stage (vehicle → plate → OCR) flow that only fires a heavy decision when a vehicle has reversed into a parking ROI and stands still, voting across frames to emit one stable AUTHORIZED/MISMATCH/UNREGISTERED decision based on plate + color, plus a multi-model benchmark harness.

**Architecture:** Five focused units — `VehicleDetector` (stage 1), `ParkingTrigger` (gate, pure NumPy state machine), `PlateReader` (stage 2+3: plate detect → OCR), `DecisionEngine` (vote + DB match on plate+color), and `ParkingSession` (orchestrator the UI calls). A generic `ModelBenchmark` measures accuracy/latency/size for color-CNN and plate-detector candidates. Bug-fix bundle A (onnxruntime dep, OCR two-line regression, unused imports) is folded into Task 1.

**Tech Stack:** Python 3.12, Ultralytics YOLOv8 (ONNX), EasyOCR, TensorFlow/Keras, OpenCV, pandas, pytest. Brand classifier is removed from the decision path.

**Conventions:**
- Tests run from the `main/` directory: `cd main && python -m pytest tests/<file> -v` (the repo's `conftest.py` puts `main/` on `sys.path`, so imports are `from src.…`).
- New code uses plain dicts with documented keys (matches the existing detector/result dict style). Trigger states are string constants.
- Commit after every green task. Work happens on branch `feature/pipeline-overhaul` (already created).

---

## File Structure

**Create:**
- `main/src/models/vehicle_detector.py` — stage-1 vehicle detector (YOLO + class filter + crop).
- `main/src/models/plate_reader.py` — stage-2+3: plate detect within a vehicle crop, then OCR.
- `main/src/engine/parking_trigger.py` — pure-NumPy parking-gate state machine.
- `main/src/engine/decision_engine.py` — multi-frame vote + DB verification.
- `main/src/engine/parking_session.py` — orchestrator wiring the four units.
- `main/src/engine/benchmark.py` — generic accuracy/latency/size benchmark harness.
- `main/tests/test_parking_trigger.py`
- `main/tests/test_decision_engine.py`
- `main/tests/test_plate_reader.py`
- `main/tests/test_parking_session.py`
- `main/tests/test_vehicle_detector.py`
- `main/tests/test_benchmark.py`

**Modify:**
- `main/requirements.txt` — add `onnxruntime`.
- `main/src/models/ocr.py` — restore `readtext()` (fix two-line regression).
- `main/src/ui/dashboard.py` — remove unused imports; wire `ParkingSession` into video + webcam modes.
- `main/src/utils/matching.py` — `verify_vehicle(plate, color)` (drop brand).
- `main/tests/test_matching.py` — update to new signature.
- `main/configs/config.yaml` — add `pipeline`, `plate_detector`, `ocr.engine`.

---

## Task 1: Bug-fix bundle A (onnxruntime + OCR two-line + dead imports)

**Files:**
- Modify: `main/requirements.txt`
- Modify: `main/src/models/ocr.py:120-134`
- Modify: `main/src/ui/dashboard.py` (inside `_ensure_sample_video`)
- Test: `main/tests/test_ocr.py` (existing)

- [ ] **Step 1: Add the missing runtime dependency**

Edit `main/requirements.txt`, add under the "Core Deep Learning & Computer Vision" block (after the `ultralytics` line):

```
onnxruntime>=1.16.0
```

- [ ] **Step 2: Restore correct OCR (fixes two-line plate regression)**

In `main/src/models/ocr.py`, replace the body of the `try:` block in `read_plate` (currently lines ~120-131, the `recognize(...)` version) with the original detect+recognize call:

```python
        try:
            results = self.reader.readtext(plate_image)
        except Exception as exc:  # noqa: BLE001
            print(f"[PlateOCR] OCR inference error: {exc}")
            return ""
```

Rationale: `readtext()` runs detection then recognition, returning `(bbox, text, conf)` tuples that `_sort_and_merge` already expects for two-line Vietnamese plates. The `recognize()` single-box variant squashed both lines into one strip.

- [ ] **Step 3: Remove dead imports in dashboard**

In `main/src/ui/dashboard.py`, inside `_ensure_sample_video`, delete these two unused lines:

```python
            from urllib.error import HTTPError
            from pathlib import Path
```

(`Path` is already imported at module top; `HTTPError` is never referenced — the loop catches bare `Exception`.)

- [ ] **Step 4: Run the existing OCR test to verify nothing broke**

Run: `cd main && python -m pytest tests/test_ocr.py -v`
Expected: PASS (the test mocks/uses the reader; `_sort_and_merge`/`_clean_text` behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add main/requirements.txt main/src/models/ocr.py main/src/ui/dashboard.py
git commit -m "fix: add onnxruntime dep, restore readtext OCR, drop dead imports"
```

---

## Task 2: `verify_vehicle(plate, color)` — drop brand from the decision

**Files:**
- Modify: `main/src/utils/matching.py:26-80`
- Modify: `main/tests/test_matching.py`

- [ ] **Step 1: Update the matcher test to the new signature (write failing test first)**

Replace the whole body of `main/tests/test_matching.py` with:

```python
import unittest
import os
import pandas as pd
from src.utils.matching import DatabaseMatcher


class TestDatabaseMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_db_path = "tests_temp_database.csv"
        db_data = {
            'license_plate': ['30F-12345', '51G-67890'],
            'car_brand': ['Toyota Vios', 'Hyundai Accent'],
            'car_color': ['White', 'Black'],
        }
        pd.DataFrame(db_data).to_csv(cls.temp_db_path, index=False)
        cls.matcher = DatabaseMatcher(cls.temp_db_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db_path):
            os.remove(cls.temp_db_path)

    def test_exact_match(self):
        result = self.matcher.verify_vehicle("30F-12345", "White")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertEqual(result['action'], 'ALLOW')

    def test_case_and_symbol_insensitivity(self):
        result = self.matcher.verify_vehicle("30f - 123.45", "white")
        self.assertEqual(result['status'], 'AUTHORIZED')

    def test_unregistered_plate(self):
        result = self.matcher.verify_vehicle("30H-99999", "Blue")
        self.assertEqual(result['status'], 'UNREGISTERED')
        self.assertEqual(result['action'], 'DENY_ALERT')

    def test_color_mismatch(self):
        result = self.matcher.verify_vehicle("30F-12345", "Black")
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(result['action'], 'DENY_ALERT')
        self.assertIn("Color Mismatch", result['message'])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_matching.py -v`
Expected: FAIL with `TypeError` (verify_vehicle still requires 3 args).

- [ ] **Step 3: Update `verify_vehicle` to plate + color**

In `main/src/utils/matching.py`, replace the `verify_vehicle` method (lines 26-80) with:

```python
    def verify_vehicle(self, detected_plate: str, detected_color: str) -> dict:
        """Verify a detected vehicle against the registered database.

        Matching is plate-first; colour is a secondary verification layer
        to catch a real plate cloned onto a different vehicle.

        Args:
            detected_plate: The recognized plate sequence.
            detected_color: The classified car colour.

        Returns:
            dict with 'status', 'action', and 'message' keys.
        """
        if self.db is None:
            return {'status': 'ERROR', 'action': 'DENY', 'message': 'Database not loaded'}

        clean_plate = str(detected_plate).replace(' ', '').replace('-', '').replace('.', '').upper()
        clean_color = str(detected_color).strip().upper()

        record = self.db[self.db['license_plate'] == clean_plate]

        if record.empty:
            return {
                'status': 'UNREGISTERED',
                'action': 'DENY_ALERT',
                'message': f"Plate {detected_plate} is not registered in the system.",
            }

        registered_color = record.iloc[0]['car_color']
        color_match = clean_color == registered_color

        if color_match:
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': f"Vehicle {detected_plate} authorized: Match confirmed.",
            }
        return {
            'status': 'MISMATCH',
            'action': 'DENY_ALERT',
            'message': f"Color Mismatch (Detected: {detected_color}, Registered: {record.iloc[0]['car_color']})",
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_matching.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/utils/matching.py main/tests/test_matching.py
git commit -m "feat: matcher verifies on plate + color (drop brand from decision)"
```

---

## Task 3: `ParkingTrigger` — parking-gate state machine

**Files:**
- Create: `main/src/engine/parking_trigger.py`
- Test: `main/tests/test_parking_trigger.py`

State constants: `IDLE`, `TRACKING`, `READY_TO_DECIDE`, `DECIDED`. The gate opens only when the largest vehicle is big enough (`area_ratio ≥ min_area_ratio`), sits inside the parking ROI (default middle-bottom of frame — where a reversing car ends up), and has held still for `stable_frames` samples. "Reversed in" is captured by the size + low-ROI precondition, so no optical flow is needed.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_parking_trigger.py`:

```python
from src.engine.parking_trigger import ParkingTrigger, IDLE, TRACKING, READY_TO_DECIDE, DECIDED

FRAME = (480, 640)  # H, W


def _det(x1, y1, x2, y2):
    return [{"bbox": (x1, y1, x2, y2), "conf": 0.9}]


def test_idle_when_no_detection():
    t = ParkingTrigger()
    assert t.update([], FRAME) == IDLE


def test_idle_when_vehicle_too_small_or_outside_roi():
    t = ParkingTrigger()
    # tiny box top-left: fails area + ROI
    assert t.update(_det(0, 0, 40, 40), FRAME) == IDLE


def test_tracking_then_ready_when_large_low_and_stable():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=3, move_eps=0.02)
    # big box centered low (cx≈0.5, cy≈0.75, area≈0.25)
    box = _det(220, 240, 420, 480)
    states = [t.update(box, FRAME) for _ in range(3)]
    assert states[0] == TRACKING
    assert states[-1] == READY_TO_DECIDE


def test_jitter_keeps_tracking_not_ready():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=3, move_eps=0.01)
    boxes = [_det(220, 240, 420, 480), _det(120, 140, 320, 380), _det(260, 260, 460, 500)]
    states = [t.update(b, FRAME) for b in boxes]
    assert states[-1] == TRACKING  # moved too much to be "parked"


def test_mark_decided_and_reset_on_leave():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05)
    box = _det(220, 240, 420, 480)
    t.update(box, FRAME)
    t.update(box, FRAME)
    assert t.state == READY_TO_DECIDE
    t.mark_decided()
    assert t.update(box, FRAME) == DECIDED   # stays decided while parked
    assert t.update([], FRAME) == IDLE       # car gone -> reset
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_parking_trigger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.engine.parking_trigger'`.

- [ ] **Step 3: Implement `ParkingTrigger`**

Create `main/src/engine/parking_trigger.py`:

```python
"""Parking-gate state machine.

Pure NumPy heuristic that decides when a vehicle has reversed into the
parking ROI and stands still, so the heavy plate/OCR/colour pipeline only
runs at the right moment. No model dependency — unit-testable with fake
bounding boxes.
"""

from __future__ import annotations

from typing import Any

IDLE = "IDLE"
TRACKING = "TRACKING"
READY_TO_DECIDE = "READY_TO_DECIDE"
DECIDED = "DECIDED"

# Default ROI in normalized coords (x_min, y_min, x_max, y_max): middle-bottom.
_DEFAULT_ROI = (0.2, 0.4, 0.8, 1.0)


class ParkingTrigger:
    """Gate that opens when a vehicle is parked inside the ROI.

    Attributes:
        state: Current state (IDLE/TRACKING/READY_TO_DECIDE/DECIDED).
    """

    def __init__(
        self,
        roi: tuple[float, float, float, float] | None = None,
        min_area_ratio: float = 0.15,
        stable_frames: int = 5,
        move_eps: float = 0.02,
    ) -> None:
        self.roi = roi if roi is not None else _DEFAULT_ROI
        self.min_area_ratio = min_area_ratio
        self.stable_frames = stable_frames
        self.move_eps = move_eps
        self.state: str = IDLE
        self._centers: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.state = IDLE
        self._centers = []

    def mark_decided(self) -> None:
        self.state = DECIDED

    def update(self, detections: list[dict[str, Any]], frame_shape) -> str:
        veh = self._largest(detections)
        if veh is None:
            self.reset()
            return self.state

        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = veh["bbox"]
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h

        if area_ratio < self.min_area_ratio or not self._in_roi(cx, cy):
            self.reset()
            return self.state

        if self.state == DECIDED:
            return self.state  # latch until the car leaves

        self._centers.append((cx, cy))
        if len(self._centers) > self.stable_frames:
            self._centers.pop(0)

        if len(self._centers) >= self.stable_frames and self._is_stable():
            self.state = READY_TO_DECIDE
        else:
            self.state = TRACKING
        return self.state

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _largest(detections: list[dict[str, Any]]):
        if not detections:
            return None
        return max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

    def _in_roi(self, cx: float, cy: float) -> bool:
        x0, y0, x1, y1 = self.roi
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _is_stable(self) -> bool:
        xs = [c[0] for c in self._centers]
        ys = [c[1] for c in self._centers]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        return all(
            abs(x - mx) <= self.move_eps and abs(y - my) <= self.move_eps
            for x, y in self._centers
        )
```

Also create an empty `main/src/engine/__init__.py` if it does not already exist (it does — `evaluator.py` lives there — so skip if present).

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_parking_trigger.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/parking_trigger.py main/tests/test_parking_trigger.py
git commit -m "feat: ParkingTrigger parking-gate state machine"
```

---

## Task 4: `DecisionEngine` — multi-frame vote + verification

**Files:**
- Create: `main/src/engine/decision_engine.py`
- Test: `main/tests/test_decision_engine.py`

Aggregates per-frame `{plate_text, color}` readings: plate = strict-majority vote (tie → `UNCERTAIN`), colour = mode, then calls the injected matcher's `verify_vehicle(plate, color)`.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_decision_engine.py`:

```python
from src.engine.decision_engine import DecisionEngine


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        if plate == "30F12345" and color == "WHITE":
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}


def _engine():
    return DecisionEngine(matcher=FakeMatcher())


def test_no_plate_when_all_empty():
    d = _engine().aggregate([{"plate_text": "", "color": "WHITE"}])
    assert d["status"] == "NO_PLATE"
    assert d["action"] == "LOG"


def test_majority_vote_authorized():
    frames = [
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "30F12340", "color": "WHITE"},
    ]
    d = _engine().aggregate(frames)
    assert d["plate"] == "30F12345"
    assert d["color"] == "WHITE"
    assert d["status"] == "AUTHORIZED"


def test_tie_is_uncertain():
    frames = [
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "99X99999", "color": "WHITE"},
    ]
    d = _engine().aggregate(frames)
    assert d["status"] == "UNCERTAIN"
    assert d["action"] == "LOG"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_decision_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `DecisionEngine`**

Create `main/src/engine/decision_engine.py`:

```python
"""Multi-frame decision aggregation.

Votes plate/colour across the collected frames, then verifies the winning
pair against the registration database. Keeps the heavy models out: it
only consumes already-extracted per-frame readings.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class DecisionEngine:
    """Aggregates per-frame readings into one stable decision.

    Args:
        matcher: Object exposing ``verify_vehicle(plate, color) -> dict``.
    """

    def __init__(self, matcher) -> None:
        self.matcher = matcher

    def aggregate(self, frames_data: list[dict[str, Any]]) -> dict[str, Any]:
        plates = [
            str(f.get("plate_text", "")).strip()
            for f in frames_data
            if str(f.get("plate_text", "")).strip()
        ]
        if not plates:
            return {
                "plate": "",
                "color": "",
                "status": "NO_PLATE",
                "action": "LOG",
                "message": "No readable plate across sampled frames.",
                "votes_meta": {"plate_votes": {}, "n_frames": len(frames_data)},
            }

        plate_counts = Counter(plates)
        top = plate_counts.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            return {
                "plate": "",
                "color": "",
                "status": "UNCERTAIN",
                "action": "LOG",
                "message": "Plate votes did not converge; ask the vehicle to re-park.",
                "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
            }

        plate = top[0][0]
        colors = [
            str(f.get("color", "")).strip().upper()
            for f in frames_data
            if str(f.get("color", "")).strip()
        ]
        color = Counter(colors).most_common(1)[0][0] if colors else ""

        verdict = self.matcher.verify_vehicle(plate, color)
        return {
            "plate": plate,
            "color": color,
            "status": verdict["status"],
            "action": verdict["action"],
            "message": verdict["message"],
            "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_decision_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/decision_engine.py main/tests/test_decision_engine.py
git commit -m "feat: DecisionEngine multi-frame vote + verification"
```

---

## Task 5: `PlateReader` — plate detect → OCR (stage 2+3)

**Files:**
- Create: `main/src/models/plate_reader.py`
- Test: `main/tests/test_plate_reader.py`

Takes an injected plate detector (exposing `detect(crop) -> list[{bbox, conf, crop}]`) and OCR reader (exposing `read_plate(img) -> str`). Picks the highest-confidence plate box, OCRs it; if no plate box is found, OCRs the whole vehicle crop as a fallback.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_plate_reader.py`:

```python
import numpy as np
from src.models.plate_reader import PlateReader


class FakePlateDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, crop):
        return self._dets


class FakeOCR:
    def __init__(self, text):
        self._text = text

    def read_plate(self, img):
        return self._text


def _crop():
    return np.zeros((50, 100, 3), dtype=np.uint8)


def test_reads_best_plate_box():
    dets = [
        {"bbox": (1, 1, 9, 9), "conf": 0.4, "crop": _crop()},
        {"bbox": (2, 2, 8, 8), "conf": 0.8, "crop": _crop()},
    ]
    reader = PlateReader(FakePlateDetector(dets), FakeOCR("51F12345"))
    out = reader.read(_crop())
    assert out["text"] == "51F12345"
    assert out["conf"] == 0.8
    assert out["plate_bbox"] == (2, 2, 8, 8)


def test_fallback_to_whole_crop_when_no_plate_box():
    reader = PlateReader(FakePlateDetector([]), FakeOCR("30A99999"))
    out = reader.read(_crop())
    assert out["text"] == "30A99999"
    assert out["plate_bbox"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_plate_reader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `PlateReader`**

Create `main/src/models/plate_reader.py`:

```python
"""Plate reading: detect the plate inside a vehicle crop, then OCR it.

Dependencies are injected (plate detector + OCR reader) so the merge logic
is unit-testable without loading real models.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PlateReader:
    """Stage 2+3 of the pipeline.

    Args:
        plate_detector: Object exposing ``detect(crop) -> list[dict]`` where
            each dict has ``bbox``, ``conf``, ``crop`` keys.
        ocr_reader: Object exposing ``read_plate(img) -> str``.
    """

    def __init__(self, plate_detector, ocr_reader) -> None:
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop: np.ndarray) -> dict[str, Any]:
        dets = self.plate_detector.detect(vehicle_crop)
        if not dets:
            text = self.ocr_reader.read_plate(vehicle_crop)
            return {"text": text, "conf": 0.0, "plate_bbox": None}

        best = max(dets, key=lambda d: d["conf"])
        text = self.ocr_reader.read_plate(best["crop"])
        return {"text": text, "conf": best["conf"], "plate_bbox": best["bbox"]}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_plate_reader.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/models/plate_reader.py main/tests/test_plate_reader.py
git commit -m "feat: PlateReader (plate detect -> OCR) with whole-crop fallback"
```

---

## Task 6: `VehicleDetector` — stage 1 (YOLO + class filter + crop)

**Files:**
- Create: `main/src/models/vehicle_detector.py`
- Test: `main/tests/test_vehicle_detector.py`

Wraps Ultralytics YOLO, keeps only vehicle COCO classes (car=2, bus=5, truck=7), and returns `{bbox, conf, crop}` dicts. The same class is reused for the plate detector by passing a plate model and `vehicle_classes=None` (keep all classes).

- [ ] **Step 1: Write the failing test (skips cleanly without runtime/model)**

Create `main/tests/test_vehicle_detector.py`:

```python
import os
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("ultralytics")

MODEL = os.path.join("data", "models", "yolov8n.onnx")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="ONNX model not present")
def test_detect_returns_well_formed_list():
    from src.models.vehicle_detector import VehicleDetector

    det = VehicleDetector(model_path=MODEL, conf=0.25)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert isinstance(out, list)
    for d in out:
        assert set(("bbox", "conf", "crop")).issubset(d.keys())
        assert len(d["bbox"]) == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_vehicle_detector.py -v`
Expected: FAIL with `ModuleNotFoundError` (or SKIP if onnxruntime/model absent — if it skips, the implementation step still proceeds).

- [ ] **Step 3: Implement `VehicleDetector`**

Create `main/src/models/vehicle_detector.py`:

```python
"""Stage-1 vehicle detector.

Thin Ultralytics YOLO wrapper that filters to vehicle classes and returns
crops for downstream plate-reading and colour classification. Reused for
the dedicated plate model by passing ``vehicle_classes=None``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("ultralytics is required for VehicleDetector.") from exc

_VEHICLE_CLASSES = (2, 5, 7)  # COCO: car, bus, truck


class VehicleDetector:
    """Detect vehicles (or any object, if ``vehicle_classes=None``)."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.3,
        vehicle_classes: tuple[int, ...] | None = _VEHICLE_CLASSES,
        crop_padding: float = 0.02,
    ) -> None:
        self.conf = conf
        self.vehicle_classes = vehicle_classes
        self.crop_padding = crop_padding
        self.model: YOLO | None = None
        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[VehicleDetector] WARNING: could not load '{model_path}': {exc}")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=frame, conf=self.conf, device="cpu", verbose=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[VehicleDetector] inference error: {exc}")
            return []

        out: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if self.vehicle_classes is not None and cls_id not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                out.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf,
                        "crop": self._crop(frame, x1, y1, x2, y2),
                    }
                )
        return out

    def _crop(self, image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * self.crop_padding)
        pad_y = int((y2 - y1) * self.crop_padding)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return image[cy1:cy2, cx1:cx2].copy()
```

- [ ] **Step 4: Run to verify it passes (or skips cleanly)**

Run: `cd main && python -m pytest tests/test_vehicle_detector.py -v`
Expected: PASS or SKIP (never ERROR/FAIL).

- [ ] **Step 5: Commit**

```bash
git add main/src/models/vehicle_detector.py main/tests/test_vehicle_detector.py
git commit -m "feat: VehicleDetector stage-1 (class filter + crop)"
```

---

## Task 7: `ParkingSession` — orchestrator

**Files:**
- Create: `main/src/engine/parking_session.py`
- Test: `main/tests/test_parking_session.py`

Wires the four units. Every `process_frame` call increments an internal counter; heavy work runs only on sampled frames (`frame % sample_interval == 0`). On a sampled frame: run vehicle detection → `trigger.update`. While `READY_TO_DECIDE` (and not yet decided), collect `collect_frames` per-frame readings (plate via `PlateReader`, colour via the colour classifier on the largest vehicle crop); once enough are collected, aggregate → cache decision → `trigger.mark_decided`. The cached decision is returned until the car leaves and the trigger resets. All collaborators are injected for testability.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_parking_session.py`:

```python
import numpy as np
from src.engine.parking_session import ParkingSession


class FakeVehicleDetector:
    def detect(self, frame):
        # one big, low, centered vehicle
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((10, 10, 3), dtype=np.uint8)}]


class FakePlateReader:
    def read(self, crop):
        return {"text": "30F12345", "conf": 0.9, "plate_bbox": (0, 0, 5, 5)}


class FakeColorClf:
    def predict(self, crop):
        return ("White", 0.95)


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}


def _session():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    return ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=FakePlateReader(),
        color_clf=FakeColorClf(),
        decision_engine=DecisionEngine(FakeMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05),
        sample_interval=1,
        collect_frames=2,
    )


def test_eventually_produces_one_authorized_decision():
    sess = _session()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    decision = None
    for _ in range(10):
        out = sess.process_frame(frame)
        if out["decision"] is not None:
            decision = out["decision"]
            break
    assert decision is not None
    assert decision["status"] == "AUTHORIZED"
    assert decision["plate"] == "30F12345"


def test_non_sampled_frames_skip_heavy_work():
    sess = _session()
    sess.sample_interval = 5
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = sess.process_frame(frame)  # frame 1, not a multiple of 5
    assert out["decision"] is None
    assert out["state"] in ("IDLE",)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_parking_session.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ParkingSession`**

Create `main/src/engine/parking_session.py`:

```python
"""Parking-session orchestrator.

Drives one parked-vehicle decision from a frame stream. Heavy models run
only on sampled frames and only once the ParkingTrigger gate opens.
Collaborators are injected so the control flow is unit-testable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engine.parking_trigger import READY_TO_DECIDE, DECIDED, IDLE


class ParkingSession:
    def __init__(
        self,
        vehicle_detector,
        plate_reader,
        color_clf,
        decision_engine,
        trigger,
        sample_interval: int = 5,
        collect_frames: int = 5,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_reader = plate_reader
        self.color_clf = color_clf
        self.decision_engine = decision_engine
        self.trigger = trigger
        self.sample_interval = sample_interval
        self.collect_frames = collect_frames

        self._frame_idx = 0
        self._collected: list[dict[str, Any]] = []
        self._decision: dict[str, Any] | None = None
        self._last_detections: list[dict[str, Any]] = []

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        self._frame_idx += 1
        if self._frame_idx % self.sample_interval != 0:
            return self._output()

        detections = self.vehicle_detector.detect(frame)
        self._last_detections = detections
        state = self.trigger.update(detections, frame.shape)

        if state == IDLE:
            self._collected = []
            self._decision = None
        elif state == READY_TO_DECIDE:
            self._collect(detections)
            if len(self._collected) >= self.collect_frames:
                self._decision = self.decision_engine.aggregate(self._collected)
                self.trigger.mark_decided()

        return self._output()

    def _collect(self, detections: list[dict[str, Any]]) -> None:
        if not detections:
            return
        veh = max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )
        plate = self.plate_reader.read(veh["crop"])
        color, _conf = self.color_clf.predict(veh["crop"])
        self._collected.append({"plate_text": plate["text"], "color": color})

    def _output(self) -> dict[str, Any]:
        return {
            "state": self.trigger.state,
            "overlay_results": self._last_detections,
            "decision": self._decision,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_parking_session.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/parking_session.py main/tests/test_parking_session.py
git commit -m "feat: ParkingSession orchestrator (sampling + gated decision)"
```

---

## Task 8: `ModelBenchmark` — accuracy/latency/size harness

**Files:**
- Create: `main/src/engine/benchmark.py`
- Test: `main/tests/test_benchmark.py`

Generic harness: each candidate exposes `name`, `predict(X) -> np.ndarray` (class indices), `num_params`, `size_mb`. `run` measures top-1 accuracy and mean CPU latency per sample; `to_report` renders a Markdown table. The real color-CNN / plate-detector candidates are built by a separate thin script (`scripts/benchmark_color.py`, documented below) — the harness itself stays model-agnostic and fully unit-tested.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_benchmark.py`:

```python
import numpy as np
from src.engine.benchmark import ModelBenchmark, BenchmarkCandidate


class ConstModel(BenchmarkCandidate):
    def __init__(self, name, cls):
        self.name = name
        self.num_params = 1000
        self.size_mb = 0.1
        self._cls = cls

    def predict(self, X):
        return np.full(len(X), self._cls, dtype=int)


def test_run_builds_dataframe_with_expected_columns():
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    bench = ModelBenchmark()
    df = bench.run([ConstModel("always0", 0), ConstModel("always1", 1)], X, y)
    assert list(df["name"]) == ["always0", "always1"]
    for col in ("accuracy", "latency_ms", "num_params", "size_mb"):
        assert col in df.columns
    # always0 gets 2/4 correct
    assert abs(df.loc[df["name"] == "always0", "accuracy"].iloc[0] - 0.5) < 1e-9


def test_to_report_is_markdown_table():
    X = np.zeros((2, 3), dtype=np.float32)
    y = np.array([0, 1])
    bench = ModelBenchmark()
    df = bench.run([ConstModel("m", 0)], X, y)
    md, _plots = bench.to_report(df)
    assert "| name" in md and "accuracy" in md
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ModelBenchmark`**

Create `main/src/engine/benchmark.py`:

```python
"""Generic model-comparison harness.

Measures top-1 accuracy, mean CPU latency, parameter count, and on-disk
size for a set of candidates over a labelled dataset, and renders a
Markdown comparison table. Model-agnostic: candidates just implement the
``BenchmarkCandidate`` protocol.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd


class BenchmarkCandidate(Protocol):
    name: str
    num_params: int
    size_mb: float

    def predict(self, X: np.ndarray) -> np.ndarray:  # class indices
        ...


class ModelBenchmark:
    """Run accuracy/latency/size comparison across candidates."""

    def run(self, candidates, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        rows = []
        y = np.asarray(y)
        for c in candidates:
            t0 = time.perf_counter()
            preds = np.asarray(c.predict(X))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            accuracy = float((preds == y).mean()) if len(y) else 0.0
            latency_ms = elapsed_ms / max(1, len(X))
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            rows.append(
                {
                    "name": c.name,
                    "accuracy": round(accuracy, 4),
                    "latency_ms": round(latency_ms, 3),
                    "fps": round(fps, 1),
                    "num_params": int(c.num_params),
                    "size_mb": round(float(c.size_mb), 2),
                }
            )
        return pd.DataFrame(rows)

    def to_report(self, df: pd.DataFrame) -> tuple[str, list[str]]:
        md = df.to_markdown(index=False)
        return md, []
```

Note: `DataFrame.to_markdown` needs the `tabulate` package. Add it to `main/requirements.txt` under "Data Processing & Utilities":

```
tabulate>=0.9.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && python -m pytest tests/test_benchmark.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/benchmark.py main/tests/test_benchmark.py main/requirements.txt
git commit -m "feat: ModelBenchmark accuracy/latency/size harness"
```

---

## Task 9: Wire `ParkingSession` into the dashboard + config

**Files:**
- Modify: `main/configs/config.yaml`
- Modify: `main/src/ui/dashboard.py` (model-loading block ~88-153; video block ~448-540; webcam block ~542+)

- [ ] **Step 1: Add pipeline config**

Append to `main/configs/config.yaml`:

```yaml
# Two-stage parking pipeline
pipeline:
  frame_sample_interval: 5
  collect_frames: 5
  trigger:
    min_area_ratio: 0.15
    stable_frames: 5
    move_eps: 0.02
    roi: null   # null => default middle-bottom ROI

# Dedicated license-plate detector (stage 2)
plate_detector:
  model_name: "plate_yolov8n.onnx"
  conf_threshold: 0.3

# OCR engine selection
ocr:
  languages: ["en"]
  gpu: false
  engine: "easyocr"   # easyocr | ppocr
```

- [ ] **Step 2: Build a `ParkingSession` in the model loader**

In `main/src/ui/dashboard.py`, at the end of `_load_models` (just before `st.session_state["models"] = models`), add construction of the session from already-loaded parts. Insert:

```python
    # Two-stage parking session (best-effort; UI still works if parts missing)
    try:
        from src.models.vehicle_detector import VehicleDetector
        from src.models.plate_reader import PlateReader
        from src.engine.parking_trigger import ParkingTrigger
        from src.engine.decision_engine import DecisionEngine
        from src.engine.parking_session import ParkingSession

        pcfg = cfg.get("pipeline", {})
        tcfg = pcfg.get("trigger", {})
        det_model = str(_PROJECT_ROOT / cfg["paths"]["model_save_dir"] / cfg["detector"]["model_name"])
        plate_model = str(_PROJECT_ROOT / cfg["paths"]["model_save_dir"] / cfg["plate_detector"]["model_name"])

        vehicle_det = VehicleDetector(model_path=det_model, conf=cfg["detector"].get("conf_threshold", 0.3))
        plate_det = VehicleDetector(
            model_path=plate_model,
            conf=cfg["plate_detector"].get("conf_threshold", 0.3),
            vehicle_classes=None,
        )
        if models.get("ocr") is not None and models.get("color_clf") is not None and models.get("matcher") is not None:
            session = ParkingSession(
                vehicle_detector=vehicle_det,
                plate_reader=PlateReader(plate_det, models["ocr"]),
                color_clf=models["color_clf"],
                decision_engine=DecisionEngine(models["matcher"]),
                trigger=ParkingTrigger(
                    roi=tcfg.get("roi"),
                    min_area_ratio=tcfg.get("min_area_ratio", 0.15),
                    stable_frames=tcfg.get("stable_frames", 5),
                    move_eps=tcfg.get("move_eps", 0.02),
                ),
                sample_interval=pcfg.get("frame_sample_interval", 5),
                collect_frames=pcfg.get("collect_frames", 5),
            )
            models["session"] = session
    except Exception:
        logger.exception("ParkingSession construction failed.")
```

- [ ] **Step 3: Use the session in the video loop**

In the video block of `main/src/ui/dashboard.py`, replace the per-frame `_run_pipeline(...)` call and its result handling inside the `while cap.isOpened() and not stop:` loop with the session-driven version:

```python
                    session = models.get("session")
                    if session is not None:
                        out = session.process_frame(frame)
                        for d in out["overlay_results"]:
                            x1, y1, x2, y2 = d["bbox"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 135, 90), 2)
                        cv2.putText(
                            frame, f"STATE: {out['state']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 135, 90), 2,
                        )
                        if out["decision"] is not None:
                            dec = out["decision"]
                            st.session_state["total_processed"] += 1
                            if dec["status"] in ("MISMATCH", "UNREGISTERED"):
                                st.session_state["alert_count"] += 1
                            cv2.putText(
                                frame, f"{dec['status']}: {dec['plate']}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (222, 53, 11), 2,
                            )

                    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(display_rgb, use_container_width=True)
```

Remove the now-unused `_current_results, _current_latency = _run_pipeline(...)` block and its `draw_detection_overlay` call within this loop. (Leave `_run_pipeline` and `draw_detection_overlay` defined for the still-image "Upload Image" mode.)

- [ ] **Step 4: Mirror the same handling in the Webcam block**

In the `elif mode == "Webcam":` block, after `cam_input = st.camera_input(...)`, when a frame is captured decode it to `frame` (BGR) and run the same `session.process_frame(frame)` overlay code from Step 3 (single frame, no loop). Use this exact adaptation:

```python
        if cam_input is not None:
            file_bytes = np.frombuffer(cam_input.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            session = models.get("session")
            if session is not None and frame is not None:
                out = session.process_frame(frame)
                for d in out["overlay_results"]:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 135, 90), 2)
                if out["decision"] is not None:
                    dec = out["decision"]
                    cv2.putText(
                        frame, f"{dec['status']}: {dec['plate']}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (222, 53, 11), 2,
                    )
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
```

- [ ] **Step 5: Smoke-run the full test suite**

Run: `cd main && python -m pytest -v`
Expected: PASS (existing tests + all new unit tests; `test_vehicle_detector` may SKIP). Dashboard is import-only here — no Streamlit run in CI.

- [ ] **Step 6: Manual UI check (documented, not automated)**

Run: `cd main && bash run_ui.sh` (macOS/Linux) or `streamlit run src/ui/dashboard.py`. Choose "Upload Video" → "Play Default Parking Video"; confirm the overlay shows `STATE:` transitions and a single decision banner once a vehicle settles. If the plate model `plate_yolov8n.onnx` is absent, the session is not built and the UI still runs other modes — this is expected until the model is added (see Notes).

- [ ] **Step 7: Commit**

```bash
git add main/configs/config.yaml main/src/ui/dashboard.py
git commit -m "feat: wire ParkingSession into video + webcam UI modes"
```

---

## Task 10: Benchmark scripts for the real candidates (color CNN + plate detector)

**Files:**
- Create: `main/scripts/benchmark_color.py`
- Create: `main/scripts/benchmark_plate.py`

These are thin runnable drivers that construct real candidates and call the tested `ModelBenchmark`. They are smoke-checked by running with a tiny `--limit`, not unit-tested (training/real models are environment-heavy).

- [ ] **Step 1: Color-CNN benchmark driver**

Create `main/scripts/benchmark_color.py`:

```python
"""Benchmark color-classifier backbones (Group A).

Wraps each Keras backbone in a BenchmarkCandidate and compares accuracy,
latency, params, and size on the processed color dataset. Writes
docs/benchmarks/color_benchmark.{csv,md}.
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from src.engine.benchmark import ModelBenchmark


class KerasCandidate:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.num_params = model.count_params()
        self.size_mb = round(self.num_params * 4 / 1e6, 2)  # float32 estimate

    def predict(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)


def build_candidates(input_shape, num_classes):
    import tensorflow as tf
    from tensorflow import keras

    def head(base):
        inp = keras.Input(shape=input_shape)
        x = base(inp, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inp, out)

    return [
        KerasCandidate("MobileNetV3Small", head(tf.keras.applications.MobileNetV3Small(include_top=False, weights=None, input_shape=input_shape))),
        KerasCandidate("EfficientNetB0", head(tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape))),
        KerasCandidate("ResNet50", head(tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap samples for smoke runs")
    args = ap.parse_args()

    # Smoke dataset if no processed data wired yet.
    X = np.random.rand(args.limit or 8, 224, 224, 3).astype("float32")
    y = np.random.randint(0, 8, size=len(X))

    cands = build_candidates((224, 224, 3), 8)
    bench = ModelBenchmark()
    df = bench.run(cands, X, y)
    md, _ = bench.to_report(df)

    os.makedirs(os.path.join("..", "docs", "benchmarks"), exist_ok=True)
    out_dir = os.path.join("..", "docs", "benchmarks")
    df.to_csv(os.path.join(out_dir, "color_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, "color_benchmark.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the color driver**

Run: `cd main && python scripts/benchmark_color.py --limit 4`
Expected: prints a Markdown table; writes `docs/benchmarks/color_benchmark.csv` and `.md`. (Random smoke data — numbers are not meaningful; this only proves the harness wiring.)

- [ ] **Step 3: Plate-detector benchmark driver**

Create `main/scripts/benchmark_plate.py`:

```python
"""Benchmark plate detectors (Group B): pretrained-finetune vs trained.

Compares two YOLO plate models on a labelled validation set using
Ultralytics' built-in mAP, plus latency/size. Writes
docs/benchmarks/plate_benchmark.{csv,md}.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd


def measure(model_path: str, data_yaml: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device="cpu", verbose=False)
    size_mb = round(os.path.getsize(model_path) / 1e6, 2) if os.path.exists(model_path) else 0.0
    # crude latency probe on the val set is provided by metrics.speed (ms)
    speed = getattr(metrics, "speed", {}) or {}
    latency_ms = round(float(speed.get("inference", 0.0)), 3)
    return {
        "name": os.path.basename(model_path),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "latency_ms": latency_ms,
        "size_mb": size_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to plate data.yaml")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--trained", required=True)
    args = ap.parse_args()

    rows = [measure(args.pretrained, args.data), measure(args.trained, args.data)]
    df = pd.DataFrame(rows)
    out_dir = os.path.join("..", "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "plate_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, "plate_benchmark.md"), "w") as fh:
        fh.write(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add main/scripts/benchmark_color.py main/scripts/benchmark_plate.py
git commit -m "feat: benchmark drivers for color CNN and plate detector"
```

---

## Notes & Follow-ups (not blocking B)

- **Plate model artifact:** `plate_yolov8n.onnx` must be produced (pretrained-finetune or trained-from-scratch via `benchmark_plate.py` winner) and dropped in `main/data/models/`. Until then `ParkingSession` degrades gracefully (not built; other UI modes still work).
- **Real benchmark datasets:** wire `benchmark_color.py` to the processed color dataset (`src/datasets/vehicle_dataset.py`) replacing the random smoke data once Group-A training is run.
- **Downstream:** Sub-project E (repo split + docs/README sync) and D (slide audit) consume `docs/benchmarks/*` and the new architecture — they get their own plans.

---

## Self-Review

- **Spec coverage:** 2-stage detection (Tasks 5,6) ✓; PA1 trigger (Task 3) ✓; plate+color decision & `verify_vehicle` change (Tasks 2,4) ✓; sampling + gated heavy work (Task 7,9) ✓; vote/UNCERTAIN/NO_PLATE error handling (Task 4) ✓; benchmark A+B (Tasks 8,10) ✓; bug-fix A (Task 1) ✓; config (Task 9) ✓; UI integration video+webcam (Task 9) ✓. Brand removed from decision (Tasks 2,9) ✓.
- **Placeholder scan:** none — every code step has full code; smoke datasets are explicitly labelled as such.
- **Type consistency:** detection dicts use `{bbox, conf, crop}` everywhere (VehicleDetector, fakes, PlateReader, ParkingSession); decision dicts use `{plate, color, status, action, message, votes_meta}` (DecisionEngine ↔ session ↔ UI); trigger states `IDLE/TRACKING/READY_TO_DECIDE/DECIDED` imported from one module; `verify_vehicle(plate, color)` signature matches across matcher, DecisionEngine fakes, and tests.
