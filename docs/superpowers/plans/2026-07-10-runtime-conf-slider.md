# Runtime Detection-Confidence Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sidebar "Detection Confidence" slider actually control stage-1 vehicle-detection confidence at runtime in Upload Image, Webcam, and Upload Video modes, while the API `/verify` endpoint keeps its config-default behavior exactly.

**Architecture:** Per-call confidence override threaded as an optional kwarg: `VehicleDetector.detect(frame, conf=None)` → `infer_single_image(..., conf_override=None)` and `ParkingSession.process_frame(frame, conf_override=None)` → dashboard passes the slider value. The kwarg is forwarded **only when an override is supplied**, so every existing caller (API, tests with fakes whose `detect(self, frame)` has no conf param) keeps the identical call `detect(frame)`. `infer_single_image` additionally returns the chosen vehicle's bbox (`vehicle_bbox`) so image mode can draw the detection box — making the slider's effect visible (box at low conf, gone at high conf). No pipeline rebuild, no mutation of shared detector state.

**Tech Stack:** Python 3.12 (miniforge), Streamlit 1.35.0, Ultralytics YOLO (ONNX), pytest.

**Non-goals (do NOT do):**
- Do NOT touch `main/src/api/app.py` — `/verify` must keep calling `infer_single_image(image, pipeline, cfg)` with no override.
- Do NOT change the plate detector's threshold (`plate_detector.conf_threshold` stays config-fixed; the slider is stage-1 vehicle detection only).
- Do NOT remove the empty-detections → whole-image fallback in `infer_single_image`.
- Do NOT pass `conf=None` explicitly to `detect()` — call `detect(frame)` with no kwarg when there is no override (test fakes don't accept the kwarg).

**Environment (every command):**
- Python is `/opt/homebrew/Caskroom/miniforge/base/bin/python` (NEVER plain `python3` — Homebrew python3 swallows stdout).
- Prefix every run with `KMP_DUPLICATE_LIB_OK=TRUE`.
- Tests run from the `main/` directory.
- Baseline: `87 passed, 7 skipped`. The bar is: no NEW failures, all new tests pass.

---

### Task 1: `VehicleDetector.detect` per-call conf override

**Files:**
- Modify: `main/src/models/vehicle_detector.py:43-49`
- Test: `main/tests/test_vehicle_detector.py`

- [ ] **Step 1: Write the failing tests**

Append to `main/tests/test_vehicle_detector.py`:

```python
class _PredictSpy:
    """Records the conf kwarg passed to model.predict; returns no boxes."""

    def __init__(self):
        self.confs = []

    def predict(self, source, conf, device, verbose):
        self.confs.append(conf)
        return []


def _detector_with_spy():
    from src.models.vehicle_detector import VehicleDetector

    det = VehicleDetector(model_path="nonexistent.onnx", conf=0.25)
    det.model = _PredictSpy()
    return det


def test_detect_uses_constructor_conf_by_default():
    det = _detector_with_spy()
    det.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert det.model.confs == [0.25]


def test_detect_conf_override_wins_for_that_call_only():
    det = _detector_with_spy()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    det.detect(frame, conf=0.9)
    det.detect(frame)
    assert det.model.confs == [0.9, 0.25]
```

(`VehicleDetector(model_path="nonexistent.onnx", ...)` prints a load warning and leaves `model=None`; the test then installs the spy. `pytest.importorskip` at the top of the file already guards the imports.)

- [ ] **Step 2: Run the tests to verify the new one fails**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_vehicle_detector.py -q`
Expected: `test_detect_conf_override_wins_for_that_call_only` FAILS with `TypeError: detect() got an unexpected keyword argument 'conf'`; `test_detect_uses_constructor_conf_by_default` passes.

- [ ] **Step 3: Implement the override**

In `main/src/models/vehicle_detector.py`, change `detect` (currently `def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:`) to:

```python
    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[dict[str, Any]]:
        """``conf`` overrides the constructor threshold for THIS call only
        (dashboard's live confidence slider); ``None`` keeps the configured
        default, so existing callers are unaffected."""
        if self.model is None:
            return []
        effective_conf = self.conf if conf is None else conf
        try:
            results = self.model.predict(
                source=frame, conf=effective_conf, device="cpu", verbose=False
            )
```

(The rest of the method body is unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_vehicle_detector.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
git add main/src/models/vehicle_detector.py main/tests/test_vehicle_detector.py
git commit -m "feat(models): VehicleDetector.detect accepts per-call conf override"
```

---

### Task 2: `infer_single_image` conf_override + vehicle_bbox

**Files:**
- Modify: `main/src/engine/pipeline_factory.py:98-161`
- Test: `main/tests/test_pipeline_factory.py`

- [ ] **Step 1: Write the failing tests**

Append to `main/tests/test_pipeline_factory.py`:

```python
class ConfSpyVehicleDetector:
    """Fake accepting the per-call conf kwarg, recording what each call got."""

    def __init__(self, dets):
        self._dets = dets
        self.confs = []

    def detect(self, image, conf=None):
        self.confs.append(conf)
        return self._dets


def test_infer_single_image_forwards_conf_override(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    spy = ConfSpyVehicleDetector([{"bbox": (0, 0, 10, 10), "conf": 0.9, "crop": img}])
    pipeline["vehicle_detector"] = spy
    infer_single_image(img, pipeline, cfg={}, conf_override=0.9)
    assert spy.confs == [0.9]


def test_infer_single_image_default_does_not_pass_conf(fake_db):
    """No override -> detector is called WITHOUT the conf kwarg, so the API
    default path keeps working with detectors that don't accept it
    (FakeVehicleDetector.detect has no conf parameter)."""
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)  # detect(self, image) only
    result = infer_single_image(img, pipeline, cfg={})  # must not raise
    assert result["status"] == "AUTHORIZED"


def test_infer_single_image_returns_vehicle_bbox(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    result = infer_single_image(img, pipeline, cfg={})
    assert result["vehicle_bbox"] == (0, 0, 10, 10)


def test_infer_single_image_vehicle_bbox_none_when_no_detection(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    pipeline["vehicle_detector"] = FakeVehicleDetector([])
    result = infer_single_image(img, pipeline, cfg={})
    assert result["vehicle_bbox"] is None


def test_infer_single_image_no_plate_includes_vehicle_bbox(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    pipeline["plate_reader"] = FakePlateReader({"text": "", "conf": 0.0, "plate_bbox": None})
    result = infer_single_image(img, pipeline, cfg={})
    assert result["status"] == "NO_PLATE"
    assert result["vehicle_bbox"] == (0, 0, 10, 10)
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_pipeline_factory.py -q`
Expected: the conf_override test fails with `TypeError: infer_single_image() got an unexpected keyword argument 'conf_override'`; the vehicle_bbox tests fail with `KeyError: 'vehicle_bbox'`. All pre-existing tests still pass.

- [ ] **Step 3: Implement**

In `main/src/engine/pipeline_factory.py`, change `infer_single_image`:

Signature (was `def infer_single_image(image: np.ndarray, pipeline: dict, cfg: dict) -> dict:`):

```python
def infer_single_image(
    image: np.ndarray,
    pipeline: dict,
    cfg: dict,
    conf_override: float | None = None,
) -> dict:
```

Extend the docstring with:

```
    conf_override, when given, replaces the config-fixed stage-1 vehicle-
    detection confidence FOR THIS CALL ONLY (the dashboard's live slider).
    ``None`` (the API /verify default) keeps the pipeline-build threshold,
    so UI and API agree by default. The kwarg is only forwarded to the
    detector when supplied — injected detectors (tests, older callers) may
    not accept ``conf``. The result carries ``vehicle_bbox`` (chosen
    vehicle's box, or None) so the UI can visualize what was detected.
```

Body — replace the detection block (currently `dets = pipeline["vehicle_detector"].detect(image)` and the `if not dets:` branch):

```python
    if conf_override is None:
        dets = pipeline["vehicle_detector"].detect(image)
    else:
        dets = pipeline["vehicle_detector"].detect(image, conf=conf_override)
    if not dets:
        vehicle_crop = image
        vehicle_bbox = None
    else:
        chosen = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        vehicle_crop = chosen["crop"]
        vehicle_bbox = chosen["bbox"]
```

Add `"vehicle_bbox": vehicle_bbox,` to BOTH return dicts (the NO_PLATE early return and the final return).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_pipeline_factory.py tests/test_api.py -q`
Expected: all PASS (test_api.py proves the API path is untouched).

- [ ] **Step 5: Commit**

```bash
cd "/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
git add main/src/engine/pipeline_factory.py main/tests/test_pipeline_factory.py
git commit -m "feat(engine): infer_single_image accepts conf_override, returns vehicle_bbox"
```

---

### Task 3: `ParkingSession.process_frame` conf_override

**Files:**
- Modify: `main/src/engine/parking_session.py:71-76`
- Test: `main/tests/test_parking_session.py`

- [ ] **Step 1: Write the failing test**

Append to `main/tests/test_parking_session.py`:

```python
class ConfRecordingDetector:
    """Accepts the per-call conf kwarg and records what each call got."""

    def __init__(self):
        self.confs = []

    def detect(self, frame, conf=None):
        self.confs.append(conf)
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((10, 10, 3), dtype=np.uint8)}]


def test_process_frame_forwards_conf_override_only_when_given():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    spy = ConfRecordingDetector()
    sess = ParkingSession(
        vehicle_detector=spy,
        plate_reader=FakePlateReader(),
        color_clf=FakeColorClf(),
        decision_engine=DecisionEngine(FakeMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05),
        sample_interval=1,
        collect_frames=2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sess.process_frame(frame, conf_override=0.9)
    sess.process_frame(frame)
    assert spy.confs == [0.9, None]
```

(The pre-existing tests, whose `FakeVehicleDetector.detect(self, frame)` has no conf param, pin that the no-override path really passes no kwarg.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_parking_session.py -q`
Expected: new test FAILS with `TypeError: process_frame() got an unexpected keyword argument 'conf_override'`.

- [ ] **Step 3: Implement**

In `main/src/engine/parking_session.py`, change `process_frame` (currently `def process_frame(self, frame: np.ndarray) -> dict[str, Any]:` followed by `detections = self.vehicle_detector.detect(frame)`):

```python
    def process_frame(self, frame: np.ndarray, conf_override: float | None = None) -> dict[str, Any]:
        self._frame_idx += 1
        if self._frame_idx % self.sample_interval != 0:
            return self._output()

        # conf_override: per-call stage-1 detection threshold (the dashboard's
        # live slider). Only forward the kwarg when supplied — injected
        # detectors (tests, older callers) may not accept ``conf``.
        if conf_override is None:
            detections = self.vehicle_detector.detect(frame)
        else:
            detections = self.vehicle_detector.detect(frame, conf=conf_override)
```

(The rest of the method is unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_parking_session.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
git add main/src/engine/parking_session.py main/tests/test_parking_session.py
git commit -m "feat(engine): ParkingSession.process_frame accepts per-call conf_override"
```

---

### Task 4: Dashboard wiring — slider drives all three modes

**Files:**
- Modify: `main/src/ui/dashboard.py` (`_run_pipeline` at 177-226, slider at 415-421, video loop at 609)

No new unit test (Streamlit script; verified live via the preview server by the controller afterwards). Full test suite must still pass.

- [ ] **Step 1: Rewire `_run_pipeline`**

Replace the signature + docstring + call (lines 177-203). New version:

```python
def _run_pipeline(
    image: np.ndarray,
    models: dict[str, Any],
    conf_threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    """Execute the shared 2-stage vehicle->plate->OCR + colour-gated verify
    pipeline via ``infer_single_image`` (same function the API /verify
    endpoint uses), so Upload-Image and the API always agree on a verdict.

    Args:
        image: BGR input image.
        models: Loaded model dict (must contain ``"pipeline"`` from
            ``build_pipeline``).
        conf_threshold: Stage-1 vehicle-detection confidence from the
            sidebar slider, forwarded per call so the slider works at
            runtime. The API /verify path passes no override and keeps
            the config.yaml default, so UI and API agree by default.

    Returns:
        A tuple of (list with 0 or 1 result dict, latency in ms).
    """
    pipeline = models.get("pipeline")
    if pipeline is None:
        return [], 0.0

    cfg = _load_config()
    result = infer_single_image(image, pipeline, cfg, conf_override=conf_threshold)
```

The `# noqa: ARG001 — kept for call-site compat...` comment on the parameter must be deleted.

In the `mapped = {...}` dict in the same function, change `"bbox": None,` to:

```python
        "bbox": result.get("vehicle_bbox"),
```

(`draw_detection_overlay` already draws `det["bbox"]` when it is not None, so image mode now shows the vehicle box — and the box disappears when the slider exceeds the detection's confidence.)

- [ ] **Step 2: Give the slider a help tooltip**

Change the slider (line 415-421) to:

```python
    conf_threshold = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence for stage-1 vehicle detection. Applies "
        "live to Upload Image, Webcam and Upload Video. The plate "
        "detector's threshold stays fixed in config.yaml.",
    )
```

- [ ] **Step 3: Thread the slider into the video loop**

In the Upload Video block, change `out = session.process_frame(frame)` (line 609) to:

```python
                        out = session.process_frame(frame, conf_override=conf_threshold)
```

- [ ] **Step 4: Sanity checks**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q`
Expected: `95 passed, 7 skipped` (87 baseline + 8 new) — every pre-existing test and every new test passes; no NEW failures vs the `87 passed, 7 skipped` baseline.

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -c "import ast; ast.parse(open('src/ui/dashboard.py').read()); print('dashboard OK')"`
Expected: `dashboard OK`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project"
git add main/src/ui/dashboard.py
git commit -m "feat(ui): Detection Confidence slider controls detection at runtime (image+video)"
```

---

## Verification (controller, after all tasks)

1. Full suite: no NEW failures vs `87 passed, 7 skipped`.
2. Live dashboard (preview config `dashboard`, port 8502): `main/data/test/test_authorized.jpg` at conf 0.25 → vehicle box drawn, verdict AUTHORIZED/51F06532; at 0.95 → box gone (detections respond).
3. API regression: `/verify` on the same image must equal the pre-change baseline (scratchpad `api_baseline_before.json`) on every pre-existing field: status AUTHORIZED, action ALLOW, plate 51F06532, color Black, color_conf 0.3810521364212036, color_warning false, brand_diagnostic null, message unchanged. The only permitted difference is the new additive `vehicle_bbox` key.
