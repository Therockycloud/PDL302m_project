# Demo Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default parking-video demo make a trustworthy, repeatable decision on a fresh checkout.

**Architecture:** OCR returns a structured reading with recognition confidence, while `PlateReader` keeps detector and OCR confidence separate. Video playback may skip rendering frames, but it must continue feeding skipped source frames into the decision session so gate evidence is not lost. The sample video remains an external artifact, verified by URL and SHA-256 before use.

**Tech Stack:** Python 3.12, PaddleOCR 2.x/3.x compatibility layer, OpenCV, Streamlit, pytest, Docker Compose.

---

## File Structure

- Modify `main/src/models/ppocr_reader.py`: return OCR text plus recognition confidence for both PaddleOCR APIs.
- Modify `main/src/models/plate_reader.py`: expose `ocr_conf` and `plate_det_conf` without losing the existing plate bounding box.
- Modify `main/src/engine/parking_session.py`: lock from OCR confidence and allow non-rendered source frames to be processed.
- Modify `main/src/engine/pipeline_factory.py` and `main/src/api/app.py`: preserve the two confidence values in image results/API output.
- Create `main/src/engine/video_pacing.py`: consume skipped source frames without coupling tests to Streamlit's top-level UI.
- Modify `main/src/ui/dashboard.py`: process source frames skipped for display through the parking session.
- Modify `main/src/utils/download_sample_video.py`: add a stable URL, checksum verification, timeout and CLI result code.
- Modify `README.md` and `main/README.md`: document video setup and correct test instructions.
- Modify/add tests in `main/tests/test_ppocr_reader.py`, `main/tests/test_plate_reader.py`, `main/tests/test_decision_engine.py`, `main/tests/test_parking_session.py`, and `main/tests/test_sample_video.py`.

### Task 1: OCR recognition-confidence contract

**Files:**
- Modify: `main/src/models/ppocr_reader.py:25-132`
- Modify: `main/src/models/plate_reader.py:14-38`
- Test: `main/tests/test_ppocr_reader.py`
- Test: `main/tests/test_plate_reader.py`

- [ ] **Step 1: Write failing v2 confidence tests**

```python
def test_map_v2_result_returns_mean_recognition_confidence():
    reading = map_v2_result_to_plate_reading([
        [[BOX_TOP, ("30M-71854", 0.90)], [BOX_BOTTOM, ("", 0.70)]]
    ])
    assert reading == {"text": "30M71854", "ocr_conf": 0.90}
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_ppocr_reader.py -q`

Expected: FAIL because `map_v2_result_to_plate_reading` does not exist.

- [ ] **Step 3: Implement the smallest structured OCR result**

Add a pure helper `map_v2_result_to_plate_reading(v2_result) -> dict[str, float | str]` that sorts lines as today, cleans the joined text, and returns `ocr_conf=0.0` for empty text. Keep `map_v2_result_to_plate_text` as a compatibility wrapper returning `reading["text"]`.

Change `PaddleOCRReader.read_plate` to return `{ "text": str, "ocr_conf": float }`; for v3, average `rec_scores` only for recognised non-empty text.

- [ ] **Step 4: Add a failing PlateReader contract test**

```python
def test_plate_reader_keeps_detector_and_ocr_confidences_separate():
    reader = PlateReader(FakeDetector(conf=0.95), FakeOCR({"text": "30M71854", "ocr_conf": 0.42}))
    assert reader.read(IMAGE) == {
        "text": "30M71854", "ocr_conf": 0.42,
        "plate_det_conf": 0.95, "plate_bbox": (1, 2, 3, 4),
    }
```

- [ ] **Step 5: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_plate_reader.py -q`

Expected: FAIL because `ocr_conf` and `plate_det_conf` are not returned.

- [ ] **Step 6: Implement the PlateReader adapter**

Accept both legacy string OCR readers and structured Paddle readers. Return `ocr_conf` from the structured reading, default `0.0` for a legacy string, and return plate-detector confidence only as `plate_det_conf`.

- [ ] **Step 7: Verify GREEN and commit**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_ppocr_reader.py tests/test_plate_reader.py -q`

Run: `git add main/src/models/ppocr_reader.py main/src/models/plate_reader.py main/tests/test_ppocr_reader.py main/tests/test_plate_reader.py && git commit -m "fix: separate OCR and detector confidences"`

### Task 2: Decision and API observability

**Files:**
- Modify: `main/src/engine/parking_session.py:96-136`
- Modify: `main/src/engine/pipeline_factory.py:133-180`
- Modify: `main/src/api/app.py:137-170`
- Test: `main/tests/test_decision_engine.py`
- Test: `main/tests/test_parking_session.py`
- Test: `main/tests/test_api.py`

- [ ] **Step 1: Write failing lock-confidence tests**

```python
def test_low_ocr_confidence_does_not_lock_even_when_detector_confidence_is_high():
    out = engine.aggregate([
        {"plate_text": "30M71854", "plate_conf": 0.20, "plate_det_conf": 0.99, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.20, "plate_det_conf": 0.99, "color": "YELLOW", "color_conf": 0.9},
    ], lock_conf=0.60, lock_repeat=2)
    assert out["status"] == "UNCERTAIN"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_decision_engine.py -q`

Expected: FAIL until `ParkingSession` passes OCR confidence instead of detector confidence.

- [ ] **Step 3: Wire OCR confidence through the session**

In `_collect`, set `plate_conf=plate["ocr_conf"]` and retain `plate_det_conf` separately. Do not change `DecisionEngine`'s lock algorithm; its existing `plate_conf` contract now means OCR confidence consistently.

- [ ] **Step 4: Add a failing API payload test**

```python
def test_verify_includes_detector_and_ocr_confidence(client_with_fake_pipeline):
    body = client_with_fake_pipeline.post("/verify", files=_make_test_image_files()).json()
    assert body["ocr_conf"] == 0.9
    assert body["plate_det_conf"] == 0.8
```

- [ ] **Step 5: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_api.py tests/test_parking_session.py -q`

Expected: FAIL because result payload omits the confidence values.

- [ ] **Step 6: Return the confidence fields from image inference and API**

Carry `ocr_conf` and `plate_det_conf` through `infer_single_image`, including the `NO_PLATE` branch, so API JSON requires no special serialization code.

- [ ] **Step 7: Verify GREEN and commit**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_decision_engine.py tests/test_parking_session.py tests/test_api.py -q`

Run: `git add main/src/engine/parking_session.py main/src/engine/pipeline_factory.py main/src/api/app.py main/tests/test_decision_engine.py main/tests/test_parking_session.py main/tests/test_api.py && git commit -m "fix: lock plates using OCR confidence"`

### Task 3: Preserve decision evidence during display-frame drops

**Files:**
- Create: `main/src/engine/video_pacing.py`
- Modify: `main/src/ui/dashboard.py:504-601,722-770`
- Test: `main/tests/test_parking_session.py`
- Create: `main/tests/test_video_pacing.py`

- [ ] **Step 1: Write a failing source-frame pacing test**

```python
def test_skipped_source_frames_are_still_seen_by_the_decision_session():
    capture = FakeCapture(FRAMES)
    session = RecordingSession()
    consume_skipped_frames(capture, count=3, process_frame=session.process_frame)
    assert session.seen_source_indexes == [0, 1, 2]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_video_pacing.py -q`

Expected: FAIL because the UI discards skipped frames using `cap.grab()`.

- [ ] **Step 3: Create a UI-independent video pacing helper**

Create `src.engine.video_pacing.consume_skipped_frames(capture, count, process_frame) -> int`. It must decode up to `count` skipped source frames with `capture.read()`, call `process_frame(frame)` for every decoded frame, and return the number consumed. It must not import Streamlit or render images.

- [ ] **Step 4: Add the real default-video regression test**

```python
def test_default_video_decides_when_rendering_is_overloaded():
    result = run_default_video_with_render_lag(max_lag_seconds=0.5, process_skipped=True)
    assert result["state"] == "DECIDED"
    assert result["decision"]["status"] == "AUTHORIZED"
```

The helper may use the actual checked-in/local demo artifact but must skip with an explicit reason when it is absent.

- [ ] **Step 5: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_video_pacing.py -q`

Expected: FAIL before the drop-path processing is wired into the UI loop.

- [ ] **Step 6: Replace `cap.grab()` drop-only behaviour**

In the lag branch, call `consume_skipped_frames` instead of `cap.grab()`. Pass a non-rendering wrapper around `_process_live_frame`, then render only the next current frame. Keep the existing stop, release and cleanup logic.

- [ ] **Step 7: Verify GREEN and commit**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_video_pacing.py tests/test_parking_session.py -q`

Run: `git add main/src/engine/video_pacing.py main/src/ui/dashboard.py main/tests/test_parking_session.py main/tests/test_video_pacing.py && git commit -m "fix: preserve video evidence during frame drops"`

### Task 4: Verify the default demo-video artifact

**Files:**
- Modify: `main/src/utils/download_sample_video.py`
- Modify: `main/src/ui/dashboard.py:633-665`
- Test: `main/tests/test_sample_video.py`
- Modify: `README.md`
- Modify: `main/README.md`

- [ ] **Step 1: Write failing artifact-validation tests**

```python
def test_validate_sample_video_rejects_missing_file(tmp_path):
    assert validate_sample_video(tmp_path / "sample_parking.mp4") is False

def test_download_sample_video_rejects_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: FakeResponse(b"wrong"))
    with pytest.raises(ValueError, match="checksum"):
        download_sample_video(tmp_path / "sample_parking.mp4")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_sample_video.py -q`

Expected: FAIL because the helper has no checksum-validation API.

- [ ] **Step 3: Implement deterministic artifact handling**

Define one source URL and SHA-256 constant (`fac033acb960b0a87e2a0e50b7532025d90c0acfe8af172b3d2b112107a4c1c5`). Download to a temporary path with a timeout, verify the digest, then atomically replace the destination. Provide `--verify` and normal download CLI modes with non-zero exit status on failure.

- [ ] **Step 4: Make the UI fail clearly before playback**

Replace `_ensure_sample_video`'s informational return with validation that emits an error containing the exact setup command. Never initiate a network download from the Streamlit request path.

- [ ] **Step 5: Verify GREEN, update docs and commit**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_sample_video.py -q`

Document `python main/src/utils/download_sample_video.py --verify` and the normal download command in both READMEs. Correct the Docker test command to `docker compose exec -T -w /app/main backend pytest -q`.

Run: `git add main/src/utils/download_sample_video.py main/src/ui/dashboard.py main/tests/test_sample_video.py README.md main/README.md && git commit -m "fix: verify default parking video artifact"`

### Task 5: Full verification

**Files:**
- Verify only: all files modified above

- [ ] **Step 1: Run native regression suite**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q`

Expected: no test failures.

- [ ] **Step 2: Run Docker regression suite in the documented working directory**

Run: `docker compose exec -T -w /app/main backend pytest -q`

Expected: no test failures; skipped tests must list only intentionally unavailable train-only dependencies.

- [ ] **Step 3: Verify source hygiene and commits**

Run: `git diff --check HEAD~4..HEAD`

Run: `git status --short`

Expected: no whitespace errors and only the user-owned `CLAUDE.md` change remains unstaged.
