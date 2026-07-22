# Vietnamese Plate Recognizer Model and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and deploy a lightweight Vietnamese license-plate recognizer that reaches at least 90% whole-plate exact match on both required real held-out sets and is materially faster than PaddleOCR on constrained Docker CPU.

**Architecture:** YOLOv8n continues to localize the vehicle and plate. A fixed-shape MobileNetV3-Small + CTC recognizer reads the localized `192 x 64` plate crop through ONNX Runtime. Runtime replacement is gated by reproducible evaluation; PaddleOCR stays deployed unless the candidate passes both real-data accuracy gates.

**Tech Stack:** Python 3.12, PyTorch, torchvision, ONNX, ONNX Runtime, OpenCV, NumPy, pandas, pytest, Docker.

---

## File Map

- Create `main/src/models/vn_plate_text.py`: normalization, Vietnamese format validation, two-line conversion, CTC decoding.
- Create `main/src/models/vn_plate_recognizer.py`: ONNX Runtime reader implementing `read_plate(image)`.
- Create `main/src/datasets/plate_ocr_dataset.py`: manifest schema, grouped split validation, preprocessing, PyTorch dataset.
- Create `main/src/models/vn_plate_ctc.py`: MobileNetV3-Small + CTC training model.
- Create `main/scripts/build_plate_ocr_dataset.py`: reproducible synthetic/pseudo-label/real manifest builder.
- Create `main/scripts/train_vn_plate_ocr.py`: training, evaluation, checkpointing, ONNX export, metadata output.
- Create `main/scripts/benchmark_vn_plate_ocr.py`: candidate-versus-Paddle evaluation and deployment-gate report.
- Create `main/tests/test_vn_plate_text.py`, `test_plate_ocr_dataset.py`, `test_vn_plate_recognizer.py`, `test_vn_plate_ctc.py`, `test_vn_ocr_benchmark.py`.
- Modify `main/src/engine/pipeline_factory.py`: config-selected OCR construction with strict ONNX readiness.
- Modify `main/src/engine/parking_session.py`: 4/8 evidence and deferred one-time colour inference.
- Modify `main/configs/config.yaml`: candidate model paths, thresholds, `collect_frames: 8`, `lock_repeat: 4`.
- Modify `main/requirements.txt`, `main/requirements-train.txt`, `Dockerfile`: ONNX model runtime and export dependencies; remove Paddle runtime only after the gate passes.
- Create model artifacts under `main/data/models/vn_plate_recognizer.onnx` and `vn_plate_recognizer.json` only after the gate passes.

### Task 1: Plate text contract and decoder

**Files:**
- Create: `main/src/models/vn_plate_text.py`
- Test: `main/tests/test_vn_plate_text.py`

- [ ] **Step 1: Write failing tests for normalization, validation, two-line layout, and CTC collapse**

```python
def test_ctc_decode_collapses_repeats_and_blanks():
    vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    logits = fake_logits_for_indices([3, 3, 36, 0, 0, 22])
    out = greedy_ctc_decode(logits, vocab)
    assert out.text == "30M"
    assert 0.0 <= out.confidence <= 1.0

def test_invalid_plate_is_not_lockable():
    assert validate_vietnamese_plate("30M71854") is True
    assert validate_vietnamese_plate("VF3") is False

def test_two_line_plate_is_stacked_in_reading_order():
    image = fixture_two_line_plate()
    strip = normalize_plate_crop(image, output_size=(192, 64))
    assert strip.shape == (64, 192, 3)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `cd main && pytest -q tests/test_vn_plate_text.py`

Expected: collection fails because `src.models.vn_plate_text` does not exist.

- [ ] **Step 3: Implement immutable `PlateReading`, strict normalization, format validation, crop padding, compact-plate row splitting, and NumPy greedy CTC decoding**

```python
@dataclass(frozen=True)
class PlateReading:
    text: str
    confidence: float

def normalize_plate_text(raw: str) -> str:
    return "".join(ch for ch in raw.upper() if ch.isalnum())

def validate_vietnamese_plate(text: str) -> bool:
    value = normalize_plate_text(text)
    return bool(re.fullmatch(r"(?:\d{2}[A-Z]\d{4,6}|\d{2}[A-Z]{1,2}\d{4,6})", value))
```

The validator rejects impossible strings; it never substitutes ambiguous characters.

- [ ] **Step 4: Run focused tests**

Run: `cd main && pytest -q tests/test_vn_plate_text.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main/src/models/vn_plate_text.py main/tests/test_vn_plate_text.py
git commit -m "feat: add Vietnamese plate text contract"
```

### Task 2: Reproducible OCR dataset manifest

**Files:**
- Create: `main/src/datasets/plate_ocr_dataset.py`
- Create: `main/scripts/build_plate_ocr_dataset.py`
- Test: `main/tests/test_plate_ocr_dataset.py`

- [ ] **Step 1: Write failing tests for required columns and group isolation**

```python
REQUIRED = {"image_path", "label", "source_type", "group_id", "split", "verified"}

def test_manifest_rejects_group_leakage(tmp_path):
    manifest = write_manifest(tmp_path, [
        row("a.png", "30M71854", "real", "vehicle-1", "train", True),
        row("b.png", "30M71854", "real", "vehicle-1", "test", True),
    ])
    with pytest.raises(ValueError, match="group_id"):
        load_plate_manifest(manifest)

def test_test_split_requires_manually_verified_real_labels(tmp_path):
    manifest = write_manifest(tmp_path, [
        row("a.png", "30M71854", "pseudo", "vehicle-1", "test", False),
    ])
    with pytest.raises(ValueError, match="verified real"):
        load_plate_manifest(manifest)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `cd main && pytest -q tests/test_plate_ocr_dataset.py`

Expected: missing-module failure.

- [ ] **Step 3: Implement manifest loading and deterministic grouped splitting**

`load_plate_manifest()` must normalize labels, resolve paths relative to the manifest, reject duplicate image paths, reject group leakage, and require `source_type=real` plus `verified=true` for validation and test claims.

- [ ] **Step 4: Implement dataset builder modes**

```bash
python main/scripts/build_plate_ocr_dataset.py synthetic --count 50000 --seed 42
python main/scripts/build_plate_ocr_dataset.py pseudo-label --source main/data/raw/plate_det --min-conf 0.95
python main/scripts/build_plate_ocr_dataset.py extract-video --video main/data/test/parking_case_real.mp4 --plate 30M71854 --every-n 6
python main/scripts/build_plate_ocr_dataset.py validate --manifest main/data/plate_ocr/manifest.csv
```

The synthetic generator renders valid one-line and two-line Vietnamese layouts and applies perspective, blur, low-light, glare, noise, downscale, occlusion, and JPEG transforms. It records every random parameter in the manifest. Pseudo-label rows are never marked verified.

- [ ] **Step 5: Build a review queue and manually verify at least 100 real test crops**

Run: `python main/scripts/build_plate_ocr_dataset.py review-sheet --manifest main/data/plate_ocr/manifest.csv --count 120`

Expected: contact sheets plus `main/data/plate_ocr/review.csv`. Correct labels by visual inspection, mark `verified=true`, assign stable `group_id`, then validate that the held-out real test set contains at least 100 crops and has no group leakage.

- [ ] **Step 6: Run tests and manifest validation**

Run: `cd main && pytest -q tests/test_plate_ocr_dataset.py && python scripts/build_plate_ocr_dataset.py validate --manifest data/plate_ocr/manifest.csv`

Expected: tests pass and validation reports at least 100 verified real test crops.

- [ ] **Step 7: Commit code and the compact manifest, excluding generated synthetic images**

```bash
git add main/src/datasets/plate_ocr_dataset.py main/scripts/build_plate_ocr_dataset.py main/tests/test_plate_ocr_dataset.py main/data/plate_ocr/manifest.csv main/data/plate_ocr/review.csv
git commit -m "feat: add reproducible plate OCR dataset"
```

### Task 3: Trainable MobileNetV3 CTC model

**Files:**
- Create: `main/src/models/vn_plate_ctc.py`
- Test: `main/tests/test_vn_plate_ctc.py`
- Modify: `main/requirements-train.txt`

- [ ] **Step 1: Write failing shape and CTC-loss tests**

```python
def test_model_outputs_time_major_character_logits():
    model = VnPlateCTC(num_classes=37)
    logits = model(torch.zeros(2, 3, 64, 192))
    assert logits.ndim == 3
    assert logits.shape[1] == 2
    assert logits.shape[2] == 37

def test_tiny_batch_backpropagates():
    loss = ctc_loss_for_batch(model, images, labels, lengths)
    loss.backward()
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd main && pytest -q tests/test_vn_plate_ctc.py`

Expected: missing-module failure.

- [ ] **Step 3: Implement MobileNetV3-Small feature extraction, height pooling, sequence projection, and CTC logits**

Use torchvision MobileNetV3-Small without downloading weights during runtime. Convert the final feature map from `[N,C,H,W]` to `[T,N,C]`, pool height only, and project channels to `len(vocab)+1` classes. Keep the model free of training-only global state so ONNX export is deterministic.

- [ ] **Step 4: Run focused tests**

Run: `cd main && pytest -q tests/test_vn_plate_ctc.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main/src/models/vn_plate_ctc.py main/tests/test_vn_plate_ctc.py main/requirements-train.txt
git commit -m "feat: add lightweight plate CTC model"
```

### Task 4: Training, evaluation, and ONNX export

**Files:**
- Create: `main/scripts/train_vn_plate_ocr.py`
- Modify: `main/tests/test_vn_plate_ctc.py`

- [ ] **Step 1: Add failing smoke-training and export-parity tests**

```python
def test_smoke_training_reduces_loss(tmp_path):
    report = train(smoke_config(tmp_path, epochs=2))
    assert report["final_train_loss"] < report["initial_train_loss"]

def test_onnx_logits_match_torch(tmp_path):
    torch_logits, onnx_logits = export_and_compare(tmp_path, fixed_batch())
    np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `cd main && pytest -q tests/test_vn_plate_ctc.py -k 'smoke or onnx'`

Expected: training/export entry points are missing.

- [ ] **Step 3: Implement deterministic train/evaluate/export CLI**

```bash
python main/scripts/train_vn_plate_ocr.py --manifest main/data/plate_ocr/manifest.csv --output main/data/models/vn_plate_run --seed 42 --epochs 40 --batch-size 64
```

Checkpoint by real-validation exact match, then CER. Calibrate sequence confidence
on the verified real validation split and record the lowest threshold whose
whole-plate precision is at least 99%; this becomes `recommended_lock_conf`.
Emit `best.pt`, `vn_plate_recognizer.onnx`, `vn_plate_recognizer.json`,
`training_history.json`, and `training_curves.png`. Metadata includes vocabulary,
input size, mean/std, blank index, calibrated threshold, model SHA-256, and
training manifest SHA-256.

- [ ] **Step 4: Run smoke training and parity tests**

Run: `cd main && pytest -q tests/test_vn_plate_ctc.py`

Expected: all tests pass without downloading data.

- [ ] **Step 5: Run full training and retain the best measured candidate**

Run the full CLI above. Expected: a completed report with no split leakage and ONNX parity within tolerance. Do not copy the ONNX file into the deployed model path yet.

- [ ] **Step 6: Commit training code and measured training evidence**

```bash
git add main/scripts/train_vn_plate_ocr.py main/tests/test_vn_plate_ctc.py main/data/models/vn_plate_run/training_history.json main/data/models/vn_plate_run/training_curves.png main/data/models/vn_plate_run/vn_plate_recognizer.json
git commit -m "feat: train Vietnamese plate recognizer"
```

### Task 5: ONNX runtime reader

**Files:**
- Create: `main/src/models/vn_plate_recognizer.py`
- Test: `main/tests/test_vn_plate_recognizer.py`

- [ ] **Step 1: Write failing tests for metadata validation and reader output**

```python
def test_reader_rejects_wrong_model_checksum(tmp_path):
    with pytest.raises(RuntimeError, match="checksum"):
        VnPlateRecognizer(model_path, bad_metadata_path)

def test_reader_returns_plate_reader_contract(fake_session):
    reader = VnPlateRecognizer("model.onnx", "model.json", session=fake_session)
    assert reader.read_plate(sample_crop()) == {
        "text": "30M71854", "ocr_conf": pytest.approx(0.93)
    }
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd main && pytest -q tests/test_vn_plate_recognizer.py`

Expected: missing-module failure.

- [ ] **Step 3: Implement strict model loading, preprocessing, decoding, timing, and readiness**

The constructor verifies file existence, checksum, opset-compatible ONNX loading, vocabulary, and input metadata. `read_plate()` returns the existing `{text, ocr_conf}` contract; invalid formats return empty text with the measured confidence preserved in diagnostics.

- [ ] **Step 4: Run focused tests**

Run: `cd main && pytest -q tests/test_vn_plate_recognizer.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main/src/models/vn_plate_recognizer.py main/tests/test_vn_plate_recognizer.py
git commit -m "feat: serve Vietnamese plate OCR with ONNX"
```

### Task 6: Candidate benchmark and hard deployment gate

**Files:**
- Create: `main/scripts/benchmark_vn_plate_ocr.py`
- Create: `main/tests/test_vn_ocr_benchmark.py`
- Create after run: `docs/benchmarks/vn_plate_ocr_gate.json`

- [ ] **Step 1: Write failing metric and gate tests**

```python
def test_gate_requires_both_real_sets_at_90_percent():
    result = deployment_gate(frozen_exact=15/16, expanded_exact=0.89, cer=0.02)
    assert result.passed is False

def test_character_accuracy_cannot_replace_exact_match():
    result = deployment_gate(frozen_exact=14/16, expanded_exact=0.95, cer=0.001)
    assert result.passed is False
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `cd main && pytest -q tests/test_vn_ocr_benchmark.py`

Expected: missing benchmark module or gate function.

- [ ] **Step 3: Implement side-by-side evaluation**

Compute whole-plate exact match, CER, invalid-format rate, model size, cold latency, and warm p50/p95. Report frozen-16 and expanded-real metrics separately. `passed=true` requires frozen exact `>=0.90`, expanded exact `>=0.90`, CER `<=0.031`, and candidate p95 lower than Paddle p95.

- [ ] **Step 4: Run native benchmark, then constrained Docker benchmark**

```bash
python main/scripts/benchmark_vn_plate_ocr.py --manifest main/data/plate_ocr/manifest.csv --candidate main/data/models/vn_plate_run/vn_plate_recognizer.onnx --metadata main/data/models/vn_plate_run/vn_plate_recognizer.json --output docs/benchmarks/vn_plate_ocr_gate.json
docker compose exec -T backend python main/scripts/benchmark_vn_plate_ocr.py --manifest main/data/plate_ocr/manifest.csv --candidate main/data/models/vn_plate_run/vn_plate_recognizer.onnx --metadata main/data/models/vn_plate_run/vn_plate_recognizer.json --output docs/benchmarks/vn_plate_ocr_gate_docker.json
```

Expected: reports contain raw per-sample results and an explicit `passed` boolean. Do not deploy when either report fails.

- [ ] **Step 5: Run tests and commit benchmark evidence**

```bash
cd main && pytest -q tests/test_vn_ocr_benchmark.py
cd ..
git add main/scripts/benchmark_vn_plate_ocr.py main/tests/test_vn_ocr_benchmark.py docs/benchmarks/vn_plate_ocr_gate*.json
git commit -m "test: gate Vietnamese OCR deployment at 90 percent"
```

### Task 7: Runtime integration after the gate passes

**Files:**
- Modify: `main/src/engine/pipeline_factory.py`
- Modify: `main/src/engine/parking_session.py`
- Modify: `main/configs/config.yaml`
- Modify: `main/tests/test_pipeline_factory.py`
- Modify: `main/tests/test_parking_session.py`
- Modify: `main/tests/test_parking_session_factory.py`
- Modify: `main/src/utils/warmup.py`
- Modify: `main/tests/test_warmup.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_factory_builds_vn_onnx_reader_from_config(candidate_cfg):
    reader = _build_ocr_reader(candidate_cfg)
    assert isinstance(reader, VnPlateRecognizer)

def test_session_requires_four_matching_reads_within_eight_frames(session):
    for _ in range(3):
        assert session.process_frame(frame())["decision"] is None
    assert session.process_frame(frame())["decision"]["plate"] == "30M71854"

def test_colour_runs_once_after_plate_lock(session):
    for _ in range(8):
        session.process_frame(frame())
    assert session.color_clf.calls == 1
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `cd main && pytest -q tests/test_pipeline_factory.py tests/test_parking_session.py tests/test_parking_session_factory.py tests/test_warmup.py`

Expected: reader selection, 4/8, or deferred-colour assertions fail.

- [ ] **Step 3: Deploy artifacts only when gate reports pass**

Copy the passing ONNX and metadata to `main/data/models/vn_plate_recognizer.onnx` and `.json`; verify their SHA-256 values match the gate report. If the gate did not pass, stop this task and leave `ocr.engine: ppocr` unchanged.

- [ ] **Step 4: Implement strict config selection and readiness**

```yaml
ocr:
  engine: vn_ctc
  model_name: vn_plate_recognizer.onnx
  metadata_name: vn_plate_recognizer.json
  diagnostic_fallback: ppocr
pipeline:
  collect_frames: 8
  lock:
    lock_repeat: 4
```

Set `pipeline.lock.lock_conf` to the measured `recommended_lock_conf` from the
passing metadata; do not reuse PaddleOCR's `0.60` threshold. `_build_ocr_reader()`
raises `RuntimeError` for missing or invalid candidate assets. Warmup calls the
new recognizer and exposes failure through API readiness.

- [ ] **Step 5: Defer colour until lock**

Store the best vehicle crop by plate confidence during collection. Once four matching votes lock, call `color_clf.predict(best_crop)` once, attach the result to locked frames used by `DecisionEngine`, and never authorize on an invalid plate format.

- [ ] **Step 6: Run focused and full native tests**

Run: `cd main && pytest -q tests/test_pipeline_factory.py tests/test_parking_session.py tests/test_parking_session_factory.py tests/test_warmup.py && pytest -q`

Expected: focused tests pass; full suite has no regression beyond documented environment skips.

- [ ] **Step 7: Commit runtime integration and passing artifacts**

```bash
git add main/src/engine/pipeline_factory.py main/src/engine/parking_session.py main/src/utils/warmup.py main/configs/config.yaml main/tests/test_pipeline_factory.py main/tests/test_parking_session.py main/tests/test_parking_session_factory.py main/tests/test_warmup.py main/data/models/vn_plate_recognizer.onnx main/data/models/vn_plate_recognizer.json
git commit -m "feat: deploy fast Vietnamese plate recognizer"
```

### Task 8: Docker runtime verification

**Files:**
- Modify: `main/requirements.txt`
- Modify: `Dockerfile`
- Test: full Docker suite and API smoke tests

- [ ] **Step 1: Remove Paddle runtime dependencies only when diagnostic fallback is disabled**

Keep Paddle packages when `diagnostic_fallback: ppocr`. Otherwise remove Paddle installation and build-time priming together; never leave documentation and image behavior inconsistent.

- [ ] **Step 2: Rebuild without cache-sensitive stale artifacts**

Run: `docker compose build backend frontend && docker compose up -d backend frontend`

Expected: both services become healthy and backend logs show successful candidate warmup.

- [ ] **Step 3: Run Docker tests**

Run: `docker compose exec -T -w /app/main backend pytest -q`

Expected: all supported Docker tests pass; environment skips are listed.

- [ ] **Step 4: Exercise real image and video APIs**

Run the repository API smoke command against `main/data/test/test_authorized.jpg`, then process both parking videos through the demo endpoint. Record actual gate-to-verdict and per-stage timing; verify no verdict occurs before four matching votes.

- [ ] **Step 5: Commit Docker dependency changes**

```bash
git add main/requirements.txt Dockerfile
git commit -m "build: package Vietnamese OCR runtime"
```

### Task 9: Model-plan completion review

**Files:**
- Review all files and commits from Tasks 1–8

- [ ] **Step 1: Run final native verification**

Run: `cd main && pytest -q`

Expected: all supported native tests pass.

- [ ] **Step 2: Run final gate verification**

Read both gate JSON files and confirm `passed=true`, frozen exact match `>=0.90`, expanded exact match `>=0.90`, CER `<=0.031`, and Docker candidate p95 below Docker Paddle p95.

- [ ] **Step 3: Inspect git scope**

Run: `git status --short && git log --oneline --decorate -12`

Expected: `CLAUDE.md` remains the only unrelated user-owned modification; no generated caches or synthetic image corpus are staged.

- [ ] **Step 4: Hand off measured artifacts and commit IDs to the primary agent**

Report exact metrics, commands, failures, skipped tests, artifact SHA-256, and every commit created. Do not claim runtime deployment if the 90% gate failed.
