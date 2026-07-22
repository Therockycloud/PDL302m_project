# Single Product Camera, Benchmark, and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present one synchronized Product camera, verify the deployed recognizer in a real Docker/browser workflow, and update all project documentation and presentations with measured evidence.

**Architecture:** The existing hidden browser video remains the sole decoder and media clock. A Product canvas draws every decoded frame while a one-request-in-flight sampler sends only fresh evidence frames. Documentation and slides are generated only after the model deployment gate and end-to-end benchmark are known.

**Tech Stack:** HTML/CSS/JavaScript Streamlit component, Python, FastAPI, Streamlit, pytest, Docker Compose, browser automation, Markdown, HTML presentations.

---

### Task 1: One visible Product camera with independent controls

**Files:**
- Modify: `main/src/ui/components/media_clock_video/index.html`
- Modify: `main/src/ui/media_clock_video.py`
- Modify: `main/tests/test_media_clock_video.py`

- [ ] **Step 1: Write failing component-contract tests**

```python
def test_component_hides_source_and_exposes_product_controls():
    html = component_entrypoint().read_text()
    assert "Source video — native playback" not in html
    assert 'id="origin-video"' in html
    assert 'id="product-canvas"' in html
    assert 'id="play-toggle"' in html
    assert 'id="seek-slider"' in html

def test_wrapper_defaults_to_100_ms_sampling(monkeypatch):
    media_clock_video("/video.mp4", "http://localhost:8000", "session")
    assert captured["sample_interval_ms"] == 100
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd main && pytest -q tests/test_media_clock_video.py`

Expected: source caption remains, controls are missing, and interval is 200 ms.

- [ ] **Step 3: Implement layout A**

Hide the origin video visually while keeping it in the DOM as the only clock. Render the Product caption, canvas, diagnostics, and accessible play/pause plus range seek controls below the canvas. Mirror loaded duration, current time, play state, ended state, keyboard operation, and seek reset to the hidden video.

- [ ] **Step 4: Keep native display FPS and latest-frame-only inference**

Draw every `requestVideoFrameCallback` frame. Set the default sample attempt interval to 100 ms, retain one request in flight, and never enqueue pending blobs. Seeking and new activations abort the current request and clear evidence.

- [ ] **Step 5: Run focused tests and commit**

```bash
cd main && pytest -q tests/test_media_clock_video.py
cd ..
git add main/src/ui/components/media_clock_video/index.html main/src/ui/media_clock_video.py main/tests/test_media_clock_video.py
git commit -m "feat: show one synchronized Product camera"
```

### Task 2: Real Docker and browser acceptance

**Files:**
- Create: `docs/benchmarks/vn_plate_e2e_docker.json`
- Create: `docs/benchmarks/vn_plate_e2e_docker.md`

- [ ] **Step 1: Rebuild and run constrained Docker services**

Run: `docker compose build backend frontend && docker compose up -d backend frontend`

Expected: backend and frontend stay running; backend readiness reports the deployed OCR engine and successful warmup.

- [ ] **Step 2: Run native and Docker regression suites**

```bash
cd main && pytest -q
cd ..
docker compose exec -T -w /app/main backend pytest -q
```

Record pass, skip, and failure counts exactly.

- [ ] **Step 3: Test both real parking videos in the browser**

Open `http://localhost:8501`, run `parking_case_real.mp4` and `sample_parking.mp4`, and verify only Product cam is visible; play/pause/seek works; media time continues during inference; seek issues a session reset; and the rendered Product frame matches the hidden media time.

- [ ] **Step 4: Capture end-to-end evidence**

For each video record motion/approach gate time, eight attempted evidence timestamps, successful reads, vote sequence, lock timestamp, OCR per-frame latency, gate-to-verdict latency, display FPS, and final verdict. Verify authorization never occurs before the fourth matching read.

- [ ] **Step 5: Write machine-readable and human-readable benchmark evidence**

The JSON contains environment, image hashes, model hashes, raw trials, p50/p95, and verdicts. The Markdown summarizes measured results and explicitly distinguishes cold, warm, OCR-only, and end-to-end latency.

- [ ] **Step 6: Commit benchmark evidence**

```bash
git add docs/benchmarks/vn_plate_e2e_docker.json docs/benchmarks/vn_plate_e2e_docker.md
git commit -m "test: benchmark Vietnamese OCR end to end"
```

### Task 3: Synchronize READMEs, reports, and model specification

**Files:**
- Modify: `README.md`
- Modify: `main/README.md`
- Modify: `docs/model_specifications.md`
- Create: `docs/benchmarks/vn_plate_ocr_benchmark.csv`
- Create: `docs/benchmarks/vn_plate_ocr_benchmark.md`
- Create: `docs/benchmarks/vn_plate_ocr_benchmark.png`
- Modify: `reports/documents/Report_2_Data_Tasks.md`
- Modify: `reports/documents/Report_3_Model_Results.md`
- Modify: `reports/documents/Report_4_Final_Report.md`

- [ ] **Step 1: Generate the benchmark table and chart from gate JSON**

Use one script path and the recorded JSON as the only source. Include PaddleOCR and candidate exact match, CER, model size, cold, warm p50, and warm p95. If the gate failed, label the candidate experimental and keep PaddleOCR as deployed.

- [ ] **Step 2: Update architecture and operating instructions**

Document MobileNetV3-Small + CTC, `192 x 64`, ONNX Runtime, 4/8 voting, deferred colour, single Product cam, Docker rebuild commands, model checksum, and measured limitations.

- [ ] **Step 3: Update Reports 2–4**

Report 2 covers synthetic, pseudo-labelled, and manually verified data plus grouped split rules. Report 3 covers model architecture, training curves, exact-match/CER/latency comparisons, and the 90% gate. Report 4 covers deployed runtime, safety behavior, Docker/browser results, and whether the target 500–1000 ms was met.

- [ ] **Step 4: Check claims mechanically**

Run searches for stale deployed-Paddle claims, `81%`, `423 ms`, `0.73 s`, and `0.96 s`. Historical baseline statements may remain only when clearly labelled as baseline; current claims must match benchmark JSON.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md main/README.md docs/model_specifications.md docs/benchmarks/vn_plate_ocr_benchmark.* reports/documents/Report_2_Data_Tasks.md reports/documents/Report_3_Model_Results.md reports/documents/Report_4_Final_Report.md
git commit -m "docs: report Vietnamese OCR model results"
```

### Task 4: Update and visually verify presentations

**Files:**
- Modify: `reports/presentations/Report_3_Presentation.html`
- Modify: `reports/presentations/Report_3_Script.md`
- Modify: `reports/presentations/Report_4_Presentation.html`
- Modify: `reports/presentations/Report_4_Script.md`
- Modify: `reports/presentations/professional_deck/index.html`
- Modify: `reports/presentations/professional_deck/Report_4_Script.md`
- Modify: presentation benchmark assets and release exports

- [ ] **Step 1: Invoke the presentations skill and read its full instructions**

Use the project presentation sources, preserve their design language, and update only slides affected by OCR architecture, data, benchmarks, runtime, and limitations.

- [ ] **Step 2: Replace current-runtime claims with measured candidate results**

Update diagrams from PaddleOCR to Vietnamese MobileNetV3-CTC only if the deployment gate passed. Show PaddleOCR as the baseline comparison. Include the 90% hard gate, real held-out set sizes, Docker p95, 4/8 policy, and one Product camera.

- [ ] **Step 3: Regenerate charts and release exports**

Use the same benchmark data as Task 3. Re-export the relevant presentation PDFs so distributed files match HTML sources.

- [ ] **Step 4: Render every affected slide and inspect visually**

Check overflow, clipped text, unreadable charts, stale labels, broken images, and slide numbering. Correct issues in source, rerender, and repeat until clean.

- [ ] **Step 5: Commit presentation sources and exports**

```bash
git add reports/presentations reports/release
git commit -m "docs: update OCR result presentations"
```

### Task 5: Final verification and handoff

**Files:**
- Review complete repository scope

- [ ] **Step 1: Invoke verification-before-completion and run its required checks**

Run fresh native tests, Docker tests, gate benchmark validation, browser acceptance, documentation claim searches, and presentation rendering checks.

- [ ] **Step 2: Inspect git state and commits**

Run: `git status --short && git log --oneline --decorate -20`

Expected: user-owned `CLAUDE.md` remains untouched; all project changes are committed in focused commits.

- [ ] **Step 3: Report the outcome without rounding across the 90% boundary**

State exact-match numerators and denominators, CER, cold/p50/p95, end-to-end trials, model checksum, test totals, Docker environment, presentation exports, and any target that was not met.
