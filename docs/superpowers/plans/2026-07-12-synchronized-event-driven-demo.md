# Synchronized Event-Driven Parking Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Streamlit-rerun video processing with one browser-owned media clock that renders synchronized Origin/Product panes and sends event-driven sampled frames directly to FastAPI.

**Architecture:** FastAPI owns per-browser `ParkingSession` state and exposes frame/reset endpoints. A vanilla Streamlit component renders one HTML5 video into Origin and Product surfaces, submits at most one sampled JPEG request at a time, draws returned overlays, and notifies Streamlit only for final decisions. Upload Image and Webcam remain unchanged.

**Tech Stack:** Streamlit 1.35 custom components, vanilla HTML Canvas/JavaScript, FastAPI, OpenCV, pytest, Docker Compose.

---

### Task 1: Extract the shared ParkingSession factory

**Files:**
- Modify: `main/src/engine/pipeline_factory.py`
- Modify: `main/src/ui/dashboard.py`
- Create: `main/tests/test_parking_session_factory.py`

- [ ] **Step 1: Write the failing factory test**

```python
from src.engine.pipeline_factory import build_parking_session


def test_build_parking_session_uses_pipeline_collaborators(monkeypatch):
    pipeline = {
        "vehicle_detector": object(),
        "plate_reader": type("Reader", (), {"ocr_reader": object()})(),
        "color_clf": object(),
        "decision_engine": object(),
        "matcher": object(),
    }
    cfg = {
        "pipeline": {
            "trigger": {"stable_frames": 3},
            "lock": {"lock_conf": 0.7, "lock_repeat": 2},
        }
    }

    session = build_parking_session(pipeline, cfg)

    assert session.vehicle_detector is pipeline["vehicle_detector"]
    assert session.plate_reader is pipeline["plate_reader"]
    assert session.color_clf is pipeline["color_clf"]
    assert session.lock_conf == 0.7
    assert session.lock_repeat == 2
```

- [ ] **Step 2: Verify RED**

Run: `cd main && pytest tests/test_parking_session_factory.py -q`

Expected: FAIL because `build_parking_session` does not exist.

- [ ] **Step 3: Implement the shared factory**

Add this public function in `pipeline_factory.py`:

```python
def build_parking_session(pipeline: dict, cfg: dict):
    from src.engine.parking_session import ParkingSession
    from src.engine.parking_trigger import ParkingTrigger

    plate_reader = pipeline.get("plate_reader")
    required = (
        pipeline.get("vehicle_detector"),
        plate_reader,
        pipeline.get("color_clf"),
        pipeline.get("decision_engine"),
    )
    if any(item is None for item in required):
        return None

    pcfg = cfg.get("pipeline", {})
    tcfg = pcfg.get("trigger", {})
    lcfg = pcfg.get("lock", {})
    return ParkingSession(
        vehicle_detector=pipeline["vehicle_detector"],
        plate_reader=plate_reader,
        color_clf=pipeline["color_clf"],
        decision_engine=pipeline["decision_engine"],
        trigger=ParkingTrigger(
            roi=tcfg.get("roi"),
            min_area_ratio=tcfg.get("min_area_ratio", 0.15),
            stable_frames=tcfg.get("stable_frames", 5),
            move_eps=tcfg.get("move_eps", 0.02),
        ),
        sample_interval=pcfg.get("sample_interval", 5),
        collect_frames=pcfg.get("collect_frames", 5),
        lock_conf=lcfg.get("lock_conf", 0.60),
        lock_repeat=lcfg.get("lock_repeat", 2),
    )
```

Replace the dashboard's duplicate `ParkingSession` construction with this factory.

- [ ] **Step 4: Verify GREEN and regression tests**

Run: `cd main && pytest tests/test_parking_session_factory.py tests/test_parking_session.py tests/test_pipeline_factory.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/pipeline_factory.py main/src/ui/dashboard.py main/tests/test_parking_session_factory.py
git commit -m "refactor: share parking session construction"
```

### Task 2: Add isolated backend demo sessions

**Files:**
- Create: `main/src/engine/demo_session_manager.py`
- Create: `main/tests/test_demo_session_manager.py`

- [ ] **Step 1: Write failing isolation/reset tests**

```python
from src.engine.demo_session_manager import DemoSessionManager


class FakeSession:
    def __init__(self):
        self.frames = []
        self.reset_count = 0

    def process_frame(self, frame):
        self.frames.append(frame)
        return {"state": "TRACKING", "overlay_results": [], "decision": None}

    def reset(self):
        self.reset_count += 1


def test_sessions_are_isolated():
    created = []
    manager = DemoSessionManager(lambda: created.append(FakeSession()) or created[-1])

    first = manager.get("session-a")
    second = manager.get("session-b")

    assert first is not second


def test_reset_discards_trajectory_history():
    manager = DemoSessionManager(FakeSession)
    original = manager.get("session-a")

    manager.reset("session-a")

    assert original.reset_count == 1
    assert manager.get("session-a") is not original
```

- [ ] **Step 2: Verify RED**

Run: `cd main && pytest tests/test_demo_session_manager.py -q`

Expected: FAIL because the manager module does not exist.

- [ ] **Step 3: Implement the manager**

Implement a lock-protected dictionary keyed by a validated session ID. `get()`
creates one session through the injected factory. `reset()` calls the old
session's `reset()` and removes it. Store `last_access` and implement
`expire(now, max_idle_s=300)` to reset/remove inactive sessions.

- [ ] **Step 4: Add and pass expiry test**

```python
def test_expire_removes_inactive_sessions():
    manager = DemoSessionManager(FakeSession, clock=lambda: 0.0)
    old = manager.get("session-a")
    manager.expire(now=301.0, max_idle_s=300.0)
    assert old.reset_count == 1
```

Run: `cd main && pytest tests/test_demo_session_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/demo_session_manager.py main/tests/test_demo_session_manager.py
git commit -m "feat: isolate event-driven demo sessions"
```

### Task 3: Expose FastAPI frame and reset endpoints

**Files:**
- Modify: `main/src/api/app.py`
- Modify: `main/tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Add a fake demo manager whose session returns one bbox and a decision. Test:

```python
def test_demo_frame_returns_source_timestamp_and_json_safe_overlay(client_with_fake_demo):
    response = client_with_fake_demo.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "11.24"},
        files=_make_test_image_files(),
    )
    assert response.status_code == 200
    assert response.json()["source_time_s"] == 11.24
    assert response.json()["overlay_results"][0]["bbox"] == [1, 2, 3, 4]


def test_demo_frame_rejects_invalid_session_id(client_with_fake_demo):
    response = client_with_fake_demo.post(
        "/demo/frame",
        data={"session_id": "../bad", "source_time_s": "1.0"},
        files=_make_test_image_files(),
    )
    assert response.status_code == 400


def test_demo_reset_clears_session(client_with_fake_demo):
    response = client_with_fake_demo.delete("/demo/session/browser-123")
    assert response.status_code == 204
```

- [ ] **Step 2: Verify RED**

Run: `cd main && pytest tests/test_api.py -q`

Expected: FAIL with 404 for the new routes.

- [ ] **Step 3: Implement endpoints and CORS**

Add CORS middleware for `http://localhost:8501` and
`http://127.0.0.1:8501`. Create the demo manager in lifespan using
`build_parking_session(pipeline, cfg)`.

Implement:

```python
@app.post("/demo/frame")
def process_demo_frame(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    source_time_s: float = Form(...),
) -> JSONResponse:
    # validate ^[A-Za-z0-9_-]{8,64}$ and finite nonnegative timestamp
    # decode image, get session, call process_frame sequentially
    # return source_time_s, state, JSON-safe bbox/conf/class, decision,
    # and measured latency_ms
```

Implement `DELETE /demo/session/{session_id}` returning status 204. Map
`READY_TO_DECIDE` to UI state `REVERSING_VERIFYING`; keep other trigger
states unchanged. Strip crops/NumPy arrays from responses.

- [ ] **Step 4: Verify GREEN**

Run: `cd main && pytest tests/test_api.py tests/test_demo_session_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/api/app.py main/tests/test_api.py
git commit -m "feat: process synchronized demo frames in api"
```

### Task 4: Replace the clock-reporting component with a synchronized dual renderer

**Files:**
- Modify: `main/src/ui/components/media_clock_video/index.html`
- Modify: `main/src/ui/media_clock_video.py`
- Modify: `main/tests/test_media_clock_video.py`

- [ ] **Step 1: Write failing component-contract tests**

```python
def test_component_uses_one_video_clock_for_both_panes():
    html = component_entrypoint().read_text()
    assert "requestVideoFrameCallback" in html
    assert 'id="origin-video"' in html
    assert 'id="product-canvas"' in html


def test_component_posts_frames_directly_to_api():
    html = component_entrypoint().read_text()
    assert "fetch(frameEndpoint" in html
    assert "requestInFlight" in html
    assert "canvas.toBlob" in html


def test_component_resets_backend_on_seek():
    html = component_entrypoint().read_text()
    assert 'addEventListener("seeked"' in html
    assert 'method: "DELETE"' in html
```

- [ ] **Step 2: Verify RED**

Run: `cd main && pytest tests/test_media_clock_video.py -q`

Expected: FAIL because the current component reports `currentTime` to Streamlit.

- [ ] **Step 3: Implement the dual renderer**

The HTML must create one visible Origin `<video id="origin-video">` and one
`<canvas id="product-canvas">`. On each
`requestVideoFrameCallback`, draw the exact decoded video frame into Product
canvas, then draw the latest cached bbox/state/decision overlay.

At a configurable sampling interval (default 200 ms), copy the current video
frame to a small offscreen canvas, encode JPEG, and POST multipart fields
`session_id`, `source_time_s`, and `file` directly to
`http://localhost:8000/demo/frame`. If `requestInFlight` is true, drop the
sample. Do not call `streamlit:setComponentValue` for clock ticks.

On a final decision only, call `streamlit:setComponentValue` with the
decision payload and evidence timestamp so Streamlit may update its durable
results panel without driving playback.

On seek, clear cached overlays and send DELETE to the session endpoint.

- [ ] **Step 4: Update Python wrapper contract**

Change `media_clock_video` to accept `media_url`, `api_base_url`, and a
stable `session_id`. Normalize only final-decision events; remove
`should_process_revision` and `should_run_inference`.

- [ ] **Step 5: Verify GREEN and commit**

Run: `cd main && pytest tests/test_media_clock_video.py -q`

Expected: PASS.

```bash
git add main/src/ui/components/media_clock_video/index.html main/src/ui/media_clock_video.py main/tests/test_media_clock_video.py
git commit -m "feat: render synchronized event-driven product video"
```

### Task 5: Simplify Upload Video dashboard integration

**Files:**
- Modify: `main/src/ui/dashboard.py`
- Delete: `main/src/engine/inference_worker.py`
- Delete: `main/src/engine/media_clock_sync.py`
- Delete: `main/tests/test_inference_worker.py`
- Delete: `main/tests/test_video_pacing.py`
- Modify: `main/tests/test_media_clock_video.py`

- [ ] **Step 1: Write failing dashboard helper tests**

Add tests for stable browser session IDs and final event mapping:

```python
def test_demo_session_id_is_stable_for_same_video():
    assert demo_session_id("default-parking-video", "browser") == demo_session_id(
        "default-parking-video", "browser"
    )


def test_final_event_maps_evidence_timestamp():
    mapped = map_demo_decision({
        "decision": {"status": "AUTHORIZED", "plate": "30M71854"},
        "source_time_s": 11.24,
    })
    assert mapped["evidence_time_s"] == 11.24
```

- [ ] **Step 2: Verify RED**

Run: `cd main && pytest tests/test_media_clock_video.py -q`

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Replace the Upload Video loop**

Keep file validation/upload persistence and media URL registration. Replace all
OpenCV capture, media revision, Product `st.image`, and background worker code
with one stable component call:

```python
event = media_clock_video(
    media_url,
    api_base_url=os.getenv("DPL_DEMO_API_URL", "http://localhost:8000"),
    session_id=demo_session_id(video_id, st.session_state["demo_browser_id"]),
    key=f"demo-{video_id}",
)
```

When `event` contains a new final decision, update the existing Streamlit
results log once using its event ID. Do not update Streamlit for frame clocks.
Delete obsolete media-clock and background-worker modules/tests after
`rg` confirms no remaining imports.

- [ ] **Step 4: Verify targeted tests**

Run: `cd main && pytest tests/test_media_clock_video.py tests/test_api.py tests/test_parking_session.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/ui/dashboard.py main/src/ui/media_clock_video.py main/tests/test_media_clock_video.py
git rm main/src/engine/inference_worker.py main/src/engine/media_clock_sync.py main/tests/test_inference_worker.py main/tests/test_video_pacing.py
git commit -m "fix: decouple synchronized video from streamlit reruns"
```

### Task 6: Docker and browser acceptance verification

**Files:**
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `main/README.md`

- [ ] **Step 1: Configure the browser-facing API URL**

Set frontend environment:

```yaml
- DPL_DEMO_API_URL=http://localhost:8000
```

Remove obsolete Compose `version` to eliminate its warning. Document that
ports 8501 and 8000 must both be reachable by the browser.

- [ ] **Step 2: Run full native and Docker tests**

Run:

```bash
cd main
KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests -q
docker compose exec -T -w /app/main backend pytest -q
```

Expected: both suites pass with only documented optional skips.

- [ ] **Step 3: Restart services and verify endpoints**

Run:

```bash
docker compose restart backend frontend
curl -fsS http://localhost:8000/status
curl -I http://localhost:8501
```

Expected: healthy API JSON and HTTP 200 from Streamlit.

- [ ] **Step 4: Browser acceptance test**

Use the default parking video. Verify through component diagnostics:

- Origin and Product media times differ by no more than `1 / source_fps`.
- Source playback continues while a fake/real frame request is in flight.
- Only one request is in flight.
- Seeking resets the backend session and both panes jump together.
- A returned verdict displays its evidence timestamp.
- Browser console has no uncaught errors.

- [ ] **Step 5: Commit documentation/config**

```bash
git add docker-compose.yml README.md main/README.md
git commit -m "docs: describe synchronized event-driven demo"
```

