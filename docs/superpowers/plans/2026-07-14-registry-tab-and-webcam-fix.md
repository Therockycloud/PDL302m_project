# Registry Tab + Webcam Fix Implementation Plan

> **For agentic workers:** Implement task-by-task. Checkbox steps for tracking.

**Goal:** Add isolated Registry mode (view/add/delete with photos) and fix Webcam in Docker via browser camera → `/demo/frame`.

**Architecture:** Small `registry_store` helper owns CSV+photo IO. Dashboard gains a fourth Input Mode. Webcam uses a lightweight Streamlit component (or media_clock live mode) with `getUserMedia`, reusing demo session API.

**Tech Stack:** Streamlit, FastAPI `/demo/frame`, pandas CSV, OpenCV only for image encode/decode of uploads.

---

### Task 1: Registry store helper + tests

**Files:**
- Create: `main/src/utils/registry_store.py`
- Create: `main/tests/test_registry_store.py`

- [x] List vehicles from CSV + resolve photo path by normalized plate
- [x] `add_vehicle(plate, brand, color, image_bytes|None)` — reject duplicates
- [x] `delete_vehicle(plate)` — remove CSV row + photo if present
- [x] Unit tests on temp dir/CSV

### Task 2: Registry UI mode in dashboard

**Files:**
- Modify: `main/src/ui/dashboard.py`
- Modify: `main/README.md` (short note)

- [x] Add `Registry` to Input Mode selectbox
- [x] When Registry: render gallery + add form only (skip live pipeline chrome where practical; results panel can show empty/caption)
- [x] Wire add/delete → registry_store → reload matcher in `st.session_state["models"]`
- [x] Do not break Image/Video/Webcam paths

### Task 3: Browser webcam component

**Files:**
- Create or extend: `main/src/ui/components/live_webcam/index.html` (preferred new small component)
- Create/Modify: `main/src/ui/live_webcam.py` wrapper
- Modify: `main/src/ui/dashboard.py` Webcam branch
- Tests: thin wrapper test if easy; otherwise manual/Docker smoke

- [x] `getUserMedia` → canvas sample ~10Hz → `POST {api}/demo/frame` with session_id
- [x] Draw overlays from API response like media_clock
- [x] Emit final decision events compatible with `normalize_demo_event` / existing results panel wiring (reuse patterns from Upload Video event handling)
- [x] Remove blocking `cv2.VideoCapture(0)` loop as primary path
- [x] Show permission / no-camera errors in component or Streamlit caption

### Task 4: Verify + restart

- [x] `pytest main/tests/test_registry_store.py main/tests/test_matching.py -q` (and any new UI helper tests)
- [x] `docker compose restart backend frontend`
- [x] Commit with explicit paths (no `git add -A`), no push
