# Media-Clock Video Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the raw Upload Video player at native browser FPS while Product cam renders the source frame selected by the browser's actual media clock.

**Architecture:** A local Streamlit custom component contains the HTML5 player and returns `{time_s, is_playing, revision}`. A pure `MediaClockSynchronizer` maps that time to source-frame indices and seeks/reopens its OpenCV capture when necessary. The dashboard persists the upload/capture identity in session state and runs exactly one annotated inference per new playback revision.

**Tech Stack:** Streamlit 1.35 custom components, vanilla HTML/JavaScript, OpenCV, NumPy, pytest.

---

### Task 1: Add deterministic media-clock synchronizer

**Files:**
- Create: `main/src/engine/media_clock_sync.py`
- Modify: `main/tests/test_video_pacing.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.engine.media_clock_sync import MediaClockSynchronizer


def test_selects_frame_at_browser_media_time():
    sync = MediaClockSynchronizer(source_fps=30.0)

    assert sync.frame_index_for(11.24) == 337


def test_clamps_negative_media_time_to_first_frame():
    sync = MediaClockSynchronizer(source_fps=30.0)

    assert sync.frame_index_for(-0.1) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_video_pacing.py -q`

Expected: FAIL because `src.engine.media_clock_sync` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class MediaClockSynchronizer:
    def __init__(self, source_fps: float) -> None:
        self.source_fps = source_fps if source_fps > 0 else 30.0

    def frame_index_for(self, time_s: float) -> int:
        return max(0, int(max(0.0, time_s) * self.source_fps))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_video_pacing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/media_clock_sync.py main/tests/test_video_pacing.py
git commit -m "feat: map media clock to source frames"
```

### Task 2: Add seek-aware capture positioning and source timestamp formatting

**Files:**
- Modify: `main/src/engine/media_clock_sync.py`
- Modify: `main/tests/test_video_pacing.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_reopens_capture_for_backward_seek():
    reopened = []
    sync = MediaClockSynchronizer(source_fps=10.0)
    sync.current_frame_index = 50

    target = sync.position_capture(10, reopen_capture=lambda: reopened.append(True))

    assert reopened == [True]
    assert target == 10


def test_formats_rendered_source_time():
    assert MediaClockSynchronizer.format_source_time(11.24) == "00:11.24"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_video_pacing.py -q`

Expected: FAIL because positioning and formatting methods do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def position_capture(self, target_index: int, reopen_capture: Callable[[], None]) -> int:
    if target_index < self.current_frame_index:
        reopen_capture()
        self.current_frame_index = 0
    return target_index

@staticmethod
def format_source_time(time_s: float) -> str:
    centiseconds = max(0, round(time_s * 100))
    minutes, remainder = divmod(centiseconds, 6000)
    seconds, hundredths = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_video_pacing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/engine/media_clock_sync.py main/tests/test_video_pacing.py
git commit -m "feat: handle media seeks in product stream"
```

### Task 3: Build the browser media-clock component

**Files:**
- Create: `main/src/ui/media_clock_video.py`
- Create: `main/src/ui/components/media_clock_video/index.html`
- Create: `main/tests/test_media_clock_video.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.ui.media_clock_video import component_entrypoint, normalize_media_state


def test_component_entrypoint_exists():
    assert component_entrypoint().is_file()


def test_normalizes_valid_component_value():
    assert normalize_media_state({"time_s": 11.24, "is_playing": True, "revision": 7}) == {
        "time_s": 11.24, "is_playing": True, "revision": 7,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_media_clock_video.py -q`

Expected: FAIL because the wrapper module does not exist.

- [ ] **Step 3: Write minimal implementation**

Create a vanilla component frontend that sends `streamlit:componentReady`, handles `streamlit:render`, preserves its `<video>` source while reruns occur, and sends `streamlit:setComponentValue` with `time_s`, `is_playing`, and incrementing `revision` on metadata, play, pause, seek, and 100 ms playback ticks. Create the Python wrapper with `components.declare_component(..., path=...)`; it exposes `media_clock_video(media_url, key)` and rejects malformed values with `normalize_media_state` returning `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_media_clock_video.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/ui/media_clock_video.py main/src/ui/components/media_clock_video/index.html main/tests/test_media_clock_video.py
git commit -m "feat: report browser media clock to dashboard"
```

### Task 4: Replace wall-clock Upload Video loop with media-clock processing

**Files:**
- Modify: `main/src/ui/dashboard.py:38,646-720`
- Modify: `main/tests/test_video_pacing.py`
- Modify: `main/tests/test_media_clock_video.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_same_component_revision_does_not_request_another_product_frame():
    assert should_process_revision(last_revision=7, media_state={"revision": 7, "is_playing": True}) is False


def test_paused_video_does_not_request_product_frame():
    assert should_process_revision(last_revision=6, media_state={"revision": 7, "is_playing": False}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_media_clock_video.py main/tests/test_video_pacing.py -q`

Expected: FAIL because the revision gate does not exist.

- [ ] **Step 3: Write minimal implementation**

Persist `video_path`, `video_id`, capture state, and last component revision in `st.session_state`. Convert the source file into a Streamlit media URL once using the media manager, render `media_clock_video`, and process only its newest playing revision. Position the OpenCV capture at `floor(time_s * fps)`, reopen it for backward seeks, process one frame, and display `Product cam — <measured fps>; source 00:MM.ss`. Stop resets only the processing state, not the browser player.

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/test_media_clock_video.py main/tests/test_video_pacing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/src/ui/dashboard.py main/src/engine/media_clock_sync.py main/tests/test_video_pacing.py main/tests/test_media_clock_video.py
git commit -m "fix: synchronize product cam to browser video clock"
```

### Task 5: Verify native and Docker runtime behavior

**Files:**
- Modify: `README.md:126-134`
- Modify: `main/README.md:124-132`

- [ ] **Step 1: Add the user-facing behavior note**

Document that Upload Video has a native source player and Product cam displays its own measured inference FPS with a visible source timestamp, including pause/seek behavior.

- [ ] **Step 2: Run targeted and complete native tests**

Run: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests -q`

Expected: PASS (existing optional/model tests may remain skipped).

- [ ] **Step 3: Run complete Docker tests**

Run: `docker compose exec -T -w /app/main backend pytest -q`

Expected: PASS (Docker-only skips are acceptable).

- [ ] **Step 4: Manually validate the Docker dashboard**

Run: `docker compose up -d frontend backend`

At `http://localhost:8501`, start the default video, confirm Product cam timestamp follows raw video during play, stops while paused, and jumps with a seek.

- [ ] **Step 5: Commit**

```bash
git add README.md main/README.md
git commit -m "docs: explain synchronized product video playback"
```

