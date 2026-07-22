# Real-Footage UNCERTAIN Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Soften real-footage plate locking, widen trigger ROI defaults, add `parking_case_real_v2.mp4` demo checkbox, and keep barrier closed when no plate is localized.

**Architecture:** Extend lock-aware `DecisionEngine.aggregate` with an `allow_soft_lock` path used only when `ParkingSession` exhausts `collect_frames`. Wire new lock knobs through `config.yaml` → `pipeline_factory` → `ParkingSession`. Dashboard gains a second bundled demo video selector.

**Tech Stack:** Python, pytest, Streamlit dashboard, existing YOLO/PaddleOCR pipeline.

**Spec:** `docs/superpowers/specs/2026-07-14-real-footage-uncertain-robustness-design.md`

---

### Task 1: Soft-lock in DecisionEngine + unit tests

**Files:**
- Modify: `main/src/engine/decision_engine.py`
- Modify: `main/tests/test_decision_engine.py`

- [x] **Step 1:** Add failing tests:
  - 2 reads at conf 0.45 with `allow_soft_lock=True`, `soft_conf=0.40` → locks (AUTHORIZED/UNREGISTERED via matcher)
  - same frames with `allow_soft_lock=False` → UNCERTAIN
  - 1 read at 0.90 with `allow_soft_lock=True`, `single_lock_conf=0.85` → locks
  - empty plates → still NO_PLATE (caller maps to UNCERTAIN)

- [x] **Step 2:** Implement in `_aggregate_lock_aware`:
  - new kwargs `soft_conf=0.40`, `single_lock_conf=0.85`, `allow_soft_lock=False`
  - after hard-lock fails: if `allow_soft_lock`, apply soft rules from the spec; else existing UNCERTAIN message

- [x] **Step 3:** Run `pytest main/tests/test_decision_engine.py -q` and fix until green.

- [x] **Step 4:** Commit with explicit paths (if committing in this session).

---

### Task 2: Wire config through ParkingSession + factory

**Files:**
- Modify: `main/configs/config.yaml`
- Modify: `main/src/engine/parking_session.py`
- Modify: `main/src/engine/pipeline_factory.py`
- Modify: `main/tests/test_parking_session.py`

- [x] **Step 1:** Update config:
  - `pipeline.collect_frames: 10`
  - `pipeline.trigger.roi: [0.20, 0.20, 0.80, 1.0]`
  - `pipeline.trigger.min_area_ratio: 0.10`
  - `pipeline.trigger.approach_min_area: 0.10`
  - `pipeline.lock.lock_conf: 0.50`
  - `pipeline.lock.soft_conf: 0.40`
  - `pipeline.lock.single_lock_conf: 0.85`

- [x] **Step 2:** `ParkingSession.__init__` accepts `soft_conf`, `single_lock_conf`; when calling `aggregate`, pass `allow_soft_lock=(len(self._collected) >= self.collect_frames)`.

- [x] **Step 3:** `build_parking_session` reads and forwards the new knobs; update default `lock_conf` fallback to `0.50` and `collect_frames` fallback to `10` to match config.

- [x] **Step 4:** Adjust parking_session tests that assumed collect_frames=5 / old finalize timing; add one test that a single 0.90 read finalizes via soft lock after budget.

- [x] **Step 5:** `pytest main/tests/test_parking_session.py main/tests/test_decision_engine.py -q`

---

### Task 3: Install demo video + dashboard selector + DB row

**Files:**
- Create/copy: `main/data/test/parking_case_real_v2.mp4` from  
  `/Users/konalyn/Downloads/1783988959999_896412636915093981_896412636915093981.mp4`
- Modify: `main/src/ui/dashboard.py`
- Modify: `main/data/database.csv` (only if plate text is known/readable)
- Modify: `main/README.md` (one short note on the new checkbox)

- [x] **Step 1:** Copy the mp4 into `main/data/test/parking_case_real_v2.mp4`.

- [x] **Step 2:** In Upload Video mode, add checkbox **Play Real-World Case** that plays `parking_case_real_v2.mp4` when present; mutually exclusive with **Play Default Parking Video**. Use a distinct `video_id` (e.g. `real-world-case-v2`).

- [x] **Step 3:** If a headless/manual read yields a stable plate for that clip, append it to `database.csv` with plausible brand/color (White Kia). If unknown, skip DB change and leave UNREGISTERED as acceptable demo outcome.

- [x] **Step 4:** Smoke-check imports / syntax for dashboard helpers if tests exist; otherwise rely on Task 4.

---

### Task 4: Automated verification

- [x] **Step 1:** `pytest main/tests/test_decision_engine.py main/tests/test_parking_session.py -q`

- [x] **Step 2:** If `main/scripts/run_on_video.py` (or equivalent) exists and models load, run headless on `parking_case_real_v2.mp4` and report the final decision status. Prefer lock over UNCERTAIN when OCR evidence exists; empty plate detector miss → UNCERTAIN is OK.

- [x] **Step 3:** Summarize results for the parent agent.
