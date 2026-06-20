# WS-1: Đọc biển pha-lùi + ROI chuồng-mình + latency <1s — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: dùng superpowers:subagent-driven-development. Mỗi Task = 1 Sonnet subagent (`model: "sonnet"`). Opus verify giữa các task. Steps dùng checkbox `- [ ]`.

**Goal:** Hệ thống đọc đầy đủ biển trong pha xe đang lùi (trước khi đỗ hẳn), chỉ với xe ở chuồng của mình, latency <1s.

**Architecture:** `ParkingTrigger` lọc ROI-first (loại xe bên cạnh/đi ngang) + mở cửa sổ quyết định trong pha *approach* (không chờ đứng yên). `ParkingSession` gom + OCR biển trong pha lùi, **chốt khi 2 read trùng ở conf≥0.6**, rồi `mark_decided`. `DecisionEngine` lock-aware.

**Tech Stack:** Python 3.12 (miniforge), pytest, NumPy, OpenCV, ONNX YOLOv8, PaddleOCR. Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q`.

**Spec nguồn:** [docs/superpowers/specs/2026-06-20-plate-read-approach-latency-fa-api-design.md](../specs/2026-06-20-plate-read-approach-latency-fa-api-design.md) §3 WS-1.

**Bằng chứng dữ liệu (probe thật):** biển `30M71854` đọc được frame 492–534 (area 0.16–0.79, conf 0.85–0.87); đỗ-đỉnh frame 666 (area 0.98) biển mù. Tới 14 box xe/khung.

---

## File structure

| File | Trách nhiệm | Thay đổi |
|---|---|---|
| `main/src/engine/parking_trigger.py` | State machine gate | ROI-first + approach-window (bỏ điều kiện đứng yên) |
| `main/src/engine/parking_session.py` | Orchestrate gom/đọc | Capture plate+conf+color+conf mỗi frame; gọi lock |
| `main/src/engine/decision_engine.py` | Gộp → verdict | Lock-aware: chốt khi plate trùng ≥lock_repeat @conf≥lock_conf |
| `main/configs/config.yaml` | Tham số | conf 0.3→0.15, roi, approach/lock constants |
| `main/data/database.csv` | CSDL demo | +`30M-718.54` + mở rộng ~15–20 xe |
| `main/src/ui/dashboard.py` | UI + load models | Warmup; truyền constants xuống trigger/session |
| `main/tests/test_parking_trigger.py` | test trigger | Thêm ROI-first + approach; cập nhật test "đứng yên" cũ |
| `main/tests/test_decision_engine.py` | test lock | Thêm lock-on-repeat |
| `main/tests/test_parking_session.py` | test session | Thêm capture+lock |
| `main/tests/test_matching.py` | test DB | AUTHORIZED cho biển demo mới |

**Hằng số config (đọc với default an toàn):** `plate_detector.conf_threshold: 0.15`; `pipeline.trigger.roi: [0.35,0.30,0.65,1.0]`, `approach_min_area: 0.15`, `min_persist_frames: 3`; `pipeline.lock: {capture_conf: 0.50, lock_conf: 0.60, lock_repeat: 2, late_area: 0.85}`.

---

## Task 1: ROI-first own-slot gating (loại xe bên cạnh/đi ngang)

**Files:** Modify `main/src/engine/parking_trigger.py`; Modify `main/tests/test_parking_trigger.py`

- [ ] **Step 1 — Viết test fail.** Thêm vào `test_parking_trigger.py` (helper `_det` hiện chỉ trả 1 box; thêm helper đa-box):

```python
def _box(x1, y1, x2, y2, conf=0.9):
    return {"bbox": (x1, y1, x2, y2), "conf": conf}

def test_picks_in_roi_vehicle_over_larger_outside_roi():
    # ROI = middle band. A BIGGER car sits centered OUTSIDE roi (far left);
    # a smaller car is centered INSIDE roi. Trigger must track the in-ROI one.
    t = ParkingTrigger(roi=(0.35, 0.30, 0.65, 1.0), min_area_ratio=0.10, min_persist_frames=3)
    big_outside = _box(0, 0, 250, 460)         # center x≈0.20 (outside roi), large
    small_inside = _box(300, 260, 430, 470)    # center x≈0.57, y≈0.76 (inside roi)
    state = t.update([big_outside, small_inside], FRAME)
    assert state in (TRACKING, READY_TO_DECIDE)  # did NOT reset to IDLE on the outside car

def test_all_vehicles_outside_roi_is_idle():
    t = ParkingTrigger(roi=(0.35, 0.30, 0.65, 1.0), min_area_ratio=0.10)
    assert t.update([_box(0, 0, 120, 200)], FRAME) == IDLE
```

Run: `... -m pytest tests/test_parking_trigger.py -q` → **FAIL** (`ParkingTrigger.__init__` chưa nhận `min_persist_frames`; chưa lọc ROI-first).

- [ ] **Step 2 — Implement.** Trong `parking_trigger.py`:
  - `__init__` thêm tham số `min_persist_frames: int = 3` (lưu `self.min_persist_frames`, `self._persist = 0`).
  - Thêm `_largest_in_roi(detections, frame_shape)`: tính center từng box, **lọc các box có center trong ROI và area ≥ `min_area_ratio`**, trả box lớn nhất trong số đó (None nếu rỗng). Dùng nó thay cho `_largest` ở đầu `update`.
  - Nếu không có box hợp lệ trong ROI → `reset()` → IDLE.
  - Logic ROI dùng `_in_roi(cx, cy)` sẵn có.

- [ ] **Step 3 — Run.** `... -m pytest tests/test_parking_trigger.py -q` → **PASS** (giữ các test cũ chưa đụng tới approach vẫn xanh ở task này nếu chưa đổi gate; nếu test cũ phụ thuộc `_largest` toàn-khung fail thì chuyển sang Task 2 xử lý).

- [ ] **Step 4 — Commit.**
```bash
git add main/src/engine/parking_trigger.py main/tests/test_parking_trigger.py
git commit -m "feat(trigger): ROI-first own-slot gating; ignore adjacent/passing vehicles (WS-1)"
```

---

## Task 2: Approach-window gate (mở quyết định trong pha lùi, bỏ điều kiện đứng yên)

**Files:** Modify `main/src/engine/parking_trigger.py`; Modify `main/tests/test_parking_trigger.py`

- [ ] **Step 1 — Viết/cập nhật test.**
  - **Thêm:**
```python
def test_ready_after_persist_even_while_moving():
    # In reverse approach the car is MOVING (center drifts) but must still open
    # the decision window once it has persisted in ROI for min_persist_frames.
    t = ParkingTrigger(roi=(0.35,0.30,0.65,1.0), min_area_ratio=0.10, min_persist_frames=3)
    boxes = [_box(300,250,430,470), _box(305,258,438,478), _box(298,262,432,472)]  # in-ROI, jittering
    states = [t.update([b], FRAME) for b in boxes]
    assert states[-1] == READY_TO_DECIDE   # motion does NOT block readiness

def test_tracking_before_persist_threshold():
    t = ParkingTrigger(roi=(0.35,0.30,0.65,1.0), min_area_ratio=0.10, min_persist_frames=3)
    s1 = t.update([_box(300,250,430,470)], FRAME)
    assert s1 == TRACKING  # only 1 frame persisted < 3
```
  - **Cập nhật test cũ đã lỗi thời** (giả định "chờ đứng yên"): `test_tracking_then_ready_when_large_low_and_stable` → đổi kỳ vọng sang "READY sau khi persisted min_persist_frames"; `test_jitter_keeps_tracking_not_ready` → **đổi ý nghĩa**: jitter KHÔNG còn chặn READY (ta muốn đọc lúc đang lùi). Đổi tên thành `test_leaving_roi_resets_to_idle` kiểm tra: xe rời ROI → IDLE.

Run → **FAIL**.

- [ ] **Step 2 — Implement.** Trong `update()`:
  - Bỏ gate `_is_stable()` để mở READY. Thay bằng đếm persist: mỗi frame có target hợp lệ trong ROI → `self._persist += 1`; nếu không hợp lệ → `reset()` (đặt `_persist=0`).
  - `state = READY_TO_DECIDE` khi `self._persist >= self.min_persist_frames`, ngược lại `TRACKING`.
  - Giữ `DECIDED` latch (mark_decided) tới khi target rời ROI thì `reset`.
  - Có thể giữ `_is_stable`/`_centers` (không dùng để gate) hoặc xoá nếu không test nào cần — DRY: xoá nếu thừa.

- [ ] **Step 3 — Run.** `... -m pytest tests/test_parking_trigger.py -q` → **PASS** (toàn bộ, gồm test cũ đã cập nhật).

- [ ] **Step 4 — Commit.**
```bash
git add main/src/engine/parking_trigger.py main/tests/test_parking_trigger.py
git commit -m "feat(trigger): open decision window during approach (persist-based, not stillness) (WS-1)"
```

---

## Task 3: Capture + plate-lock trong pha lùi (session + decision engine)

**Files:** Modify `main/src/engine/decision_engine.py`, `main/src/engine/parking_session.py`; Modify `main/tests/test_decision_engine.py`, `main/tests/test_parking_session.py`

- [ ] **Step 1 — Viết test fail (decision_engine lock).** Thêm vào `test_decision_engine.py`:

```python
def test_locks_on_two_consistent_high_conf_reads():
    # readings carry plate_text + plate_conf; lock when same plate repeats
    # >= lock_repeat at conf >= lock_conf.
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.8},
        {"plate_text": "",          "plate_conf": 0.0,  "color": "WHITE", "color_conf": 0.5},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30M71854"
    assert out["status"] == "AUTHORIZED"

def test_single_high_conf_read_does_not_lock():
    frames = [{"plate_text": "30M71854", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9}]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] in ("UNCERTAIN", "NO_PLATE")  # not enough evidence to lock

def test_low_conf_reads_do_not_lock():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.40, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.45, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] in ("UNCERTAIN", "NO_PLATE")
```
(Định nghĩa `_FakeMatcher` nếu chưa có: `verify_vehicle(plate,color)` → AUTHORIZED nếu plate∈registered & màu khớp, else UNREGISTERED.)

Run → **FAIL** (`aggregate` chưa nhận `lock_conf/lock_repeat`, chưa dùng `plate_conf`).

- [ ] **Step 2 — Implement decision_engine.** `aggregate(self, frames_data, lock_conf=0.60, lock_repeat=2)`:
  - Lọc reads có `plate_text` không rỗng **và** `plate_conf >= lock_conf`; đếm theo text.
  - Nếu text nào đạt count ≥ `lock_repeat` → đó là plate chốt; vote color (majority, kèm color_conf trung bình của màu thắng); gọi `matcher.verify_vehicle(plate, color)` → verdict (giữ field `votes_meta` + thêm `color_conf`).
  - Nếu không có text nào đạt lock nhưng có ≥1 read hợp lệ → `UNCERTAIN`/LOG (chưa đủ tin để chốt). Nếu không có read nào → `NO_PLATE` (như cũ).
  - Giữ tương thích: tham số mới có default.

- [ ] **Step 3 — Run decision tests.** `... -m pytest tests/test_decision_engine.py -q` → **PASS**.

- [ ] **Step 4 — Viết test fail (session capture+lock).** Thêm vào `test_parking_session.py` (theo pattern fakes hiện có): fake `plate_reader.read(crop)` trả `{"text":"30M71854","conf":0.85,...}`, fake `color_clf.predict` trả `("WHITE",0.9)`; cho trigger luôn READY; chạy đủ frame → `process_frame` cuối có `decision["plate"]=="30M71854"` và `trigger.state=="DECIDED"`. Thêm 1 test: nếu plate_reader chỉ trả 1 lần rồi rỗng → không DECIDED (chưa lock).

Run → **FAIL**.

- [ ] **Step 5 — Implement session.** `_collect`: lưu dict đầy đủ `{"plate_text","plate_conf","color","color_conf"}` (lấy `plate=self.plate_reader.read(crop)` → text+conf; `color,color_conf=self.color_clf.predict(crop)`). Sau mỗi collect khi READY: gọi `decision_engine.aggregate(self._collected, lock_conf, lock_repeat)`; nếu kết quả **đã lock** (status ∈ {AUTHORIZED, UNREGISTERED} — `matching.py` không có status `MISMATCH`; màu lệch là AUTHORIZED + action ALLOW_WARN) → set `self._decision`, `trigger.mark_decided()`. Trạng thái `UNCERTAIN`/`NO_PLATE` → chưa chốt, tiếp tục gom. Bỏ điều kiện `len>=collect_frames` cứng (chốt theo lock, không theo đếm đủ frame). Truyền `lock_conf/lock_repeat` vào session từ config (thêm tham số `__init__` default).

- [ ] **Step 6 — Run.** `... -m pytest tests/test_parking_session.py tests/test_decision_engine.py -q` → **PASS**.

- [ ] **Step 7 — Commit.**
```bash
git add main/src/engine/decision_engine.py main/src/engine/parking_session.py main/tests/test_decision_engine.py main/tests/test_parking_session.py
git commit -m "feat(decision): lock plate on repeated high-conf reads during approach (WS-1)"
```

---

## Task 4: Config + wiring + plate conf 0.15

**Files:** Modify `main/configs/config.yaml`, `main/src/ui/dashboard.py`

- [ ] **Step 1 — Config.** Thêm/sửa: `plate_detector.conf_threshold: 0.15`; dưới `pipeline.trigger`: `roi: [0.35, 0.30, 0.65, 1.0]`, `approach_min_area: 0.15`, `min_persist_frames: 3`; thêm block `pipeline.lock: {capture_conf: 0.50, lock_conf: 0.60, lock_repeat: 2, late_area: 0.85}`.

- [ ] **Step 2 — Wiring.** Trong `dashboard.py` chỗ dựng `ParkingTrigger`/`ParkingSession` (dòng ~194–202): truyền `min_persist_frames`, `lock_conf`, `lock_repeat` từ config (đọc với `.get(..., default)`).

- [ ] **Step 3 — Calibrate ROI.** Chạy `KMP_DUPLICATE_LIB_OK=TRUE <py> main/scripts/calibrate_roi.py --source main/data/test/sample_parking.mp4 --frame-frac 0.7` để xác nhận/điều chỉnh `roi` sao cho xe target hiện GATE còn box bên cạnh bị loại; cập nhật `config.yaml` nếu cần.

- [ ] **Step 4 — Run full suite.** `... -m pytest -q` → **≥ 44 passed** (+ test mới), 0 failed.

- [ ] **Step 5 — Commit.**
```bash
git add main/configs/config.yaml main/src/ui/dashboard.py
git commit -m "feat(config): plate conf 0.15, own-slot ROI, approach/lock constants (WS-1)"
```

---

## Task 5: Warmup models (xe đầu không dính cold-start)

**Files:** Modify `main/src/ui/dashboard.py` (hàm load models)

- [ ] **Step 1 — Implement.** Sau khi dựng xong các model, chạy mỗi model 1 lần trên ảnh giả `np.zeros((320,320,3),uint8)` trong `try/except` (vehicle_det.detect, plate_det.detect, color_clf.predict, ocr.read_plate). Log "warmup done".

- [ ] **Step 2 — Smoke.** Import dashboard module + gọi hàm load (mock streamlit nếu cần) không raise; hoặc test nhẹ rằng warmup helper chạy với fake models. `... -m pytest -q` xanh.

- [ ] **Step 3 — Commit.**
```bash
git add main/src/ui/dashboard.py
git commit -m "perf(runtime): warmup models at load to remove first-vehicle cold-start (WS-1)"
```

---

## Task 6: DB demo entry + mở rộng

**Files:** Modify `main/data/database.csv`; Modify `main/tests/test_matching.py`

- [ ] **Step 1 — Xác định màu xe target.** Chạy color_clf trên crop xe target của clip (frame ~516) lấy màu dự đoán → dùng làm `car_color` cho `30M-718.54` (để demo ra AUTHORIZED không cảnh báo).
- [ ] **Step 2 — Test fail.** Trong `test_matching.py`: `verify_vehicle("30M71854", "<màu vừa xác định>")` → `status=="AUTHORIZED"`, `color_warning==False`.
- [ ] **Step 3 — Sửa CSV.** Thêm dòng `30M-718.54,<brand>,<màu>` + ~14 dòng xe mẫu khác (biển VN hợp lệ đa dạng màu) → ~15–20 xe.
- [ ] **Step 4 — Run.** `... -m pytest tests/test_matching.py -q` → **PASS**.
- [ ] **Step 5 — Commit.**
```bash
git add main/data/database.csv main/tests/test_matching.py
git commit -m "feat(data): register demo plate 30M-718.54 + expand DB to ~18 vehicles (WS-1)"
```

---

## Verify cuối (Opus, không phải Sonnet)

- [ ] `cd main && KMP_DUPLICATE_LIB_OK=TRUE <py> -m pytest -q` → ≥44 passed, 0 failed.
- [ ] **E2E headless** trên `sample_parking.mp4`: chạy `ParkingSession` thật → `decision["plate"]=="30M71854"`, **frame quyết định < 666** (đỗ-đỉnh), latency tới chốt <1s (đo & in). Loại được box xe khác.
- [ ] Đối chiếu G1–G3 của spec. Tick plan + ghi Progress log.
```
