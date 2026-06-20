# Spec: Đọc biển pha-lùi + latency <1s + báo động giả <5% + đồng bộ API/Dashboard

**Ngày:** 2026-06-20 · **Tác giả:** Opus (plan) · **Trạng thái:** Approved design → chờ writing-plans
**Thực thi:** Sonnet subagent mỗi WS (`model: sonnet`), Opus verify. Commit explicit-path, không push.

---

## 1. Bối cảnh & bằng chứng (data-grounded)

Hệ thống ALPR bãi xe, plate-primary. Camera gắn ở góc chuồng nhìn ra; xe lùi vào, đuôi (biển) hướng camera.

Chẩn đoán bằng probe thật trên `main/data/test/sample_parking.mp4` (898 frame, = bản copy của `parking_case_real.mp4`):

| Pha | Frame | area xe (ratio) | Biển đọc được? |
|---|---|---|---|
| **Đang lùi (approach)** | 492–534 | 0.16 → 0.79 | ✅ `30M71854` (plate-conf 0.85–0.87) |
| **Đã đỗ hẳn (parked)** | 540–894 (đỉnh 666) | 0.94 → 0.98 | ❌ rỗng (plate-conf ~0) |

**Kết luận chẩn đoán:**
1. Biển CHỈ đọc được trong **pha lùi** (area ~0.16–0.8). Khi xe đã vào hẳn (area >0.9) biển quá gần/khuất → mù. Trigger hiện tại chờ `_is_stable()` (đứng yên = đã đỗ) → OCR đúng lúc mù → **NO_PLATE**. Đây là gốc rễ thật.
2. Khung hình có **tới 14 box xe** cùng lúc (xe chuồng bên/đi ngang). `ParkingTrigger._largest()` chọn xe to nhất TRƯỚC khi lọc ROI (parking_trigger.py:51,81) → dễ chọn nhầm xe khác.
3. Latency đo thật: vehicle-detect 42ms, plate-detect 40ms, OCR PaddleOCR ~315ms (steady). Một lần đọc biển ≈ **~400ms**; chốt sau 2 frame trùng ≈ **~750ms < 1s**.
4. Báo động giả 14.5% (`docs/benchmarks/security_eval.md`) sinh từ `matching.py:verify_vehicle` cảnh báo mỗi khi màu đoán ≠ màu đăng ký; nguồn lớn nhất là nhầm trong cụm trung tính (Grey↔Silver↔White↔Black, confusion matrix Report 3 §5.1).
5. API (`main/src/api/app.py`) là code-path CŨ riêng: dùng `PlateOCR` (EasyOCR, 0% exact-match Benchmark C), còn tính hãng trong luồng, OCR 1 tầng — lệch hẳn dashboard (PaddleOCR, 2 tầng, plate-primary).

---

## 2. Mục tiêu / Ngoài phạm vi

**Mục tiêu (đo được):**
- G1. Đọc đầy đủ biển `30M71854` trên clip mặc định, **chốt trong pha lùi** (frame quyết định < frame đỗ-đỉnh 666).
- G2. Latency tới lúc chốt biển **< 1s** (compute), warmup để xe đầu không dính cold-start.
- G3. **Chỉ xe ở chuồng của mình** (trong ROI) được đọc; loại xe bên cạnh / đi ngang (13+ box còn lại không gây quyết định).
- G4. Báo động giả **< 5%** (từ 14.5%), kèm **bằng chứng before/after** (bảng + biểu đồ) cho report/presentation.
- G5. API & Dashboard **dùng chung pipeline**, cùng verdict trên cùng ảnh; **hãng = cảnh báo phụ/diagnostic, không vào quyết định**.

**Ngoài phạm vi:** retrain model (plate/color/brand); thu thập dữ liệu CCTV thật; chạy GPU. (Caveat domain VCoR→CCTV giữ nguyên, ghi trung thực.)

---

## 3. Thiết kế

### WS-1 — Đọc biển pha-lùi + ROI chuồng-mình + <1s

**File:** `main/src/engine/parking_trigger.py`, `main/src/engine/parking_session.py`, `main/src/engine/decision_engine.py`, `main/configs/config.yaml`, `main/data/database.csv`; (warmup) `main/src/ui/dashboard.py` + API.

**A. Không gian — ROI-first (chuồng của mình):**
- Sửa `ParkingTrigger`: lọc detections theo tâm trong ROI **trước**, rồi `_largest_in_roi`. Xe tâm ngoài ROI → bỏ qua. (Sửa thứ tự largest-trước-lọc.)
- ROI chuồng cho clip demo: calibrate bằng `main/scripts/calibrate_roi.py`; giá trị khởi đề xuất `[0.35, 0.30, 0.65, 1.0]` (tâm target cx≈0.5, cy 0.38→0.5). Sonnet verify ROI loại được các box bên cạnh.
- Chống xe đi ngang: target phải xuất hiện liên tục ≥ `min_persist_frames` (vd 3) trong ROI mới tính (xe lướt qua không sustained → loại).

**B. Thời gian — chốt trong pha lùi:**
- State machine: `IDLE → APPROACHING` (target trong ROI & area ≥ `approach_min_area` ~0.15 & đang lớn dần) → gom + OCR biển **ngay trong pha lùi**, KHÔNG chờ `_is_stable`.
- `parking_session._collect`: khi APPROACHING, chạy plate-detect trên crop target; nếu có box conf ≥ `capture_conf` (~0.5) → OCR crop biển; lưu `(plate_text, plate_conf)` + `(color, color_conf)`.
- **Chốt** khi cùng một `plate_text` (chuẩn hoá) xuất hiện ≥ `lock_repeat` (2) lần ở conf ≥ `lock_conf` (0.6) → `decision_engine` finalize + `trigger.mark_decided()`. (Clip demo: chốt ~frame 498, area 0.20.)
- Nếu area vượt `late_area` (~0.85) mà chưa chốt → log "đọc biển trễ/không kịp" (không bịa). Pha approach ~1.7s đủ rộng để kịp.
- Latch DECIDED tới khi xe rời ROI thì reset.

**C. Warmup:** lúc load model (dashboard + API), chạy mỗi model 1 lần trên ảnh giả để dời cold-start khỏi xe thật đầu tiên.

**D. DB demo:** thêm `30M-718.54` (màu = màu model dự đoán cho xe này, xác định lúc impl, để demo ra AUTHORIZED sạch) + mở rộng `database.csv` lên ~15–20 xe (gộp việc R3).

**Tiêu chí nghiệm thu WS-1:**
- Chốt `plate=30M71854`, frame quyết định < 666 (đỉnh area).
- Latency tới chốt < 1s (đo & in ra).
- 13+ box còn lại không tạo quyết định; xe ngoài ROI bị bỏ.
- Test: unit (ROI-first chọn đúng target khi có box to hơn ngoài ROI; approach-lock chốt khi 2 read trùng; reject khi chỉ 1 read) + E2E headless trên clip.

### WS-2 — Báo động giả < 5% ("cả hai" cơ chế)

**File:** `main/src/utils/matching.py`, `main/src/engine/parking_session.py` + `decision_engine.py` (luồn color_conf), `main/src/ui/dashboard.py` & `main/src/api/app.py` (callers), `main/configs/config.yaml`, `main/scripts/eval_security.py`.

- `verify_vehicle(detected_plate, detected_color, color_conf=None)`:
  - Cụm trung tính: `NEUTRAL = {BLACK, GREY, SILVER, WHITE}`. Helper `_colors_equivalent(c1,c2) = (c1==c2) or (c1 in NEUTRAL and c2 in NEUTRAL)`.
  - Bật `color_warning` **chỉ khi** `not _colors_equivalent(detected, registered)` **VÀ** (`color_conf is None or color_conf >= color_warn_conf` (0.60)).
  - `color_conf=None` giữ tương thích ngược (mặc định coi như đủ tin để cảnh báo) — nhưng runtime sẽ luôn truyền conf thật.
- Luồn `color_conf`: `_collect` lấy cả `color_conf` (hiện đang vứt `_conf`); `aggregate` vote màu + tính conf trung bình của màu thắng → truyền vào `verify_vehicle`. Callers (dashboard image-path, API) truyền conf.
- Hằng số ra config (`decision:` block mới): `color_warn_conf: 0.60`, `neutral_colors: [Black, Grey, Silver, White]`.
- `eval_security.py`: cập nhật truyền `color_conf` (đã có từ `predict`); chạy lại đo cặp (FA, detection) MỚI. Giữ số CŨ (14.5%/98.5%) để so.

**Tiêu chí nghiệm thu WS-2:**
- `eval_security.py` cho **FA < 5%**; báo cáo kèm detection rate mới (chấp nhận giảm từ 98.5% — đo trung thực).
- Test: unit `verify_vehicle` — (Grey reg + Silver det, conf cao → KHÔNG cảnh báo); (Red reg + Blue det, conf cao → cảnh báo); (Red reg + Blue det, conf thấp <0.6 → KHÔNG cảnh báo); (khớp → không cảnh báo).

### WS-3 — API dùng chung pipeline (hãng diagnostic-phụ)

**File:** mới `main/src/engine/pipeline_factory.py`; sửa `main/src/api/app.py`; refactor nhẹ `main/src/ui/dashboard.py` để dùng chung factory.

- `build_pipeline(cfg) -> dict`: dựng vehicle_det, plate_det, PlateReader(PaddleOCR, fallback EasyOCR theo config), color_clf, matcher, DecisionEngine, ParkingSession — **một nguồn sự thật** cho cả dashboard & API.
- `/verify`: vehicle-detect → plate-read 2 tầng → color → `verify_vehicle` (logic WS-2). Trả `status/action/color_warning` + `brand_diagnostic` + `brand_confidence` (tính nhưng **KHÔNG** vào quyết định; ghi rõ trong response + docstring).
- Bỏ phụ thuộc `PlateOCR` (EasyOCR) làm engine chính trong API; EasyOCR chỉ còn fallback theo config.

**Tiêu chí nghiệm thu WS-3:**
- Test smoke API: `/verify` 1 ảnh → verdict plate-primary; field `brand_diagnostic` có mặt nhưng đổi brand không đổi `status/action`.
- API & dashboard cho cùng `status/action` trên cùng 1 ảnh test.

### WS-4 — Dashboard: đường Upload Image 2 tầng

**File:** `main/src/ui/dashboard.py` (`_run_pipeline`).

- Thay OCR-cả-box-COCO + color-full-frame bằng: vehicle-detect → plate-detect trên vehicle crop → OCR plate crop (PlateReader) → color trên body-crop → `verify_vehicle`. Hãng diagnostic-only. Đồng nhất đường video.

**Tiêu chí:** Upload 1 ảnh xe có biển → đọc biển đúng (không đọc badge), không còn train/serve mismatch (color ăn body-crop).

### Gói bằng chứng (yêu cầu của user, cho presentation)

**File:** `docs/benchmarks/security_eval.md` + `.json` (cập nhật), mới `docs/benchmarks/security_fa_before_after.png`, `reports/documents/Report_4_Final_Report.md` (§4.1 + §4.3), deck trong `reports/presentations/`.

- Chạy lại `eval_security.py` → bảng **before/after**: FA 14.5%→(mới), detection 98.5%→(mới).
- Sinh **biểu đồ cột PNG** (FA trước/sau + detection trước/sau) → nhúng Report 4 §4.3 + slide bảo mật.
- Chạy lại **Report 4 §4.1 E2E** với biển đọc được (`30M71854`) + color 86% → thay bảng "Silver 0.18" cũ.

---

## 4. Hằng số & config (đề xuất)

| Tên | Giá trị | Vị trí |
|---|---|---|
| `plate_detector.conf_threshold` | 0.30 → **0.15** | config.yaml |
| `pipeline.trigger.roi` | `[0.35, 0.30, 0.65, 1.0]` (calibrate) | config.yaml |
| `pipeline.trigger.approach_min_area` | 0.15 | config.yaml |
| `pipeline.trigger.min_persist_frames` | 3 | config.yaml |
| `pipeline.lock.capture_conf` | 0.50 | config.yaml |
| `pipeline.lock.lock_conf` | 0.60 | config.yaml |
| `pipeline.lock.lock_repeat` | 2 | config.yaml |
| `pipeline.lock.late_area` | 0.85 | config.yaml |
| `decision.color_warn_conf` | 0.60 | config.yaml |
| `decision.neutral_colors` | `[Black, Grey, Silver, White]` | config.yaml |

---

## 5. Thực thi & kiểm thử

- **Test-first** mỗi WS; `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q` phải **≥ 44 passed, 7 skipped** (thêm test mới → tổng tăng), 0 failed. Interpreter: `/opt/homebrew/Caskroom/miniforge/base/bin/python`.
- **Thứ tự:** WS-1 + WS-2 (cùng đụng `_collect`/`aggregate`, làm chung 1 nhánh) → WS-3 → WS-4 → gói bằng chứng + reports.
- Mỗi WS = 1 Sonnet brief self-contained (file/mục tiêu/ràng buộc/verify). Opus verify: đọc diff + pytest + E2E headless + đọc số eval.
- Commit explicit-path sau mỗi WS. **Không push.**

**Cổng nghiệm thu tổng:** G1–G5 đạt; pytest xanh; FA<5% có bằng chứng; latency<1s đo được; API≡dashboard verdict.

---

## 6. Rủi ro & giảm thiểu

- **Chốt nhầm do misread 1 lần** → `lock_repeat=2` + `lock_conf=0.6`.
- **ROI clip-specific** → tài liệu hoá `calibrate_roi.py` cho camera thật; ROI ra config không hard-code.
- **FA<5% kéo detection xuống nhiều** → đo & báo cáo trung thực (đúng yêu cầu user về bằng chứng); nếu detection rơi quá sâu, trình bày trade-off ở report.
- **Latency phụ thuộc máy** → đo trên máy dev, ghi rõ điều kiện.
- **Hỏng dashboard đang chạy khi refactor factory** → giữ pytest + E2E headless làm lưới an toàn; refactor tối thiểu, không đổi hành vi video-path đang đúng.
