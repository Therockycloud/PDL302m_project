# WS-3 + WS-4: Hợp nhất API & Dashboard về một pipeline dùng chung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Sonnet thực thi từng task (`model: sonnet`), Opus verify. Steps checkbox `- [ ]`.

**Goal:** API `/verify` và dashboard (Upload Image) cho **cùng một verdict** trên cùng ảnh, qua **một nguồn sự thật**: 2 tầng vehicle→plate PaddleOCR + TorchColor body-crop + `verify_vehicle` WS-2 (có color_conf). **Hãng = field diagnostic phụ, KHÔNG vào quyết định.**

**Architecture:** Tách `build_pipeline(cfg)` (dựng components dùng chung) + `infer_single_image(image, pipeline, cfg)` (inference đơn-ảnh 2 tầng → decision dict). Dùng ở CẢ API `/verify` lẫn dashboard `_run_pipeline`. Đường video của dashboard giữ `ParkingSession` (dựng từ cùng components — đã đúng từ WS-1).

**Spec nguồn:** [specs/...api-design.md](../specs/2026-06-20-plate-read-approach-latency-fa-api-design.md) §3 WS-3, WS-4. User chốt: hãng là cảnh báo phụ không ảnh hưởng quyết định.

**Bối cảnh (đã đọc code):**
- API `app.py` hiện: `PlateDetector` 1 tầng chạy cả ảnh + `PlateOCR`(EasyOCR) + `ColorClassifier`(Keras!) + `verify_vehicle(plate,color)` thiếu color_conf. Hoàn toàn lệch dashboard.
- Dashboard `_run_pipeline` (Upload Image, dòng ~231-330) cũng 1 tầng `PlateDetector` + verify thiếu color_conf. Đường video (ParkingSession) thì đã đúng 2 tầng + WS-1/WS-2.
- **Ràng buộc TF/PaddleOCR:** KHÔNG load Keras (TF) cùng tiến trình PaddleOCR (xung đột — lý do dự án đã pivot sang PyTorch). ⇒ color dùng `TorchColorClassifier`; brand diagnostic chỉ best-effort qua model PyTorch nếu có, else null.

---

## File structure

| File | Thay đổi |
|---|---|
| `main/src/engine/pipeline_factory.py` (mới) | `build_pipeline(cfg)` + `infer_single_image(image, pipeline, cfg)` |
| `main/src/api/app.py` | startup dùng `build_pipeline`; `/verify` dùng `infer_single_image`; bỏ PlateDetector/PlateOCR/Keras |
| `main/src/ui/dashboard.py` | `_load_models` dùng `build_pipeline` cho components; `_run_pipeline` (image) dùng `infer_single_image`; video path giữ ParkingSession từ cùng components |
| `main/tests/test_pipeline_factory.py` (mới) | test build_pipeline + infer_single_image |
| `main/tests/test_api.py` (mới/sửa) | smoke `/verify` + brand-không-đổi-quyết-định + API≡image-path |

---

## Task 1: `build_pipeline(cfg)` — factory components dùng chung

**Files:** Create `main/src/engine/pipeline_factory.py`; Create `main/tests/test_pipeline_factory.py`

- [ ] **Step 1 — Test đỏ.** `test_pipeline_factory.py`: gọi `build_pipeline(cfg)` với cfg thật (đọc `config.yaml`) → trả dict có keys `{vehicle_detector, plate_reader, color_clf, matcher, decision_engine}` không None; `plate_reader` là `PlateReader`; engine OCR theo `cfg.ocr.engine` (ppocr→PaddleOCRReader, fallback easyocr). (Có thể mark `@pytest.mark.slow` nếu load model chậm; tối thiểu assert cấu trúc.)
- [ ] **Step 2 — Implement.** `build_pipeline(cfg) -> dict`: dựng `VehicleDetector(yolov8n, conf=cfg.detector.conf_threshold)`, `VehicleDetector(plate_yolov8n, conf=cfg.plate_detector.conf_threshold, vehicle_classes=None)`, OCR (PaddleOCRReader nếu `cfg.ocr.engine=='ppocr'` else PlateOCR/EasyOCR fallback), `PlateReader(plate_det, ocr)`, `TorchColorClassifier(color_MobileNetV3Small.pt)`, `DatabaseMatcher(db)`, `DecisionEngine(matcher)`. Trả dict. **DRY:** đây chính là logic đang nằm rải rác trong `dashboard._load_models` — gom về đây.
- [ ] **Step 3 — Run** `... pytest tests/test_pipeline_factory.py -q` → PASS.
- [ ] **Step 4 — Commit.** `feat(pipeline): shared build_pipeline factory (single source of truth for API+UI) (WS-3)`

## Task 2: `infer_single_image` — inference đơn-ảnh 2 tầng

**Files:** Modify `main/src/engine/pipeline_factory.py`; Modify `main/tests/test_pipeline_factory.py`

- [ ] **Step 1 — Test đỏ.** Với fakes (vehicle_detector trả 1 box+crop; plate_reader trả `{"text":"30M71854","conf":0.9}`; color_clf trả `("YELLOW",0.8)`; matcher thật trên DB tạm có 30M71854→Yellow): `infer_single_image(img, fakes, cfg)` → dict `{plate_text:"30M71854", color:"YELLOW", color_conf:0.8, status:"AUTHORIZED", action:"ALLOW", color_warning:False, brand_diagnostic:..., latency_ms:...}`. Thêm test: đổi `brand_diagnostic` KHÔNG làm đổi `status/action`.
- [ ] **Step 2 — Implement.** `infer_single_image(image, pipeline, cfg) -> dict`:
  1. `dets = vehicle_detector.detect(image)`; chọn vehicle crop lớn nhất; **nếu rỗng → dùng cả ảnh làm vehicle_crop** (ảnh close-up).
  2. `plate = plate_reader.read(vehicle_crop)` (2 tầng: plate-detect trên vehicle crop → OCR). `plate_text, plate_conf = plate["text"], plate["conf"]`.
  3. `color, color_conf = color_clf.predict(vehicle_crop)` (body-crop nội bộ trong TorchColor).
  4. `brand_diagnostic`: best-effort PyTorch brand model nếu pipeline có; else `None` (KHÔNG load Keras cạnh PaddleOCR). **Không bao giờ đưa vào verify.**
  5. Nếu `plate_text` rỗng → `{status:"NO_PLATE", action:"LOG", ...}` (không OCR badge — đã có từ WS-1 plate_reader).
  6. `verdict = matcher.verify_vehicle(plate_text, color, color_conf)` (truyền conf — logic WS-2).
  7. Trả dict gồm verdict + plate_text/color/color_conf/brand_diagnostic/latency_ms.
- [ ] **Step 3 — Run** → PASS.
- [ ] **Step 4 — Commit.** `feat(pipeline): infer_single_image (2-stage plate + WS-2 colour gating; brand diagnostic-only) (WS-3)`

## Task 3: API `/verify` dùng pipeline chung

**Files:** Modify `main/src/api/app.py`; Create/Modify `main/tests/test_api.py`

- [ ] **Step 1 — Test đỏ.** `test_api.py` dùng `fastapi.testclient.TestClient`: POST `/verify` 1 ảnh (ảnh nhỏ tạo bằng cv2, hoặc 1 frame test) → 200, JSON có `status/action/color_warning/plate_text/color/brand_diagnostic`. Test 2: monkeypatch để brand_diagnostic khác nhau → `status/action` KHÔNG đổi (brand không vào quyết định). (Có thể mock pipeline để khỏi load model thật trong unit test.)
- [ ] **Step 2 — Implement.** `app.py`: lifespan dựng `_models["pipeline"] = build_pipeline(cfg)` (bỏ load PlateDetector/PlateOCR/Brand/Color Keras riêng). `/verify`: decode ảnh → `infer_single_image(image, _models["pipeline"], cfg)` → JSONResponse. `/status`: cập nhật models_loaded theo pipeline keys. Bỏ import `PlateOCR`/`classifiers` Keras khỏi đường chính (giữ try/except an toàn nếu cần).
- [ ] **Step 3 — Run** `... pytest tests/test_api.py -q` → PASS; `... pytest -q` toàn bộ xanh.
- [ ] **Step 4 — Commit.** `refactor(api): /verify uses shared pipeline (PaddleOCR 2-stage + WS-2 colour); brand diagnostic-only (WS-3)`

## Task 4: Dashboard Upload-Image dùng pipeline chung

**Files:** Modify `main/src/ui/dashboard.py`

- [ ] **Step 1 — Refactor `_load_models`:** thay phần dựng components rải rác bằng `pipeline = build_pipeline(cfg)`; `ParkingSession` dựng từ `pipeline` components (giữ trigger/lock params WS-1). `models["pipeline"]=pipeline`. Giữ warmup (WS-1).
- [ ] **Step 2 — Refactor `_run_pipeline` (Upload Image):** thay toàn bộ thân bằng gọi `infer_single_image(image, models["pipeline"], cfg)`; map kết quả về cấu trúc UI cần (giữ format hiển thị hiện có). Bỏ đường `PlateDetector` 1 tầng + verify-thiếu-conf cũ.
- [ ] **Step 3 — Verify import + smoke.** `... pytest -q` xanh (dashboard import-time chạy streamlit nên test gián tiếp qua pipeline_factory/api; nếu có test dashboard hiện có, giữ xanh).
- [ ] **Step 4 — Commit.** `refactor(ui): Upload-Image path uses shared infer_single_image (2-stage, WS-2 colour) (WS-4)`

## Task 5: Test nhất quán API ≡ Dashboard image-path

**Files:** Modify `main/tests/test_api.py` (hoặc mới `test_consistency.py`)

- [ ] **Step 1 — Test.** Trên cùng 1 ảnh, `infer_single_image` (đường dashboard dùng) và `/verify` (API) trả **cùng `status/action/plate_text/color`** (vì cùng hàm). Vì cả hai gọi cùng `infer_single_image`, test này khẳng định không còn 2 code-path lệch.
- [ ] **Step 2 — Run + Commit.** `... pytest -q` xanh. Commit: `test(pipeline): API and dashboard image-path produce identical verdict (WS-3/WS-4)`

---

## Verify cuối (Opus) — ✅ ĐẠT (2026-06-21)
- [x] `... pytest -q` → **76 passed, 7 skipped, 0 failed** (baseline 68 → +8 test: pipeline_factory, api, consistency).
- [x] **E2E thật** qua đường hợp nhất (`build_pipeline` + `infer_single_image`, model thật, frame xe target): `plate=30M71854`, color Yellow (0.91), **status=AUTHORIZED action=ALLOW**, `color_warning=False`, `brand_diagnostic=None` (diagnostic-only, KHÔNG vào quyết định). Engine = PaddleOCR (ppocr) — hết EasyOCR/Keras ở đường chính.
- [x] API ≡ dashboard image-path: `test_consistency.py` khẳng định cùng verdict (cùng gọi `infer_single_image`). `test_api.py`: brand đổi KHÔNG đổi status/action. Đường video dashboard (ParkingSession + warmup WS-1) còn nguyên hành vi. API warmup lúc startup (hết cold-start request đầu).

## Progress log
- **2026-06-21 — WS-3+WS-4 XONG.** Tasks 1–4 (Sonnet, commits `ec68a56`/`77db088`/`0c77cec`/`05e7790`): `pipeline_factory.py` (`build_pipeline`+`infer_single_image` 2-tầng PaddleOCR + TorchColor body-crop + verify WS-2 có color_conf; brand=None diagnostic vì Keras xung đột PaddleOCR), API `/verify` + dashboard `_run_pipeline` cùng dùng `infer_single_image`, dashboard `_load_models` + ParkingSession dựng từ `build_pipeline`. Việc cuối (Sonnet, commits `cf8e19d`/`36cc65d`): warmup API startup + `test_consistency.py` (Task 5). Opus verify: pytest 76/7/0 + E2E thật (AUTHORIZED 30M71854 qua đường hợp nhất) + đọc diff (video path nguyên vẹn, brand không vào quyết định). **Kết quả: API & dashboard một nguồn sự thật, hết lệch engine/logic; hãng = diagnostic phụ.** **Lưu ý nhỏ:** slider conf không còn tác dụng ở đường Upload-Image (infer dùng conf cố định từ config — Sonnet đã ghi chú). **Còn nợ:** reports finalization (Report 4 §5.1 latency 1.6s→<1s sau WS-1; §4.1 E2E chạy lại; README đối chiếu API=PaddleOCR).
```
