# Remediation + Color-80% — Master Plan (2026-06-18)

> **For agentic workers:** mỗi Work Stream (WS) = **1 Sonnet subagent** (`model: "sonnet"`). Dùng `superpowers:subagent-driven-development` để thực thi từng WS. **Opus chỉ PLAN + VERIFY**, không tự sửa code. Train nặng chạy trên máy bạn/Colab → **không tốn token**.

**Nguồn lỗi:** tổng hợp từ 3 audit (engineering / tester-repro / academic) + 1 lần chạy thử UI trực tiếp (native :8502 trên video mặc định) ngày 2026-06-18.

**Baseline hiện tại (đã verify):** `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q` → **44 passed, 7 skipped**. Docker (frontend :8501 + backend :8000) **đã build & chạy được** trên máy bạn.

**Run convention:** `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q`. Interpreter = miniforge base. `KMP_DUPLICATE_LIB_OK=TRUE` bắt buộc.

---

## 0. Giao thức chạy nhiều phiên (multi-session) — đọc trước mỗi phiên

Vì bạn chỉ có Pro + usage giới hạn (5h window), **đừng làm tất cả trong 1 phiên**. Mỗi phiên:

1. **Opus đọc file này** → chọn WS chưa `[x]` theo thứ tự session bên dưới.
2. **Opus viết test-first brief ngắn** (nếu cần) rồi **spawn Sonnet** với prompt nhúng sẵn trong WS (`model: sonnet`, reasoning vừa đủ).
3. **Sonnet sửa + chạy test cục bộ.** Commit theo explicit path.
4. **Opus verify** (đọc diff / chạy test / chạy E2E nếu là UI) → nếu đạt thì **tick `[x]`** vào file này + ghi 1 dòng vào §Progress log → **commit file plan** → **DỪNG phiên**.
5. Phiên sau lặp lại. Nhờ file này, phiên mới **không phải dò lại context** (đây mới là phần tốn token nhất).

**Quy tắc tiết kiệm token (quan trọng):**
- Opus **không đọc file lớn** nếu Sonnet làm được; giao Sonnet đọc + sửa, Opus chỉ đọc *diff*.
- Verify UI tốn token (screenshot) → chỉ làm cho WS-UIMETRICS & WS-OCR, các WS khác verify bằng `pytest` + đọc diff.
- **Color training KHÔNG chạy qua agent** — Sonnet chỉ viết script; bạn chạy `python` trên máy/Colab giữa 2 phiên.
- Tối đa **1–2 WS/phiên**. Commit sau mỗi WS để không mất việc nếu hết usage giữa chừng.

---

## 1. Master issue list (theo lăng kính + ưu tiên)

Lăng kính: 🎓 = người chấm · 🧪 = tester · 👤 = người dùng

| ID | Vấn đề | File / bằng chứng | Lens | P | WS |
|----|--------|-------------------|------|---|----|
| C1 | Color classifier ~55% (gần ngẫu nhiên trên frame thật) | `scripts/benchmark_color.py:82-86` freeze toàn bộ backbone | 🎓🧪 | **P0** | WS-COLOR |
| C2 | "Fine-tune" giả: vẫn freeze backbone do bug MPS | `scripts/retrain_color.py:76-77` | 🎓 | **P0** | WS-COLOR |
| C3 | Augmentation phá nhãn màu (saturation/hue jitter) | `scripts/retrain_color.py:65` | 🎓 | **P0** | WS-COLOR |
| D1 | Report 4 ghi runtime màu = TF/Keras (thực tế PyTorch) | `Report_4` vs `torch_color.py` | 🎓 | **P0** | WS-DOCS |
| D2 | Bảng CLO Report 1 ghi ResNet50/MobileNetV2 (đã thay) | `Report_1_Proposal` | 🎓 | **P0** | WS-DOCS |
| D3 | Số liệu dataset README (1209/1130) ≠ Report 2 (792/783) | `README.md:74-77` vs `Report_2` | 🎓 | **P0** | WS-DOCS |
| D4 | Acc màu Report 3 = 54.2% ≠ JSON 55.1% (số cũ) | `Report_3` vs `*_test_report.json` | 🎓 | P1 | WS-DOCS |
| D5 | Sơ đồ Report 4 vẫn vẽ brand như thành phần "sống" | `Report_4` ASCII diagram | 🎓 | P1 | WS-DOCS |
| D6 | Latency: KPI <1.0s, thực 1.6–2.2s, README spin "cực thấp" | `README.md:170`, `Report_4` | 🎓👤 | P1 | WS-DOCS |
| D7 | README test "28 passed, 5 skipped" (thực 44/7) | `README.md:133,150` | 🎓🧪 | P1 | WS-DOCS |
| D8 | Reports thiếu confusion matrix (đã có trong JSON) | `Report_3`, JSON reports | 🎓 | P2 | WS-DOCS |
| S1 | Mục tiêu an ninh ≥95% chống tráo biển **chưa hề đo** | tập E2E chỉ 5 ảnh | 🎓 | **P0** | WS-SECEVAL |
| O1 | OCR đọc nhầm chữ "VF3" trên cốp → UNREGISTERED sai | live test 2026-06-18 | 🧪👤 | **P1** | WS-OCR |
| O2 | API dùng EasyOCR còn dashboard dùng PaddleOCR (lệch) | `api/app.py` vs `ui/dashboard.py` | 🧪 | P1 | WS-RUNTIME |
| O3 | OCR row-threshold magic `/3`, chưa test edge-case | `ocr.py:170` | 🧪 | P2 | WS-REPRO |
| U1 | Metric đếm mỗi frame: TOTAL=ALERTS=334 (1 xe) | live test; `dashboard.py:586-650` | 🧪👤 | **P1** | WS-UIMETRICS |
| U2 | FPS / AVG LATENCY luôn 0.0 khi chạy video | live test | 🧪👤 | P1 | WS-UIMETRICS |
| U3 | Slider "Detection Confidence" không tác dụng | `detector.py:125`, `dashboard.py:232-235` | 🧪👤 | P1 | WS-UIMETRICS |
| U4 | Panel "Detection Results" không cập nhật ở chế độ video | live test | 👤 | P2 | WS-UIMETRICS |
| E1 | `_select_device()` luôn trả "cpu" (docstring sai) | `detector.py:44-50` | 🧪 | P2 | WS-RUNTIME |
| E2 | `ParkingSession._collect()` nuốt mọi exception | `parking_session.py:82` | 🧪 | P1 | WS-RUNTIME |
| E3 | Brand classifier tính nhưng bị bỏ khỏi quyết định | `app.py`, `dashboard.py` | 🧪 | P2 | WS-RUNTIME |
| E4 | Hard-coded path trong `run_evaluation.py` | lines 22,26,28,31 | 🧪 | P2 | WS-REPRO |
| R1 | `requirements.txt` thiếu lib (torch/easyocr/...); không có setup spec | audit repro | 🧪👤 | P2 | WS-REPRO |
| R2 | Dockerfile preload `import tensorflow` (audit đoán fail) — **NHƯNG bạn build OK** → cần kiểm chứng | `Dockerfile` | 🧪 | P2 | WS-REPRO |
| R3 | `database.csv` chỉ 6 xe (demo mỏng) | `data/database.csv` | 👤 | P2 | WS-REPRO |

---

## 2. Work streams (mỗi WS = 1 brief dispatch-được cho Sonnet)

### WS-DOCS — Đồng bộ tài liệu (P0, RẺ, KHÔNG đụng code) → covers D1–D8
**Goal:** xoá mọi mâu thuẫn giữa README / 4 reports / JSON / code. Đây là điểm cộng rẻ nhất với người chấm.
**Files:** `README.md`, 4 report (định vị: `reports/documents/Report_*.md` và/hoặc `docs/Report_*.md`), `docs/model_specifications.md`. Đọc đối chiếu: `main/src/models/torch_color.py`, `configs/config.yaml`, `docs/benchmarks/*`, `*_test_report.json`.
**Việc:** (1) runtime màu = **PyTorch MobileNetV3** ở mọi nơi (sửa D1); (2) bảng CLO → EfficientNet-B0/MobileNetV3-Small (D2); (3) thống nhất dataset: nêu rõ "raw ~1209/1130 → sau cap ~100/lớp = 792/783 dùng để train" ở cả README & Report 2 (D3); (4) đồng bộ acc màu theo JSON (D4); (5) sơ đồ Report 4 đánh dấu brand **(đã loại khỏi quyết định)** (D5); (6) bỏ chữ "cực thấp", nêu thẳng 1.6s vs mục tiêu 1.0s + lý do (D6); (7) test 28/5 → **44/7** (D7); (8) chèn confusion matrix từ JSON vào Report 3 (D8).
**Constraint:** chỉ sửa text/markdown, **không đổi code**. Giữ nguyên giọng văn báo cáo.
**Verify (Opus):** grep lại các cụm sai ("TF/Keras", "ResNet50", "1,209", "28 passed", "cực thấp") → không còn; đọc diff.
**Commit:** `docs: reconcile reports/README with code & test JSON (D1-D8)`

### WS-COLOR — Pipeline train màu đạt 80% (P0, giá trị cao nhất) → covers C1–C3
**Goal:** thay regimen "freeze backbone" bằng full fine-tune để kéo test acc 55% → **≥80%**.
**Files:** tạo `main/scripts/train_color.py`; (tuỳ chọn) cập nhật `main/src/models/torch_color.py` nếu đổi tiền xử lý.
**Recipe bắt buộc:**
- **Full fine-tune** MobileNetV3-Small (mở băng backbone). **Device fallback `cuda → cpu`, KHÔNG dùng MPS** (né bug backward). 780 ảnh nên CPU vẫn chạy; kèm hướng dẫn Colab T4.
- **Discriminative LR:** head 1e-3, backbone 1e-4; warmup head 2–3 epoch rồi mở băng; cosine decay; EarlyStopping theo held-out.
- **Augmentation đã sửa:** BỎ saturation/hue jitter. Giữ: flip, RandomResizedCrop(0.8–1.0), brightness/contrast nhẹ, GaussianBlur + downscale (giả CCTV).
- **Body-crop:** crop vùng giữa thân xe (bỏ ~20% trên = kính/trời, ~15% dưới = lốp/đường) trước khi resize.
- **Eval đúng chuẩn:** split **train/val/test** (test giữ-riêng, seed=42); xuất accuracy + macro-F1 + **confusion matrix** + per-class → `docs/benchmarks/color_finetune_report.json` + `.md`.
- **Export** state_dict ra `main/data/models/color_MobileNetV3Small.pt` (đúng format runtime đang load) — nhưng **lưu tên mới** `color_MobileNetV3Small_ft.pt` để chưa ghi đè cho tới khi Opus verify đạt 80%.
**Constraint:** script chạy được từ `main/`; không phụ thuộc TF; không sửa runtime cho tới khi verify.
**Verify (Opus):** chạy **smoke 2 epoch trên CPU** xác nhận train+eval chạy & acc tăng so baseline; bạn chạy full offline; Opus đọc JSON kết quả, xác nhận ≥80% test giữ-riêng **trước khi** cho swap model runtime.
**Commit:** `feat(color): full fine-tune training pipeline (target 80% test acc)`

### WS-OCR — Sửa OCR đọc nhầm badge (P1) → covers O1
**Goal:** không để OCR đọc chữ trang trí ("VF3") thành biển số.
**Files:** `main/src/models/ocr.py`, pipeline gọi OCR (`parking_session.py` / `api/app.py` / `ui/dashboard.py` — định vị nơi truyền crop vào OCR).
**Việc:** OCR **chỉ chạy trên crop của plate-detector (stage-2 `plate_yolov8n`)**, không chạy trên full-frame/vehicle box. Thêm lọc hình học: tỉ lệ khung biển VN (~3–5:1 dòng đơn / ~1.3:1 hai dòng), vị trí trong nửa dưới xe, ngưỡng OCR confidence; nếu không có plate hợp lệ → trả rỗng (→ KHÔNG kết luận UNREGISTERED sai). Thêm test hồi quy với frame "VF3".
**Verify (Opus):** chạy E2E trên `sample_parking.mp4` (native :8502) → overlay **không còn "VF3"**; hoặc đọc đúng biển, hoặc rỗng+trạng thái "không đọc được biển".
**Commit:** `fix(ocr): read plate-detector crop only; reject decorative badge text (O1)`

### WS-UIMETRICS — Sửa metric & tương tác dashboard (P1) → covers U1–U4, U3
**Goal:** số liệu UI đúng & slider có tác dụng.
**Files:** `main/src/ui/dashboard.py` (+ `detector.py` cho signature conf_threshold).
**Việc:** (U1) đếm **mỗi sự kiện xe** (mỗi lần GATE: DECIDED), không mỗi frame; ALERTS chỉ tăng khi UNREGISTERED/MISMATCH. (U2) nối **FPS + AVG LATENCY** vào vòng lặp video qua `st.empty()` placeholder cập nhật mỗi N frame. (U3) cho `PlateDetector.detect()` nhận `conf_threshold` thật (sửa `detector.py:125`), bỏ fallback nuốt TypeError ở `dashboard.py:232-235`. (U4) panel "Detection Results" hiển thị kết quả quyết định gần nhất ở cả chế độ video.
**Verify (Opus):** chạy video E2E → TOTAL PROCESSED ≈ số xe thật (không 334); FPS/latency ≠ 0; kéo slider thấy số box đổi.
**Commit:** `fix(ui): per-event metrics, live FPS/latency, working conf slider (U1-U4)`

### WS-RUNTIME — Đồng bộ & bền runtime (P1) → covers O2, E1, E2, E3
**Goal:** API & dashboard hành xử giống nhau; lỗi không bị nuốt.
**Files:** `main/src/api/app.py`, `main/src/models/detector.py`, `main/src/pipeline/parking_session.py`.
**Việc:** (O2) API đọc `ocr.engine` từ config như dashboard (dùng PaddleOCR). (E2) `_collect()` log/đưa lỗi ra thay vì `except Exception: return`. (E1) `_select_device()` thật (cuda→cpu) hoặc xoá + sửa docstring. (E3) brand: bỏ tính thừa hoặc đánh dấu rõ "diagnostic-only, không vào quyết định".
**Verify (Opus):** `pytest -q` xanh; smoke import API + 1 request giả.
**Commit:** `fix(runtime): API uses configured OCR engine; surface pipeline errors (O2,E1-E3)`

### WS-SECEVAL — Đo mục tiêu an ninh (P0 học thuật) → covers S1
**Goal:** có **một con số thật** cho năng lực chống tráo biển — lý do tồn tại của đề tài.
**Files:** tạo `main/scripts/eval_security.py` + `docs/benchmarks/security_eval.md`; cập nhật Report 4 (mục Limitations + kết quả).
**Việc:** dựng bộ kịch bản có kiểm soát từ ảnh/clip sẵn có: (a) biển khớp DB → kỳ vọng AUTHORIZED; (b) **biển tráo** (gán biển không khớp) → kỳ vọng UNREGISTERED/cảnh báo; (c) biển lạ. Tối thiểu ~20–30 ca. Đo precision/recall của trạng thái cảnh báo; ghi bảng + **mục "Giới hạn đánh giá"** thừa nhận thẳng cỡ mẫu.
**Verify (Opus):** đọc script + bảng kết quả; methodology hợp lý, không rò rỉ.
**Commit:** `feat(eval): controlled plate-swap security benchmark + limitations (S1)`

### WS-REPRO — Tái lập & vệ sinh (P2) → covers R1–R3, E4, O3
**Goal:** máy sạch chạy được; bỏ rác.
**Files:** `main/requirements.txt`/`requirements-train.txt`, `Dockerfile`, `data/database.csv`, `scripts/run_evaluation.py`, `tests/`.
**Việc:** (R1) bổ sung đủ runtime deps hoặc thêm `environment.yml` + mục Setup README. (R2) **kiểm chứng `docker compose build` trên trạng thái sạch** (bạn báo build OK → xác nhận lại; nếu bước `import tensorflow` thực sự fail thì bọc try/except hoặc bỏ). (R3) thêm ~15–20 xe vào `database.csv`. (E4) bỏ hard-code path. (O3) test edge-case OCR single-line.
**Verify (Opus):** `pytest -q` xanh; (nếu được) build docker sạch.
**Commit:** `chore(repro): complete deps, expand DB, de-hardcode paths, edge tests`

---

## 3. Lịch chạy theo session (token-aware)

> Mỗi dòng là **một phiên Claude riêng**. Bắt đầu phiên = mở file này, làm WS chưa tick, rồi dừng.

- [x] **Session 1 — P0 rẻ + khởi động long-pole** · WS-DOCS ✅ (commit d9cd827) ∥ WS-COLOR script ✅ (commit sau). **CÒN LẠI:** bạn chạy full train offline → khi acc test ≥80% thì Session 3 mới swap `color_MobileNetV3Small_ft.pt` → `color_MobileNetV3Small.pt`.
- [x] **Session 2 — P1 bug nhìn-thấy-ngay** · WS-UIMETRICS ✅ `0e0ca2b` (E2E: TOTAL 334→1, FPS 0→96, slider+panel) · WS-OCR ✅ `2cd3697` (E2E: decision NO_PLATE thay vì UNREGISTERED:VF3). **Còn lại (P2):** image-path `_run_pipeline` vẫn OCR box COCO yolov8n (Upload Image mode) — fix sau.
- [ ] **Session 3 — P1 runtime + P0 an ninh + chốt màu** · WS-RUNTIME + WS-SECEVAL; nếu train màu xong → Opus verify ≥80%, swap `color_*.pt` runtime, cập nhật report.
- [ ] **Session 4 — P2 polish + nghiệm thu** · WS-REPRO + full verification (pytest + E2E + đọc lại report tìm mâu thuẫn còn sót).

**Nếu chỉ còn 1 phiên:** làm **WS-DOCS + WS-COLOR(script)** (Session 1) — rẻ nhất, giá trị học thuật cao nhất, và mở khoá mục tiêu 80%.

---

## 4. Verify checklist (Opus, cuối mỗi phiên)
- [ ] `cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q` ≥ baseline (44/7), 0 failed.
- [ ] Đọc diff của WS vừa làm, đối chiếu Goal.
- [ ] WS UI (OCR/UIMETRICS): chạy E2E `sample_parking.mp4` xác nhận hành vi.
- [ ] Tick `[x]` WS + dòng Progress log + commit explicit path.

---

## 5. Progress log
- 2026-06-18 — Plan tạo bởi Opus sau 3 audit + 1 live UI test. Baseline 44/7. Chưa WS nào chạy.
- 2026-06-18 — **Session 1 xong.** WS-DOCS (D1–D8) commit `d9cd827`. WS-COLOR `main/scripts/train_color.py` commit `7e9a121`. Verify: pytest 44/7 xanh; class order khớp `torch_color.py` (drop-in); phase-1→phase-2 chạy OK (train acc 0.19→0.80 trong 6 epoch short-run). Runtime `color_MobileNetV3Small.pt` CHƯA đụng. **Chờ:** full train offline → verify ≥80% ở Session 3 rồi swap.
- 2026-06-18 — **Session 2 (một phần).** WS-UIMETRICS commit `0e0ca2b`. E2E live verify: TOTAL 334→**1**, ALERTS 334→**1**, FPS 0→**96.2**, AVG lat 0→**10.4ms**, panel hiện verdict. pytest 44/7. Lỗi gốc U1: gate latch DECIDED + `_decision` persist mỗi frame → đã sửa bằng rising-edge (`out["state"]=="DECIDED" và prev!=DECIDED`). WS-OCR HOÃN.
  **Findings cho WS-OCR (VF3) — phiên sau:** video path DÙNG 2-stage (VehicleDetector→plate_yolov8n→OCR) qua `ParkingSession` nhưng vẫn đọc "VF3" → lỗi ở `src/engine/parking_session.py:79` `_collect` đang gọi `plate_reader.read(crop)` trên **vehicle crop** (`veh.get("crop")`), KHÔNG phải plate crop từ plate_yolov8n. Image path `_run_pipeline` (dashboard.py) còn OCR thẳng box của detector COCO `yolov8n` (cả xe). Fix: chèn bước plate-detect trên vehicle crop rồi OCR plate crop; nếu không có plate hợp lệ → no-plate (đừng kết luận UNREGISTERED).
  **Findings cho Session 3 (swap model màu):** train/serve mismatch — `_run_pipeline` (dashboard.py:272) cho `color_clf.predict(image)` ăn **FULL frame**; session path ăn vehicle crop; `train_color.py` train trên body-crop. Khi swap model 80% phải cho runtime color_clf ăn vehicle/body crop, nếu không 80% test cũng vô dụng ở runtime.
- 2026-06-18 — **Train màu full fine-tune: KHÔNG đạt 80% (thực ra TỆ HƠN baseline).** 30 epoch (early-stop @15), full unfreeze trên 547 train / 116 val / 120 test. Kết quả **test acc 0.433, macro-F1 0.41** vs baseline frozen ~0.55. Train acc 0.89 nhưng val kẹt 0.41–0.50 + val loss tăng → **overfit nặng** (~68 ảnh/lớp quá ít cho full fine-tune). Confusion matrix: **White/Silver/Grey/Black đổ dồn vào Silver** (Grey recall 0.07). KL: unfreeze KHÔNG phải lever đủ — bottleneck là **DỮ LIỆU + nhập nhằng màu trung tính**, không phải frozen-vs-unfrozen. **KHÔNG swap** (giữ runtime 55%). Hướng đạt 80%: (A) thêm dữ liệu công khai (VCoR ~10k ảnh), (B) gộp lớp trung tính → taxonomy thô 5–6 lớp, (C) chấp nhận ~55% + ghi limitation trung thực. → CHỜ user chọn.
- 2026-06-18 — **Session 2b xong (WS-OCR / VF3).** Lỗi gốc: `src/models/plate_reader.py:30` — khi `plate_yolov8n` không thấy biển trong vehicle crop, code OCR **cả xe** → đọc badge "VF3" → UNREGISTERED giả. Fix: trả text rỗng → `DecisionEngine` log **NO_PLATE** (action LOG, không alert). TDD: test cũ `test_fallback_to_whole_crop_*` (vốn khẳng định hành vi sai) → đổi thành `test_no_ocr_fallback_when_no_plate_box`; red→green. Commit `2cd3697`, suite 44/7. Verify E2E headless trên `sample_parking.mp4` (565 frame, gate IDLE→DECIDED): decision = NO_PLATE, plate=''. **Lưu ý:** `plate_yolov8n` thực sự KHÔNG bắt được biển VinFast rear ở clip này → kết quả trung thực là "không đọc được biển" (không phải đọc đúng biển). Nếu muốn đọc đúng: cần tinh chỉnh plate detector / thêm dữ liệu — để sau.
- 2026-06-19 — **Breakthrough màu: thêm data VCoR.** User tải VCoR (Kaggle) → `/Users/konalyn/Downloads/archive`. Script `main/scripts/build_color_dataset.py` merge 8 lớp VCoR khớp (bỏ beige/gold/green/orange/pink/purple/tan) + data cũ → `main/data/raw/car_colors_vcor` = **6664 ảnh** (Black679 Blue1160 Brown898 Grey711 Red1009 Silver616 White675 Yellow916). Retrain 300/class balanced (2400 ảnh): **test 0.733, macro-F1 0.72** — overfit GIẢM MẠNH (train0.92/val0.74, val loss giảm vs 0.89/0.48 lúc data ít). → "thêm data" = lever ĐÚNG. Grey vẫn yếu (recall0.33, lẫn Black/Silver). Đang chạy 600/class balanced (4800) để đẩy ≥80%. Model lưu `color_MobileNetV3Small_ft.pt`, **CHƯA swap** runtime (giữ 55% tới khi ≥80% + áp crop-fix dashboard.py:272).
- 2026-06-19 — **600/class → 77.6%; finisher Colab sẵn sàng.** Retrain 600/class balanced (4800 ảnh): test **0.776**, macro-F1 0.776, best val 0.796 (vs 73.3% @300). Yellow0.99/Red0.94/Blue0.87 mạnh; **Silver(recall0.54)/Grey(0.64) = trần** (neutral confusion). Diminishing returns (+4.3 khi gấp đôi data) → cần nhắm cụm trung tính chứ không phải thêm data. **Finisher** `main/scripts/colab_train_color.py`: self-contained, full 6.6k, 3 lever (class-weight Silver1.29/White1.18/Grey1.12 + label-smoothing0.1 + TTA-hflip), drop-in khớp `torch_color.py`, smoke CPU OK. User chọn chạy **Colab GPU**. CHỜ: user chạy → gửi `test_accuracy_tta` + `.pt` → Opus verify ≥80% → swap runtime + crop-fix `dashboard.py:272`. Nếu <80%: chốt 77.6% + limitation.
- 2026-06-19 — **🎯 ĐẠT MỤC TIÊU: deploy model màu 86%.** User chạy finisher Colab (full VCoR 5881 ảnh, 3 lever): **test 85.3% plain / 86.3% TTA**, macro-F1 0.84, best val 0.857 (từ 55%!). Verify drop-in OK (load qua `TorchColorClassifier`, class order khớp, sanity 0.87 no-crop). **Swap** `color_MobileNetV3Small.pt` ← model mới (backup cũ → `color_MobileNetV3Small_pre-vcor-backup.pt`; file git-tracked nên cũng có trong history). **Deploy runtime** commit `a313823`: (1) dashboard ưu tiên PyTorch (was Keras-first), (2) `torch_color.predict()` thêm body-crop khớp train → sanity 0.87→**0.96**, (3) image-path cho color_clf ăn vehicle bbox crop (video/session path vốn đã đúng). pytest 44/7. **CAVEAT thành thật:** 86% là trên VCoR (ảnh web SẠCH), KHÔNG phải CCTV bãi xe → runtime thật sẽ thấp hơn (domain gap); muốn bền ánh sáng cần white-balance preprocessing + ít data CCTV. **CÒN LẠI:** cập nhật reports (README/Report 3/benchmarks/model_specs) với số mới + caveat — cần Colab `_report.json` cho per-class/confusion đầy đủ.
- 2026-06-19 — **Reports cập nhật lên 86% (commit `8bcaeaa`). COLOR THREAD XONG.** Regenerate held-out eval của model ĐANG DEPLOY trên VCoR test (889 ảnh) qua `main/scripts/eval_color_deployed.py` (đọc lại weights runtime, KHÔNG retrain, cùng split seed 42 như Colab): **85.3% plain / 86.3% TTA**, macro-F1 0.84 — tái lập đúng số Colab. Cập nhật README/Report 3/model_specs/color_benchmark: 55%→86% + phương pháp (VCoR + class-weight + label-smooth + TTA) + journey 55→86 + caveat domain CCTV thành thật. 55% chỉ còn là baseline lịch sử có nhãn. pytest 44/7. Deployed `.pt` không đụng.
  **Ngoài lề (đã commit riêng):** MPS enable (`bad3e35`, train GPU local ~8×), num_workers chỉ cho cuda vì macOS+MPS flaky (`fbe4adc`); đúc kết PyTorch-trên-máy → memory + (chờ user paste vào CLAUDE.md vì file bị guard khoá).
  **CÒN LẠI:** WS-SECEVAL (đo chống tráo biển — tử huyệt học thuật), WS-RUNTIME (API/OCR đồng bộ), WS-REPRO (deps/Docker/DB).
