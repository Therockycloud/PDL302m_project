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

- [ ] **Session 1 — P0 rẻ + khởi động long-pole** · WS-DOCS ∥ WS-COLOR(viết script). Chạy song song (khác file). Sau phiên: **bạn kick off train màu offline (CPU/Colab)** để nó "nấu" trong lúc làm phiên khác.
- [ ] **Session 2 — P1 bug nhìn-thấy-ngay** · WS-UIMETRICS, rồi WS-OCR. Verify bằng E2E video. (Đây là phần tốn token nhất vì có screenshot → để riêng 1 phiên.)
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
