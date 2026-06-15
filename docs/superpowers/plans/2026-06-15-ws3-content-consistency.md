# WS3 — Content Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all narrative (4 reports + 4 slides + README + PROJECT.md) tell one truthful story — the **pivot narrative**: a 3-factor cross-verification was *proposed*, experiments showed brand/colour too weak for hard denial, so the *delivered* system is **plate-primary** (PaddleOCR OCR, colour = soft warning, brand dropped).

**Architecture:** Reports 1–2 are *historical proposal/data* docs — keep their original 3-factor / ResNet50 / MobileNetV2 / EasyOCR framing as proposed, but add a short forward "pivot note" pointing to Reports 3–4. Reports 3–4, README, and PROJECT.md describe the *delivered* system and MUST be accurate (PaddleOCR, EfficientNet-B0/MobileNetV3-Small, plate-primary). Slides mirror their report. Instructor name unified to **Lương Trung Kiên**. Suspicious citations web-verified and replaced with real sources.

**Tech Stack:** Markdown docs, HTML slides, WebSearch for citation verification.

**Verification style:** content edits are verified with `grep` consistency assertions (no unit tests). Each task ends by re-reading the changed region and running the grep checks in Task 8.

**No code changes** in this plan — docs/slides only. Do not touch `main/src/**` or `visual.py`.

---

### Task 1: Report_1 (Proposal) + Report_2 (Data) — add pivot note, keep history

**Files:** Modify `docs/Report_1_Proposal.md`, `docs/Report_2_Data_Tasks.md`

**Rationale:** These are historical proposal/data-stage docs. Do NOT rewrite their model names — proposing ResNet50/MobileNetV2/3-factor is what genuinely happened. Add a forward pointer so a reader knows the final system pivoted.

- [ ] **Step 1: Insert a pivot note at the top of `Report_1_Proposal.md`**

After the title block (immediately after the line `**Trường:** FPT University  ` and before the `---` that precedes `## 1. Đặt vấn đề`), insert:

```markdown
> **🔄 Ghi chú phiên bản (đọc trước):** Đây là **đề xuất ban đầu** (Giai đoạn 1) mô tả hướng *xác thực đa nhân tố (biển số + hãng + màu)* với ResNet50/MobileNetV2/EasyOCR. Trong quá trình thực nghiệm (Report 3) các bộ phân loại hãng (~29%) và màu (~14% lúc đầu) cho thấy quá yếu để chặn cứng, nên **bản giao cuối đã pivot sang quyết định *plate-primary*** (OCR là khoá chính bằng **PaddleOCR**, màu chỉ là **cảnh báo mềm**, bỏ phân loại hãng), và đổi backbone sang **EfficientNet-B0/MobileNetV3-Small**. Xem hành trình & lý do trong Report 3 và Report 4.
```

- [ ] **Step 2: Insert a one-line pivot pointer in `Report_2_Data_Tasks.md`**

After the first paragraph of section `## 1. Đặt vấn đề và Mục tiêu nghiên cứu` (the paragraph ending `...Màu sắc xe (Color).`), insert a new line:

```markdown

> **Ghi chú:** Bộ ba đặc trưng dưới đây là theo *đề xuất ban đầu*. Bản giao cuối dùng biển số làm khoá chính (plate-primary), màu là cảnh báo phụ, bỏ hãng — xem Report 3/4.
```

- [ ] **Step 3: Verify**

Run: `grep -n "Ghi chú phiên bản\|pivot" docs/Report_1_Proposal.md docs/Report_2_Data_Tasks.md`
Expected: the inserted notes appear. No other lines changed (`git diff --stat` shows only these two files, small insertions).

- [ ] **Step 4: Commit**

```bash
git add docs/Report_1_Proposal.md docs/Report_2_Data_Tasks.md
git commit -m "docs(reports): add pivot note to proposal/data reports (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Report_3 (Model & Results) — OCR engine + colour/brand reconciliation

**Files:** Modify `docs/Report_3_Model_Results.md`

- [ ] **Step 1: Fix the intro OCR mention (line ~4)**

Find: `trích xuất văn bản biển số (EasyOCR)`
Replace: `trích xuất văn bản biển số (ban đầu EasyOCR, sau chuyển sang **PaddleOCR** — xem §3.2)`

- [ ] **Step 2: Rewrite section 3.2 (OCR engine) to reflect the PaddleOCR pivot**

Replace the whole block:

```markdown
### 3.2. Bộ nhận diện ký tự biển số (EasyOCR Engine)
*   **Kiến trúc**: Sử dụng mạng ResNet kết hợp với mạng hồi quy tuần hoàn LSTM và tầng giải mã CTC (Connectionist Temporal Classification).
*   **Cấu hình**: Chạy hoàn toàn ngoại tuyến (`download_enabled=False`) để loại bỏ hoàn toàn độ trễ kiểm tra phiên bản qua mạng của EasyOCR.
```

with:

```markdown
### 3.2. Bộ nhận diện ký tự biển số (OCR Engine: PaddleOCR)
*   **Lựa chọn engine (Benchmark C)**: Ban đầu nhóm dùng **EasyOCR** (ResNet + LSTM + CTC). Tuy nhiên benchmark trên 16 biển CCTV thật cho thấy EasyOCR đọc đúng **0%** chuỗi (exact-match), trong khi **PaddleOCR (PP-OCRv4, CRNN+CTC)** đạt **81%** (CER 0.28 → 0.03). Vì vậy **PaddleOCR là engine chính**, EasyOCR giữ làm fallback. Chi tiết: `docs/benchmarks/ocr_benchmark.md`.
*   **Cấu hình**: Chạy ngoại tuyến hoàn toàn; engine cấu hình ở `main/configs/config.yaml` (`ocr.engine: ppocr`, fallback `easyocr`).
```

- [ ] **Step 3: Add the model-pivot sentence at the end of section 3.3 (Brand Classifier)**

After the `Dense(8, activation="softmax")` bullet of §3.3, append a bullet:

```markdown
    *   *Lưu ý pivot:* khác với ResNet50 trong đề xuất ban đầu (Report 1), nhóm chọn **EfficientNet-B0** vì cùng độ chính xác nhưng nhẹ và nhanh hơn nhiều trên CPU (xem Benchmark A màu, cùng kết luận về kích thước/độ trễ).
```

- [ ] **Step 4: Reconcile section 5 (results) — explain the two colour numbers and the decision**

After section `### 5.2` (the colour `14.16%` paragraph, ending with the training-curve link), insert a new subsection:

```markdown

### 5.3. Hệ quả thực nghiệm → Quyết định pivot
Hai kết quả ban đầu (hãng ~29%, màu ~14.16%) cho thấy bộ phân loại đặc trưng **quá yếu để chặn cứng** một xe. Nhóm đã:
1.  **Bỏ phân loại hãng** khỏi quyết định (giữ lại như thử nghiệm).
2.  **Chuyển màu sang mô hình PyTorch MobileNetV3-Small thích ứng miền CCTV** — đạt **accuracy 59.73% / macro-F1 0.625** trên 226 ảnh val (Benchmark A, `docs/benchmarks/color_benchmark.md`), cao hơn hẳn 14.16% ban đầu; con số 14.16% phản ánh lần huấn luyện Keras đầu chưa thích ứng miền.
3.  **Hạ màu xuống "cảnh báo mềm"**: biển số (PaddleOCR) là khoá chính (plate-primary); màu lệch chỉ cảnh báo, không từ chối cứng.
Đây là cơ chế quyết định **delivered** của hệ thống (xem Report 4).
```

- [ ] **Step 5: Verify**

Run: `grep -n "PaddleOCR\|plate-primary\|59.73\|cảnh báo mềm" docs/Report_3_Model_Results.md`
Expected: PaddleOCR present in §3.2, the 59.73% reconciliation and plate-primary decision present in §5.3.
Run: `grep -n "EasyOCR Engine" docs/Report_3_Model_Results.md` → expected: no match (heading renamed).

- [ ] **Step 6: Commit**

```bash
git add docs/Report_3_Model_Results.md
git commit -m "docs(report3): PaddleOCR pivot + colour/brand reconciliation (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Report_4 (Final) — delivered-system framing, honest conclusion, latency

**Files:** Modify `docs/Report_4_Final_Report.md`

- [ ] **Step 1: Reframe the intro (section 1, line ~4)**

Find: `tích hợp các mô hình học sâu thành phần (YOLOv8, EasyOCR, EfficientNet-B0, MobileNetV3-Small) thành một hệ thống an ninh bãi xe khép kín, tự động và bảo mật cao. Hệ thống thực hiện quy trình suy luận tuần tự kết hợp đối sánh 3 nhân tố với cơ sở dữ liệu mẫu`
Replace: `tích hợp các mô hình học sâu thành phần (YOLOv8, **PaddleOCR**, MobileNetV3-Small) thành một hệ thống an ninh bãi xe khép kín, tự động. Sau thực nghiệm (Report 3), hệ thống dùng quyết định **plate-primary**: biển số (PaddleOCR) là khoá chính, **màu xe là cảnh báo mềm**, và **bỏ phân loại hãng**; đối chiếu với cơ sở dữ liệu mẫu`

- [ ] **Step 2: Fix the OCR engine box in the ASCII diagram (line ~28)**

Find: `| EasyOCR Engine (Plate)  |`
Replace: `| PaddleOCR Engine (Plate)|`

- [ ] **Step 3: Clarify latency in section 4.2**

Find the line: `*   **Thời gian phản hồi trung bình (Average Latency)**: **2,190.89 ms / xe**`
Replace:

```markdown
*   **Thời gian phản hồi trung bình (Average Latency)**: **2,190.89 ms / xe** trên 5 ảnh test — con số này gồm *cold-start* (~4.49 s ở ảnh đầu do nạp model lần đầu). Từ ảnh thứ hai trở đi, độ trễ ổn định ở mức **~1.6 s / xe** (xem §5.1). Mục tiêu <1.0 s ở đề xuất chưa đạt do chạy thuần CPU + tải PaddleOCR.
```

- [ ] **Step 4: Rewrite the conclusion (section 6) to match real metrics**

Replace the whole paragraph under `## 6. Kết luận`:

```markdown
Hệ thống đã hoàn thiện một **pipeline ALPR biên, ngoại tuyến, plate-primary**: phát hiện biển số (YOLOv8n, mAP@0.5 ~0.98) và đọc biển bằng **PaddleOCR** (Benchmark C: 81% exact-match) hoạt động tốt và là lớp quyết định chính. Phân loại **màu** (MobileNetV3-Small thích ứng CCTV, ~60%) được dùng làm **cảnh báo mềm**; phân loại **hãng** (~29%) bị loại khỏi quyết định do còn yếu trước khoảng cách miền dữ liệu. So với đề xuất ban đầu (đa nhân tố chặn cứng, mục tiêu ≥95%), bản giao đã **pivot có chủ đích** sang plate-primary — trung thực với năng lực thực đo của từng mô hình. Hệ thống chạy ổn định ngoại tuyến 100% trên CPU (~1.6 s/xe sau cold-start), giao diện Streamlit cảnh báo trực quan khi biển không khớp hoặc màu lệch. Hướng cải thiện: thu dữ liệu in-domain + fine-tuning sâu để nâng màu/hãng (xem Report 3 §6).
```

- [ ] **Step 5: Verify**

Run: `grep -n "EasyOCR\|3 nhân tố\|hoàn thành mục tiêu thiết kế ban đầu" docs/Report_4_Final_Report.md`
Expected: no match for "EasyOCR" as the delivered engine in §1/diagram (it may still appear in §5's thread-conflict/offline notes describing history — that is acceptable and correct); no "3 nhân tố" as the delivered decision; the overclaiming conclusion sentence gone.
Run: `grep -n "plate-primary\|PaddleOCR\|cảnh báo mềm" docs/Report_4_Final_Report.md` → expected: present.

- [ ] **Step 6: Commit**

```bash
git add docs/Report_4_Final_Report.md
git commit -m "docs(report4): plate-primary framing + honest conclusion + latency (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: PROJECT.md (runtime reality) + README.md (test count)

**Files:** Modify `PROJECT.md`, `README.md`

- [ ] **Step 1: Update `PROJECT.md` architecture lines**

Find: `  - License Plate OCR: EasyOCR wrapper in `main/src/models/ocr.py`.`
Replace: `  - License Plate OCR: PaddleOCR (primary, won Benchmark C) with EasyOCR fallback, in `main/src/models/ocr.py`.`

Find: `  - Attributes classification: Brand & Color classifiers in `main/src/models/classifiers.py`.`
Replace: `  - Attributes: Vehicle colour classifier (PyTorch MobileNetV3-Small) as a soft-warning layer in `main/src/models/torch_color.py`. Brand classification was dropped from the decision after weak results.`

Find: `- `main/src/models/ocr.py`: EasyOCR processor.`
Replace: `- `main/src/models/ocr.py`: PaddleOCR processor (EasyOCR fallback).`

Find: `- `main/src/models/classifiers.py`: Keras classifiers for vehicle brand and color.`
Replace: `- `main/src/models/torch_color.py`: PyTorch MobileNetV3-Small colour classifier (runtime). `main/src/models/classifiers.py`: Keras brand/colour classifiers (training/eval only).`

Find: `| 4 | Pipeline Optimization | CPU optimizations for YOLOv8, EasyOCR, Brand/Color classifiers | Exploration | PLANNED |`
Replace: `| 4 | Pipeline Optimization | CPU optimizations for YOLOv8, PaddleOCR, colour classifier | Exploration | PLANNED |`

- [ ] **Step 2: Fix the README test-count claim**

In `README.md`, find: `(gồm 15 bài test kiểm tra OCR, so khớp CSDL, logic tiền xử lý)`
Replace: `(bộ test tự động — hiện **28 passed, 5 skipped** — kiểm tra OCR, so khớp CSDL, logic tiền xử lý)`

- [ ] **Step 3: Verify**

Run: `grep -n "EasyOCR processor\|PaddleOCR\|torch_color" PROJECT.md` → expected: PaddleOCR/torch_color present.
Run: `grep -n "15 bài test\|28 passed" README.md` → expected: "28 passed" present, "15 bài test" gone.

- [ ] **Step 4: Commit**

```bash
git add PROJECT.md README.md
git commit -m "docs(project,readme): align to delivered runtime + fix test count (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Slides — narrative mirror + instructor name unification

**Files:** Modify `presentations/Report_2_Presentation.html`, `presentations/Report_3_Presentation.html`, `presentations/Report_4_Presentation.html`

**Note:** Report_1 slide stays as the historical proposal (matches Report_1.md). Only fix the *delivered-system* claims in R3/R4 slides + unify the instructor name in R2/R3/R4.

- [ ] **Step 1: Unify instructor name (R2, R3, R4 slides)**

In each of `Report_2_Presentation.html`, `Report_3_Presentation.html`, `Report_4_Presentation.html`, find the text `Thầy Trần Đức Anh` and replace with `Thầy Lương Trung Kiên`.

- [ ] **Step 2: Fix R3 slide OCR section heading (line ~473) and architecture text**

In `Report_3_Presentation.html`:
- Find `EasyOCR & Hậu xử lý hình học` → replace `PaddleOCR & Hậu xử lý hình học`
- Find `Kiến trúc EasyOCR: Sử dụng mạng ResNet làm bộ trích xuất đặc trưng kết hợp mạng LSTM nhận diện chuỗi ký tự theo thời gian.` → replace `Kiến trúc PaddleOCR (PP-OCRv4): CRNN + CTC. Benchmark C: PaddleOCR đọc đúng 81% (exact-match) vs EasyOCR 0%, nên PaddleOCR là engine chính, EasyOCR fallback.`
- Find `lỗi đọc sai ký tự của EasyOCR xuống mức thấp nhất.` → replace `lỗi đọc sai ký tự của OCR xuống mức thấp nhất.`
- Find `khiến EasyOCR bị nhầm lẫn số `8` và chữ `B`.` → replace `khiến OCR bị nhầm lẫn số `8` và chữ `B`.`

- [ ] **Step 3: Fix R4 slide "3-factor" delivered framing**

In `Report_4_Presentation.html`, locate the slide that lists "Factor 1 / Factor 2 / Factor 3" cross-verification (the "Hệ thống đối chiếu chéo đa nhân tố" slide). Read that slide block, then reframe its lead text to plate-primary. Specifically, find the descriptive line that says the system cross-verifies 3 factors and replace its emphasis so it reads: biển số (PaddleOCR) là khoá chính; màu là **cảnh báo mềm**; hãng đã **loại** sau thực nghiệm. Keep the factor cards but add a one-line caption under the heading: `Bản giao: plate-primary — màu là cảnh báo mềm, hãng đã loại (xem Report 3).` If the exact wording differs, preserve surrounding markup and only adjust the text content.

- [ ] **Step 4: Verify**

Run: `grep -rn "Trần Đức Anh" presentations/` → expected: no matches.
Run: `grep -rn "Lương Trung Kiên" presentations/` → expected: present in R1, R2, R3, R4.
Run: `grep -rn "Kiến trúc EasyOCR" presentations/Report_3_Presentation.html` → expected: no match.
Run: `grep -rni "plate-primary\|cảnh báo mềm" presentations/Report_4_Presentation.html` → expected: present.

- [ ] **Step 5: Commit**

```bash
git add presentations/Report_2_Presentation.html presentations/Report_3_Presentation.html presentations/Report_4_Presentation.html
git commit -m "docs(slides): plate-primary mirror + unify instructor name (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Citations — web-verify and replace placeholder-looking references

**Files:** Modify `docs/Report_1_Proposal.md`, `docs/Report_2_Data_Tasks.md`, `docs/Report_3_Model_Results.md`, `docs/Report_4_Final_Report.md` (references sections)

**Suspicious refs to verify (no venue/DOI, placeholder-style names):** Lima et al. (2026) arXiv:2604.05271; Hu et al. (2017) arXiv:1702.01721; Chen et al. (2019) "Vehicle Color Recognition in Urban Surveillance"; Wang & Choi (2021); Jang & Lim (2020); Lin et al. (2022); Smith & Patel (2023) "Offline-First Intelligent Edge Architectures".

- [ ] **Step 1: Verify each reference with WebSearch**

For each suspicious ref, run a WebSearch for the exact title + authors. Record for each: REAL (found, with correct citation) / WRONG-METADATA (real paper, wrong details → fix details) / FABRICATED (no such paper found).

- [ ] **Step 2: Keep REAL refs; fix WRONG-METADATA refs to the correct citation.**

- [ ] **Step 3: For FABRICATED refs, replace with a real, relevant source.** Prefer reusing the real, link-bearing sources already in `docs/related_work.md` (e.g. the YOLO-ALPR survey, Layout-Independent ALPR arXiv:1909.01754, Visual-Rhythm ALPR arXiv:2501.02270, NVIDIA DeepStream, Nature Sci. Reports 2025). Match each replacement to the claim it supports (e.g. a fabricated "offline edge" ref → the real DeepStream/edge ALPR source). Never swap one fabricated ref for another unverified one.

- [ ] **Step 4: Verify**

Run: `grep -n "Smith & Patel\|Wang & Choi\|Jang & Lim" docs/Report_*.md`
Expected: no matches remain UNLESS web-verified as real (document the verification outcome in the commit message).

- [ ] **Step 5: Commit**

```bash
git add docs/Report_1_Proposal.md docs/Report_2_Data_Tasks.md docs/Report_3_Model_Results.md docs/Report_4_Final_Report.md
git commit -m "docs(reports): verify and replace placeholder citations with real sources (WS3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Final consistency sweep

**Files:** none (verification only)

- [ ] **Step 1: Delivered-engine consistency**

Run: `grep -rn "EasyOCR" docs/Report_3_Model_Results.md docs/Report_4_Final_Report.md PROJECT.md presentations/Report_3_Presentation.html presentations/Report_4_Presentation.html`
Expected: EasyOCR only ever appears as *fallback* or *historical/offline-config* mention, never as the delivered primary engine.

- [ ] **Step 2: Model-name consistency for the delivered system**

Run: `grep -rn "ResNet50\|MobileNetV2" docs/Report_3_Model_Results.md docs/Report_4_Final_Report.md PROJECT.md`
Expected: no matches (these belong only to Report_1/2 + slide R1 as the historical proposal).

- [ ] **Step 3: Decision-logic consistency**

Run: `grep -rn "plate-primary\|cảnh báo mềm" docs/Report_4_Final_Report.md PROJECT.md presentations/Report_4_Presentation.html`
Expected: present in all three.

- [ ] **Step 4: Instructor + test-count**

Run: `grep -rn "Trần Đức Anh\|15 bài test" presentations/ README.md`
Expected: no matches.

- [ ] **Step 5: Report any residual inconsistency** found by the greps above; if found, fix in the owning file and re-run. If clean, WS3 is complete.

---

## Self-Review

**Spec coverage (spec §6 WS3):**
- OCR EasyOCR→PaddleOCR (Report 3 §3.2, Report 4 diagram, slide R3) → Task 2 Step 2, Task 3 Step 2, Task 5 Step 2 ✓
- Models unify (delivered) + note change → Report 3 model-pivot bullet (Task 2 Step 3); proposal kept historical with pivot note (Task 1) — deliberate deviation from "rewrite Report 1 models", documented here and approved as Option C ✓
- Decision logic plate-primary across reports/slides/PROJECT → Tasks 2–5 ✓
- Report 4 §6 conclusion rewrite → Task 3 Step 4 ✓
- Latency reconciled → Task 3 Step 3 ✓
- Colour accuracy 14.16% vs 59.73% reconciled → Task 2 Step 4 ✓
- Citations verify/replace → Task 6 ✓
- Dataset scope planned-vs-delivered → covered by Report_1 pivot note (Task 1 Step 1) which flags the proposal as aspirational; the §6 dataset table stays as the proposal's plan ✓
- PROJECT.md runtime reality → Task 4 Step 1 ✓
- README test count → Task 4 Step 2 ✓
- Instructor name unify (Lương Trung Kiên) → Task 5 Step 1 ✓

**Placeholder scan:** Task 5 Step 3 and Task 6 are necessarily judgment-based (exact slide wording / web results unknown ahead of time); both give explicit decision rules and verification greps rather than blind edits. All mechanical edits have exact find/replace text.

**Consistency:** "plate-primary", "cảnh báo mềm", "PaddleOCR", "EfficientNet-B0/MobileNetV3-Small" used identically across tasks. Instructor name "Lương Trung Kiên" consistent.

**Note for executor:** edits are in Vietnamese; preserve diacritics and surrounding Markdown/HTML markup exactly. Commit per task. No code under `main/src/**` is touched.
