# Design: Cải thiện Data & Độ chính xác + Dọn Git + Video thật

- **Ngày:** 2026-06-17
- **Dự án:** DPL302m — Smart Parking Lot Security Monitoring (Vehicle Anti-Theft)
- **Tinh thần:** Dự án nhỏ → **giữ đơn giản**, không over-engineer. Commit sau mỗi bước, làm trên `main`.

---

## 0. Phạm vi (3 mảng)
1. **Dọn git**: gộp/dọn branch về `main` cho gọn (housekeeping, làm trước).
2. **Cải thiện độ chính xác** Color (deploy) + Brand (experimental) qua **làm sạch nhãn + recipe**.
3. **Tận dụng video thật** `parking_case_real.mp4` cho eval/demo + domain-gap + calibrate.
Cuối cùng: cập nhật Report 2 cho khớp số thật.

## 1. Chẩn đoán hiện trạng (số thật trên đĩa)

| Model | Backbone | Test Acc | Macro-F1 | Lớp tệ | Lớp tốt |
|---|---|---|---|---|---|
| Color (deploy) | MobileNetV3-Small | 54.2% | 0.540 | Brown 0.23, Grey 0.46 | Yellow 0.79 |
| Brand (experimental) | EfficientNetB0 | 35.3% | 0.337 | Mazda 0.17, Kia 0.19 | Hyundai 0.50 |

**Lưu ý R2:** slide đang ghi số CŨ (color 48.3 / brand 32.8, trước fine-tune) — sẽ cập nhật ở bước cuối.

**Nguyên nhân chính:** data ít (~70 ảnh/lớp train, test 15/lớp → nhiễu); augmentation yếu (chỉ flip+rotate0.1+zoom0.1); brand bias VinFast; pipeline làm sạch cũ không bắt **nhãn sai**; lệch crop train↔infer (một số path classify full frame).

## 2. Quyết định đã chốt (brainstorming)

| Vấn đề | Lựa chọn |
|---|---|
| Ngân sách data | Làm sạch nhãn sai (data hiện có) + tối ưu recipe. KHÔNG crawl mới quy mô lớn |
| Brand | Cải thiện color; brand làm **nhánh thực nghiệm**, KHÔNG đưa lại hệ thống |
| Report 2 | Cập nhật **sau** khi có số cuối |
| Crop consistency | **Có** sửa code inference (dashboard/app) classify trên vehicle crop |
| Git | Gộp/dọn branch về `main`; commit sau mỗi bước; **KHÔNG** merge 2 branch stripped |
| Độ phức tạp | **Giữ đơn giản** — bỏ ML machinery nặng (xem mục 5) |

## 3. Video thật `parking_case_real.mp4` (asset domain thật)

854×480, 30fps, ~30s (898 frames). Nội dung: camera đặt **thấp ở góc nhìn ra**; xe **VinFast VF3 màu vàng lùi vào**, đuôi + biển số quay vào camera, to dần rồi dừng — **đúng kịch bản** mô tả trong `config.yaml`.

**Ground truth 1 case:** `color=Yellow`, `brand=VinFast`, `plate` hiện rõ.

**Cách dùng (ưu tiên đơn giản):**
- **(A) Demo/eval end-to-end thật** — chạy `ParkingSession` lên video: detect → đọc biển → color có ra "Yellow"? → trigger có bắn? Đây là phép thử đúng miền, mạnh hơn test web. → đáp ứng **roadmap R2 item 1** không cần crawl mới.
- **(B) Domain-gap check** — color model (train ảnh web) có nhận đúng "Yellow" trên crop CCTV không.
- **(C) Calibrate ROI/trigger** bằng `calibrate_roi.py` cho camera thật.
- **(D) Tùy chọn, thận trọng** — trích vài crop làm mini real eval. CHỈ 1 xe = Yellow/VinFast → **không** dùng cùng frames vừa train vừa test (tránh leak).

## 4. Git consolidation (làm trước, housekeeping)

**Trạng thái:** 6 branch local, 2 worktree, main ahead origin/main 81 commits, working tree đang dirty.

| Branch | Thực chất | Hành động |
|---|---|---|
| `feature/improve-data-pipeline` | ancestor của main | **xóa** |
| `redesign/ui-content-code-overhaul` | ancestor của main | **xóa** |
| `training2` (worktree) | ancestor của main | gỡ worktree + **xóa** |
| `fix-report-2` (worktree) | sửa 1 file `docs/Report_2_Data_Tasks.md` (rewrite học thuật) | **merge vào main** → gỡ worktree + xóa |
| `docs-and-presentations` | docs-only export, **xóa hết code** (−5365 dòng) | **ĐỂ YÊN** (có remote) — KHÔNG merge |
| `test/streamlit-only` | deploy build, **bỏ core source** | **ĐỂ YÊN** — KHÔNG merge |

**Bước:** (1) commit working tree đang dở (docs spec + thay đổi presentation R2) — *xác nhận với user phần presentation là WIP hay commit được*; (2) merge `fix-report-2`; (3) gỡ 2 worktree; (4) xóa 3 branch ancestor + `fix-report-2`; (5) để yên 2 branch stripped. Mỗi bước 1 commit. KHÔNG push (trừ khi user yêu cầu).

## 5. Cải thiện độ chính xác — recipe ĐƠN GIẢN

> Đã cắt bỏ so với bản nặng: bỏ ensemble, bỏ k-fold CV bắt buộc, bỏ cleanlab nặng, bỏ ablation matrix lớn, bỏ confidence-threshold/TTA khỏi core (để "optional/tương lai").

**Eval (đơn giản, tin được):**
- **Đóng băng** `data/processed/classifiers/*/test` (anchor so sánh R2). Chỉ clean train/val.
- Báo cáo before/after trên test đóng băng + **chạy video thật** làm phép thử miền. (k-fold CV = optional nếu còn thời gian.)

**Làm sạch nhãn (nhẹ, human-in-the-loop):**
- Train model hiện tại → liệt kê ảnh **train bị đoán sai với confidence cao** (dấu hiệu nhãn sai) → người review nhanh, sửa/loại. Không cần thư viện nặng.
- Color: thêm sanity-check màu trội vùng xe (rẻ). Chỉ động train/val.

**Recipe (lực chính, rẻ):**
- **Augmentation tách theo task** trong `vehicle_dataset.py`:
  - Color (nhạy màu): thêm translation + brightness/contrast NHẸ. **CẤM hue/saturation jitter.**
  - Brand (nhạy hình/logo): mạnh hơn — color jitter, RandAugment.
- **Fine-tune** theo macro-F1 (EarlyStopping); tinh chỉnh LR/epochs. Cấu trúc 2-stage sẵn có.

**Crop/domain consistency:**
- Sửa path "Upload Image" (`dashboard.py:272`) và API (`app.py`): `VehicleDetector` → crop → mới classify. (Path video/ParkingSession đã đúng.)

**Brand experimental:** áp dụng cleaning + recipe ở nhánh riêng, báo cáo số. **KHÔNG** đụng matcher/decision_engine/parking_session.

## 6. Mục tiêu (validate, không hứa chắc)
- Color: 54.2% → **65–70%** (F1 ≥ 0.63) + chạy đúng trên video thật (ra "Yellow").
- Brand (experimental): 35.3% → **45–50%** (F1 ≥ 0.45).

## 7. File sẽ đụng
- Git: branch/worktree ops (không sửa file nội dung trừ merge `fix-report-2`).
- `src/datasets/vehicle_dataset.py` (aug tách task), `train.py` (fine-tune theo F1), `src/ui/dashboard.py` + `src/api/app.py` (crop consistency).
- Script nhẹ (tùy chọn): `scripts/audit_labels.py` (liệt kê train nghi sai nhãn), `scripts/run_on_video.py` (chạy pipeline lên video thật & in kết quả) — hoặc tái dùng tooling sẵn có.
- `presentations/Report_2_Presentation.html` (bước cuối).
- **KHÔNG đụng:** `matching.py`, `decision_engine.py`, `parking_session.py`, test set.

## 8. Rủi ro & giảm thiểu
- Merge nhầm branch stripped → mất code: **đã chặn** (để yên 2 branch đó).
- Test nhỏ → cải tiến là nhiễu: dùng video thật + before/after; CV nếu cần.
- Hue jitter phá nhãn color: cấm cho color.
- Crop fix hồi quy path video đang đúng: chỉ sửa upload-image & API, test trước/sau.
- Video 1 xe → leak nếu train+test cùng frames: chỉ dùng eval/demo, không train-test trùng.

## 9. Nhất quán Report 2
- Plan = đúng roadmap R2 item 1 (domain gap qua video thật) + item 2 (fine-tune backbone color).
- Brand giữ "ngoài hệ thống" → không mâu thuẫn quyết định đã trình bày.
- R2 cập nhật ở bước cuối (số color/brand mới + sửa lập luận brand + verify citation Yang 2025).

## 10. Open Questions
- Phần thay đổi `presentations/Report_2_Presentation.html` đang uncommitted: là WIP hay commit luôn khi dọn git? (xác nhận lúc thực thi)
- Giữ hay xóa `docs-and-presentations` / `test/streamlit-only`? (mặc định: để yên)
