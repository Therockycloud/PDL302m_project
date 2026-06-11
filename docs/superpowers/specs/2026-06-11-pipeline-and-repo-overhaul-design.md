# Thiết kế: Nâng cấp Pipeline Nhận diện Xe Đỗ + Dọn Repo & Tài liệu

- **Ngày:** 2026-06-11
- **Trạng thái:** Đã duyệt thiết kế (chờ review spec)
- **Phạm vi:** Sub-project B (cốt lõi) + gộp A (bug fixes); kèm thiết kế cho E (repo/docs) và D (slides).
- **Lưu ý vị trí file:** File này nằm trong `docs/`. Sau khi thực hiện Sub-project E, `docs/` chuyển sang branch `docs-presentation`; spec này sẽ sống ở branch đó.

> Mục tiêu tổng: từ webcam / video "lùi chuồng", hệ thống nhận diện **xe đang đỗ** và đưa ra **một quyết định ổn định, nhanh** (AUTHORIZED / MISMATCH / UNREGISTERED) dựa trên **biển số + màu xe**; model nhẹ phù hợp phần cứng edge tương lai và **chính xác**; có **so sánh đa model** để chứng minh hiệu năng.

---

## 0. Quyết định đã chốt (Q&A)

| Chủ đề | Quyết định |
|---|---|
| Ưu tiên brainstorm trước | Sub-project B — Pipeline ML đúng |
| Cơ chế quyết định | Lấy mẫu N frame; **chỉ kích hoạt khi xe đang lùi vào chuồng**; gom kết quả (vote) → 1 quyết định khi xe đứng yên |
| Yếu tố quyết định | **Biển số + màu xe** (bỏ phân loại hãng khỏi luồng) |
| Chiến lược detect | **2 tầng:** Xe (YOLO) → Biển số (model riêng) → OCR |
| Engine biển số | Model biển số chuyên dụng riêng + OCR (recognition mặc định EasyOCR, có cờ chuyển PP-OCR) |
| Nguồn model biển | **Để mở:** spec liệt kê cả pretrained-fine-tune lẫn train-from-scratch; chốt sau benchmark B |
| Cơ chế kích hoạt | **PA1 — Heuristic vị trí + chuyển động** (gộp kiểm tra hướng lùi dạng nhẹ) |
| Benchmark | So sánh **CNN màu (A) + plate detector (B)**; bỏ so OCR |
| Gộp bug-fix A | Có — gộp vào đợt B |
| Đổi `verify_vehicle` | Có — bỏ tham số brand, cập nhật `test_matching.py` |
| Cấu trúc repo | `main` = code; branch `docs-presentation` riêng giữ docs/presentations (giữ lịch sử, không rewrite) |
| Nguồn chân lý tài liệu | **docs/** canonical; README phải có hướng dẫn dùng kỹ, **đa nền tảng macOS + Windows** |
| Slide | Giữ **HTML**, chuẩn lại theo **high-end-visual-design**; xoá tên thành viên; thêm Related Work + trích dẫn |
| Related Work | **Nghiên cứu bài báo thật** (academic/deep-research): cách nước ngoài làm + lý do thích nghi cho VN |
| Phần cứng đích | Edge/CPU nhẹ (lớp Jetson Nano / Raspberry Pi 5 / mini-PC); model export ONNX, ưu tiên nano/mobile |

---

## 1. Kiến trúc & luồng dữ liệu (Sub-project B)

State machine cho **một phiên đỗ xe**:

```
Video / Webcam
   │  (lấy mẫu mỗi N frame, mặc định N=5)
   ▼
[Tầng 1] VehicleDetector (YOLOv8n, lọc class xe)  → bbox xe + crop xe
   ▼
[Gate] ParkingTrigger (PA1 heuristic)
   │   theo dõi tâm/diện tích bbox qua K mẫu:
   │   • xe nằm trong ROI "chuồng"?  • đủ lớn/gần?  • đứng yên (đã lùi xong)?
   │   ─ chưa thỏa  → chỉ vẽ overlay "đang theo dõi…", KHÔNG chạy tầng nặng
   │   ─ thỏa       → chuyển COLLECTING
   ▼
[COLLECTING] với K frame ổn định, mỗi frame:
   │   ├─ [Tầng 2] PlateDetector (model biển riêng) → crop biển
   │   ├─ [Tầng 3] OCR → chuỗi biển số
   │   └─ ColorClassifier (CNN) trên crop XE → màu
   ▼
[Aggregator] vote: biển số = mode(chuỗi hợp lệ); màu = mode
   ▼
[Decision] DatabaseMatcher.verify(plate, color) → 1 quyết định ổn định
   ▼
[Khóa kết quả] giữ tới khi xe rời ROI → reset về IDLE
```

**Khác biệt cốt lõi so với hiện trạng:**
1. Pipeline nặng (biển + OCR + màu) **chỉ chạy khi gate mở** → giảm CPU, "nhanh" vì không phí tài nguyên mỗi frame.
2. Màu & biển lấy từ **crop xe / crop biển đúng vùng** (hiện tại classify trên full frame, OCR trên crop sai).
3. Kết quả là **một quyết định voted** thay vì nhấp nháy mỗi frame.

---

## 2. Components, interface & file layout (Sub-project B)

Nguyên tắc: mỗi class một file, interface hẹp, test cô lập được, tái dùng tối đa code hiện có.

### 2.1 `VehicleDetector` — `main/src/models/vehicle_detector.py` (mới, tách từ `detector.py`)
```python
class VehicleDetector:
    def __init__(self, model_path: str, conf: float = 0.3,
                 vehicle_classes: tuple[int, ...] = (2, 5, 7)): ...  # car, bus, truck (COCO)
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
    # Detection = {bbox:(x1,y1,x2,y2), conf:float, crop:np.ndarray}
```
- YOLOv8n nhưng **lọc đúng class xe** (hiện tại không lọc). Trả crop xe cho màu + làm input tầng biển.
- Giữ lại logic resolve path ONNX trong `detector.py` cũ (tái dùng).

### 2.2 `ParkingTrigger` — `main/src/engine/parking_trigger.py` (mới)
```python
class ParkingTrigger:
    def __init__(self, roi=None, min_area_ratio=0.15,
                 stable_frames=5, move_eps=0.02): ...
    def update(self, detections, frame_shape) -> TriggerState: ...
    # TriggerState ∈ {IDLE, TRACKING, READY_TO_DECIDE, DECIDED}
    def reset(self): ...
```
- Thuần Python/NumPy, **không phụ thuộc model** → unit-test bằng chuỗi bbox giả lập.
- `roi=None` → mặc định ROI = vùng giữa-dưới khung (nơi xe lùi tới). Chỉnh trong `config.yaml`.
- Kiểm tra hướng lùi dạng nhẹ: theo dõi dấu của thay đổi tâm-y qua K mẫu để xác nhận xe tiến vào chuồng rồi dừng.

### 2.3 `PlateReader` — `main/src/models/plate_reader.py` (gộp tầng 2+3, thay vai trò `ocr.py`)
```python
class PlateReader:
    def __init__(self, plate_model_path: str, ocr_engine: str = "easyocr"): ...
    def read(self, vehicle_crop: np.ndarray) -> PlateRead: ...
    # PlateRead = {text:str, conf:float, plate_bbox|None}
```
- Bên trong: detect biển trong crop xe → OCR vùng biển sạch.
- **Khôi phục OCR đúng:** dùng `readtext()` (detect+recognize) để xử lý biển 1 dòng/2 dòng — sửa regression đã phát hiện ở review (việc thay bằng `recognize()` một-box làm hỏng biển 2 dòng).
- Giữ helper `_sort_and_merge` / `_clean_text` (đang đúng & có test) đưa vào lớp này.

### 2.4 `DecisionEngine` — `main/src/engine/decision_engine.py` (mới)
```python
class DecisionEngine:
    def __init__(self, matcher: DatabaseMatcher, color_clf: ColorClassifier): ...
    def aggregate(self, frames_data: list[FrameData]) -> Decision: ...
    # vote: plate = mode(text hợp lệ), color = mode; bỏ frame OCR rỗng
    # Decision = {plate, color, status, action, message, votes_meta}
```
- Dùng lại `DatabaseMatcher.verify_vehicle` nhưng **đổi chữ ký** `verify_vehicle(plate, color)` (bỏ brand). Matching plate là chính, color là lớp xác thực phụ.
- Dùng lại `ColorClassifier` (đã có) nhưng **chạy trên crop xe**.

### 2.5 Orchestrator — `main/src/engine/parking_session.py` (mới)
```python
class ParkingSession:
    """Nối 4 đơn vị trên thành một vòng đời phiên đỗ xe."""
    def process_frame(self, frame) -> SessionOutput: ...
    # SessionOutput = {state, overlay_results, decision|None}
```
- `dashboard.py` gọi cái này (thay `_run_pipeline`) cho cả video & webcam. Overlay + counters giữ ở UI.

### 2.6 Thành phần bị loại / đổi vai
- `BrandClassifier` (`classifiers.py`): **gỡ khỏi luồng quyết định** (giữ file, không gọi).
- `detector.py` cũ: thay bằng `vehicle_detector.py` + `plate_reader.py`.
- `SystemEvaluator` (`evaluator.py`) & `run_evaluation.py`: cập nhật để khớp chữ ký mới (bỏ brand).

### 2.7 Cấu hình thêm `config.yaml`
```yaml
pipeline:
  frame_sample_interval: 5
  trigger: {min_area_ratio: 0.15, stable_frames: 5, move_eps: 0.02, roi: null}
plate_detector: {model_name: "plate_yolov8n.onnx", conf_threshold: 0.3}
ocr: {engine: "easyocr"}   # easyocr | ppocr
# brand_classifier giữ trong file config nhưng KHÔNG dùng trong luồng quyết định
```

---

## 3. Benchmark đa model — chứng minh hiệu năng (Sub-project B)

### 3.1 Component: `main/src/engine/benchmark.py` (mới)
```python
class ModelBenchmark:
    def run(self, candidates: list[ModelSpec], dataset) -> pd.DataFrame: ...
    # mỗi candidate đo: accuracy/F1 hoặc mAP · #params · size(MB) ·
    #                   CPU latency ms/ảnh · FPS · peak RAM
    def to_report(self, df) -> tuple[str, list[str]]: ...  # markdown table + đường dẫn plots
```
- Tái dùng `ModelTrainer` (train từng backbone) và `SystemEvaluator` (đo end-to-end).
- Xuất `docs/benchmarks/*.csv` + `*.png` (accuracy-vs-latency, bar charts) → dùng thẳng cho docs & slides.

### 3.2 Nhóm A — CNN phân loại màu
| Candidate | Vai trò |
|---|---|
| MobileNetV3-Small | baseline nhẹ (đang dùng) |
| EfficientNet-B0 | cân bằng accuracy/size |
| ResNet50 | accuracy cao, nặng — mốc trên |
| CNN tự xây nhỏ | mốc dưới, cực nhẹ cho edge |

### 3.3 Nhóm B — Plate detector
- YOLOv8n **train-from-scratch** (dataset Kaggle "Vietnamese Car License Plate Detection") **vs** YOLOv8n-LP **pretrained fine-tune**.
- So mAP@0.5, latency, size → đây cũng là cách **chốt câu "pretrained vs train"** còn để mở.

### 3.4 Chuẩn metric & kết luận
- Mọi nhóm đo: Accuracy/F1 hoặc mAP · #params · size(MB) · CPU latency(ms) · FPS · RAM.
- Chọn model theo **Pareto accuracy–latency**, ghi rõ "chọn X vì …" (gắn với mục tiêu phần cứng edge).

---

## 4. Xử lý lỗi & Kiểm thử (Sub-project B)

### 4.1 Xử lý lỗi (suy biến mềm)
- **Thiếu model / `onnxruntime`:** thêm `onnxruntime` vào `requirements.txt` (bug A); nếu vẫn thiếu → log cảnh báo, gate chạy chế độ "chỉ theo dõi", overlay báo "model chưa sẵn sàng", không crash.
- **OCR rỗng ở 1 frame:** bỏ frame khỏi vote; cả K frame rỗng → `NO_PLATE`, action `LOG`, không AUTHORIZED nhầm.
- **Vote không hội tụ:** trả `UNCERTAIN`, đề nghị xe đỗ lại / chụp gần hơn.
- **Màu lệch nhưng biển khớp:** → `MISMATCH` (lớp xác thực phụ), message rõ biển khớp/màu lệch.
- **ROI/ngưỡng sai cảnh:** `roi=null` mặc định vùng giữa-dưới; chỉnh qua `config.yaml`, không sửa code.

### 4.2 Kiểm thử
| Đơn vị | Loại | Cách test (không cần GPU) |
|---|---|---|
| `ParkingTrigger` | unit | Chuỗi bbox giả lập → khẳng định chuyển state IDLE→…→READY |
| `DecisionEngine.aggregate` | unit | List FrameData giả → kiểm tra vote + status; mở rộng `test_matching.py` cho `verify(plate,color)` |
| `PlateReader` | unit (mock) | Mock model biển + OCR; giữ test `_sort_and_merge`/`_clean_text` |
| `VehicleDetector` | smoke | 1 ảnh `data/test/*.jpg` → có ≥1 bbox xe |
| `ParkingSession` | integration | Chạy `sample_parking.mp4` → ra đúng 1 Decision, không lỗi |
| `ModelBenchmark` | smoke | 1 epoch / 2 candidate nhỏ → xuất DataFrame + file báo cáo |

### 4.3 Tiêu chí "xong" Sub-project B
1. Video lùi chuồng → đúng 1 quyết định ổn định (biển+màu) khi xe đứng yên trong ROI.
2. Pipeline nặng chỉ chạy khi gate mở (đo được CPU giảm so với mỗi-frame).
3. Bảng benchmark A+B (CSV + biểu đồ) sinh ra được, kèm câu kết luận chọn model.
4. Bug A đã vá (`onnxruntime`, OCR 2 dòng, import thừa, đặt tên CSS); toàn bộ `pytest` xanh.

---

## 5. Dọn repo & đồng bộ tài liệu (Sub-project E)

> Triển khai ở branch riêng, **tách hẳn commit** khỏi code B.

### 5.1 Tách cấu trúc git (giữ lịch sử, không rewrite)
- Tạo branch `docs-presentation` từ `main` hiện tại → nơi sống của `docs/`, `presentations/`, `course_details/`, `zip/`.
- Trên `main`: `git rm` các thư mục đó (lịch sử vẫn truy ra được), chỉ giữ:
  `main/`, `README.md`, `Dockerfile`, `docker-compose.yml`, `.gitignore`.
- Dọn rác cả hai branch: `.DS_Store`, `.pytest_cache/`, `__pycache__/`, `zip/` thừa → cập nhật `.gitignore`.
- README ghi rõ: "Tài liệu học thuật & slide ở branch `docs-presentation`".

### 5.2 Đồng bộ nội dung (docs = nguồn chân lý)
Thứ tự cập nhật sau khi B đổi kiến trúc:
1. **`docs/model_specifications.md`** + **Report_3 / Report_4**: bỏ Brand/ResNet50 khỏi luồng quyết định; thêm kiến trúc 2 tầng, plate detector, quyết định biển+màu, **bảng benchmark A+B**.
2. **Slides** `presentations/Report_*.html`: tóm tắt đúng theo docs; nhúng biểu đồ benchmark; vá placeholder ảnh trống (xem Phần 6).
3. **Script thuyết trình**: tạo/đồng bộ **đủ 4 report** (`Report_1..4_Script.md`); hiện chỉ có Report_1 & Report_4 → thêm Report_2 & Report_3, cập nhật khớp slide mới.

### 5.3 README đa nền tảng (làm kỹ)
Mục **Cài đặt & Chạy** cho 3 đường:
- **macOS/Linux:** `main/run.sh`, `main/run_ui.sh` (kèm tạo venv + cài `requirements.txt`).
- **Windows:** `main/run_ui.bat` + lệnh PowerShell tương đương (venv / cài đặt / chạy).
- **Docker (mọi OS):** `docker compose up` — khuyến nghị cho người mới.
- **Troubleshooting:** thiếu `onnxruntime`, model chưa tải, quyền webcam trên mac/win.

### 5.4 Nghiệm thu E
- Clone sạch `main` → cài theo README (cả mac & win) → `pytest` xanh, UI chạy.
- `git ls-files` trên `main` không còn file rác / docs / presentations.
- Không còn nhắc "brand classifier trong quyết định" ở bất kỳ README/docs/slide nào.

---

## 6. Audit & nâng cấp slide (Sub-project D)

- **Định dạng:** giữ **HTML**, chuẩn lại toàn bộ theo skill **high-end-visual-design** (font, spacing, shadow, card, animation cao cấp).
- **Xoá tên thành viên** trên mọi slide (nhóm tự điền sau).
- **Thêm mục Related Work / Literature:**
  - **Nghiên cứu bài báo thật** (academic-research-skills / deep-research): chọn các công trình ALPR / nhận diện biển số / quản lý bãi đỗ ở nước ngoài.
  - Trình bày: nước ngoài đã triển khai ý tưởng này **dưới hình thức nào** (ví dụ ANPR cho thu phí/bãi đỗ ở EU/US/TQ), và **tại sao chọn cho điều kiện Việt Nam** (biển 2 dòng, nhiều xe máy, ánh sáng/thời tiết, chi phí thấp, chạy edge, dataset nội địa).
  - Kèm **trích dẫn chuẩn** (dùng citation format của academic skill).
- **Vá placeholder ảnh** trống / link hỏng; nhúng biểu đồ benchmark + sơ đồ kiến trúc 2 tầng sinh từ B.
- Nghiệm thu D: không còn ô ảnh trống/link hỏng; mọi slide khớp docs; có slide Related Work với ≥3 trích dẫn thật; không còn tên thành viên.

---

## 7. Thứ tự triển khai & quan hệ phụ thuộc

```
B (code + benchmark + bug A)
      │  (đổi kiến trúc → tạo sự thật kỹ thuật mới)
      ▼
Cập nhật docs theo B  ──►  E (tách repo + đồng bộ + README đa nền tảng)
      │
      ▼
D (audit slide sâu: high-end-visual-design + Related Work nghiên cứu thật)
```

- **B là sub-project được lập plan & thực thi trước.** Mỗi sub-project sau (E, D) sẽ có vòng plan → implement riêng để giữ phạm vi gọn.
- A được gộp vào B (cùng đụng `detector.py`, `ocr.py`, `requirements.txt`).

---

## 8. Rủi ro & giả định

- **Giả định phần cứng đích:** edge/CPU nhẹ; nếu nhóm có thiết bị cụ thể khác (vd Coral TPU) cần chỉnh lựa chọn export model.
- **Rủi ro dataset biển VN:** chất lượng/độ phủ ảnh hưởng option train-from-scratch; benchmark B sẽ quyết định.
- **Rủi ro ROI heuristic:** phụ thuộc góc camera; mặc định vùng giữa-dưới + cấu hình; có thể cần hiệu chỉnh theo cảnh thực.
- **Rủi ro tách git:** thao tác trên branch riêng, không rewrite history → an toàn, có thể hoàn tác bằng revert.
