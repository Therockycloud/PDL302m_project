# 🚗 Smart Parking Security System via Cross-Verification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống giám sát an ninh bãi đỗ xe thông minh ứng dụng Học Sâu (Deep Learning) để phòng chống các hành vi gian lận hoặc tráo đổi biển số xe tinh vi. Cơ chế quyết định lấy **Biển số xe (OCR)** làm khoá chính; **Màu sắc xe (Color)** đóng vai trò cảnh báo mềm (ALLOW_WARN, gộp cụm màu trung tính Black/Grey/Silver/White + confidence gate 0.40) chống tráo biển; **Hãng xe (Brand)** chỉ mang tính chẩn đoán (diagnostic), không tham gia vào quyết định cuối cùng.

---

## 📚 Tài liệu các giai đoạn dự án (Project Reports)

Toàn bộ báo cáo kỹ thuật và hướng dẫn kiểm nghiệm chính thức được lưu trữ cấu trúc dưới thư mục `reports/documents/` (và `docs/` cho đặc tả mô hình) ở gốc dự án:

*   **[Report 1: Đề xuất dự án & Kiến trúc (Proposal)](../reports/documents/Report_1_Proposal.md)** - Mô tả vấn đề an ninh, thiết kế kiến trúc hệ thống và phân công công việc.
*   **[Report 2: Quy trình xử lý dữ liệu (Data Tasks)](../reports/documents/Report_2_Data_Tasks.md)** - Chi tiết thu thập dữ liệu Stanford Cars, Wikimedia VinFast crawler, EDA và tiền xử lý.
*   **[Report 3: Kết quả thực nghiệm mô hình (Model & Results)](../reports/documents/Report_3_Model_Results.md)** - Đặc tả thiết kế mạng EfficientNet-B0, MobileNetV3-Small, PaddleOCR và đồ thị huấn luyện.
*   **[Report 4: Tích hợp hệ thống & Đánh giá (Final Defense)](../reports/documents/Report_4_Final_Report.md)** - Kết quả đo lường độ trễ đầu cuối trên CPU, các kịch bản kiểm thử an ninh bãi xe.
*   **[Đặc tả Mô hình & Cấu hình (Model Specs)](../docs/model_specifications.md)** - File cấu hình chi tiết, dải pixel đầu vào và sơ đồ thư mục CSDL.
*   **[Hướng dẫn báo cáo (Report Guidelines)](../reports/documents/report_guidelines.md)** - Các yêu cầu, khuôn mẫu slide presentations theo tiêu chuẩn của môn học.

---

## 🏗️ Kiến trúc hệ thống (System Architecture)

Hệ thống được thiết kế theo mô hình Microservices phân tách độc lập giữa Backend (FastAPI suy luận) và Frontend (Streamlit Dashboard):

```
                  ┌──────────────────────┐
                  │   Camera giám sát    │ (Gửi ảnh xe máy/ô tô lúc ra)
                  └──────────┬───────────┘
                             │ POST /verify
                             ▼
                  ┌──────────────────────┐
                  │    FastAPI Server    │
                  └────┬────────────┬────┘
       ┌────────────────┘            └────────────────┐
       ▼ (Luồng xử lý song song, biển số = khoá chính) ▼ (Cảnh báo mềm, không quyết định)
 ┌───────────┐                                  ┌────────────┐
 │  YOLOv8   │ (Cắt vùng biển số)               │MobileNetV3 │ (Phân loại màu → ALLOW_WARN)
 └─────┬─────┘                                  └─────┬──────┘
       ▼                                              │
 ┌───────────┐                                        │ (EfficientNet-B0 Hãng xe:
 │ PaddleOCR │ (Đọc ký tự biển số)                     │  chỉ diagnostic, KHÔNG vào
 │           │                                        │  quyết định — xem log/UI)
 └─────┬─────┘                                        │
       ▼                                              │
 ┌───────────┐                                        │
 │  Spatial  │ (Sắp xếp không gian 2 dòng)            │
 │  Sorting  │                                        │
 └─────┬─────┘                                        │
       ▼                                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │              Bộ đối chiếu chéo (Cross-Verifier)          │
 │   (Biển số khớp CSDL là điều kiện chính; màu lệch chỉ    │
 │    phát cảnh báo mềm, không tự động khoá xe)             │
 └────────────────────────────┬─────────────────────────────┘
                              │ Trả về trạng thái & Báo động
                              ▼
                  ┌──────────────────────┐
                  │ Streamlit Dashboard  │ (Còi hú + chớp đỏ nếu lệch thông tin)
                  └──────────────────────┘
```

> **OCR runtime:** Demo và API vẫn dùng **PaddleOCR** (`ocr.engine: ppocr`). Thí nghiệm ONNX/CTC nhẹ (mục dưới) **chưa thay** Paddle.

---

## 🔤 Thí nghiệm OCR nhẹ (CTC/ONNX) — chưa deploy

Huấn luyện thử **MobileNetV3-Small + CTC → ONNX** (`main/data/models/vn_plate_run/vn_plate_recognizer.onnx`) trên synthetic + pseudo-label; đánh giá trên biển ô tô thật. Gate thay Paddle: **≥90% exact-match** trên held-out real — **chưa đạt** (`deployment_ready: false`).

| | |
|---|---|
| Train | `task4_train` + `pseudo_vision` (conf ≥ 0.5) |
| Val (tuning) | `real_validation.csv` (64) |
| Giữ kín | `expanded_real_test` (102), `frozen_regression` (16) |
| Kết quả | val exact-match **0/64**; val CER **~0.659** |
| Nguyên nhân | Domain gap (val/test = ô tô; pseudo chủ yếu xe máy); thiếu biển ô tô verified cho train |

Paddle baseline ~**81%** exact-match trên frozen 16 (Benchmark C) — đủ để giữ runtime; ONNX nhẹ hơn nhưng chưa đủ chính xác.

Chính sách dữ liệu & anti-leakage: [`data/plate_ocr/README.md`](data/plate_ocr/README.md).

---

## 📊 Thống kê tập dữ liệu thực nghiệm (Dataset Metrics)

Hệ thống được huấn luyện và đánh giá trên bộ dữ liệu thực tế đã qua tiền xử lý, loại bỏ hoàn toàn mock data:

*   **Tập phân loại hãng xe (Brands)**: Tổng cộng **792** hình ảnh đã làm sạch & cân bằng trên 8 lớp:
    *   *Toyota (95), Hyundai (99), Kia (99), Mazda (100), Honda (99), VinFast (100 - cào từ Wikimedia), Ford (100), Mitsubishi (100)*.
*   **Tập phân loại màu sắc xe (Colors)**: Tổng cộng **783** hình ảnh trên 8 gam màu (~100 ảnh/lớp cân bằng):
    *   *White (100), Black (100), Grey (100), Silver (100), Red (100), Blue (100), Brown (91), Yellow (92)*.
*   **Dữ liệu biển số kiểm thử (License Plates)**: **5** hình ảnh xe thực tế tại Việt Nam kèm file nhãn định dạng YOLO tương ứng để đánh giá E2E.

---

## 🛠️ Hướng dẫn cài đặt & Vận hành (Installation & Setup via Docker)

Hệ thống hỗ trợ cài đặt và chạy tức thời bằng Docker và Docker Compose trên mọi hệ điều hành (macOS, Windows, Linux) mà không cần cài đặt môi trường Python hay các thư viện OpenCV/Học sâu trên máy chủ:

### 1. Khởi động hệ thống (Start API & Dashboard)
Chạy lệnh sau từ thư mục gốc của dự án:
```bash
docker compose up --build
```
*Lệnh này sẽ tự động tải các dependencies; cache mô hình PaddleOCR đã được "mồi" (prime) sẵn ngay tại **thời điểm build image** (không phải lúc chạy), giúp container khởi động và vận hành **hoàn toàn zero-network** (đã kiểm chứng bằng `docker run --network none`), song song khởi động Backend FastAPI (cổng 8000) và Dashboard Streamlit (cổng 8501).*

*Lưu ý: trên nền linux/aarch64 (Docker Desktop trên Apple Silicon), image dùng engine PaddleOCR legacy 2.7.3 (model PP-OCRv3-det/PP-OCRv4-rec) do bug segfault của loader PIR paddle 3.x trên nền tảng này; số đo Benchmark C (81% exact-match) được đo với stack 3.x/PP-OCRv6 chạy native — độ chính xác OCR trong container có thể khác và chưa được benchmark riêng.*

*   **API Documentation (Swagger UI)**: Truy cập tại `http://localhost:8000/docs`
*   **Vận hành Dashboard UI**: Truy cập tại `http://localhost:8501`

> **Upload Video — Product view đồng bộ:** một canvas Product duy nhất (đã bỏ pane
> Source video riêng) làm media clock; vẽ decoded frame và POST frame mẫu trực tiếp
> tới FastAPI với tối đa một request đang chạy. Streamlit chỉ rerun khi có verdict
> cuối. Vì request phát từ trình duyệt, cả `http://localhost:8501` và
> `http://localhost:8000` phải truy cập được từ máy người dùng; Compose đặt
> `DPL_DEMO_API_URL=http://localhost:8000`. Seek sẽ reset trajectory/evidence
> session trước khi tiếp tục lấy mẫu.

**Registry** (Input Mode thứ tư): quản lý `main/data/database.csv` và ảnh tham chiếu
tại `main/data/registry/photos/` — thêm/xóa xe, không chạy pipeline detect. Sau mỗi
thay đổi, matcher trong session được reload để Upload Image/Video/Webcam dùng CSDL mới.

**Webcam** dùng camera trình duyệt (`getUserMedia`) và POST frame tới `/demo/frame`
(giống Upload Video), nên hoạt động khi dashboard chạy trong Docker. Cần cấp quyền
camera cho trang `http://localhost:8501`; HTTPS hoặc `localhost` là bắt buộc trên
hầu hết trình duyệt hiện đại.

Nếu hết cửa sổ `collect_frames` mà biển vẫn không đọc chắc chắn, session chốt
**UNCERTAIN** và giữ barrier đóng thay vì treo ở trạng thái verifying. Backend
warmup model khi container khởi động; giới hạn một thread vẫn được giữ để mô
phỏng máy production cấu hình thấp mà không đẩy cold-start vào sự kiện xe lùi.

### 🎬 Video demo (Upload Video)

Radio **Demo video** chỉ chọn một clip; để trống (—) thì không phát video.

| Lựa chọn | File | Biển (OCR) | Trong `database.csv` | Kết quả demo |
|----------|------|------------|----------------------|--------------|
| **1. Unregistered** | `main/data/test/parking_case_real.mp4` | `30M-718.54` | Không | **UNREGISTERED** |
| **2. Registered** | `main/data/test/parking_case_real_v2.mp4` | `30K-439.36` | Có (`Kia Sonet`, White) | **AUTHORIZED** |
| **3. Mismatched** | `main/data/demo_videos/sequence_01_1.mp4` | `29B-625.61` | Có (`Ford Transit`, Red — sai màu) | **AUTHORIZED** + **ALLOW_WARN** |
| **4. Registered (Seq)** | `main/data/demo_videos/sequence_01_2.mp4` | `29F-019.51` | Có (`VinBus`, Grey) | **AUTHORIZED** + **ALLOW** |

`data/test/sample_parking.mp4` là artifact tải ngoài Git (tùy chọn). Nếu cần,
chạy từ thư mục gốc dự án:

```bash
python main/src/utils/download_sample_video.py
python main/src/utils/download_sample_video.py --verify
```

### 2. Dừng và dọn dẹp môi trường (Clear Environment)
Để dừng các container và giải phóng bộ nhớ, chạy:
```bash
docker compose down
```

---

## 🏃 Kiểm thử hệ thống (Running Tests inside Container)

Để thực thi bộ unit test tự động trong đúng working directory của application bên trong Docker:

```bash
docker compose exec -T -w /app/main backend pytest -q
```

---

## ⚡ Tối ưu hóa suy luận trên CPU ngoại tuyến (Offline CPU Optimizations)

Để hệ thống hoạt động với hiệu năng ổn định nhất trên các PC bãi đỗ chạy CPU thông thường và không cần kết nối Internet, dự án tích hợp các giải pháp tối ưu sau:

### 1. Khắc phục nghẽn luồng / Treo CPU (Thread Deadlock Fix)
Khi TensorFlow và PyTorch chạy song song, việc tranh chấp luồng tính toán có thể làm đơ hệ thống. Chúng ta bắt buộc phải cấu hình giới hạn đơn luồng cho các thư viện xử lý trước khi chạy:
*   Được tự động cấu hình trong `Dockerfile` và `docker-compose.yml` thông qua các biến môi trường:
    *   `OMP_NUM_THREADS=1`
    *   `MKL_NUM_THREADS=1`
    *   `OPENBLAS_NUM_THREADS=1`
    *   `VECLIB_MAXIMUM_THREADS=1`
    *   `NUMEXPR_NUM_THREADS=1`
*   *Giải pháp này đưa độ trễ suy luận ổn định về mức cực thấp chỉ **~1.6 giây / xe**.*

### 2. Cấu hình chạy ngoại tuyến hoàn toàn (100% Offline Mode)
*   **PaddleOCR** (`ocr.engine: "ppocr"` — engine runtime duy nhất): cache mô hình được **mồi sẵn ở thời điểm build Docker image** (build-time prime), không tải qua mạng khi container chạy. Nếu PaddleOCR không khởi tạo được, hệ thống **raise `RuntimeError`** ngay (không còn silent fallback sang EasyOCR như bản cũ) — EasyOCR nay chỉ dùng cho train/eval/benchmark (`requirements-train.txt`), không có mặt trong runtime.
*   **YOLOv8**: Tắt tính năng đồng bộ telemetry của Ultralytics qua: `settings.update({"sync": False})` và biến môi trường `YOLO_OFFLINE=True`. Sao chép thủ công tệp font hệ thống `Arial.ttf` vào thư mục mặc định `~/.config/Ultralytics/` để chặn việc YOLOv8 tự động tải font mỗi lần suy luận.
*   *Lưu ý: Dockerfile đã tự động mồi (prime) cache PaddleOCR và tải trước các model weights YOLOv8 trong quá trình build để container hoạt động hoàn toàn ngoại tuyến (zero-network runtime).*
