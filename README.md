# 🚗 Smart Parking Security System via Cross-Verification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống giám sát an ninh bãi đỗ xe thông minh ứng dụng Học Sâu để phòng chống tráo đổi biển số. Pipeline 2 tầng **YOLOv8n (xe → biển số) → PaddleOCR**, kèm phân loại **màu xe (MobileNetV3)** làm lớp xác thực phụ, và một **cơ chế gate "xe đã đỗ"** chỉ chạy suy luận nặng một lần mỗi xe.

> **Quyết định theo biển số (plate-primary):** biển số là khoá chính — biển khớp ⇒ AUTHORIZED; màu lệch chỉ là **cảnh báo mềm** (không từ chối cứng); biển không có trong CSDL ⇒ UNREGISTERED. Xem quá trình & lý do thay đổi so với thiết kế ban đầu trong các slide (`presentations/`), số liệu benchmark trong `docs/benchmarks/`, và đối chiếu công nghệ thế giới trong [`docs/related_work.md`](docs/related_work.md).

---

## 📚 Tài liệu các giai đoạn dự án (Project Reports)

Toàn bộ báo cáo kỹ thuật và hướng dẫn kiểm nghiệm chính thức được lưu trữ cấu trúc dưới thư mục `docs/`:

*   **[Report 1: Đề xuất dự án & Kiến trúc (Proposal)](docs/Report_1_Proposal.md)** - Mô tả vấn đề an ninh, thiết kế kiến trúc hệ thống và phân công công việc.
*   **[Report 2: Quy trình xử lý dữ liệu (Data Tasks)](docs/Report_2_Data_Tasks.md)** - Chi tiết thu thập dữ liệu Stanford Cars, Wikimedia VinFast crawler, EDA và tiền xử lý.
*   **[Report 3: Kết quả thực nghiệm mô hình (Model & Results)](docs/Report_3_Model_Results.md)** - Đặc tả thiết kế mạng EfficientNet-B0, MobileNetV3-Small, EasyOCR và đồ thị huấn luyện.
*   **[Report 4: Tích hợp hệ thống & Đánh giá (Final Defense)](docs/Report_4_Final_Report.md)** - Kết quả đo lường độ trễ đầu cuối trên CPU, các kịch bản kiểm thử an ninh bãi xe.
*   **[Đặc tả Mô hình & Cấu hình (Model Specs)](docs/model_specifications.md)** - File cấu hình chi tiết, dải pixel đầu vào và sơ đồ thư mục CSDL.
*   **[Hướng dẫn báo cáo (Report Guidelines)](docs/report_guidelines.md)** - Các yêu cầu, khuôn mẫu slide presentations theo tiêu chuẩn của môn học.

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
       ▼ (Luồng xử lý song song)                      ▼ (Luồng xử lý song song)
 ┌───────────┐                                  ┌───────────┐
 │  YOLOv8   │ (Cắt vùng biển số)               │EfficientNet│ (Phân loại hãng)
 └─────┬─────┘                                  └─────┬─────┘
       ▼                                              │ (Phân loại màu)
 ┌───────────┐                                        ▼
 │  EasyOCR  │ (Đọc ký tự biển số)              ┌───────────┐
 │           │                                  │MobileNetV3│
 └─────┬─────┘                                  └─────┬─────┘
       ▼                                              │
 ┌───────────┐                                        │
 │  Spatial  │ (Sắp xếp không gian 2 dòng)            │
 │  Sorting  │                                        │
 └─────┬─────┘                                        │
       ▼                                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │              Bộ đối chiếu chéo (Cross-Verifier)          │
 │   (Kiểm tra chéo thông tin đầu ra với CSDL lịch sử CSV)  │
 └────────────────────────────┬─────────────────────────────┘
                              │ Trả về trạng thái & Báo động
                              ▼
                  ┌──────────────────────┐
                  │ Streamlit Dashboard  │ (Còi hú + chớp đỏ nếu lệch thông tin)
                  └──────────────────────┘
```

---

## 📊 Thống kê tập dữ liệu thực nghiệm (Dataset Metrics)

Hệ thống được huấn luyện và đánh giá trên bộ dữ liệu thực tế đã qua tiền xử lý, loại bỏ hoàn toàn mock data:

*   **Tập phân loại hãng xe (Brands)**: Tổng cộng **1,209** hình ảnh được làm sạch phân chia trên 8 lớp:
    *   *Toyota (168), Hyundai (200), Kia (120), Mazda (120), Honda (161), VinFast (120 - cào từ Wikimedia), Ford (200), Mitsubishi (120)*.
*   **Tập phân loại màu sắc xe (Colors)**: Tổng cộng **1,130** hình ảnh trên 8 gam màu:
    *   *White (185), Black (200), Grey (200), Silver (175), Red (110), Blue (200), Brown (35), Yellow (25)*.
*   **Dữ liệu biển số kiểm thử (License Plates)**: **5** hình ảnh xe thực tế tại Việt Nam kèm file nhãn định dạng YOLO tương ứng để đánh giá E2E.

---

## 🛠️ Hướng dẫn cài đặt & Vận hành (Installation & Setup via Docker)

Hệ thống hỗ trợ cài đặt và chạy tức thời bằng Docker và Docker Compose trên mọi hệ điều hành (macOS, Windows, Linux) mà không cần cài đặt môi trường Python hay các thư viện OpenCV/Học sâu trên máy chủ:

### 1. Khởi động hệ thống (Start API & Dashboard)
Chạy lệnh sau từ thư mục gốc của dự án:
```bash
docker compose up --build
```
*Lệnh này sẽ tự động tải các dependencies, tải trước các mô hình YOLOv8 và EasyOCR để chạy ngoại tuyến, khởi động Backend FastAPI (cổng 8000) và Dashboard Streamlit (cổng 8501) song song.*

*   **API Documentation (Swagger UI)**: Truy cập tại `http://localhost:8000/docs`
*   **Vận hành Dashboard UI**: Truy cập tại `http://localhost:8501`

### 2. Dừng và dọn dẹp môi trường (Clear Environment)
Để dừng các container và giải phóng bộ nhớ, chạy:
```bash
docker compose down
```

---

## 💻 Chạy trực tiếp trên máy — Native (macOS / Windows, không cần Docker)

Yêu cầu: **Python 3.12**. Mô hình đã kèm trong repo (`main/data/models/`: `plate_yolov8n.onnx`, `color_MobileNetV3Small.pt`, `yolov8n.onnx`).

### 🍎 macOS / Linux
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r main/requirements.txt

# Chạy Dashboard (script tự dò interpreter + đặt biến môi trường)
bash main/run_ui.sh
# …hoặc chạy thủ công:
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=main streamlit run main/src/ui/dashboard.py
```

### 🪟 Windows (PowerShell)
```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r main\requirements.txt

$env:KMP_DUPLICATE_LIB_OK = "TRUE" ; $env:PYTHONPATH = "main"
streamlit run main\src\ui\dashboard.py
# …hoặc: main\run_ui.bat
```

Dashboard mở tại `http://localhost:8501`. Chọn **Upload Video → “Play Default Parking Video”**, hoặc **Webcam**, để chạy pipeline end-to-end.

### 🧪 Chạy test
```bash
cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q     # 28 passed, 5 skipped
```

### 🛠️ Khắc phục sự cố (Troubleshooting)
| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| Tiến trình **treo 0% CPU** hoặc crash `mutex lock failed` | Xung đột OpenMP. **Bắt buộc** đặt `KMP_DUPLICATE_LIB_OK=TRUE`. Runtime không dùng TensorFlow (đã chuyển sang PyTorch) để tránh xung đột với PaddleOCR. |
| Detector trả về rỗng (không phát hiện gì) | Thiếu `onnxruntime` → chạy lại `pip install -r main/requirements.txt`. |
| Lần đầu chạy OCR hơi lâu / cần mạng | PaddleOCR tự tải model (~vài chục MB) lần đầu. Engine cấu hình ở `main/configs/config.yaml` (`ocr.engine: ppocr`, fallback `easyocr`). |
| Cài `paddlepaddle` lỗi | Đảm bảo Python 3.12; trên Apple Silicon dùng bản CPU mặc định. Nếu không cài được, đặt `ocr.engine: easyocr` để dùng fallback. |
| Webcam không hoạt động (macOS) | Cấp quyền **System Settings → Privacy & Security → Camera** cho terminal/trình duyệt. |
| Sai góc camera (không bắt được xe đỗ) | Hiệu chỉnh ROI/ngưỡng không cần sửa code: `python main/scripts/calibrate_roi.py --source <clip> ` rồi chỉnh `pipeline.trigger` trong `config.yaml`. |

---

## 🏃 Kiểm thử hệ thống (Running Tests inside Container)

Để thực thi bộ unit test tự động (gồm 15 bài test kiểm tra OCR, so khớp CSDL, logic tiền xử lý) bên trong môi trường Docker đang chạy:

```bash
docker compose exec backend pytest main/tests/
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
*   **EasyOCR**: Tắt tính năng tự động tải hoặc kiểm tra mô hình qua mạng bằng cách khởi tạo: `easyocr.Reader(..., download_enabled=False)`.
*   **YOLOv8**: Tắt tính năng đồng bộ telemetry của Ultralytics qua: `settings.update({"sync": False})`. Sao chép thủ công tệp font hệ thống `Arial.ttf` vào thư mục mặc định `~/.config/Ultralytics/` để chặn việc YOLOv8 tự động tải font mỗi lần suy luận.
*   *Lưu ý: Dockerfile đã tự động tải trước các model weights này trong quá trình build để container hoạt động hoàn toàn ngoại tuyến.*
