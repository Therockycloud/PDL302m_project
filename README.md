# 🚗 Smart Parking Security System via Cross-Verification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống giám sát an ninh bãi đỗ xe thông minh ứng dụng Học Sâu (Deep Learning) để phòng chống các hành vi gian lận hoặc tráo đổi biển số xe tinh vi. Hệ thống thực hiện đối chiếu chéo đồng thời 3 nhân tố sinh trắc học trực quan của phương tiện: **Biển số xe (OCR)**, **Hãng xe (Brand)** và **Màu sắc xe (Color)**.

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
 └─────┬─────┘                                  │MobileNetV3│
       ▼                                        └─────┬─────┘
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

## ⚡ Tối ưu hóa suy luận trên CPU ngoại tuyến (Offline CPU Optimizations)

Để hệ thống hoạt động với hiệu năng ổn định nhất trên các PC bãi đỗ chạy CPU thông thường và không cần kết nối Internet, dự án tích hợp các giải pháp tối ưu sau:

### 1. Khắc phục nghẽn luồng / Treo CPU (Thread Deadlock Fix)
Khi TensorFlow và PyTorch chạy song song, việc tranh chấp luồng tính toán có thể làm đơ hệ thống. Chúng ta bắt buộc phải cấu hình giới hạn đơn luồng cho các thư viện xử lý trước khi chạy:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
*Giải pháp này đưa độ trễ suy luận ổn định về mức cực thấp chỉ **~1.6 giây / xe**.*

### 2. Cấu hình chạy ngoại tuyến hoàn toàn (100% Offline Mode)
*   **EasyOCR**: Tắt tính năng tự động tải hoặc kiểm tra mô hình qua mạng bằng cách khởi tạo: `easyocr.Reader(..., download_enabled=False)`.
*   **YOLOv8**: Tắt tính năng đồng bộ telemetry của Ultralytics qua: `settings.update({"sync": False})`. Sao chép thủ công tệp font hệ thống `Arial.ttf` vào thư mục mặc định `~/.config/Ultralytics/` để chặn việc YOLOv8 tự động tải font mỗi lần suy luận.

---

## 🛠️ Hướng dẫn cài đặt đa nền tảng (Installation Setup)

Khuyến khích sử dụng **Miniconda** (Python 3.12) để tránh xung đột thư viện hệ thống.

### 🍏 1. Hướng dẫn cho macOS
1.  Khởi tạo và kích hoạt môi trường:
    ```bash
    conda create -n dpl302m python=3.12 -y
    conda activate dpl302m
    ```
2.  Cài đặt dependencies:
    ```bash
    pip install -r main/requirements.txt
    ```
3.  Thiết lập môi trường chạy cục bộ:
    ```bash
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH=main
    ```

### 🐧 2. Hướng dẫn cho Linux (Ubuntu/Debian)
1.  Cài đặt thư viện đồ họa hệ thống cho OpenCV & EasyOCR:
    ```bash
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    ```
2.  Khởi tạo môi trường và cài đặt:
    ```bash
    conda create -n dpl302m python=3.12 -y
    conda activate dpl302m
    pip install -r main/requirements.txt
    export PYTHONPATH=main
    ```

### 🪟 3. Hướng dẫn cho Windows
1.  Mở Anaconda Prompt (Run as Administrator).
2.  Tạo và kích hoạt môi trường:
    ```cmd
    conda create -n dpl302m python=3.12 -y
    conda activate dpl302m
    pip install -r main/requirements.txt
    ```
3.  Cấu hình môi trường CMD:
    ```cmd
    set KMP_DUPLICATE_LIB_OK=TRUE
    set PYTHONPATH=main
    ```

---

## 🏃 Vận hành & Kiểm thử (Execution & Testing)

Tất cả lệnh dưới đây phải chạy từ **thư mục gốc** (`PDL302m_project`):

### 1. Khởi động Backend API (FastAPI)
```bash
python -m uvicorn main.src.api.app:app --reload --port 8000
```
*Tài liệu hướng dẫn endpoints trực quan có sẵn tại: `http://localhost:8000/docs`.*

### 2. Khởi động Giao diện Giám sát (Streamlit)
Mở một Terminal mới và chạy:
```bash
streamlit run main/src/ui/dashboard.py
```
*Giao diện Dashboard vận hành sẽ mở tại: `http://localhost:8501`.*

### 3. Chạy Unit Tests tự động
Hệ thống tích hợp 15 bài kiểm thử unit test chạy qua pytest:
```bash
pytest main/tests/
```
