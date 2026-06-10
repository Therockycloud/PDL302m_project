# Smart Parking Security System via Cross-Verification

Hệ thống giám sát an ninh bãi đỗ xe thông minh ứng dụng Học Sâu (Deep Learning) để phòng chống tráo đổi biển số xe tinh vi bằng phương pháp đối chiếu chéo 3 nhân tố: **Biển số xe (OCR)**, **Hãng xe (Brand)** và **Màu sắc xe (Color)**.

---

## 🏗️ Kiến trúc hệ thống (System Architecture)

Hệ thống được phát triển tách biệt giữa Backend (FastAPI) và Frontend (Streamlit Dashboard) để tối ưu hiệu năng chạy CPU thời gian thực:

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

## 🛠️ Hướng dẫn cài đặt đa nền tảng (Multi-Platform Setup)

Dự án yêu cầu cài đặt trình quản lý môi trường **Miniconda** hoặc **Anaconda** để quản lý các thư viện song song một cách an toàn.

### 🍏 1. Hướng dẫn cho macOS

1. **Khởi tạo môi trường Conda:**
   ```bash
   conda create -n dpl302m python=3.10 -y
   conda activate dpl302m
   ```

2. **Cài đặt dependencies:**
   ```bash
   pip install -r main/requirements.txt
   ```

3. **Cấu hình biến môi trường (Bắt buộc để tránh lỗi OpenMP Crash):**
   Thêm dòng sau vào file `~/.zshrc` hoặc chạy trực tiếp trên Terminal trước khi mở ứng dụng:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   export PYTHONPATH=main
   ```

4. **Sử dụng đúng Python path:**
   Nếu Python mặc định hệ thống bị lỗi hoặc xung đột thư viện, chạy các script trực tiếp bằng đường dẫn Conda:
   ```bash
   /opt/homebrew/Caskroom/miniforge/base/envs/dpl302m/bin/python -m uvicorn main.src.api.app:app
   ```

---

### 🐧 2. Hướng dẫn cho Linux (Ubuntu/Debian)

1. **Cài đặt thư viện hệ thống cần thiết (cho OpenCV & EasyOCR):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```

2. **Khởi tạo và kích hoạt Conda:**
   ```bash
   conda create -n dpl302m python=3.10 -y
   conda activate dpl302m
   ```

3. **Cài đặt thư viện:**
   ```bash
   pip install -r main/requirements.txt
   ```

4. **Thiết lập PYTHONPATH:**
   ```bash
   export PYTHONPATH=main
   ```

---

### 🪟 3. Hướng dẫn cho Windows

1. **Khởi chạy Anaconda Prompt** (với quyền Administrator).

2. **Khởi tạo môi trường Conda:**
   ```cmd
   conda create -n dpl302m python=3.10 -y
   conda activate dpl302m
   ```

3. **Cài đặt thư viện:**
   ```cmd
   pip install -r main/requirements.txt
   ```

4. **Thiết lập biến môi trường trên Windows CMD:**
   ```cmd
   set KMP_DUPLICATE_LIB_OK=TRUE
   set PYTHONPATH=main
   ```
   *(Nếu dùng PowerShell, chạy lệnh: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`; `$env:PYTHONPATH="main"`)*

---

## 🏃 Hướng dẫn chạy và kiểm thử (Execution & Testing)

Mọi lệnh chạy dưới đây đều thực hiện từ **thư mục gốc của dự án** (`PDL302m_project`).

### 1. Tạo dữ liệu giả lập (Mock Data Generator)
Trước khi chạy mô hình lần đầu tiên, hãy tạo dữ liệu giả lập để phục vụ kiểm thử:
* **macOS/Linux:**
  ```bash
  export PYTHONPATH=main
  python main/src/utils/mock_generator.py
  ```
* **Windows CMD:**
  ```cmd
  set PYTHONPATH=main
  python main/src/utils/mock_generator.py
  ```

### 2. Chạy Unit Tests
Hệ thống tích hợp 15 bài kiểm thử tự động đo lường độ chính xác OCR, xử lý chuỗi và tải tệp dữ liệu:
```bash
pytest main/tests/
```

### 3. Khởi động Backend API (FastAPI)
Chạy server API REST tiếp nhận ảnh quét:
```bash
./run.sh api
```

Hoặc chạy trực tiếp:
```bash
uvicorn main.src.api.app:app --reload --port 8000
```
Tài liệu hướng dẫn API trực quan có sẵn tại: `http://localhost:8000/docs`.

### 4. Khởi động Giao diện Giám sát (Streamlit)
Mở một Terminal mới và chạy một trong hai lệnh sau để mở Dashboard:
```bash
./run.sh ui
```

Hoặc, nếu bạn đã kích hoạt đúng môi trường Python:
```bash
streamlit run main/src/ui/dashboard.py
```
Giao diện sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## ⚠️ Khắc phục lỗi thường gặp (Troubleshooting)

### 🔴 Lỗi 1: `OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.`
* **Nguyên nhân:** Xảy ra trên macOS khi cả TensorFlow và PyTorch (Torchvision) cùng khởi tạo thư viện xử lý song song OpenMP.
* **Khắc phục:** Thêm biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` trước lệnh khởi chạy uvicorn hoặc streamlit. Hoặc đặt đầu file script python:
  ```python
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  ```

### 🔴 Lỗi 2: `ModuleNotFoundError: No module named 'src'`
* **Nguyên nhân:** Python không tìm thấy thư mục gốc chứa các gói module.
* **Khắc phục:** Đảm bảo bạn đã xuất biến môi trường `PYTHONPATH=main` và đang chạy lệnh từ thư mục gốc của dự án.

---

## 👥 Hướng dẫn làm việc với Git (Git Branching & Dev Workflow)

Đội ngũ phát triển tuân thủ quy trình Git chặt chẽ để đảm bảo tính ổn định của mã nguồn.

### 1. Quy tắc đặt tên Branch
Mọi tính năng hoặc sửa lỗi đều phải tạo nhánh con từ nhánh `main`:
* Nhánh tính năng mới: `feature/[tên-tính-năng-slug]` (Ví dụ: `feature/resnet-upgrade`)
* Nhánh sửa lỗi: `fix/[tên-lỗi-slug]` (Ví dụ: `fix/openmp-crash`)

### 2. Nhánh chạy thử nghiệm độc lập (`test/streamlit-only`)
Để phục vụ việc chạy thử nghiệm nhanh và demo trực tiếp mà không cần tải toàn bộ mã nguồn huấn luyện nặng nề:
* Dự án cấu hình nhánh đặc biệt **`test/streamlit-only`**.
* Nhánh này sử dụng một file `.gitignore` tùy chỉnh để bỏ qua (ignore) các thư mục dữ liệu thô, tài liệu huấn luyện nặng (`main/src/engine/`, `main/src/datasets/`).
* Nhánh chỉ lưu giữ: mô hình đã đóng gói (`main/data/models/*.keras`), file cấu hình (`main/configs/config.yaml`), mock data generator (`main/src/utils/mock_generator.py`) và ứng dụng Streamlit UI/uvicorn API.
* **Quy trình đẩy bản dựng (Build & Test Push):**
  ```bash
  # 1. Chuyển sang nhánh kiểm thử
  git checkout -b test/streamlit-only
  
  # 2. Đảm bảo file weights mô hình đã nằm trong main/data/models/
  # 3. Commit và push nhánh kiểm thử lên máy chủ
  git add main/data/models/ main/src/ui/ dashboard.py
  git commit -m "deploy: update lightweight streamlit test build"
  git push origin test/streamlit-only
  ```
