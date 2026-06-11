# Kế hoạch Thiết lập Dữ liệu Thực tế (Real Data Setup Plan)

## Overview
Dự án "Hệ thống giám sát an ninh bãi đỗ xe thông minh" hiện tại đang sử dụng các hình ảnh giả lập (mock solid-color images) để chạy thử nghiệm và kiểm thử luồng. Để đáp ứng yêu cầu nghiệm thu thực tế, kế hoạch này vạch ra các bước cần thiết để tìm kiếm, tải xuống các bộ dữ liệu ảnh thực tế nhỏ gọn về biển số xe Việt Nam, hãng xe và màu sắc xe từ các nguồn công cộng (GitHub, Kaggle, Roboflow). Sau đó, tiến hành sắp xếp dữ liệu vào thư mục `main/data/raw/`, chạy huấn luyện lại các mô hình phân loại hãng xe/màu sắc xe, kiểm tra đánh giá hệ thống tích hợp, kiểm thử giao diện/API và cập nhật lại toàn bộ tài liệu báo cáo (Reports 2, 3, 4) cũng như README chính thức của dự án.

## Project Type
WEB & BACKEND (Streamlit Frontend Dashboard & FastAPI Backend Server with Deep Learning Pipeline)

## Success Criteria
1. **Nguồn dữ liệu thực tế**: Xác định và kiểm chứng được các URL tải dữ liệu sạch, dung lượng nhẹ cho biển số xe, hãng xe, và màu sắc xe.
2. **Thu thập dữ liệu mẫu**: Tải thành công tập mẫu thực tế gồm 10-20 ảnh cho mỗi nhãn hiệu xe (8 hãng) và màu sắc xe (8 màu), kèm một số ảnh biển số có nhãn định dạng YOLO.
3. **Cấu trúc dữ liệu chuẩn hóa**: Tổ chức dữ liệu đúng cấu trúc thư mục `main/data/raw/` và vượt qua bài unit test tải dữ liệu của hệ thống (`test_dataset.py`).
4. **Huấn luyện mô hình**: Chạy thành công script `main/train.py` để tạo ra các file trọng số `.keras` mới cho cả `brand` và `color` từ tập dữ liệu thực tế.
5. **Đánh giá hệ thống**: Chạy pipeline đánh giá bằng `main/src/engine/evaluator.py`, xuất ra báo cáo số liệu độ chính xác và độ trễ thực tế của hệ thống đối chiếu chéo.
6. **Xác thực API & Dashboard**: Đảm bảo cổng API Uvicorn (`/verify`) và Streamlit UI hoạt động trơn tru với các mô hình mới được huấn luyện.
7. **Cập nhật báo cáo & tài liệu**: Hoàn thiện slide thuyết trình Report 2 (Dữ liệu), Report 3 (Mô hình), Report 4 (Tích hợp) và `main/README.md` với các số liệu thật, biểu đồ loss/accuracy và nhật ký huấn luyện thật.

## Tech Stack
- **Python Environment**: `/opt/homebrew/Caskroom/miniforge/base/bin/python`
- **Mô hình học sâu**:
  - Phân loại hãng xe (Brand): TensorFlow/Keras Transfer Learning với EfficientNet-B0
  - Phân loại màu xe (Color): TensorFlow/Keras Transfer Learning với MobileNetV3-Small
  - Phát hiện biển số (Plate Detection): PyTorch & Ultralytics YOLOv8-Nano (`yolov8n.pt`)
  - Nhận diện ký tự (OCR): EasyOCR (kết hợp sắp xếp không gian 2 dòng biển số Việt Nam)
- **Web App / Service**: Streamlit (Dashboard UI), FastAPI & Uvicorn (Backend API)
- **Testing Framework**: pytest (kiểm thử OCR, matching logic, dataset loader)
- **Slide Decks**: HTML5 / CSS3 (Editorial Serif & Sans-Serif, Sharp corners, no-purple theme)

## File Structure
Các file và thư mục chịu tác động chính trong quá trình thực hiện kế hoạch này:
```
.
├── real-data-setup.md                  # [NEW] File kế hoạch này
├── main/
│   ├── README.md                      # [MODIFY] Cập nhật kết quả huấn luyện và metrics thực tế (tiếng Việt)
│   ├── train.py                       # [RUN] Thực thi huấn luyện mô hình phân loại hãng và màu xe
│   ├── run_ui.sh                      # [NEW] Script một click chạy Streamlit UI (macOS/Linux)
│   ├── run_ui.bat                     # [NEW] Script một click chạy Streamlit UI (Windows)
│   ├── configs/
│   │   └── config.yaml                # [MODIFY] Điều chỉnh các siêu tham số huấn luyện (epochs, lr) nếu cần
│   ├── data/
│   │   ├── raw/
│   │   │   ├── car_brands/            # [REPLACE] Chứa các ảnh xe thật được tổ chức theo thư mục con các hãng
│   │   │   ├── car_colors/            # [REPLACE] Chứa các ảnh xe thật được tổ chức theo thư mục con các màu
│   │   │   └── license_plates/        # [REPLACE] Chứa ảnh biển số xe thật kèm file tọa độ .txt (YOLO format)
│   │   ├── models/                    # [OUTPUT] Lưu checkpoint mô hình mới (*.keras và *.pt)
│   │   └── database.csv               # [MODIFY] Điều chỉnh thông tin đăng ký mẫu khớp với biển số/hãng/màu thật để test
│   └── src/
│       ├── engine/
│       │   └── evaluator.py           # [RUN] Chạy đánh giá batch trên thư mục ảnh test thực tế
│       ├── ui/
│       │   └── dashboard.py           # [VERIFY] Giao diện Streamlit đối chiếu xe thời gian thực
│       └── utils/
│           ├── download_real_data.py  # [NEW] Script tự động tải và giải nén các tập dữ liệu nhỏ gọn
│           └── matching.py            # [VERIFY] Logic đối chiếu 3 nhân tố với database
└── presentations/
    ├── Report_2_Presentation.html     # [MODIFY] Thay thế thông tin dữ liệu giả lập bằng EDA dữ liệu thật
    ├── Report_2_Details.md            # [NEW] Báo cáo chi tiết dạng tài liệu cho Report 2 (Dữ liệu)
    ├── Report_3_Presentation.html     # [MODIFY] Cập nhật biểu đồ loss/accuracy và metrics huấn luyện thật
    ├── Report_3_Details.md            # [NEW] Báo cáo chi tiết dạng tài liệu cho Report 3 (Mô hình)
    ├── Report_4_Presentation.html     # [MODIFY] Cập nhật kết quả đo lường độ trễ và tỷ lệ khớp hệ thống thật
    └── Report_4_Details.md            # [NEW] Báo cáo chi tiết dạng tài liệu cho Report 4 (Final Defense)
```

---

## Task Breakdown

### Phase 1: Research & Setup (Nghiên cứu & Thiết lập)
#### Task 1.1: Tìm kiếm nguồn dữ liệu thực tế mở cho biển số, hãng xe và màu xe
*   **Agent**: `project-planner`
*   **Skills**: `brainstorming`, `plan-writing`
*   **Priority**: P0
*   **Dependencies**: None
*   **Description**: Tìm kiếm và lựa chọn các liên kết tải xuống dữ liệu (ZIP/Direct URL) cho:
    1.  Biển số xe Việt Nam (định dạng YOLO detection) từ Kaggle/GitHub.
    2.  Hãng xe (8 nhãn: Toyota, Hyundai, Kia, Mazda, Honda, VinFast, Ford, Mitsubishi) từ Stanford Cars hoặc Google Images scraped datasets.
    3.  Màu sắc xe (8 nhãn: White, Black, Grey, Silver, Red, Blue, Brown, Yellow) từ Kaggle Vehicle Color dataset.
*   **INPUT**: Các trang tìm kiếm dataset (Kaggle, Roboflow, GitHub).
*   **OUTPUT**: Danh sách URL dữ liệu thực tế được ghi nhận rõ ràng trong tài liệu.
*   **VERIFY**: Các đường dẫn URL hoạt động bình thường, không yêu cầu xác thực đăng nhập phức tạp, có thể tải bằng lệnh `curl`, `wget` hoặc thư viện `requests` của Python.

---

### Phase 2: Data Acquisition & Preprocessing (Thu thập & Tiền xử lý)
#### Task 2.1: Viết script tự động tải dữ liệu mẫu thực tế
*   **Agent**: `backend-specialist`
*   **Skills**: `python-patterns`, `clean-code`
*   **Priority**: P0
*   **Dependencies**: Task 1.1
*   **Description**: Xây dựng một script Python gọn nhẹ `main/src/utils/download_real_data.py` để tự động tải các tệp ZIP của tập dữ liệu mẫu và giải nén chúng vào thư mục tạm `main/data/temp/`.
*   **INPUT**: Các URL đã chọn từ Task 1.1.
*   **OUTPUT**: Script `main/src/utils/download_real_data.py`.
*   **VERIFY**: Chạy script bằng môi trường Python `/opt/homebrew/Caskroom/miniforge/base/bin/python` và kiểm tra xem dữ liệu được tải xuống và giải nén thành công mà không gây lỗi phân đoạn bộ nhớ.

#### Task 2.2: Phân chia và tổ chức lại cấu trúc dữ liệu thô
*   **Agent**: `backend-specialist`
*   **Skills**: `clean-code`
*   **Priority**: P1
*   **Dependencies**: Task 2.1
*   **Description**: Chuyển đổi và tổ chức lại các thư mục ảnh thật được tải xuống vào đúng vị trí:
    -   `main/data/raw/car_brands/<Brand>/*.jpg` (10-20 ảnh/thương hiệu, tổng cộng 8 thương hiệu).
    -   `main/data/raw/car_colors/<Color>/*.jpg` (10-20 ảnh/màu sắc, tổng cộng 8 màu sắc).
    -   `main/data/raw/license_plates/*.jpg` và file annotations `.txt` tương ứng (3-5 ảnh mẫu có biển số thật).
*   **INPUT**: Dữ liệu giải nén tại `main/data/temp/`.
*   **OUTPUT**: Cấu trúc thư mục dữ liệu sạch sẽ trong `main/data/raw/`. Thư mục tạm `main/data/temp/` được dọn dẹp sạch.
*   **VERIFY**: Chạy lệnh kiểm thử đơn vị `pytest main/tests/test_dataset.py` bằng Python môi trường base để xác nhận Keras Dataset Loader đọc được dữ liệu thật với số lượng nhãn chính xác.

---

### Phase 3: Model Training & Evaluation (Huấn luyện & Đánh giá)
#### Task 3.1: Thực thi huấn luyện lại bộ phân loại hãng và màu xe
*   **Agent**: `backend-specialist`
*   **Skills**: `python-patterns`
*   **Priority**: P1
*   **Dependencies**: Task 2.2
*   **Description**: Kích hoạt huấn luyện hai bộ phân loại trên dữ liệu thật bằng cách tinh chỉnh tham số epoch trong `main/configs/config.yaml` hoặc qua dòng lệnh CLI:
    ```bash
    export PYTHONPATH=main
    # Huấn luyện hãng xe (EfficientNet-B0)
    /opt/homebrew/Caskroom/miniforge/base/bin/python main/train.py brand --data_dir main/data/raw/car_brands --epochs 10
    # Huấn luyện màu xe (MobileNetV3-Small)
    /opt/homebrew/Caskroom/miniforge/base/bin/python main/train.py color --data_dir main/data/raw/car_colors --epochs 10
    ```
*   **INPUT**: Dữ liệu ảnh thực tế tại `main/data/raw/` và cấu hình hệ thống.
*   **OUTPUT**: File trọng số mới `main/data/models/brand_classifier.keras` và `main/data/models/color_classifier.keras` được tạo ra.
*   **VERIFY**: Kiểm tra sự tồn tại của hai tệp mô hình và log huấn luyện xuất ra có hiển thị độ chính xác (val_accuracy) tăng dần qua các epoch.

#### Task 3.2: Thực thi đánh giá tích hợp hệ thống đầu cuối (End-to-End)
*   **Agent**: `backend-specialist`
*   **Skills**: `verify-changes`
*   **Priority**: P1
*   **Dependencies**: Task 3.1
*   **Description**: Cập nhật cơ sở dữ liệu mẫu `main/data/database.csv` với các biển số xe thực tế có trong tập dữ liệu thử nghiệm để đảm bảo khớp thông tin. Tiến hành chạy pipeline đánh giá hệ thống để đo lường độ chính xác tổng thể và thời gian xử lý (latency) của từng thành phần (YOLOv8 -> OCR -> Classifiers -> Matcher).
*   **INPUT**: Ảnh kiểm thử thực tế và các mô hình vừa được cập nhật.
*   **OUTPUT**: Log đánh giá hiển thị chi tiết thời gian đáp ứng (ms) và kết quả phân loại/so khớp.
*   **VERIFY**: Đảm bảo các mô hình hoạt động ổn định trên CPU/GPU của macOS mà không gặp lỗi rò rỉ bộ nhớ hoặc lỗi OpenMP.

---

### Phase 4: Integration Verification & Demo (Xác thực tích hợp & Demo)
#### Task 4.1: Kiểm thử Backend API và Streamlit Dashboard
*   **Agent**: `test-engineer`
*   **Skills**: `verify-changes`, `webapp-testing`
*   **Priority**: P2
*   **Dependencies**: Task 3.2
*   **Description**: Chạy khởi động đồng thời cả API Server và Streamlit UI bằng lệnh `./run.sh all` (hoặc chạy các terminal riêng biệt). Thực hiện tải ảnh xe thật lên giao diện dashboard để kiểm thử xem:
    1.  Biển số xe thật được phát hiện và OCR nhận dạng đúng ký tự.
    2.  Hãng xe và màu sắc xe được phân loại chính xác trên ảnh thật.
    3.  Hệ thống đối chiếu đúng trạng thái đăng ký (AUTHORIZED / MISMATCH / UNREGISTERED).
    4.  Còi báo động hoạt động khi xảy ra trạng thái cảnh báo an ninh.
*   **INPUT**: Giao diện UI Dashboard và API chạy tại localhost.
*   **OUTPUT**: Hệ thống hoạt động hoàn chỉnh, ghi lại video/screenshot demo.
*   **VERIFY**: API `/verify` trả về mã JSON chuẩn 200 OK với dữ liệu phân tích đúng. Giao diện Streamlit cập nhật biểu đồ và trạng thái khớp xe tức thời.

#### Task 4.2: Tạo script tự động khởi chạy Dashboard trên đa nền tảng (Multi-Platform UI Auto-Run)
*   **Agent**: `devops-engineer`
*   **Skills**: `bash-linux`, `powershell-windows`
*   **Priority**: P1
*   **Dependencies**: None
*   **Description**: Tạo file chạy tự động một click cho Streamlit UI trên cả macOS/Linux (`main/run_ui.sh`) và Windows (`main/run_ui.bat`) nhằm đơn giản hóa việc triển khai kiểm thử cho người dùng mới mà không cần nhớ các cấu hình PYTHONPATH hay KMP_DUPLICATE_LIB_OK.
*   **INPUT**: Lệnh chạy Streamlit và cấu hình môi trường.
*   **OUTPUT**: `main/run_ui.sh` và `main/run_ui.bat`.
*   **VERIFY**: Thực thi thành công trên môi trường cục bộ, kiểm chứng các biến môi trường được inject đúng.

#### Task 4.3: Dọn dẹp và tối giản hóa mã nguồn (Code Cleanliness & Simplification)
*   **Agent**: `code-archaeologist`
*   **Skills**: `simplify-code`, `clean-code`
*   **Priority**: P2
*   **Dependencies**: Task 4.1
*   **Description**: Rà soát lại toàn bộ mã nguồn hệ thống, loại bỏ các file mock dư thừa, dọn sạch tài nguyên tạm, loại bỏ các khối code trùng lặp hoặc không sử dụng, và tối giản hóa cấu trúc import để nâng cao tính ổn định và bảo trì lâu dài.
*   **INPUT**: Mã nguồn hiện tại của dự án.
*   **OUTPUT**: Mã nguồn được tối giản hóa và sạch sẽ.
*   **VERIFY**: Chạy script linter và pytest để kiểm thử toàn diện sau khi tái cấu trúc.

---

### Phase 5: Documentation & Presentation (Tài liệu & Thuyết trình)
#### Task 5.1: Cập nhật tài liệu README.md và Slide thuyết trình
*   **Agent**: `documentation-writer` / `frontend-specialist`
*   **Skills**: `frontend-design`, `i18n-localization`
*   **Priority**: P2
*   **Dependencies**: Task 4.1
*   **Description**: Cập nhật lại các báo cáo để phản ánh kết quả thực tế thay vì dữ liệu mock:
    1.  **README.md**: Thêm bảng so sánh thông số hiệu năng thật, số lượng mẫu thật và hướng dẫn chạy script tải dữ liệu mới.
    2.  **Report_2_Presentation.html**: Thay thế bảng dữ liệu mock bằng thông tin dữ liệu thật đã thu thập (số lượng, phân bố lớp hãng/màu, EDA phân tích biển số xe thật).
    3.  **Report_3_Presentation.html**: Cập nhật log huấn luyện thật, đồ thị loss/accuracy của EfficientNet-B0 và MobileNetV3-Small.
    4.  **Report_4_Presentation.html**: Cập nhật các bảng số liệu latency của hệ thống tích hợp chạy trên máy Mac M-series, kết hợp chèn ảnh chụp màn hình UI thật hoạt động với dữ liệu thực tế.
*   **INPUT**: Nhật ký huấn luyện và hình ảnh chạy ứng dụng thực tế.
*   **OUTPUT**: Slide decks HTML và file README.md được cập nhật đầy đủ thông tin chuẩn xác.
*   **VERIFY**: Mở các trang HTML trên trình duyệt Safari/Chrome, đảm bảo tuân thủ thiết kế Light Editorial, các góc nhọn sắc nét (`border-radius: 0px`), và quy tắc Purple Ban (không chứa màu tím).

#### Task 5.2: Tạo tài liệu báo cáo chi tiết đi kèm từng slide thuyết trình
*   **Agent**: `documentation-writer`
*   **Skills**: `documentation-templates`
*   **Priority**: P2
*   **Dependencies**: Task 5.1
*   **Description**: Tạo các file tài liệu chi tiết định dạng Markdown (`Report_2_Details.md`, `Report_3_Details.md`, `Report_4_Details.md`) đặt ngay cạnh các file HTML slide thuyết trình trong thư mục `presentations/` để diễn giải chuyên sâu bằng tiếng Việt về dữ liệu, mô hình và kết quả tích hợp hệ thống.
*   **INPUT**: Số liệu thực nghiệm dữ liệu và mô hình thật.
*   **OUTPUT**: Các file `presentations/Report_2_Details.md`, `presentations/Report_3_Details.md`, và `presentations/Report_4_Details.md`.
*   **VERIFY**: Xác nhận các file được tạo thành công, nội dung có tính cấu trúc rõ ràng, đầy đủ các phân tích kỹ thuật.

---

## Phase X: Final Verification

> [!IMPORTANT]
> Toàn bộ quá trình kiểm nghiệm tự động phải được thực hiện bằng Python có sẵn trong môi trường base của máy Mac: `/opt/homebrew/Caskroom/miniforge/base/bin/python`.

### 1. Run All Verifications
Chạy script checklist tổng để kiểm tra toàn bộ chất lượng mã nguồn và slide HTML:
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python .agents/scripts/checklist.py .
```

### 2. Individual Automated Tests
Chạy kiểm thử unit test cho luồng dữ liệu, OCR và so khớp:
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest main/tests/
```

### 3. Rule Compliance check
- [ ] Xác nhận không có mã màu tím/violet (`#800080`, `#4b0082`, etc.) được đưa vào các slide HTML.
- [ ] Xác nhận tất cả các slide, card và bảng biểu có thuộc tính `border-radius: 0px`.
- [ ] Xác nhận script tải dữ liệu thật chạy độc lập tốt và không phụ thuộc vào thư viện bên ngoài chưa khai báo trong `main/requirements.txt`.

### 4. Phase X Completion Marker
*Sau khi hoàn thành tất cả các bài kiểm tra, hãy thêm phần này vào cuối file:*

## ✅ PHASE X COMPLETE
- Lint & Code Quality: ✅ Pass
- Dataset Tests: ✅ Pass
- UI & Presentation Auditing: ✅ Pass
- Date: 2026-06-11

