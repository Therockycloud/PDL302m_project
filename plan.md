# Kế hoạch chi tiết và Tiến trình thực hiện dự án DPL302m

Tài liệu này trình bày lộ trình thực hiện dự án **Hệ thống chống trộm xe thông minh bằng đối chiếu chéo thông tin xe (Ô tô)**. Các công việc được sắp xếp nghiêm ngặt theo **tiến trình thời gian biểu (Chronological Order)** của môn học DPL302m tại FPT University, đi từ khâu đề xuất ý tưởng đến thu thập dữ liệu, huấn luyện mô hình, kiểm thử và bảo vệ dự án.

---

## GIAI ĐOẠN 1: ĐỀ XUẤT DỰ ÁN & THIẾT KẾ KIẾN TRÚC (REPORT 1 & PRESENTATION 1)
*Mục tiêu: Hoàn thành Proposal và định hình giải pháp (Thời gian: Session 8 - 18).*

* [x] **Task 1.1: Định nghĩa bài toán & Ý nghĩa thực tế**
  * Phân tích thực trạng trộm cắp ô tô và thủ đoạn hoán đổi biển số xe (biển số giả) tại các bãi gửi xe thông minh ở Việt Nam.
  * Thống nhất giải pháp đối chiếu chéo 3 nhân tố thời gian thực: Biển số xe (OCR) $\leftrightarrow$ Nhãn hiệu xe (Brand) $\leftrightarrow$ Màu sắc xe (Color).
* [x] **Task 1.2: Thiết kế sơ đồ kiến trúc hệ thống (System Architecture Diagram)**
  * Vẽ sơ đồ luồng dữ liệu chi tiết:
    `Video Stream / Webcam` $\rightarrow$ `YOLOv8 Plate Detector` $\rightarrow$ `Cropped Bounding Box` $\rightarrow$ Song song: `OCR Engine` + `Brand Classifier` + `Color Classifier` $\rightarrow$ `Logic Matcher` $\rightarrow$ Tra cứu file `database.csv` $\rightarrow$ `Giao diện điều khiển / Còi báo động`.
* [x] **Task 1.3: Lựa chọn Công nghệ & Framework**
  * Thống nhất sử dụng PyTorch & Ultralytics YOLOv8 cho phần Object Detection.
  * Thống nhất sử dụng PaddleOCR hoặc EasyOCR cho nhận diện ký tự biển số.
  * Thống nhất sử dụng TensorFlow/Keras cho phần Classifier (đảm bảo yêu cầu kỹ thuật của Syllabus).
  * Lựa chọn Streamlit để phát triển Web Dashboard Demo thời gian thực.
* [x] **Task 1.4: Xác định các chỉ số đo lường hiệu năng (Key Metrics)**
  * **Độ trễ hệ thống (Latency):** $< 1.0$ giây/xe.
  * **Độ chính xác mô hình đơn lẻ:** YOLOv8 (mAP $\ge 90\%$), OCR (Word accuracy $\ge 90\%$), Brand Classifier (Accuracy $\ge 85\%$), Color Classifier (Accuracy $\ge 92\%$).
  * **Độ chính xác toàn hệ thống:** Tỷ lệ phát hiện biển giả/sai thông tin $\ge 95\%$.
* [x] **Task 1.5: Soạn thảo văn bản và Slide thuyết trình Report 1**
  * Viết tài liệu `docs/Report_1_Proposal.md` theo mẫu đề cương.
  * Thiết kế slide thuyết trình Report 1 (15 slides) với nội dung chữ đầy đủ dễ đọc và trực quan.

---

## GIAI ĐOẠN 2: THIẾT LẬP MÔI TRƯỜNG, THU THẬP & PHÂN TÍCH DỮ LIỆU (REPORT 2 & PRESENTATION 2)
*Mục tiêu: Cài đặt thư viện, chuẩn bị dữ liệu sạch và phân tích đặc trưng dữ liệu (Thời gian: Session 28 - 33).*

* [ ] **Task 2.1: Cấu hình môi trường lập trình nhóm**
  * Cài đặt các thư viện từ file [requirements.txt](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/requirements.txt) bằng Python trong môi trường Conda.
  * Kiểm tra và kiểm thử khả năng tăng tốc phần cứng (CUDA trên NVIDIA GPU hoặc Metal trên Apple Silicon).
* [ ] **Task 2.2: Tải các bộ dữ liệu thô (Raw Datasets) từ Kaggle**
  * Tải dữ liệu biển số xe ô tô Việt Nam về `main/data/raw/license_plates/`.
  * Tải dữ liệu nhãn hiệu xe (Stanford Cars) về `main/data/raw/car_brands/`.
  * Tải dữ liệu màu sắc xe về `main/data/raw/car_colors/`.
* [ ] **Task 2.3: Viết Code Wrangling xử lý dữ liệu**
  * Viết script python đọc file annotation (XML/TXT) để tự động cắt (crop) khu vực biển số xe và khu vực toàn bộ thân xe ô tô.
  * Lưu các ảnh cắt được vào đúng cấu trúc thư mục `main/data/processed/` để huấn luyện các bộ phân loại.
* [ ] **Task 2.4: Áp dụng Tăng cường dữ liệu (Data Augmentation)**
  * Cấu hình các kỹ thuật tăng cường độ sáng (giả lập ánh sáng đèn pha ban đêm, sương mù, nắng gắt), xoay góc ảnh, thay đổi tỷ lệ kích thước.
* [ ] **Task 2.5: Tiến hành phân tích khám phá dữ liệu (EDA)**
  * Tạo notebook `01_eda_and_data_prep.ipynb` để trực quan hóa dữ liệu.
  * Vẽ biểu đồ phân bố độ phân giải ảnh, tỷ lệ mất cân bằng giữa các lớp nhãn hiệu xe (Toyota, Hyundai, VinFast,...) và màu sắc xe.
* [ ] **Task 2.6: Soạn thảo văn bản và Slide thuyết trình Report 2**
  * Viết tài liệu `docs/Report_2_Data_Tasks.md`.
  * Thiết kế slide thuyết trình Report 2 (10-12 slides) thể hiện rõ quy trình xử lý dữ liệu.

---

## GIAI ĐOẠN 3: HUẤN LUYỆN, ĐỒNG BỘ MÔ HÌNH & THỬ NGHIỆM (REPORT 3 & PRESENTATION 3)
*Mục tiêu: Đào tạo các mạng neural độc lập và tối ưu hóa siêu tham số (Thời gian: Session 43 - 57).*

* [ ] **Task 3.1: Huấn luyện mô hình phát hiện biển số (YOLOv8)**
  * Cấu hình file `yolov8_config.yaml` và chạy huấn luyện YOLOv8-nano trên tập dữ liệu biển số xe ô tô Việt Nam.
  * Theo dõi các đường cong loss trên TensorBoard để tránh overfitting.
* [ ] **Task 3.2: Tích hợp và Tối ưu hóa bộ ký tự OCR**
  * Viết wrapper `ocr_engine.py` gọi EasyOCR hoặc PaddleOCR.
  * Viết thuật toán sắp xếp tọa độ ký tự (x, y) để đọc đúng định dạng biển số xe ô tô Việt Nam (ví dụ: biển vuông 2 dòng hoặc biển dài 1 dòng).
  * Thêm các luật lọc chuỗi thô (stripping spaces, dashes, dots).
* [ ] **Task 3.3: Huấn luyện bộ phân loại nhãn hiệu xe (ResNet50)**
  * Viết code Transfer Learning sử dụng mạng ResNet50 pre-trained.
  * Đóng băng các tầng convolution gốc, huấn luyện các tầng Dense mới trên dữ liệu nhãn hiệu xe.
* [ ] **Task 3.4: Huấn luyện bộ phân loại màu sắc xe (MobileNetV2)**
  * Huấn luyện mạng MobileNetV2 để nhận diện 8 màu sắc xe cơ bản.
* [ ] **Task 3.5: Tinh chỉnh siêu tham số (Hyperparameter Tuning)**
  * Sử dụng Keras Tuner để thử nghiệm các giá trị Learning Rate (ví dụ: $1e-3, 1e-4, 1e-5$) và Dropout rates (từ $0.3$ đến $0.6$).
  * Thiết lập các Keras callbacks: `EarlyStopping` và `ModelCheckpoint` để lưu lại bộ trọng số tốt nhất.
* [ ] **Task 3.6: Đánh giá hiệu năng của từng mô hình riêng lẻ**
  * Tính toán Precision, Recall, mAP cho YOLOv8.
  * Xuất Confusion Matrix và F1-score cho bộ phân loại nhãn hiệu & màu sắc.
* [ ] **Task 3.7: Soạn thảo văn bản và Slide thuyết trình Report 3**
  * Viết tài liệu `docs/Report_3_Model_Results.md`.
  * Thiết kế slide thuyết trình Report 3 (15-18 slides) chi tiết các tham số huấn luyện và biểu đồ kết quả.

---

## GIAI ĐOẠN 4: TÍCH HỢP HỆ THỐNG, THỬ NGHIỆM REAL-TIME & BẢO VỆ DỰ ÁN (REPORT 4 & PRESENTATION 4)
*Mục tiêu: Hoàn thiện ứng dụng Web, viết unit test, kiểm tra độ trễ và thuyết trình bảo vệ (Thời gian: Session 104 - 119).*

* [ ] **Task 4.1: Xây dựng Pipeline tích hợp toàn diện**
  * Viết `evaluator.py` để liên kết luồng dữ liệu: đầu ra của YOLOv8 làm đầu vào trực tiếp cho mô hình OCR và các Classifier.
* [ ] **Task 4.2: Thiết lập hệ thống cơ sở dữ liệu đối chiếu**
  * Tạo file cơ sở dữ liệu đăng ký xe mẫu [database.csv](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/data/database.csv).
  * Gọi hàm `verify_vehicle` từ [matching.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/src/utils/matching.py) để thực hiện logic so sánh thời gian thực.
* [ ] **Task 4.3: Viết và chạy các bài kiểm thử đơn vị (Unit Tests)**
  * Hoàn thiện và chạy kiểm thử file [test_matching.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/tests/test_matching.py) trong môi trường conda base để kiểm tra độ tin cậy của thuật toán so khớp.
* [ ] **Task 4.4: Phát triển Web Dashboard Demo bằng Streamlit**
  * Thiết kế giao diện Web cho phép mở trực tiếp Webcam hoặc tải file Video lên.
  * Vẽ khung bounding box kèm nhãn nhận diện (Plate Text, Brand, Color) lên từng khung hình video theo thời gian thực.
* [ ] **Task 4.5: Phát triển Hệ thống báo động (Alarm Warning)**
  * Nếu phát hiện xe giả mạo biển số (`MISMATCH` hoặc `UNREGISTERED`):
    * Giao diện nhấp nháy banner **ĐỎ CHÓI**.
    * Tự động phát âm thanh cảnh báo còi hú ra loa máy tính.
* [ ] **Task 4.6: Đo lường độ trễ & Tối ưu hóa FPS**
  * Ghi nhận thời gian xử lý của từng phần (YOLO, OCR, Classifiers, Matcher) để tính tổng thời gian đáp ứng (End-to-End Latency).
  * Tối ưu hóa luồng đọc frame (Threading) để đảm bảo FPS đạt mức mượt mà khi demo trực tiếp.
* [ ] **Task 4.7: Hoàn thiện hồ sơ nghiệm thu & Slide bảo vệ**
  * Viết tài liệu báo cáo cuối cùng `docs/Report_4_Final_Report.md`.
  * Quay video ghi hình ứng dụng demo chạy thực tế để dự phòng khi bảo vệ.
  * Thiết kế slide thuyết trình bảo vệ cuối cùng (12-15 slides) bám sát các tiêu chuẩn nghiệm thu môn học.
