# Báo cáo Chi tiết: Thiết lập và Tiền xử lý Dữ liệu Thực tế (Report 2 Details)

## 1. Đặt vấn đề và Mục tiêu (Problem & Objective)
Trong các hệ thống giám sát an ninh bãi giữ xe thông thường, việc nhận diện chỉ dựa trên biển số xe (OCR) dễ bị vượt qua bằng cách làm giả hoặc tráo đổi biển số giữa hai phương tiện khác nhau. Dự án này đề xuất một giải pháp đối chiếu chéo đa nhân tố, xác thực đồng thời 3 thông tin: **Biển số xe (License Plate)**, **Hãng xe (Brand)**, và **Màu sắc xe (Color)**.

Mục tiêu của giai đoạn 1 (Stage 1) là chuyển đổi hệ thống từ việc sử dụng ảnh giả lập (mock data) sang thu thập và tiền xử lý một bộ dữ liệu thực tế đầy đủ nhằm huấn luyện các mô hình phân loại sâu (Deep Learning Classifiers) đạt độ chính xác thực tế cao nhất tại thị trường Việt Nam.

---

## 2. Quy trình Thu thập Dữ liệu (Data Acquisition Pipeline)
Nhóm đã tiến hành tích hợp và chuẩn hóa dữ liệu từ các nguồn mở uy tín và các công cụ cào dữ liệu tự động:

### 2.1. Tập dữ liệu Hãng xe (Car Brands Dataset)
Mục tiêu là thu thập 8 thương hiệu ô tô phổ biến nhất tại thị trường Việt Nam bao gồm: *Toyota, Hyundai, Kia, Mazda, Honda, VinFast, Ford, Mitsubishi*.
*   **Stanford Cars Dataset (Hugging Face)**: Sử dụng phiên bản Parquet sạch của bộ dữ liệu Stanford Cars (`tanganke/stanford_cars`) để tải các ảnh thuộc 7 hãng xe quốc tế (Toyota, Hyundai, Kia, Mazda, Honda, Ford, Mitsubishi). Việc dùng Parquet giúp tăng tốc độ tải và tránh lỗi Rate Limit (429) thường gặp khi gọi API lập chỉ mục của Hugging Face.
*   **VinFast Scraping (Wikimedia Commons)**: Thương hiệu xe điện quốc gia VinFast không có sẵn trong các bộ dữ liệu quốc tế lớn. Nhóm đã phát triển script cào ảnh tự động từ Wikimedia Commons với việc cấu hình `User-Agent` tùy chỉnh và cơ chế nghỉ dừng (sleep interval 1.5s) để chống chặn (rate limiting 429). Nhóm đã thu thập thành công 120 ảnh chất lượng cao đại diện cho các mẫu xe VinFast Lux A2.0, Lux SA2.0, VF8, VF9.

### 2.2. Tập dữ liệu Màu sắc xe (Car Colors Dataset)
Nhóm sử dụng bộ dữ liệu phân loại màu sắc xe gồm 8 màu phổ biến: *White, Black, Grey, Silver, Red, Blue, Brown, Yellow*.
*   Dữ liệu được lọc kỹ lưỡng để loại bỏ các ảnh nhiễu (ảnh chụp cận cảnh nội thất, ảnh chụp trong điều kiện ánh sáng đổi màu quá mức).

### 2.3. Tập dữ liệu Biển số xe Việt Nam (License Plates)
Nhóm thu thập tập mẫu gồm 5 hình ảnh biển số xe thực tế tại Việt Nam (bao gồm cả dạng biển số dài và biển số vuông 2 dòng) kèm theo file nhãn định dạng YOLO tương ứng (`.txt`) để kiểm tra toàn bộ luồng tích hợp phát hiện biển số và OCR.

---

## 3. Thống kê Dữ liệu Thu thập (Dataset Statistics)

### Phân bố lớp Hãng xe (Car Brands Distribution):
| Thương hiệu xe | Số lượng ảnh thô | Trạng thái tiền xử lý |
| :--- | :---: | :---: |
| **Toyota** | 168 | Đã làm sạch |
| **Hyundai** | 200 | Đã làm sạch |
| **Kia** | 120 | Đã làm sạch |
| **Mazda** | 120 | Đã làm sạch |
| **Honda** | 161 | Đã làm sạch |
| **VinFast** | 120 | Đã cào từ Wikimedia |
| **Ford** | 200 | Đã làm sạch |
| **Mitsubishi** | 120 | Đã làm sạch |
| **Tổng cộng** | **1,209** | **Hoàn thành** |

### Phân bố lớp Màu sắc xe (Car Colors Distribution):
| Nhóm màu sắc | Số lượng ảnh thô | Trạng thái tiền xử lý |
| :--- | :---: | :---: |
| **White** (Trắng) | 185 | Đã lọc nhiễu |
| **Black** (Đen) | 200 | Đã lọc nhiễu |
| **Grey** (Xám) | 200 | Đã lọc nhiễu |
| **Silver** (Bạc) | 175 | Đã lọc nhiễu |
| **Red** (Đỏ) | 110 | Đã lọc nhiễu |
| **Blue** (Xanh dương) | 200 | Đã lọc nhiễu |
| **Brown** (Nâu) | 35 | Đã lọc nhiễu |
| **Yellow** (Vàng) | 25 | Đã lọc nhiễu |
| **Tổng cộng** | **1,130** | **Hoàn thành** |

---

## 4. Công tác Tiền xử lý Dữ liệu (Preprocessing & Pipeline Integration)
Trước khi đưa vào huấn luyện mô hình học sâu, toàn bộ ảnh xe được đưa qua pipeline tiền xử lý tự động:
1.  **Chuẩn hóa kích thước (Resizing)**: Resize toàn bộ ảnh về độ phân giải chuẩn $224 \times 224 \times 3$ để phù hợp làm đầu vào cho EfficientNet-B0 và MobileNetV3-Small.
2.  **Khắc phục lỗi chuẩn hóa tỷ lệ (Scaling Bug Fix)**: 
    *   Trong quá trình thử nghiệm, nhóm phát hiện ra dataset loader mặc định chuẩn hóa ảnh về đoạn $[0, 1]$. Tuy nhiên, backbone `EfficientNetB0` của Keras yêu cầu đầu vào có thang điểm $[0, 255]$ (do đã tích hợp sẵn lớp chuẩn hóa nội bộ), còn `MobileNetV3` yêu cầu thang điểm $[-1, 1]$.
    *   Nhóm đã sửa lỗi này triệt để bằng cách chèn lớp `tf.keras.layers.Rescaling` thích hợp vào đầu mỗi Sequential model trong file [train.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/train.py).
3.  **Tăng cường dữ liệu (Data Augmentation)**: Áp dụng các kỹ thuật xoay nhẹ (rotation), lật ngang (horizontal flip) và thay đổi độ tương phản (contrast adjustment) trực tiếp trên TensorFlow Datasets API để tăng tính tổng quát cho mô hình phân loại trên môi trường thực tế bãi đỗ xe có ánh sáng thay đổi.

---

## 5. Kết luận và Bước tiếp theo (Conclusion & Next Steps)
Bộ dữ liệu sau khi thu thập và làm sạch đã vượt qua toàn bộ các bài kiểm tra unit test của hệ thống tại [test_dataset.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/tests/test_dataset.py). Đây là nền tảng vững chắc để chuyển sang giai đoạn huấn luyện các mô hình học sâu phân loại hãng xe và màu sắc xe ở Giai đoạn 2 (Stage 2).
