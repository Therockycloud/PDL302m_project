# Báo cáo kỹ thuật Giai đoạn 2: Quy trình thu thập và Xử lý dữ liệu thực tế (Data Tasks)

## 1. Đặt vấn đề và Mục tiêu nghiên cứu (Problem & Objective)
Trong các hệ thống giám sát an ninh bãi giữ xe truyền thống, nhận dạng biển số xe (LPR) bằng OCR là cơ chế duy nhất để xác thực phương tiện ra vào. Tuy nhiên, phương thức này dễ bị qua mặt bằng các thủ thuật như tráo đổi biển số hoặc làm giả biển số. Nhằm khắc phục lỗ hổng an ninh này, dự án DPL302m đề xuất hệ thống đối chiếu chéo đa nhân tố dựa trên ba đặc trưng sinh trắc học trực quan của phương tiện: **Biển số xe (License Plate)**, **Hãng sản xuất (Brand)**, và **Màu sắc xe (Color)**.

Mục tiêu của giai đoạn này là thiết lập một đường ống (pipeline) thu thập, làm sạch, phân tích thống kê (EDA) và tiền xử lý dữ liệu từ các nguồn thực tế để huấn luyện các bộ phân loại sâu (Deep Learning Classifiers), đảm bảo mô hình hoạt động ổn định và chính xác trên môi trường thực tế tại Việt Nam.

---

## 2. Nghiên cứu tài liệu tham khảo (Literature Review)
Trong quá trình xây dựng bộ dữ liệu và pipeline tiền xử lý, nhóm đã tham khảo các nghiên cứu khoa học và bài báo chuyên ngành sau:
1.  **Stanford Cars Dataset (Krause et al., 2013)**: Nghiên cứu giới thiệu bộ dữ liệu 16,185 hình ảnh của 196 loại ô tô, thiết lập tiêu chuẩn cho việc phân loại hãng xe và dòng xe dựa trên học sâu. Nhóm đã kế thừa phương pháp phân nhóm thương hiệu từ nghiên cứu này.
2.  **Vehicle Color Recognition in Urban Surveillance (Chen et al., 2019)**: Bài báo phân tích ảnh hưởng của điều kiện ánh sáng hầm và ngoài trời tới độ chính xác nhận diện màu sắc xe. Nghiên cứu nhấn mạnh tầm quan trọng của việc chuẩn hóa dải điểm ảnh trước khi đưa vào các mạng CNN siêu nhẹ như MobileNetV3.
3.  **Data Augmentation for Object Detection and Classification in Automated Parking Systems (Wang & Choi, 2021)**: Đề xuất các phương pháp tăng cường dữ liệu như dịch chuyển độ sáng, lật ảnh và xoay góc để cải thiện khả năng tổng quát hóa của mô hình dưới các góc camera giám sát bãi đỗ khác nhau.

---

## 3. Quy trình thu thập dữ liệu (Data Acquisition Pipeline)
Hệ thống sử dụng ba luồng dữ liệu độc lập để huấn luyện các mô hình thành phần:

### 3.1. Dữ liệu hãng xe (Car Brands)
Nhóm tập trung thu thập ảnh của 8 hãng ô tô phổ biến nhất tại thị trường Việt Nam: *Toyota, Hyundai, Kia, Mazda, Honda, VinFast, Ford, Mitsubishi*.
*   **Stanford Cars Dataset (Parquet Format)**: Để tránh giới hạn băng thông (Rate Limit 429) khi tải ảnh trực tiếp, nhóm sử dụng định dạng Parquet sạch của Stanford Cars trên Hugging Face (`tanganke/stanford_cars`). Từ đây, nhóm trích xuất hình ảnh tương ứng với 7 hãng xe quốc tế.
*   **Wikimedia Commons VinFast Crawler**: Do VinFast là thương hiệu nội địa mới và không có sẵn trong các dataset học thuật quốc tế, nhóm đã xây dựng một script cào ảnh tự động từ Wikimedia Commons. Script sử dụng các kỹ thuật cấu hình `User-Agent` mô phỏng trình duyệt và thiết lập khoảng nghỉ an toàn $1.5$ giây giữa các yêu cầu để tránh bị chặn. Nhóm đã thu thập thành công 120 ảnh chất lượng cao của các dòng xe VinFast Lux A2.0, Lux SA2.0, VF8, VF9.

### 3.2. Dữ liệu màu sắc xe (Car Colors)
Tập dữ liệu gồm 8 màu xe cơ bản phổ biến tại Việt Nam: *White (Trắng), Black (Đen), Grey (Xám), Silver (Bạc), Red (Đỏ), Blue (Xanh dương), Brown (Nâu), Yellow (Vàng)*.
*   Ảnh thô được tải từ bộ dữ liệu nhận diện màu sắc của Kaggle và được lọc thủ công để loại bỏ các ảnh nhiễu (ảnh cận cảnh vô lăng, nội thất, hoặc ảnh thiếu sáng nghiêm trọng làm sai lệch màu sắc thực tế).

### 3.3. Dữ liệu biển số xe (License Plates)
Nhóm thu thập tập mẫu gồm 5 ảnh chụp xe thực tế tại các bãi đỗ xe Việt Nam (chứa cả biển số xe dài 1 dòng và biển số vuông 2 dòng) đi kèm file nhãn định dạng YOLO tương ứng (`.txt`) làm dữ liệu kiểm thử (Test Set) cho luồng tích hợp đầu cuối.

---

## 4. Phân tích thống kê dữ liệu (Exploratory Data Analysis)

### 4.1. Phân bố nhãn Hãng xe (Car Brands Distribution)
| Hãng xe (Brand) | Số lượng ảnh thô | Trạng thái xử lý | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: |
| **Toyota** | 168 | Đã làm sạch | 13.9% |
| **Hyundai** | 200 | Đã làm sạch | 16.5% |
| **Kia** | 120 | Đã làm sạch | 9.9% |
| **Mazda** | 120 | Đã làm sạch | 9.9% |
| **Honda** | 161 | Đã làm sạch | 13.3% |
| **VinFast** | 120 | Đã cào & làm sạch | 9.9% |
| **Ford** | 200 | Đã làm sạch | 16.5% |
| **Mitsubishi** | 120 | Đã làm sạch | 9.9% |
| **Tổng cộng** | **1,209** | **Hoàn thành** | **100.0%** |

### 4.2. Phân bố nhãn Màu sắc xe (Car Colors Distribution)
| Màu sắc (Color) | Số lượng ảnh thô | Trạng thái xử lý | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: |
| **White** | 185 | Đã làm sạch | 16.4% |
| **Black** | 200 | Đã làm sạch | 17.7% |
| **Grey** | 200 | Đã làm sạch | 17.7% |
| **Silver** | 175 | Đã làm sạch | 15.5% |
| **Red** | 110 | Đã làm sạch | 9.7% |
| **Blue** | 200 | Đã làm sạch | 17.7% |
| **Brown** | 35 | Đã làm sạch | 3.1% |
| **Yellow** | 25 | Đã làm sạch | 2.2% |
| **Tổng cộng** | **1,130** | **Hoàn thành** | **100.0%** |

*Nhận xét*: Có sự mất cân bằng dữ liệu tự nhiên đối với nhóm màu Brown và Yellow do hai màu này ít phổ biến hơn trên thực tế xe lưu thông tại Việt Nam. Pipeline huấn luyện sẽ tích hợp các kỹ thuật tăng cường dữ liệu thích hợp để tránh hiện tượng mô hình bị thiên lệch (bias).

---

## 5. Tiền xử lý và Tăng cường dữ liệu (Preprocessing & Augmentation)

### 5.1. Chuẩn hóa kích thước hình ảnh (Resizing)
Toàn bộ ảnh xe toàn cảnh được tự động thay đổi kích thước về độ phân giải chuẩn $224 \times 224 \times 3$ pixel nhằm tương thích hoàn toàn với đầu vào của mạng `EfficientNet-B0` và `MobileNetV3-Small`.

### 5.2. Sửa lỗi chuẩn hóa tỷ lệ điểm ảnh (Pixel Scaling Correction)
Trong quá trình thử nghiệm ban đầu, hệ thống gặp lỗi suy luận do sự khác biệt về dải giá trị đầu vào của các mạng:
*   Mạng trích xuất đặc trưng `EfficientNetB0` yêu cầu dải điểm ảnh gốc $[0, 255]$ do đã có lớp chuẩn hóa nội bộ của Keras.
*   Mạng `MobileNetV3Small` yêu cầu dải điểm ảnh đã được chuẩn hóa về khoảng $[-1, 1]$.
*   Giải pháp: Nhóm đã chèn trực tiếp các lớp tiền xử lý tích hợp của TensorFlow (`preprocess_input` cho EfficientNet và `Rescaling(scale=1.0/127.5, offset=-1)` cho MobileNetV3) vào đầu mỗi mô hình để tự động hóa quá trình chuẩn hóa tại bước huấn luyện lẫn suy luận thực tế.

### 5.3. Tăng cường dữ liệu (Data Augmentation)
Để tăng độ bền vững (robustness) cho bộ phân loại trong điều kiện thời tiết và ánh sáng phức tạp của bãi đỗ xe:
*   **Xoay ngẫu nhiên (Random Rotation)**: Góc xoay tối đa $10^\circ$ để mô phỏng sai lệch góc lắp đặt camera.
*   **Lật ngang (Random Flip)**: Mô phỏng hướng di chuyển của xe đi vào từ hai phía khác nhau.
*   **Điều chỉnh độ tương phản (Random Contrast)**: Thay đổi độ tương phản ngẫu nhiên từ $0.8$ đến $1.2$ để mô phỏng ánh sáng ban ngày chói hoặc ánh đèn pha xe ban đêm.

---

## 6. Đánh giá và Kiểm thử dữ liệu (Verification)
Toàn bộ quy trình tải, phân tách thư mục, ánh xạ nhãn lớp và tiền xử lý dữ liệu đầu vào đã được kiểm thử tự động thông qua bộ kiểm thử unit test viết trong [test_dataset.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/tests/test_dataset.py). Kết quả kiểm thử thành công 100%, xác nhận cấu trúc dữ liệu đã sẵn sàng chuyển sang giai đoạn huấn luyện mô hình sâu.
