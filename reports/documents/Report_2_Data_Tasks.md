# Báo cáo kỹ thuật Giai đoạn 2: Quy trình thu thập và Xử lý dữ liệu thực tế (Data Tasks)

**Môn học:** DPL302m – Deep Learning · **Nhóm:** Nhóm 7
**Repository (GitHub):** [https://github.com/Therockycloud/PDL302m_project](https://github.com/Therockycloud/PDL302m_project)
**Clone:** `git clone https://github.com/Therockycloud/PDL302m_project.git`

---

## 1. Đặt vấn đề và Mục tiêu nghiên cứu (Problem & Objective)
Trong các hệ thống giám sát an ninh bãi giữ xe truyền thống, nhận dạng biển số xe (LPR) bằng OCR là cơ chế duy nhất để xác thực phương tiện ra vào. Tuy nhiên, phương thức này dễ bị qua mặt bằng các thủ thuật như tráo đổi biển số hoặc làm giả biển số. Nhằm khắc phục lỗ hổng an ninh này, dự án DPL302m đề xuất hệ thống đối chiếu chéo đa nhân tố dựa trên ba đặc trưng sinh trắc học trực quan của phương tiện: **Biển số xe (License Plate)**, **Hãng sản xuất (Brand)**, và **Màu sắc xe (Color)**.

> **Ghi chú:** Bộ ba đặc trưng dưới đây là theo *đề xuất ban đầu*. Bản giao cuối dùng biển số làm khoá chính (plate-primary), màu là cảnh báo phụ, bỏ hãng — xem Report 3/4.

Mục tiêu của giai đoạn này là thiết lập một đường ống (pipeline) thu thập, làm sạch, phân tích thống kê (EDA) và tiền xử lý dữ liệu từ các nguồn thực tế để huấn luyện các bộ phân loại sâu (Deep Learning Classifiers), đảm bảo mô hình hoạt động ổn định và chính xác trên môi trường thực tế tại Việt Nam.

---

## 2. Nghiên cứu tài liệu tham khảo (Literature Review)
Trong quá trình xây dựng bộ dữ liệu và pipeline tiền xử lý, nhóm đã tham khảo các nghiên cứu khoa học và bài báo chuyên ngành sau:
1.  **Stanford Cars Dataset (Krause et al., 2013)**: Nghiên cứu giới thiệu bộ dữ liệu 16,185 hình ảnh của 196 loại ô tô, thiết lập tiêu chuẩn cho việc phân loại hãng xe và dòng xe dựa trên học sâu. Nhóm đã kế thừa phương pháp phân nhóm thương hiệu từ nghiên cứu này.
2.  **Vehicle Color Recognition on Urban Road by Feature Context (Chen, Bai & Liu, 2014)** — *IEEE Transactions on Intelligent Transportation Systems, 15(5), 2340–2346*: Bài báo phân tích ảnh hưởng của điều kiện ánh sáng và nền đô thị tới độ chính xác nhận diện màu sắc xe, giới thiệu bộ dữ liệu 15,601 ảnh xe chụp từ camera giám sát với 8 nhóm màu. Nghiên cứu nhấn mạnh tầm quan trọng của việc chuẩn hóa dải điểm ảnh trước khi đưa vào các mạng CNN siêu nhẹ như MobileNetV3.
3.  **A Survey of Data Augmentation Techniques for Traffic Visual Elements (Yang et al., 2025)** — *Sensors, 25(21), 6672*: Tổng hợp và so sánh các phương pháp tăng cường dữ liệu (dịch chuyển độ sáng, lật ảnh, xoay góc, và các mô hình sinh ảnh) cho các đối tượng giao thông như người và phương tiện, giúp cải thiện khả năng tổng quát hóa của mô hình dưới các góc camera giám sát khác nhau.

---

## 3. Quy trình thu thập dữ liệu (Data Acquisition Pipeline)
Hệ thống sử dụng ba luồng dữ liệu độc lập để huấn luyện các mô hình thành phần:

### 3.1. Dữ liệu hãng xe (Car Brands)
Nhóm tập trung thu thập ảnh của 8 hãng ô tô phổ biến nhất tại thị trường Việt Nam: *Toyota, Hyundai, Kia, Mazda, Honda, VinFast, Ford, Mitsubishi*.
*   **Stanford Cars Dataset (Parquet Format)**: Để tránh giới hạn băng thông (Rate Limit 429) khi tải ảnh trực tiếp, nhóm sử dụng định dạng Parquet sạch của Stanford Cars trên Hugging Face (`tanganke/stanford_cars`). Từ đây, nhóm trích xuất hình ảnh tương ứng với 7 hãng xe quốc tế.
*   **VinFast — cào theo từng dòng xe (model-specific)**: Do VinFast là thương hiệu nội địa mới và không có trong các dataset học thuật quốc tế, nhóm cào ảnh tự động (Bing image crawler) theo **từ khóa chuyên biệt cho từng dòng xe** — VF8, VF9, VF5, Lux A2.0, Lux SA2.0, Fadil, VF e34, President — thay vì từ khóa "VinFast" chung (vốn trả về nhiều logo/nội thất). Nhờ vậy ảnh VinFast giữ được chất lượng và đúng đối tượng.

### 3.2. Dữ liệu màu sắc xe (Car Colors)
Tập dữ liệu gồm **8 màu** xe cơ bản phổ biến tại Việt Nam: *White, Black, Grey, Silver, Red, Blue, Brown, Yellow*. Lớp *green* (xanh lá) ban đầu được cào về nhưng đã bị loại khỏi tập huấn luyện để khớp với mô hình 8 lớp (39 ảnh green được tách sang khu cách ly, không dùng).
*   Ảnh thô được hợp nhất từ nhiều nguồn cào (Bing) và được đưa qua **pipeline làm sạch tự động** (xem mục 3.3) thay cho việc lọc thủ công.

### 3.3. Pipeline làm sạch & cân bằng dữ liệu (Data Cleaning Pipeline)
Toàn bộ ảnh phân loại được hợp nhất về **một cây thư mục chuẩn duy nhất** (`car_colors/`, `car_brands/`) rồi đưa qua 4 bước tự động (scripts trong `main/scripts/`):
1.  **Lọc ảnh hỏng** (`clean_corrupted_images.py`): loại file không đọc được.
2.  **Lọc theo ngữ nghĩa bằng YOLOv8** (`semantic_clean_images.py`): chỉ giữ ảnh thực sự chứa xe (lớp COCO car/bus/truck). Bước này loại **~38% ảnh màu** không có xe (vô lăng, nội thất, logo…).
3.  **Khử trùng lặp** (`remove_duplicates.py`): dùng perceptual-hash (pHash) loại ảnh gần-trùng.
4.  **Chuẩn hóa về JPEG RGB** (`normalize_images.py`): tái mã hóa mọi ảnh (kể cả WEBP đội lốt `.jpg`) sang JPEG RGB để `tf.keras` đọc được.

Các lớp thiếu (Brown, Yellow) được **cào bù** (`crawl_topup.py`) rồi lặp lại bước 1–4, cuối cùng **cap về ~100 ảnh/lớp cân bằng** (`cap_dataset.py`). Kết quả: **8 lớp màu (783 ảnh)** và **8 lớp hãng (792 ảnh)**, cân bằng (xem mục 4).

### 3.4. Phân chia tập Train / Validation / Test
Bộ dữ liệu phân loại được chia **vật lý** theo tỷ lệ **70 / 15 / 15** (`split_dataset.py`, seed cố định = 42) vào `data/processed/classifiers/<task>/{train,val,test}/`. Khác với pipeline cũ chỉ tách train/val bằng `validation_split` (**không có tập test riêng**), nay mỗi tác vụ có **tập test giữ-riêng ổn định**:

| Tác vụ | Train | Validation | **Test** |
| :--- | :---: | :---: | :---: |
| Màu sắc (Colors) | 547 | 118 | **118** |
| Hãng xe (Brands) | 554 | 119 | **119** |

> So với bản trước (tập kiểm thử chỉ gồm **5 ảnh** dùng cho luồng E2E), tập test phân loại nay lớn hơn ~24 lần và không bao giờ được nhìn thấy trong lúc huấn luyện.

### 3.5. Dữ liệu biển số xe (License Plates)
Nhóm thu thập tập mẫu ảnh chụp xe thực tế tại các bãi đỗ xe Việt Nam (chứa cả biển số dài 1 dòng và biển số vuông 2 dòng) đi kèm nhãn định dạng YOLO (`.txt`) làm dữ liệu kiểm thử cho luồng tích hợp đầu cuối (E2E). Bộ phát hiện biển số được huấn luyện riêng (HuggingFace license-plate dataset: 6.176 ảnh train / 1.765 ảnh val).

---

## 4. Phân tích thống kê dữ liệu (Exploratory Data Analysis)

### 4.1. Phân bố nhãn Hãng xe (Car Brands Distribution)
| Hãng xe (Brand) | Số lượng ảnh (đã làm sạch & cân bằng) | Tỷ lệ (%) |
| :--- | :---: | :---: |
| **Toyota** | 95 | 12.0% |
| **Hyundai** | 99 | 12.5% |
| **Kia** | 99 | 12.5% |
| **Mazda** | 100 | 12.6% |
| **Honda** | 99 | 12.5% |
| **VinFast** | 100 | 12.6% |
| **Ford** | 100 | 12.6% |
| **Mitsubishi** | 100 | 12.6% |
| **Tổng cộng** | **792** | **100.0%** |

### 4.2. Phân bố nhãn Màu sắc xe (Car Colors Distribution)
| Màu sắc (Color) | Số lượng ảnh (đã làm sạch & cân bằng) | Tỷ lệ (%) |
| :--- | :---: | :---: |
| **White** | 100 | 12.8% |
| **Black** | 100 | 12.8% |
| **Grey** | 100 | 12.8% |
| **Silver** | 100 | 12.8% |
| **Red** | 100 | 12.8% |
| **Blue** | 100 | 12.8% |
| **Brown** | 91 | 11.6% |
| **Yellow** | 92 | 11.7% |
| **Tổng cộng** | **783** | **100.0%** |

*Nhận xét*: Hai nhóm màu Brown và Yellow ban đầu rất ít (chỉ 35 và 25 ảnh thô) do ít phổ biến trên thực tế. Thay vì chỉ dựa vào augmentation, nhóm đã **cào bù dữ liệu chuyên biệt** rồi làm sạch để nâng hai lớp này lên ~90 ảnh, đưa toàn bộ tập về trạng thái **gần cân bằng (~100 ảnh/lớp)** — triết lý "ít nhưng chất" thay cho "nhiều nhưng nhiễu".

---

## 5. Tiền xử lý và Tăng cường dữ liệu (Preprocessing & Augmentation)

### 5.1. Chuẩn hóa kích thước hình ảnh (Resizing)
Toàn bộ ảnh xe toàn cảnh được tự động thay đổi kích thước về độ phân giải chuẩn $224 \times 224 \times 3$ pixel nhằm tương thích hoàn toàn với đầu vào của mạng `EfficientNet-B0` và `MobileNetV3-Small`.

### 5.2. Chuẩn hóa tỷ lệ điểm ảnh & sửa lỗi tiền xử lý (Pixel Scaling)
Cả hai backbone của Keras (`EfficientNetB0`, `MobileNetV3Small`) đều có **lớp tiền xử lý tích hợp** (`include_preprocessing=True`) và **yêu cầu đầu vào dải $[0, 255]$**, sau đó tự chuẩn hóa nội bộ. Dataset trả về ảnh đã rescale về $[0, 1]$, nên mỗi mô hình mở đầu bằng một lớp `Rescaling(255.0)` để đưa về $[0, 255]$ đúng kỳ vọng của backbone.

> **Hai lỗi đã phát hiện và sửa khi chuẩn hóa pipeline về TF/Keras:**
> 1.  **Double-preprocessing**: phiên bản cũ cộng thêm `Rescaling(1/127.5, -1)` cho MobileNetV3 *trên đầu* lớp nội bộ → sai dải giá trị, độ chính xác rơi về mức ngẫu nhiên (~1/8). Đã bỏ, chỉ giữ `Rescaling(255.0)`.
> 2.  **BatchNorm ở chế độ training khi backbone đông cứng**: dựng mô hình bằng `Sequential` khiến các lớp BatchNorm của backbone chạy theo thống kê từng batch, phá vỡ đặc trưng đông cứng và mô hình **không học được**. Đã chuyển sang **Functional API** với `base(x, training=False)` để BatchNorm chạy ở chế độ inference (moving-average).

### 5.3. Tăng cường dữ liệu (Data Augmentation)
Áp dụng **chỉ trên tập train** (val/test giữ nguyên ảnh gốc) để tăng độ bền vững:
*   **Lật ngang (RandomFlip horizontal)**: mô phỏng hướng xe đi vào từ hai phía.
*   **Xoay ngẫu nhiên (RandomRotation 0.1)**: mô phỏng sai lệch góc lắp camera.
*   **Phóng ngẫu nhiên (RandomZoom 0.1)**: mô phỏng khoảng cách xe–camera thay đổi.

---

## 6. Huấn luyện, Đánh giá và Kiểm thử (Train / Test)
Pipeline được **chuẩn hóa hoàn toàn về TensorFlow/Keras** và chạy trong môi trường training cô lập (tách khỏi runtime PaddleOCR để tránh xung đột OpenMP/protobuf). Cả hai bộ phân loại được huấn luyện trên tập train và **đánh giá trên tập test giữ-riêng**:

| Mô hình | Backbone | Test accuracy | Macro-F1 | n (test) |
| :--- | :--- | :---: | :---: | :---: |
| Phân loại màu | MobileNetV3-Small (frozen head) | **48.3%** | 0.479 | 118 |
| Phân loại hãng | EfficientNet-B0 (frozen head) | **32.8%** | 0.324 | 119 |

*Nhận xét*: Phân loại màu đạt ~48% trên 8 lớp (gấp ~4 lần mức ngẫu nhiên 12.5%) với head đông cứng — phù hợp vai trò "cảnh báo phụ". Phân loại hãng chỉ ~33%, củng cố quyết định **bỏ phân loại hãng** ở bản giao cuối (xem Report 3/4). Cả hai con số là kết quả trên tập test chưa từng thấy khi train; có thể nâng thêm bằng fine-tune (mở băng các tầng cuối của backbone).

Cấu trúc dữ liệu (tải, phân tách thư mục, ánh xạ nhãn) tiếp tục được kiểm thử tự động qua [test_dataset.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/tests/test_dataset.py).
