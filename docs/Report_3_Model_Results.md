# Báo cáo kỹ thuật Giai đoạn 3: Huấn luyện Mô hình Học sâu và Kết quả Thực nghiệm (Model & Results)

## 1. Đặt vấn đề và Mục tiêu huấn luyện (Objective)
Giai đoạn 3 tập trung vào thiết kế kiến trúc, cấu hình siêu tham số, thực thi huấn luyện và đánh giá chi tiết các mô hình học máy thành phần trong pipeline xác thực phương tiện. Mục tiêu là xây dựng và tối ưu hóa các mô hình nhận diện biển số (YOLOv8-nano), trích xuất văn bản biển số (ban đầu EasyOCR, sau chuyển sang **PaddleOCR** — xem §3.2) và các mô hình học sâu phân loại đặc trưng phương tiện (Hãng xe và Màu sắc xe) sử dụng phương pháp Học chuyển vị (Transfer Learning) nhằm tối ưu hóa độ trễ và tài nguyên phần cứng.

---

## 2. Nghiên cứu tài liệu tham khảo (Literature Review)
Trong quá trình phát triển mô hình, nhóm đã dựa trên các cơ sở lý thuyết và nghiên cứu thực nghiệm sau:
1.  **YOLOv8 Architecture (Jocher et al., 2023 - Ultralytics)**: Nghiên cứu giới thiệu kiến trúc YOLOv8 cải tiến với cấu hình không có neo (anchor-free), nâng cao đáng kể độ chính xác định vị vùng hộp biển số trong khi vẫn duy trì tốc độ suy luận thời gian thực ở mức cao trên phần cứng CPU/GPU yếu.
2.  **Deep Residual Learning for Image Recognition (He et al., 2016 - ResNet)**: Công trình đặt nền móng cho cơ chế kết nối tắt (shortcut connections) giúp giải quyết hiện tượng suy giảm gradient trong các mạng nơ-ron rất sâu. Nhóm đã tham khảo cấu trúc ResNet để thiết kế mạng phân loại hãng xe EfficientNet-B0 (vốn sử dụng cơ chế mở rộng Compound Scaling hiệu quả hơn).
3.  **Searching for MobileNetV3 (Howard et al., 2019)**: Giới thiệu kiến trúc MobileNetV3 kết hợp công nghệ NAS (Network Architecture Search) và thiết kế NetAdapt, mang lại giải pháp phân loại ảnh vô cùng gọn nhẹ và tiết kiệm năng lượng cho thiết bị biên. Đây là cơ sở khoa học để nhóm lựa chọn MobileNetV3-Small cho tác vụ phân loại màu sắc xe.

---

## 3. Kiến trúc các mô hình thành phần (Model Architectures)

### 3.1. Bộ phát hiện vị trí biển số (YOLOv8 Plate Detector)
*   **Kiến trúc**: YOLOv8-nano (`yolov8n.pt`).
*   **Đặc điểm**: Đây là phiên bản nhỏ nhất trong dòng YOLOv8 với số lượng tham số cực kỳ tối ưu, giúp suy luận cực nhanh trên CPU mà không cần card đồ họa chuyên dụng.
*   **Đầu ra**: Tọa độ bounding box $[x_{min}, y_{min}, x_{max}, y_{max}]$ bao quanh biển số xe.

### 3.2. Bộ nhận diện ký tự biển số (OCR Engine: PaddleOCR)
*   **Lựa chọn engine (Benchmark C)**: Ban đầu nhóm dùng **EasyOCR** (ResNet + LSTM + CTC). Tuy nhiên benchmark trên 16 biển CCTV thật cho thấy EasyOCR đọc đúng **0%** chuỗi (exact-match), trong khi **PaddleOCR (PP-OCRv4, CRNN+CTC)** đạt **81%** (CER 0.28 → 0.03). Vì vậy **PaddleOCR là engine chính**, EasyOCR giữ làm fallback. Chi tiết: `docs/benchmarks/ocr_benchmark.md`.
*   **Cấu hình**: Chạy ngoại tuyến hoàn toàn; engine cấu hình ở `main/configs/config.yaml` (`ocr.engine: ppocr`, fallback `easyocr`).

### 3.3. Bộ phân loại hãng xe (Brand Classifier)
*   **Backbone**: **EfficientNet-B0** (đã đóng băng các lớp trích xuất đặc trưng tiền huấn luyện trên ImageNet).
*   **Tầng phân loại bổ sung**:
    *   `GlobalAveragePooling2D()`: Làm phẳng bản đồ đặc trưng 2D từ backbone.
    *   `Dropout(0.5)`: Giảm tỷ lệ khớp quá mức xuống 50% bằng cách ngắt ngẫu nhiên một nửa số nơ-ron kết nối trong mỗi batch huấn luyện.
    *   `Dense(8, activation="softmax")`: Đầu ra phân lớp cho 8 thương hiệu mục tiêu.
    *   *Lưu ý pivot:* khác với ResNet50 trong đề xuất ban đầu (Report 1), nhóm chọn **EfficientNet-B0** vì cùng độ chính xác nhưng nhẹ và nhanh hơn nhiều trên CPU (xem Benchmark A màu, cùng kết luận về kích thước/độ trễ).

### 3.4. Bộ phân loại màu sắc xe (Color Classifier)
*   **Backbone**: **MobileNetV3-Small** (đã đóng băng các lớp trích xuất đặc trưng tiền huấn luyện trên ImageNet).
*   **Tầng phân loại bổ sung** (dựng bằng Functional API, gọi `base(x, training=False)` để BatchNorm chạy chế độ inference):
    *   `Rescaling(255.0)`: đưa ảnh [0,1] về [0,255] đúng kỳ vọng của MobileNetV3 (`include_preprocessing=True` tự chuẩn hoá nội bộ). *Lưu ý: phiên bản đầu cộng thêm `Rescaling(1/127.5,-1)` gây double-preprocessing — đã loại (xem mục 5.0).*
    *   `GlobalAveragePooling2D()`
    *   `Dropout(0.3)`: Tỷ lệ dropout 30% phù hợp với mạng nhỏ.
    *   `Dense(8, activation="softmax")`: Phân loại 8 nhóm màu xe.

---

## 4. Quá trình huấn luyện và Cấu hình siêu tham số (Training Configuration)
Quá trình huấn luyện được thực hiện cục bộ trên hệ điều hành macOS sử dụng CPU thông qua script [train.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/train.py).

### Chi tiết tham số cấu hình:
*   **Hai pha huấn luyện**: (1) **head đông cứng** — backbone freeze, chỉ train tầng Dense, Adam lr $=10^{-3}$; (2) **fine-tuning** — mở băng nửa trên backbone (BatchNorm vẫn freeze), Adam lr $=10^{-4}$.
*   **Hàm mất mát**: Categorical Crossentropy (phân loại đa lớp).
*   **Batch Size**: 32 mẫu/batch.
*   **Số lượng Epoch**: tối đa 15/pha với `EarlyStopping` (môi trường training cô lập TF/Keras, tách khỏi runtime PaddleOCR).
*   **Cơ chế giám sát nâng cao**:
    *   `EarlyStopping`: Tự động dừng huấn luyện nếu giá trị `val_loss` không cải thiện liên tiếp sau 5 epochs, đồng thời khôi phục trọng số tốt nhất trước đó.
    *   `ModelCheckpoint`: Tự động lưu mô hình có giá trị `val_loss` thấp nhất dưới dạng file `.keras`.

---

## 5. Kết quả thực nghiệm và Biểu đồ huấn luyện (Experimental Results)

### 5.0. Hai lỗi kỹ thuật ban đầu và cách khắc phục
Những lần huấn luyện Keras **đầu tiên** cho kết quả gần như ngẫu nhiên (hãng ~29%, màu ~14%). Khi chuẩn hoá lại pipeline về TF/Keras, nhóm xác định nguyên nhân **không phải** do domain gap mà do **hai lỗi dựng mô hình**:
1.  **Double-preprocessing**: lớp `Rescaling(1/127.5, -1)` được chèn *trên đầu* lớp tiền xử lý nội bộ của MobileNetV3 → sai dải giá trị, accuracy rơi về ~1/8. Đã bỏ, chỉ giữ `Rescaling(255.0)` cho cả hai backbone.
2.  **BatchNorm chạy ở chế độ training khi backbone đông cứng**: dựng bằng `Sequential` khiến BatchNorm của backbone dùng thống kê từng batch, phá vỡ đặc trưng đông cứng → mô hình không học. Đã chuyển sang **Functional API** với `base(x, training=False)`.
Ngoài ra `learning_rate` ban đầu (1e-4) quá thấp khiến EarlyStopping dừng sớm; nâng lên **1e-3** cho pha head đông cứng.

Sau khi sửa, đánh giá được thực hiện trên **tập test giữ-riêng** (118 ảnh màu / 119 ảnh hãng, split 70/15/15 — xem Report 2), chưa từng thấy khi huấn luyện.

### 5.1. Mô hình phân loại màu sắc xe (Color Classifier — TF/Keras MobileNetV3-Small)
| Cấu hình | Test accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Head đông cứng (frozen) | 48.3% | 0.48 |
| **+ Fine-tuning (mở băng nửa trên backbone, lr=1e-4)** | **54.2%** | **0.54** |

Fine-tuning nâng accuracy thêm ~6 điểm (gấp >4 lần mức ngẫu nhiên 12.5% trên 8 lớp). Đây là mô hình màu **đang chạy ở runtime** (`color_classifier.keras`), phục vụ qua tiến trình riêng để đồng tồn với PaddleOCR (xem Report 4).
*   **Biểu đồ huấn luyện**: [color_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/data/models/color_classifier_training_curves.png)

### 5.2. Mô hình phân loại hãng xe (Brand Classifier — TF/Keras EfficientNet-B0)
| Cấu hình | Test accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Head đông cứng | 32.8% | 0.32 |
| + Fine-tuning | 35.3% | 0.34 |

Dù đã sửa lỗi + fine-tune, phân loại hãng vẫn yếu (~35%) — bài toán 8 hãng nhìn từ phía sau với ~70 ảnh/lớp là khó. Kết quả này **củng cố quyết định bỏ phân loại hãng** khỏi cơ chế quyết định.
*   **Biểu đồ huấn luyện**: [brand_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/data/models/brand_classifier_training_curves.png)

### 5.3. Cơ chế quyết định delivered (plate-primary)
1.  **Biển số (PaddleOCR) là khoá chính** — Benchmark C: exact-match 81% so với EasyOCR 0% trên biển CCTV thật.
2.  **Màu là "cảnh báo mềm"**: màu lệch so với CSDL chỉ phát cảnh báo (`ALLOW_WARN`) chống tráo biển, không từ chối cứng.
3.  **Bỏ phân loại hãng** khỏi quyết định (giữ lại như thử nghiệm).
Đây là cơ chế **delivered** của hệ thống (xem Report 4).

---

## 6. Đề xuất cải tiến độ chính xác (Future Enhancements)
1.  **Thu thập dữ liệu in-domain**: quay và cắt ảnh xe trực tiếp tại bãi giữ xe Việt Nam với góc máy cố định để khớp phân phối train/test.
2.  **Fine-tune sâu hơn**: pha fine-tuning hiện mở nửa trên backbone ở lr=1e-4 (đã nâng màu 48.3%→54.2%); với GPU có thể mở nhiều tầng hơn, chạy 50–100 epochs và class-balanced sampling để đẩy cao hơn nữa.
3.  **Cân bằng & mở rộng dữ liệu**: nâng các lớp hiếm (Brown/Yellow) vượt ~100 ảnh và bổ sung biến thể ánh sáng bãi xe.
