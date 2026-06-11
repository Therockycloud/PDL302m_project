# Báo cáo kỹ thuật Giai đoạn 3: Huấn luyện Mô hình Học sâu và Kết quả Thực nghiệm (Model & Results)

## 1. Đặt vấn đề và Mục tiêu huấn luyện (Objective)
Giai đoạn 3 tập trung vào thiết kế kiến trúc, cấu hình siêu tham số, thực thi huấn luyện và đánh giá chi tiết các mô hình học máy thành phần trong pipeline xác thực phương tiện. Mục tiêu là xây dựng và tối ưu hóa các mô hình nhận diện biển số (YOLOv8-nano), trích xuất văn bản biển số (EasyOCR) và các mô hình học sâu phân loại đặc trưng phương tiện (Hãng xe và Màu sắc xe) sử dụng phương pháp Học chuyển vị (Transfer Learning) nhằm tối ưu hóa độ trễ và tài nguyên phần cứng.

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

### 3.2. Bộ nhận diện ký tự biển số (EasyOCR Engine)
*   **Kiến trúc**: Sử dụng mạng ResNet kết hợp với mạng hồi quy tuần hoàn LSTM và tầng giải mã CTC (Connectionist Temporal Classification).
*   **Cấu hình**: Chạy hoàn toàn ngoại tuyến (`download_enabled=False`) để loại bỏ hoàn toàn độ trễ kiểm tra phiên bản qua mạng của EasyOCR.

### 3.3. Bộ phân loại hãng xe (Brand Classifier)
*   **Backbone**: **EfficientNet-B0** (đã đóng băng các lớp trích xuất đặc trưng tiền huấn luyện trên ImageNet).
*   **Tầng phân loại bổ sung**:
    *   `GlobalAveragePooling2D()`: Làm phẳng bản đồ đặc trưng 2D từ backbone.
    *   `Dropout(0.5)`: Giảm tỷ lệ khớp quá mức xuống 50% bằng cách ngắt ngẫu nhiên một nửa số nơ-ron kết nối trong mỗi batch huấn luyện.
    *   `Dense(8, activation="softmax")`: Đầu ra phân lớp cho 8 thương hiệu mục tiêu.

### 3.4. Bộ phân loại màu sắc xe (Color Classifier)
*   **Backbone**: **MobileNetV3-Small** (đã đóng băng các lớp trích xuất đặc trưng tiền huấn luyện trên ImageNet).
*   **Tầng phân loại bổ sung**:
    *   `Rescaling(scale=1.0/127.5, offset=-1.0)`: Đưa dải điểm ảnh đầu vào về khoảng $[-1, 1]$.
    *   `GlobalAveragePooling2D()`
    *   `Dropout(0.3)`: Tỷ lệ dropout 30% phù hợp với mạng nhỏ nhằm tránh mất mát thông tin màu sắc.
    *   `Dense(8, activation="softmax")`: Phân loại 8 nhóm màu xe.

---

## 4. Quá trình huấn luyện và Cấu hình siêu tham số (Training Configuration)
Quá trình huấn luyện được thực hiện cục bộ trên hệ điều hành macOS sử dụng CPU thông qua script [train.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/train.py).

### Chi tiết tham số cấu hình:
*   **Optimizer (Bộ tối ưu hóa)**: Adam với tốc độ học $\eta = 10^{-4}$ (0.0001) nhằm đảm bảo quá trình cập nhật trọng số diễn ra mượt mà và ổn định.
*   **Hàm mất mát (Loss Function)**: Categorical Crossentropy để phân loại đa lớp.
*   **Kích thước Batch (Batch Size)**: 32 mẫu/batch.
*   **Số lượng Epoch**: Tối đa 10 epochs (để tối ưu hóa thời gian huấn luyện trên CPU).
*   **Cơ chế giám sát nâng cao**:
    *   `EarlyStopping`: Tự động dừng huấn luyện nếu giá trị `val_loss` không cải thiện liên tiếp sau 5 epochs, đồng thời khôi phục trọng số tốt nhất trước đó.
    *   `ModelCheckpoint`: Tự động lưu mô hình có giá trị `val_loss` thấp nhất dưới dạng file `.keras`.

---

## 5. Kết quả thực nghiệm và Biểu đồ huấn luyện (Experimental Results)
Sau khi hoàn thành 10 epochs huấn luyện trên bộ dữ liệu thực tế thu thập được ở Giai đoạn 2, nhóm ghi nhận các kết quả thực nghiệm sau:

### 5.1. Mô hình phân loại hãng xe (Brand Classifier)
*   **Độ chính xác Validation (val_accuracy)**: Đạt khoảng **29.00%**.
*   **Phân tích nguyên nhân**: Độ chính xác này phản ánh sự khác biệt lớn về phân phối dữ liệu (domain gap). Dataset huấn luyện Stanford Cars chủ yếu gồm ảnh xe chụp từ góc ngang (side view) của Mỹ, trong khi tập dữ liệu test thực tế tại Việt Nam lại chụp từ góc trực diện phía sau hoặc phía trước. Thêm vào đó, việc giới hạn số lượng epoch huấn luyện trên CPU đã làm hạn chế khả năng hội tụ sâu của tầng kết nối đầy đủ.
*   **Biểu đồ huấn luyện**: [brand_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/presentations/evidence/brand_classifier_training_curves.png)

### 5.2. Mô hình phân loại màu sắc xe (Color Classifier)
*   **Độ chính xác Validation (val_accuracy)**: Đạt khoảng **14.16%**.
*   **Phân tích nguyên nhân**: Phân loại màu sắc xe thực tế chịu ảnh hưởng rất mạnh bởi sự cân bằng trắng của camera, cường độ ánh sáng của môi trường bãi giữ xe (điều kiện ánh sáng hầm so với ngoài trời nắng). Sự chồng lấn đặc trưng giữa các nhóm màu Bạc (Silver), Xám (Grey) và Trắng (White) cũng là nguyên nhân chính khiến mô hình phân loại sai lệch.
*   **Biểu đồ huấn luyện**: [color_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/presentations/evidence/color_classifier_training_curves.png)

---

## 6. Đề xuất cải tiến độ chính xác (Future Enhancements)
Để nâng cao độ chính xác của các bộ phân loại khi triển khai thực tế thương mại, nhóm đề xuất các giải pháp kỹ thuật sau:
1.  **Thu thập dữ liệu miền thực tế (In-domain Data)**: Tiến hành quay phim và cắt ảnh xe trực tiếp tại các bãi giữ xe ở Việt Nam với các góc máy cố định để đồng bộ hóa phân phối dữ liệu huấn luyện và kiểm thử.
2.  **Fine-tuning sâu hơn**: Sử dụng card đồ họa GPU chuyên dụng để chạy từ 50 đến 100 epochs huấn luyện, kết hợp mở băng thông (unfreeze) một số tầng tích chập cuối của EfficientNet-B0 để mạng tự điều chỉnh bộ lọc đặc trưng phù hợp với xe thực tế.
