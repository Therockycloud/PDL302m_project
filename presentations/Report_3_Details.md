# Báo cáo Chi tiết: Huấn luyện Mô hình Học sâu và Kết quả Thực nghiệm (Report 3 Details)

## 1. Kiến trúc Mô hình (Model Architectures)
Nhóm phát triển hệ thống sử dụng kiến trúc mạng nơ-ron tích chập sâu (Deep Convolutional Neural Networks) với phương pháp Học chuyển vị (Transfer Learning) từ các mô hình đã được tiền huấn luyện trên tập ImageNet lớn nhằm tối ưu hóa tài nguyên phần cứng bãi đỗ xe:

### 1.1. Bộ phân loại Hãng xe (Vehicle Brand Classifier)
*   **Backbone**: **EfficientNet-B0**
*   **Lý do lựa chọn**: Trọng lượng mô hình nhẹ (~29 MB), lượng tham số nhỏ nhưng độ chính xác trích xuất đặc trưng sâu vượt trội hơn các kiến trúc truyền thống như ResNet50 hay VGG16.
*   **Thiết kế tầng Classifier bổ sung**:
    *   `Rescaling(255.0)`: Đảm bảo hình ảnh được đưa về thang điểm $[0, 255]$ trước khi đi vào EfficientNet.
    *   `GlobalAveragePooling2D()`: Giảm chiều không gian đặc trưng về vector 1D phẳng.
    *   `Dropout(0.5)`: Áp dụng tỷ lệ bỏ rơi 50% để hạn chế tối đa hiện tượng quá khớp (overfitting) trên tập dữ liệu nhỏ.
    *   `Dense(8, activation="softmax")`: Tầng kết nối đầy đủ đầu ra để đưa ra xác suất phân bố cho 8 hãng xe đích.
*   **Trạng thái huấn luyện**: Đóng băng toàn bộ các lớp tích chập của EfficientNet-B0, chỉ huấn luyện lại tầng Classifier tùy chỉnh ở phía cuối.

### 1.2. Bộ phân loại Màu sắc xe (Vehicle Color Classifier)
*   **Backbone**: **MobileNetV3-Small**
*   **Lý do lựa chọn**: Cực kỳ mỏng nhẹ (~6.5 MB), tối ưu hóa tốc độ xử lý phần cứng biên, đặc biệt thích hợp phân loại đặc trưng trực quan đơn giản như màu sắc xe.
*   **Thiết kế tầng Classifier bổ sinh**:
    *   `Rescaling(scale=2.0, offset=-1.0)`: Đưa dải điểm ảnh từ $[0, 1]$ về $[-1, 1]$ phù hợp với cấu hình đầu vào chuẩn của MobileNetV3.
    *   `GlobalAveragePooling2D()`
    *   `Dropout(0.3)`
    *   `Dense(8, activation="softmax")`: Phân loại 8 màu sắc xe cơ bản.

---

## 2. Quá trình Huấn luyện và Siêu tham số (Training Process & Hyperparameters)
Việc huấn luyện được thực thi thông qua script [train.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/train.py) sử dụng TensorFlow/Keras trên thiết bị CPU của macOS.

### Cấu hình siêu tham số huấn luyện (Hyperparameter Config):
*   **Bộ tối ưu hóa (Optimizer)**: Adam với tốc độ học ban đầu $\eta = 10^{-4}$ giúp kiểm soát quá trình hội tụ mượt mà, tránh dao động loss lớn.
*   **Hàm mất mát (Loss Function)**: Categorical Crossentropy.
*   **Batch Size**: 32.
*   **Số lượng Epochs tối đa**: 10.
*   **Callbacks tích hợp**:
    *   `EarlyStopping`: Theo dõi `val_loss`, dừng sớm sau 5 epochs không cải thiện và tự động khôi phục lại bộ trọng số tốt nhất.
    *   `ModelCheckpoint`: Tự động lưu trữ file weights tốt nhất dưới định dạng `.keras`.

---

## 3. Kết quả Huấn luyện thực nghiệm (Experimental Training Results)
Nhóm đã huấn luyện mô hình trực tiếp trên tập dữ liệu thực tế được chuẩn bị ở Giai đoạn 1. Kết quả thu được phản ánh rõ ràng đặc thù dữ liệu thực nghiệm cục bộ:

*   **Brand Classifier (EfficientNet-B0)**:
    *   Đạt độ chính xác Validation (val_accuracy) khoảng **29.00%**.
    *   *Đánh giá*: Độ chính xác còn hạn chế do sự phân bố ảnh xe trong bộ dữ liệu Stanford Cars chủ yếu là xe hơi chụp tại góc nhìn ngang của Mỹ, trong khi ảnh kiểm thử biển số tại Việt Nam có góc chụp cận cảnh từ phía sau hoặc phía trước. Thêm vào đó, việc hạn chế số epoch huấn luyện cục bộ (10 epochs) trên CPU để tránh thời gian chờ quá lâu đã làm giảm khả năng tinh chỉnh sâu.
*   **Color Classifier (MobileNetV3-Small)**:
    *   Đạt độ chính xác Validation (val_accuracy) khoảng **14.16%**.
    *   *Đánh giá*: Do bộ phân loại màu sắc chịu ảnh hưởng rất mạnh bởi sự cân bằng trắng của camera và điều kiện ánh sáng môi trường thực tế (trong hầm tối hoặc ngoài trời). Nhóm ghi nhận có hiện tượng nhầm lẫn mạnh giữa màu Bạc (Silver), Xám (Grey) và Trắng (White).

Các biểu đồ Loss và Accuracy chi tiết ghi lại toàn bộ nhật ký huấn luyện của hai mô hình đã được xuất ra thư mục báo cáo bằng matplotlib:
*   Đồ thị huấn luyện Hãng xe: [brand_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/presentations/evidence/brand_classifier_training_curves.png)
*   Đồ thị huấn luyện Màu xe: [color_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/presentations/evidence/color_classifier_training_curves.png)

> [!TIP]
> Để nâng cao độ chính xác trên môi trường sản xuất, cần mở rộng kích thước tập dữ liệu huấn luyện thật với nhiều góc chụp xe thực tế tại các bãi giữ xe Việt Nam và tăng số lượng epoch fine-tuning lên khoảng 50 epochs trên phần cứng hỗ trợ GPU/TPU.
