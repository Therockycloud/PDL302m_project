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

### 5.1. Mô hình phân loại màu sắc xe (Color Classifier — MobileNetV3-Small)

> **Lưu ý runtime:** Mô hình TF/Keras (`color_classifier.keras`) chỉ dùng cho **training và đánh giá** ở giai đoạn đầu. Mô hình **runtime/inference hiện tại (đã deploy)** là **PyTorch MobileNetV3-Small** (`main/data/models/color_MobileNetV3Small.pt`, được nạp bởi `main/src/models/torch_color.py`), fine-tune trên Google Colab (GPU) bằng `main/scripts/colab_train_color.py`. Kết quả đánh giá dưới đây được đo lại (tái lập, không phải số Colab tự báo) trên tập test giữ-riêng bằng script `main/scripts/eval_color_deployed.py`, chạy trực tiếp trên file `.pt` đang chạy ở runtime.

**Hành trình cải tiến (journey):** ~55% (frozen-backbone, data cũ, 783 ảnh project) → ~78% (full fine-tune + bổ sung dữ liệu VCoR, 600 ảnh/lớp) → **~86% (deploy)** nhờ full VCoR (5,881 ảnh hợp lệ) + ba lever bổ sung: class-weighted loss, label smoothing 0.1, test-time augmentation (TTA, hflip).

| Cấu hình | Test accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Head đông cứng (frozen, data cũ) | 48.3% | 0.48 |
| Fine-tuning nửa backbone (data cũ, 783 ảnh) | 55.1% | 0.545 |
| Full fine-tune + VCoR 600/lớp (chưa TTA) | 77.6% | 0.776 |
| **Full fine-tune + VCoR full (5,881 ảnh) + class-weight + label-smoothing — plain** | 85.3% | 0.829 |
| **+ Test-time augmentation (TTA, hflip) — DEPLOYED** | **86.3%** | **0.841** |

Methodology: dataset = **VCoR** (Kaggle, Vehicle Color Recognition — ~5,881 ảnh dùng được trên 8 lớp màu, gộp cùng ảnh gốc của nhóm) + full fine-tune toàn bộ backbone MobileNetV3-Small (không đông cứng) + class-weighted loss (bù lệch lớp Silver/Grey hiếm) + label smoothing 0.1 + test-time augmentation (trung bình softmax ảnh gốc + ảnh lật ngang) + body-crop tiền xử lý (bỏ 20% trên/15% dưới ảnh — loại trời/đường). Đánh giá trên tập test giữ-riêng (held-out), stratified split 70/15/15, seed=42 — **889 ảnh test**, chưa từng thấy khi huấn luyện. Mô hình màu vẫn được dùng làm **cảnh báo mềm** trong runtime (xem Report 4), không từ chối cứng.

> **Lưu ý trung thực (domain gap):** 86% là đo trên VCoR — ảnh xe chụp rõ, nền sạch, ánh sáng tốt (nguồn web/marketplace), **không phải ảnh CCTV bãi xe thật**. Hiệu năng khi triển khai trên camera giám sát thực tế nhiều khả năng **thấp hơn** do khác biệt domain (ánh sáng yếu/ngược sáng, nhiễu, góc nghiêng, độ phân giải thấp). Để bền hơn khi triển khai thật cần bổ sung tiền xử lý white-balance và một lượng nhỏ dữ liệu CCTV thật để fine-tune thêm.

*   **Script đánh giá tái lập**: `main/scripts/eval_color_deployed.py` (tái sử dụng hàm load/split/eval từ `main/scripts/colab_train_color.py`).
*   **Báo cáo đầy đủ (JSON + Markdown)**: `docs/benchmarks/color_finetune_report.json`, `docs/benchmarks/color_finetune_report.md`.
*   **Biểu đồ huấn luyện (giai đoạn đầu, TF/Keras, data cũ)**: [color_classifier_training_curves.png](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/data/models/color_classifier_training_curves.png)

#### Ma trận nhầm lẫn & hiệu năng theo lớp (Colour Classifier — model deployed, tập test VCoR 889 ảnh, TTA)

| Lớp | Precision | Recall | Support |
| :--- | :---: | :---: | :---: |
| Black | 0.840 | 0.773 | 88 |
| Blue | 0.955 | 0.924 | 159 |
| Brown | 0.980 | 0.803 | 122 |
| Grey | 0.597 | 0.763 | 93 |
| Red | 0.978 | 0.993 | 137 |
| Silver | 0.632 | 0.615 | 78 |
| White | 0.808 | 0.874 | 87 |
| Yellow | 0.976 | 0.984 | 125 |
| **Macro avg** | **0.846** | **0.841** | **889** |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán, dự đoán TTA):

| | Black | Blue | Brown | Grey | Red | Silver | White | Yellow |
|---|---|---|---|---|---|---|---|---|
| Black | 68 | 4 | 0 | 15 | 0 | 1 | 0 | 0 |
| Blue | 4 | 147 | 0 | 3 | 0 | 3 | 1 | 1 |
| Brown | 5 | 1 | 98 | 12 | 2 | 1 | 1 | 2 |
| Grey | 3 | 1 | 1 | 71 | 1 | 14 | 2 | 0 |
| Red | 0 | 0 | 1 | 0 | 136 | 0 | 0 | 0 |
| Silver | 0 | 1 | 0 | 15 | 0 | 48 | 14 | 0 |
| White | 1 | 0 | 0 | 1 | 0 | 9 | 76 | 0 |
| Yellow | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 123 |

*Nhận xét: Red/Blue/Yellow/Brown rất mạnh (F1 ≥ 0.88), màu sắc nổi bật và ít nhập nhằng. **Grey và Silver vẫn là cặp khó nhất** (Grey recall 0.763 nhưng precision chỉ 0.597 — bị Black và Silver lẫn vào; Silver precision/recall ~0.62) — ba màu trung tính (Black/Grey/Silver/White) là cụm nhập nhằng cố hữu về sắc tố, đúng như dự đoán trong các phiên thử nghiệm trước. Kết quả này củng cố vai trò "cảnh báo mềm" — không dùng màu để từ chối cứng.*

### 5.2. Mô hình phân loại hãng xe (Brand Classifier — EfficientNet-B0, diagnostic only — đã loại khỏi quyết định)
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
1.  **Thu thập dữ liệu in-domain**: quay và cắt ảnh xe trực tiếp tại bãi giữ xe Việt Nam với góc máy cố định để khớp phân phối train/test — đây là hướng còn lại quan trọng nhất, vì model màu hiện đo 86% trên VCoR (ảnh web sạch) chứ chưa có số đo trên CCTV thật.
2.  **Bộ phân loại màu đã đạt mục tiêu (~86% trên VCoR, xem §5.1)** nhờ full fine-tune + dữ liệu VCoR + class-weight/label-smoothing/TTA; hướng tiếp theo không phải "fine-tune sâu hơn" mà là **thu hẹp domain gap**: thêm white-balance tiền xử lý + một lượng nhỏ dữ liệu CCTV thật để fine-tune tiếp, và xử lý riêng cụm nhập nhằng Grey/Silver/Black/White (vẫn là điểm yếu nhất, xem confusion matrix §5.1).
3.  **Cân bằng & mở rộng dữ liệu hãng xe (Brand)**: nâng các lớp hiếm (Brown/Yellow tương ứng phía màu, hoặc các hãng ít ảnh phía brand) vượt ~100 ảnh và bổ sung biến thể ánh sáng bãi xe.
