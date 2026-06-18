# Report 1: Project Proposal
## Hệ thống chống trộm xe thông minh bằng đối chiếu chéo thông tin xe (Smart Anti-Theft Vehicle Verification System)

**Môn học:** DPL302m – Deep Learning (Học sâu)  
**Trường:** FPT University  

> **🔄 Ghi chú phiên bản (đọc trước):** Đây là **đề xuất ban đầu** (Giai đoạn 1) mô tả hướng *xác thực đa nhân tố (biển số + hãng + màu)* với ResNet50/MobileNetV2/EasyOCR. Trong quá trình thực nghiệm (Report 3) các bộ phân loại hãng (~29%) và màu (~14% lúc đầu) cho thấy quá yếu để chặn cứng, nên **bản giao cuối đã pivot sang quyết định *plate-primary*** (OCR là khoá chính bằng **PaddleOCR**, màu chỉ là **cảnh báo mềm**, bỏ phân loại hãng), và đổi backbone sang **EfficientNet-B0/MobileNetV3-Small**. Xem hành trình & lý do trong Report 3 và Report 4.

---

## 1. Đặt vấn đề (Problem Statement)

### 1.1 Tầm quan trọng của Xác thực bãi đỗ xe (Parking Authentication)
Trong xu hướng phát triển đô thị thông minh (Smart City), quản lý và kiểm soát xe ra vào bãi đỗ (Parking Access Control & Authentication) đóng vai trò sống còn trong việc đảm bảo an ninh đô thị, vận hành tòa nhà và bảo vệ tài sản công cộng/tư nhân. Một quy trình xác thực đỗ xe (Parking Auth) chuẩn mực yêu cầu phải nhận diện nhanh chóng, chính xác danh tính của phương tiện tại hai thời điểm: Check-in (đầu vào) và Check-out (đầu ra).

### 1.2 Hạn chế của hệ thống xác thực đơn nhân tố (Single-Factor Parking Auth) hiện tại
Các bãi đỗ xe thông minh phổ biến hiện nay tại Việt Nam chủ yếu dựa vào hệ thống nhận diện biển số xe tự động (ANPR - Automatic Number Plate Recognition). Quy trình này tồn tại một lỗ hổng bảo mật chí mạng:
- **Xác thực đơn nhân tố (Single-Factor Authentication):** Hệ thống chỉ dựa vào duy nhất chuỗi ký tự biển số xe làm định danh (Identity Token) để mở barie.
- **Biển số xe dễ bị giả mạo / đánh tráo:** Biển số là thông tin hiển thị công khai, kẻ gian có thể dễ dàng in ấn biển giả hoặc thực hiện hành vi **Hoán đổi biển số xe (Plate Swapping / Identity Spoofing)** từ xe hợp pháp sang xe bất hợp pháp.
- **Rủi ro an ninh cao:** Khi hệ thống ANPR phê duyệt xác thực sai (False Grant) cho một biển số bị hoán đổi, kẻ gian sẽ dễ dàng đánh cắp phương tiện ra khỏi bãi đỗ hoặc đưa các phương tiện không được phép vào các khu vực an ninh cao.

### 1.3 Giải pháp đề xuất: Xác thực bãi xe đa nhân tố (Multi-Factor Parking Authentication)
Để giải quyết triệt để lỗ hổng trên, dự án này đề xuất xây dựng hệ thống **Xác thực xe đa nhân tố (Multi-Factor Parking Auth)** thời gian thực, áp dụng Deep Learning để đồng thời trích xuất và đối chiếu chéo 3 đặc trưng độc lập của phương tiện:

1. **Biển số xe (License Plate):** Sử dụng YOLOv8 để phát hiện vị trí biển số và OCR để đọc chuỗi ký tự.
2. **Nhãn hiệu / Dòng xe (Car Brand):** Sử dụng mạng CNN (ResNet50 với Transfer Learning) để phân loại nhãn hiệu xe (Toyota, Hyundai, VinFast, v.v.).
3. **Màu sắc xe (Car Color):** Sử dụng mạng CNN (MobileNetV2) để phân loại màu sắc chủ đạo của xe.

Khi xe đi vào bãi đỗ (Check-in), hệ thống tự động ghi nhận bộ ba thông tin `{Biển số, Nhãn hiệu, Màu sắc}` vào cơ sở dữ liệu CSV. Khi xe đi ra (Check-out), hệ thống quét lại toàn bộ ba thông tin và **đối chiếu chéo** với bản ghi đã lưu. Nếu bất kỳ yếu tố nào không khớp (ví dụ: biển số đúng nhưng nhãn hiệu xe hoặc màu sắc xe khác), hệ thống sẽ:

- Từ chối mở barie.
- Phát âm thanh cảnh báo (Alarm).
- Hiển thị cảnh báo trên giao diện giám sát kèm chi tiết lý do sai lệch.

### 1.4 Các nghiên cứu và dự án liên quan (Related Work / Literature Review)
Để thiết kế một hệ thống tối ưu và khoa học, dự án kế thừa và tham khảo các nghiên cứu tiêu biểu sau:

1. **Nghiên cứu tích hợp ALPR và FGVC (Fine-Grained Vehicle Classification):**
   - *Tài liệu tham khảo:* Lima, G. E., et al. (2026). *"Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition."* arXiv preprint arXiv:2604.05271.
   - *Tóm tắt & Ứng dụng:* Bài báo này phát triển bộ dữ liệu UFPR-VeSV gồm 24,945 ảnh xe thực tế được gán nhãn 13 màu sắc, 26 hãng xe, 136 model và 14 loại xe. Nghiên cứu đề xuất việc huấn luyện song song và đồng bộ hóa các bộ phân loại đặc trưng của xe với mô hình nhận diện biển số (ALPR) để nâng cao độ chính xác của hệ thống giám sát. Dự án của chúng em kế thừa ý tưởng tích hợp luồng xử lý song song các đặc trưng này để tăng cường tính bảo mật.
2. **Nghiên cứu phân loại xe không phụ thuộc góc nhìn (View-Independent VMMR):**
   - *Tài liệu tham khảo:* Hu, C., et al. (2017). *"View Independent Vehicle Make, Model and Color Recognition Using Convolutional Neural Network."* arXiv preprint arXiv:1702.01721.
   - *Tóm tắt & Ứng dụng:* Nghiên cứu trình bày phương pháp sử dụng mạng tích chập sâu (CNN) để phân loại hãng xe, mẫu xe và màu sắc độc lập với góc nhìn của camera giám sát (view-independent). Điều này chứng minh rằng việc áp dụng các mạng pre-trained sâu như ResNet50 có khả năng học các đặc trưng hình học phức tạp tốt hơn các mạng CNN nông tự thiết kế. Dự án ứng dụng kết quả này bằng việc sử dụng phương pháp Transfer Learning trên mạng ResNet50 và MobileNetV2 pre-trained.
3. **Các dự án kiểm soát bãi đỗ xe thông minh công nghiệp (MMR Access Control):**
   - *Tổng quan thực tiễn:* Các hệ thống kiểm soát xe ra vào hiện đại tại các nước phát triển sử dụng công nghệ Make, Model, and Color Recognition (MMR) để bổ trợ cho ANPR. Khi có sự sai lệch giữa biển số xe và các đặc trưng hình dáng/màu sắc thực tế, hệ thống sẽ chặn barie và báo động để ngăn chặn hành vi tráo biển số (Plate Swapping) hoặc trộm cắp. Dự án của nhóm hiện thực hóa quy trình này thông qua module so khớp `matching.py` và cơ chế cảnh báo thời gian thực.

---

## 2. Mục tiêu dự án (Project Objectives)

### 2.1 Mục tiêu chính
- Xây dựng một hệ thống hoàn chỉnh có khả năng phát hiện hành vi giả mạo biển số xe ô tô trong thời gian thực (Real-time).
- Đạt tỷ lệ phát hiện đúng xe giả mạo/hoán đổi biển số **≥ 95%** trên tập dữ liệu kiểm thử.

### 2.2 Mục tiêu phụ
- Xử lý mỗi phương tiện trong thời gian **dưới 1.0 giây** (bao gồm phát hiện, nhận diện, phân loại và đối chiếu).
- Xây dựng giao diện Web Dashboard trực quan bằng Streamlit để demo trực tiếp qua Webcam hoặc file Video.
- Đáp ứng đầy đủ các Chuẩn đầu ra môn học (CLOs): CLO1 (xây dựng mạng neural), CLO2 (tối ưu hóa), CLO3 (chiến lược ML), CLO4 (CNN), CLO6 (pipeline dự án hoàn chỉnh), CLO7 (ứng dụng AI tools).

---

## 3. Kiến trúc hệ thống (System Architecture)

### 3.1 Sơ đồ luồng xử lý tổng quan

```
┌──────────────────┐
│  VIDEO INPUT     │  Camera CCTV / Webcam / File Video
│  (Frame Stream)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  BƯỚC 1: PHÁT HIỆN XE & BIỂN SỐ     │
│  YOLOv8 Object Detection            │
│  → Phát hiện bounding box xe ô tô   │
│  → Phát hiện bounding box biển số    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG (Chạy song song)                │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ OCR Engine      │  │ Brand Classifier │  │ Color Classif. │  │
│  │ (PaddleOCR /    │  │ (ResNet50 +      │  │ (MobileNetV2 + │  │
│  │  EasyOCR)       │  │  Transfer Learn) │  │  Transfer Lrn) │  │
│  │                 │  │                  │  │                │  │
│  │ Input: Ảnh crop │  │ Input: Ảnh crop  │  │ Input: Ảnh crop│  │
│  │ biển số         │  │ toàn bộ xe       │  │ toàn bộ xe     │  │
│  │                 │  │                  │  │                │  │
│  │ Output:         │  │ Output:          │  │ Output:        │  │
│  │ "30F12345"      │  │ "Toyota Vios"    │  │ "White"        │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  BƯỚC 3: ĐỐI CHIẾU CHÉO            │
│  Cross-Verification Logic Matcher    │
│                                      │
│  Tra cứu biển số "30F12345" trong    │
│  database.csv:                       │
│  → Đăng ký: Toyota Vios, White      │
│  → Thực tế: Toyota Vios, White      │
│  → Kết quả: ✅ AUTHORIZED           │
│                                      │
│  HOẶC:                               │
│  → Đăng ký: Toyota Vios, White      │
│  → Thực tế: Hyundai Accent, Red     │
│  → Kết quả: 🚨 MISMATCH - ALERT!   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  BƯỚC 4: GIAO DIỆN ĐIỀU KHIỂN       │
│  Streamlit Web Dashboard             │
│                                      │
│  ✅ AUTHORIZED → Banner XANH,       │
│     mở barie tự động                 │
│                                      │
│  🚨 MISMATCH → Banner ĐỎ nhấp      │
│     nháy, phát còi báo động,         │
│     hiển thị lý do sai lệch         │
└──────────────────────────────────────┘
```

### 3.2 Mô tả chi tiết từng thành phần

| Thành phần | Công nghệ | Đầu vào | Đầu ra |
|:---|:---|:---|:---|
| **Plate Detector** | YOLOv8-nano (Ultralytics) | Frame video gốc | Tọa độ bounding box biển số xe |
| **OCR Engine** | PaddleOCR hoặc EasyOCR | Ảnh biển số đã crop | Chuỗi ký tự biển số (ví dụ: `30F12345`) |
| **Brand Classifier** | EfficientNet-B0 (TensorFlow/Keras) *(đề xuất ban đầu: ResNet50 — đã thay bằng EfficientNet-B0 trong bản giao cuối; xem ghi chú phiên bản)* | Ảnh toàn bộ xe đã crop | Nhãn hiệu xe (ví dụ: `Toyota Vios`) — *(diagnostic only — đã loại khỏi quyết định)* |
| **Color Classifier** | MobileNetV3-Small (TF/Keras, training/eval) → **PyTorch MobileNetV3-Small** (runtime/inference) *(đề xuất ban đầu: MobileNetV2 — đã thay bằng MobileNetV3-Small trong bản giao cuối; xem ghi chú phiên bản)* | Ảnh toàn bộ xe đã crop | Màu sắc xe (ví dụ: `White`) — cảnh báo mềm |
| **Logic Matcher** | Python + Pandas | Bộ ba {Biển số, Nhãn hiệu, Màu sắc} | Trạng thái: AUTHORIZED / MISMATCH / UNREGISTERED |
| **Web Dashboard** | Streamlit | Kết quả từ Logic Matcher | Giao diện trực quan thời gian thực |

---

## 4. Công nghệ sử dụng (Technology Stack)

### 4.1 Ngôn ngữ lập trình
- **Python 3.10+**: Ngôn ngữ chính cho toàn bộ dự án.

### 4.2 Deep Learning Frameworks
- **PyTorch + Ultralytics:** Huấn luyện và chạy inference mô hình YOLOv8 cho phát hiện biển số xe.
- **TensorFlow / Keras:** Huấn luyện các bộ phân loại nhãn hiệu xe và màu sắc xe (đáp ứng yêu cầu framework của Syllabus DPL302m).

### 4.3 Thư viện hỗ trợ
- **PaddleOCR / EasyOCR:** Nhận diện ký tự trên ảnh biển số xe.
- **OpenCV:** Đọc, xử lý và hiển thị frame video.
- **NumPy, Pandas:** Xử lý dữ liệu số và quản lý cơ sở dữ liệu CSV.
- **Matplotlib, Seaborn:** Trực quan hóa kết quả EDA và biểu đồ đánh giá mô hình.

### 4.4 Công cụ triển khai
- **Streamlit:** Xây dựng giao diện Web Dashboard tương tác thời gian thực.
- **Jupyter Notebook:** Phân tích khám phá dữ liệu (EDA) và prototype mô hình.

---

## 5. Chỉ số đo lường hiệu năng (Key Performance Indicators)

| Chỉ số | Mô tả | Mục tiêu |
|:---|:---|:---|
| **mAP@0.5 (YOLOv8)** | Trung bình Precision-Recall của phát hiện biển số | ≥ 90% |
| **OCR Word Accuracy** | Tỷ lệ đọc đúng toàn bộ chuỗi ký tự biển số | ≥ 90% |
| **Brand Classification Accuracy** | Tỷ lệ phân loại đúng nhãn hiệu xe | ≥ 85% |
| **Color Classification Accuracy** | Tỷ lệ phân loại đúng màu sắc xe | ≥ 92% |
| **Fake Plate Detection Rate** | Tỷ lệ phát hiện đúng xe giả mạo/hoán đổi biển số trên tập test | ≥ 95% |
| **False Alarm Rate** | Tỷ lệ xe hợp lệ bị cảnh báo nhầm | ≤ 5% |
| **End-to-End Latency** | Tổng thời gian xử lý 1 phương tiện (từ frame đến quyết định) | < 1.0 giây |

---

## 6. Dữ liệu sử dụng (Data Strategy)

### 6.1 Chiến lược đa dạng hóa dữ liệu (Multi-Source Data Strategy)
Để đảm bảo mô hình có tính thực tiễn cao khi triển khai tại các bãi đỗ xe Việt Nam, dự án sử dụng chiến lược kết hợp dữ liệu từ nhiều nguồn khác nhau (Kaggle, Hugging Face, Roboflow Universe và Tự thu thập/Scraping):

| Tác vụ / Thành phần | Bộ dữ liệu | Nguồn | Số lượng & Vai trò thực tế |
|:---|:---|:---|:---|
| **Phát hiện biển số & OCR** | Vietnamese License Plates | [Kaggle](https://www.kaggle.com/datasets/datnguyen1111/vietnamese-car-license-plate-detection) | Baseline dataset: 1,000+ ảnh xe và biển số xe ô tô tại Việt Nam. |
| | Vietnamese Car License Plate | [Roboflow Universe](https://universe.roboflow.com/search?q=Vietnamese%20Car%20License%20Plate) | 5,000+ ảnh chụp xe thực tế tại Hà Nội/TP.HCM dưới trời mưa, ánh sáng kém, góc chụp xiên. |
| | license-plate-detection | [Hugging Face (UniDataPro)](https://huggingface.co/datasets/UniDataPro/license-plate-detection) | Tập dữ liệu lớn bổ trợ nhận diện biển số đa quốc gia (bao gồm Việt Nam) giúp mô hình tổng quát hóa tốt hơn. |
| **Phân loại Hãng xe** | Stanford Cars Dataset | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) | 16,185 ảnh của 196 dòng xe thông dụng trên thế giới. |
| | vehicle-classification | [Hugging Face (DrBimmer)](https://huggingface.co/datasets/DrBimmer/vehicle-classification) | Bổ sung phân loại hãng xe và kiểu dáng xe thực tế. |
| | Custom VinFast Dataset | Web Scraping (Google/Bing) & Tự chụp | ~500 ảnh các dòng xe VinFast (VF5, VF8, VF9, VFe34) đang phổ biến tại Việt Nam nhưng thiếu trong các bộ dataset quốc tế. |
| **Phân loại Màu sắc** | Car Color Recognition | [Kaggle](https://www.kaggle.com/datasets/landrykezebou/car-color-recognition-dataset) | 15,000+ ảnh xe được gán nhãn 8 màu cơ bản (Trắng, Đen, Đỏ, Bạc, Xanh, v.v.). |
| | vehicle-color | [Hugging Face (M4)](https://huggingface.co/datasets/HuggingFaceM4/vehicle-color) | Bổ sung hình ảnh xe đa dạng góc chụp và độ bão hòa màu để tăng độ ổn định của phân loại màu xe. |

### 6.2 Phương pháp tăng cường dữ liệu (Augmentation)
- Thay đổi độ sáng ngẫu nhiên (giả lập điều kiện ban ngày/ban đêm).
- Xoay ảnh nhẹ (±15°) để giả lập góc chụp camera khác nhau.
- Thêm nhiễu Gaussian để giả lập chất lượng camera thấp.
- Điều chỉnh độ tương phản và bão hòa màu.

---

## 7. Kế hoạch thực hiện (Project Timeline)

| Giai đoạn | Công việc chính | Session tương ứng |
|:---|:---|:---|
| **Giai đoạn 1** | Đề xuất dự án, thiết kế kiến trúc, lựa chọn công nghệ | Session 8 – 18 |
| **Giai đoạn 2** | Thu thập dữ liệu, tiền xử lý, EDA | Session 28 – 33 |
| **Giai đoạn 3** | Huấn luyện mô hình, tinh chỉnh siêu tham số, đánh giá | Session 43 – 57 |
| **Giai đoạn 4** | Tích hợp hệ thống, xây dựng Demo, bảo vệ dự án | Session 104 – 119 |

---

## 8. Quản lý rủi ro (Risk Management)

| Rủi ro | Mức độ | Giải pháp dự phòng |
|:---|:---|:---|
| Dữ liệu biển số Việt Nam không đủ đa dạng | Trung bình | Bổ sung dữ liệu bằng cách chụp thêm ảnh thực tế ngoài đường hoặc scraping Google Images |
| Mô hình OCR đọc sai ký tự do ảnh bị mờ/nghiêng | Cao | Áp dụng tiền xử lý ảnh (chuyển grayscale, thresholding, deskew) trước khi đưa vào OCR |
| Overfitting trên tập train nhỏ | Trung bình | Sử dụng Dropout, BatchNormalization, Early Stopping và Data Augmentation |
| Tốc độ xử lý chậm khi chạy nhiều mô hình cùng lúc | Thấp | Sử dụng mô hình nhẹ (YOLOv8-nano, MobileNetV3-Small) và tối ưu bằng threading *(đề xuất ban đầu: MobileNetV2)* |
| Không có phần cứng GPU mạnh tại local | Trung bình | Huấn luyện mô hình trên môi trường Google Colab miễn phí (sau đó tải file model .pt/.h5 về chạy), đồng thời tối ưu hóa sử dụng các mô hình gọn nhẹ (YOLOv8-nano, MobileNetV3-Small) để chạy suy luận (inference) mượt mà trên CPU máy tính cá nhân. *(Đề xuất ban đầu: MobileNetV2 — bản giao cuối dùng MobileNetV3-Small)* |

---

## 9. Mapping CLOs (Course Learning Outcomes)

| CLO | Nội dung | Áp dụng trong dự án |
|:---|:---|:---|
| **CLO1** | Xây dựng và huấn luyện mạng neural fully-connected | Các tầng Dense cuối cùng của EfficientNet-B0 và MobileNetV3-Small classifiers (đề xuất ban đầu: ResNet50/MobileNetV2 — đã thay bằng kiến trúc giao nộp thực tế) |
| **CLO2** | Phân tích bias/variance, tối ưu hóa, sử dụng TensorFlow | Hyperparameter Tuning, Regularization (Dropout, BatchNorm), sử dụng tf.keras |
| **CLO3** | Chẩn đoán lỗi trong hệ thống ML, Transfer Learning | Error Analysis trên Confusion Matrix, Transfer Learning từ ImageNet |
| **CLO4** | Xây dựng CNN cho phát hiện và nhận dạng hình ảnh | YOLOv8 Object Detection, EfficientNet-B0 (Brand), MobileNetV3-Small (Color) |
| **CLO6** | Thực hiện pipeline dự án DL hoàn chỉnh | Data Collection → Wrangling → EDA → Model Dev → Evaluation → Reporting |
| **CLO7** | Ứng dụng AI tools để hoàn thành dự án | Sử dụng PaddleOCR/EasyOCR, Streamlit, Keras Tuner |

---

## 10. Tài liệu tham khảo (References)

1. Lima, G. E., Nascimento, V., Santos, E., Nascimento Jr., E., Laroca, R., & Menotti, D. (2026). *Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition*. arXiv preprint arXiv:2604.05271.
2. Hu, C., Bai, X., Qi, L., Wang, P., Shen, G., & Wang, J. (2017). *View Independent Vehicle Make, Model and Color Recognition Using Convolutional Neural Network*. arXiv preprint arXiv:1702.01721.
3. Adaptive Recognition. (2024). *Make and Model Recognition (MMR) in Parking Access Control and Security Systems*. Retrieved from https://adaptiverecognition.com/
