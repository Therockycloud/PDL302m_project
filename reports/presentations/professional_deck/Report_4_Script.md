# Kịch bản thuyết trình bảo vệ đồ án (Report 4 Oral Script)

Tài liệu này chứa nội dung thuyết trình chi tiết bằng cả tiếng Việt và tiếng Anh cho từng slide của bài báo cáo bảo vệ đồ án cuối khóa (Report 4) môn Học Sâu (DPL302m).

---

## Slide 1: Slide tiêu đề (Index Cover - SWISS-COVER-ASCII)
*   **Visual:** Tiêu đề dự án "Tích hợp hệ thống & Đánh giá đầu cuối", thông tin giảng viên hướng dẫn (Thầy Lương Trung Kiên), thành viên thực hiện và hiệu năng tóm tắt trên nền ASCII breathing canvas.

### 🇻🇳 Lời thoại tiếng Việt
> "Kính thưa thầy Lương Trung Kiên cùng toàn thể các bạn. Hôm nay, nhóm chúng em xin phép đại diện trình bày báo cáo bảo vệ đồ án cuối kỳ môn Học Sâu (DPL302m). Đề tài báo cáo của chúng em là: **'Tích hợp hệ thống & Đánh giá đầu cuối - Hệ thống giám sát an ninh bãi xe thông minh qua đối chiếu chéo thuộc tính xe'**. 
>
> Hệ thống được thiết kế khép kín, hoạt động ngoại tuyến 100% trên CPU bãi giữ xe, tích hợp các mô hình YOLOv8, PaddleOCR và MobileNetV3 dưới sự điều phối của FastAPI và Streamlit. Để bảo đảm an toàn vận hành thực tế, nhóm đã áp dụng cơ chế đối chiếu Plate-Primary nhằm tối ưu hiệu năng và độ tin cậy đầu cuối."

### 🇬🇧 English Script
> "Good morning, Mr. Luong Trung Kien and everyone. Today, on behalf of our team, we would like to present our final defense report for the Deep Learning course (DPL302m), titled: **'System Integration & End-to-End Evaluation - Smart Parking Security System via Cross-Verification of Vehicle Attributes'**.
>
> Our system is self-contained and runs 100% offline on standard parking lot CPUs. It integrates YOLOv8, PaddleOCR, and MobileNetV3, orchestrated by FastAPI and Streamlit. To guarantee practical operational safety, we developed a Plate-Primary decision logic to optimize end-to-end performance and reliability."

---

## Slide 2: Vấn đề thực tế (Split Statement - Problem Statement - S03)
*   **Visual:** Bố cục chia đôi (Split Statement - S03) với nền tối bên trái làm nổi bật bài toán thực tiễn, và nền xám bên phải trình bày 3 ràng buộc cốt lõi (Tích hợp 3 mô hình, Chạy CPU/Offline, Quyết định Plate-Primary).

### 🇻🇳 Lời thoại tiếng Việt
> "Đầu tiên, em xin trình bày về động lực nghiên cứu và bài toán thực tế. Các bãi giữ xe truyền thống hiện nay đang đối mặt với lỗ hổng an ninh lớn do phụ thuộc vào xác thực đơn yếu tố, tức là chỉ đối chiếu thẻ từ và chữ số biển số qua OCR tĩnh. Kẻ gian có thể dễ dàng làm giả biển số hoặc tráo đổi biển số giữa các xe cùng loại trước khi qua cổng ra. 
>
> Việc phát hiện thủ công bằng mắt thường của nhân viên bảo vệ là hoàn toàn bất khả thi trước lưu lượng hàng ngàn xe mỗi ngày và dễ gây ùn tắc nghiêm trọng tại cổng kiểm soát vào giờ cao điểm. Vì vậy, nhu cầu tự động hóa đối chiếu chéo các thuộc tính ngoại quan của xe là cực kỳ cấp thiết để ngăn chặn các hành vi gian lận."

### 🇬🇧 English Script
> "First, let us examine our motivation and the practical problem. Traditional parking lots face a major security gap due to their reliance on single-factor verification, which only matches the RFID card and static license plate OCR text. Perpetrators can easily bypass this by mounting fake license plates or swapping plates between similar vehicles before exiting.
>
> Manual visual verification by security guards is highly impractical under the daily load of thousands of vehicles and inevitably causes severe traffic congestion at gates during peak hours. Therefore, automating the cross-verification of external vehicle attributes is essential to prevent fraud."

---

## Slide 3: Khung giải pháp đề xuất (Four Cards - Proposed Framework - S19)
*   **Visual:** Khung giải pháp đối chiếu chéo đa nhân tố gồm 4 khối (OCR biển số làm khóa chính, phân loại màu làm cảnh báo mềm, phân loại hãng xe bị loại bỏ, và đối chiếu cơ sở dữ liệu cục bộ).

### 🇻🇳 Lời thoại tiếng Việt
> "Để khắc phục lỗ hổng an ninh trên, chúng em đề xuất khung giải pháp đối chiếu chéo đa nhân tố. Hệ thống hoạt động dựa trên ba trụ cột kỹ thuật:
>
> 1. **Nhận diện biển số (OCR):** Cắt vùng biển số bằng YOLOv8-nano và giải mã ký tự qua PaddleOCR làm khóa chính.
> 2. **Phân loại màu sắc (Color):** Nhận diện 8 hệ màu cơ bản bằng MobileNetV3-Small để đưa ra cảnh báo mềm khi lệch thông tin.
> 3. **Đối chiếu cơ sở dữ liệu cục bộ:** So khớp tức thời biển số và màu sắc với tệp dữ liệu CSV lưu trữ lịch sử lúc xe vào bãi.
>
> Nhóm cũng đã thử nghiệm bộ phân loại thương hiệu xe EfficientNet-B0 nhưng quyết định loại bỏ thuộc tính này khỏi logic mở barrier do độ chính xác trên ảnh thực tế không đạt yêu cầu vận hành."

### 🇬🇧 English Script
> "To close this security gap, we propose a multi-factor cross-verification framework. The system operates on three technical pillars:
>
> 1. **License Plate OCR:** Extracts the plate region using YOLOv8-nano and decodes characters via PaddleOCR as the primary key.
> 2. **Color Classification:** Identifies 8 base colors using MobileNetV3-Small to trigger soft warnings on mismatch.
> 3. **Local Database Matcher:** Instantly matches the plate and color against a local CSV file storing check-in records.
>
> We also experimented with an EfficientNet-B0 brand classifier but dropped it from the gate decision logic because its accuracy on real-world images was insufficient for daily operations."

---

## Slide 4: Tổng quan nghiên cứu & Cơ sở khoa học (Tech Spec Sheet - Literature Review - S21)
*   **Visual:** Bảng cơ sở khoa học và các bài báo khoa học được trích dẫn (YOLOv8, PaddleOCR CNN+CTC, MobileNetV3, EfficientNet-B0).

### 🇻🇳 Lời thoại tiếng Việt
> "Về cơ sở khoa học, chúng em đã kế thừa và tích hợp các nghiên cứu học sâu tiên tiến nhất để tối ưu hóa tài nguyên phần cứng. 
>
> Đối với phát hiện biển số, chúng em sử dụng mô hình **YOLOv8-nano** của Jocher (2023) vì tốc độ phát hiện thời gian thực vượt trội trên thiết bị biên. Khâu nhận diện ký tự OCR sử dụng **PaddleOCR** với mạng nhận diện chuỗi CNN + CTC theo hướng nghiên cứu của Shi (2015) để mô hình hóa chuỗi ký tự, đem lại độ chính xác vượt trội trên dữ liệu CCTV thực tế. Cuối cùng, mô hình **MobileNetV3-Small** của Howard (2019) được lựa chọn cho phân loại màu sắc nhờ khả năng tối ưu hóa tính toán trên CPU bãi giữ xe."

### 🇬🇧 English Script
> "Regarding the scientific foundation, we integrated state-of-the-art deep learning architectures optimized for edge hardware.
>
> For plate detection, we utilized **YOLOv8-nano** (Jocher, 2023) due to its exceptional real-time inference speed on edge devices. For character recognition, we integrated **PaddleOCR**, which uses a CNN+CTC sequence-recognition model (in the spirit of Shi, 2015) to model character sequences, yielding superior accuracy on real CCTV images. Finally, Howard's **MobileNetV3-Small** (2019) was selected for color classification due to its hardware-friendly computation on standard edge CPUs."

---

## Slide 5: Kiến trúc hệ thống (System Diagram - Overall Architecture - S17)
*   **Visual:** Sơ đồ cấu trúc hệ thống (System Diagram - S17) trực quan hóa luồng dữ liệu giữa Streamlit Frontend, FastAPI Backend và file CSV cục bộ.

### 🇻🇳 Lời thoại tiếng Việt
> "Kiến trúc hệ thống được thiết kế phi tập trung hóa nhằm đạt hiệu năng tối đa và ngăn ngừa hiện tượng nghẽn giao diện. 
>
> *   **Phía Backend (FastAPI):** API Gateway tiếp nhận ảnh chụp từ camera cổng, điều phối bất đồng bộ (asyncio) luồng suy luận của các mô hình học sâu chạy hoàn toàn offline trên CPU và so khớp kết quả với tệp CSV cục bộ.
> *   **Phía Frontend (Streamlit):** Bảng điều khiển hiển thị luồng video, thông tin nhận diện xe, kết quả đối chiếu chéo và phát cảnh báo nhấp nháy đèn đỏ kèm còi rú khi có bất thường. Việc tách biệt này giúp giao diện luôn mượt mà trong khi các tác vụ mô hình nặng đang xử lý."

### 🇬🇧 English Script
> "The system architecture is decoupled to maximize performance and prevent interface freezing.
>
> *   **Backend (FastAPI):** The API Gateway receives gate camera images, orchestrates deep learning model inference asynchronously (via asyncio) running 100% offline on CPU, and matches results against a local CSV file.
> *   **Frontend (Streamlit):** The dashboard displays video streams, recognized vehicle details, cross-verification status, and triggers flashing red lights and siren audio upon mismatches. This decoupling keeps the UI highly responsive while heavy models are evaluating."

---

## Slide 6: Thu thập dữ liệu đa nguồn (Multi-card Brief - Dataset Crawling - S16)
*   **Visual:** Khối thẻ mô tả các nguồn dữ liệu (Stanford Cars, Kaggle VN Plates, VinFast Web Scraping, cào bổ sung màu hiếm) và tỷ lệ phân chia tập dữ liệu.

### 🇻🇳 Lời thoại tiếng Việt
> "Về khâu chuẩn bị dữ liệu, nhóm đã tiến hành thu thập từ nhiều nguồn đa dạng. Chúng em sử dụng tập Stanford Cars gồm 16,185 ảnh, 196 dòng xe (Krause, 2013) để pre-train, và tập Kaggle VN Plates gồm 6,176 ảnh huấn luyện và 1,765 ảnh xác thực biển số Việt Nam để huấn luyện bộ định vị YOLOv8. 
>
> Đối với bài toán phân loại thương hiệu và màu sắc nội địa, nhóm tiến hành cào hình ảnh thực tế các dòng xe VinFast trên mạng và mở rộng dữ liệu màu sang tập VCoR. Tập dữ liệu phân loại sau khi làm sạch gồm màu sắc VCoR 5,881 ảnh và hãng xe 792 ảnh, được phân chia theo tỷ lệ 70% huấn luyện, 15% xác thực, 15% kiểm thử với seed cố định 42."

### 🇬🇧 English Script
> "For data preparation, we gathered images from diverse sources. We leveraged the Stanford Cars dataset containing 16,185 images across 196 car types (Krause, 2013) for pre-training, and the Kaggle VN Plates dataset with 6,176 training and 1,765 validation Vietnamese plate images to train our YOLOv8 detector.
>
> For local brand and color classification, we scraped real-world VinFast vehicle images online and expanded our color data with the VCoR dataset. The final classification dataset comprised 5,881 VCoR color images and 792 brand images, split into 70% training, 15% validation, and 15% testing partitions with a fixed seed of 42."

---

## Slide 7: Pipeline làm sạch & Chuẩn hóa dữ liệu (Horizontal Timeline - Data Cleaning - S11)
*   **Visual:** Sơ đồ pipeline 4 bước làm sạch dữ liệu tự động (Lọc ảnh lỗi đọc, Lọc ngữ nghĩa YOLOv8, Khử trùng lặp pHash, Chuẩn hóa RGB JPEG).

### 🇻🇳 Lời thoại tiếng Việt
> "Để đảm bảo dữ liệu đầu vào có chất lượng cao nhất, nhóm đã xây dựng một đường ống (pipeline) làm sạch dữ liệu tự động qua 4 bước:
>
> *   **Bước 1:** `clean_corrupted.py` quét toàn bộ thư mục và loại bỏ các tệp ảnh bị hỏng định dạng hoặc lỗi đọc file vật lý.
> *   **Bước 2:** `semantic_clean.py` sử dụng YOLOv8 để lọc ngữ nghĩa, tự động loại bỏ 38% số ảnh không chứa xe.
> *   **Bước 3:** `remove_duplicates.py` áp dụng thuật toán mã băm cảm nhận pHash để xóa bỏ các ảnh gần trùng lặp.
> *   **Bước 4:** `normalize_images.py` quy chuẩn hóa toàn bộ hình ảnh về hệ màu RGB và định dạng JPEG tiêu chuẩn."

### 🇬🇧 English Script
> "To ensure the highest input data quality, we developed an automated four-step data cleaning pipeline:
>
> *   **Step 1:** `clean_corrupted.py` scans the directories and purges files with corrupted formats or physical read errors.
> *   **Step 2:** `semantic_clean.py` utilizes YOLOv8 for semantic filtering, automatically discarding 38% of images that do not contain a vehicle.
> *   **Step 3:** `remove_duplicates.py` applies a perceptual hashing (pHash) algorithm to eliminate near-duplicate images.
> *   **Step 4:** `normalize_images.py` standardizes all images into the RGB color space and JPEG format."

---

## Slide 8: Phân tích thống kê & Cân bằng tập dữ liệu (Matrix + Hero Stat - Dataset EDA - S15)
*   **Visual:** Bảng số lượng ảnh test giữ-riêng của 8 màu sắc trên tập VCoR held-out (Black: 88, Blue: 159, Brown: 122, Grey: 93, Red: 137, Silver: 78, White: 87, Yellow: 125 — tổng 889 ảnh), tổng pool huấn luyện VCoR là 5,881 ảnh.

### 🇻🇳 Lời thoại tiếng Việt
> "Dữ liệu màu ban đầu (783 ảnh cào thủ công) cho thấy sự mất cân bằng giữa các lớp màu sắc tự nhiên, đặc biệt là nâu và vàng rất khan hiếm.
>
> Để khắc phục, nhóm đã mở rộng sang tập VCoR công khai với 5,881 ảnh hợp lệ trên 8 lớp màu. Cách tiếp cận mở rộng dữ liệu trong miền này giúp giải quyết triệt để tình trạng lệch lớp dữ liệu mà không cần dùng phương pháp sinh mẫu nhân tạo, đưa độ chính xác từ mức fine-tune ban đầu 55.1% lên 86.3% trên tập test giữ-riêng 889 ảnh."

### 🇬🇧 English Script
> "Our initial color dataset (783 hand-scraped images) revealed a severe class imbalance, with Brown and Yellow being extremely rare.
>
> To resolve this, we expanded to the public VCoR dataset with 5,881 valid images across 8 color classes. This targeted in-domain data expansion resolved the class imbalance without relying on artificial sample generation, raising accuracy from the initial 55.1% fine-tune baseline to 86.3% on the 889-image held-out test split."

---

## Slide 9: Bộ phát hiện biển số YOLOv8 (Image Hero - YOLOv8 Detector - S22)
*   **Visual:** Ảnh minh họa YOLOv8 crop biển số (`img_title_hero.png`), các chỉ số hiệu năng (mAP50: 0.99, độ trễ: 110ms/ảnh CPU, kích thước: 6.2MB).

### 🇻🇳 Lời thoại tiếng Việt
> "Đối với bộ phát hiện biển số (Detector), nhóm tiến hành tự huấn luyện (fine-tune) mô hình YOLOv8-nano trên tập dữ liệu biển số Việt Nam. 
>
> Kết quả thực nghiệm rất khả quan khi độ chính xác định vị mAP50 đạt 0.99, tăng từ mức 0.979 khi huấn luyện từ đầu. Nhờ ưu điểm gọn nhẹ với dung lượng weight chỉ 6.2 MB, mô hình chạy ổn định trên CPU với độ trễ khoảng 110 ms mỗi ảnh, cung cấp các vùng cắt biển số xe máy và ô tô chuẩn xác trước khi đưa vào mô-đun OCR."

### 🇬🇧 English Script
> "For license plate detection, we fine-tuned a YOLOv8-nano model on Vietnamese plate data.
>
> The experimental results are highly positive, achieving a mAP50 of 0.99, up from 0.979 when training from scratch. With a lightweight model size of only 6.2 MB, inference latency on CPU is around 110 ms per image, providing highly accurate plate crops for both cars and motorcycles before passing them to the OCR engine."

---

## Slide 10: Mô hình hóa chuỗi ký tự (Three Layers - PaddleOCR sequence-recognition modeling - S05)
*   **Visual:** Sơ đồ 3 tầng mô hình hóa chuỗi ký tự (Tầng 1: PP-LCNet CNN Backbone, Tầng 2: Recurrent Core, Tầng 3: CTC Transcription).

### 🇻🇳 Lời thoại tiếng Việt
> "Tại khâu nhận diện ký tự OCR, chúng em áp dụng kiến trúc nhận dạng chuỗi của PaddleOCR nhằm đáp ứng chuẩn đầu ra CLO5 về mô hình hóa chuỗi ký tự. Quy trình giải mã gồm ba tầng:
>
> *   **Tầng 1 (CNN Backbone):** Mạng PP-LCNet trích xuất bản đồ đặc trưng từ ảnh biển số và cắt lát thành chuỗi vector từ trái sang phải.
> *   **Tầng 2 (Recurrent Core):** Mạng hồi quy mô hình hóa ngữ cảnh và ghi nhớ mối liên hệ tuần tự của chuỗi ký tự.
> *   **Tầng 3 (Giải mã):** Tầng CTC căn chỉnh chuỗi ký tự ngõ ra mà không cần phân đoạn ký tự thủ công ở mức pixel."

### 🇬🇧 English Script
> "For character recognition, we deployed PaddleOCR's text-recognition architecture to satisfy Course Learning Outcome 5 on sequence modeling. The pipeline operates in three layers:
>
> *   **Layer 1 (CNN Backbone):** PP-LCNet extracts feature maps from the plate crop, slicing them into a sequence of vectors from left to right.
> *   **Layer 2 (Recurrent Core):** A recurrent network captures sequential dependencies and context across the character sequence.
> *   **Layer 3 (Transcription):** A Connectionist Temporal Classification (CTC) layer decodes the sequence without requiring manual pixel-level segmentation."

---

## Slide 11: Sắp xếp không gian & Chuẩn hóa Regex (Three Forces - Spatial Sorting - S13)
*   **Visual:** Ba khối thẻ giải thích giải thuật sắp xếp tọa độ ký tự và chuẩn hóa Regex cho biển số 2 dòng tại Việt Nam.

### 🇻🇳 Lời thoại tiếng Việt
> "Một thử thách đặc trưng ở Việt Nam là biển số dạng vuông 2 dòng trên xe máy và một số dòng ô tô. Nếu đưa trực tiếp vào mô hình OCR thông thường, thứ tự chữ số sẽ bị đảo lộn do góc chụp nghiêng.
>
> Để khắc phục triệt để, chúng em thiết kế giải thuật sắp xếp không gian. Đầu tiên, hệ thống phân chia các hộp ký tự thành hai dòng riêng biệt bằng cách so sánh tọa độ Y trung tâm. Tiếp theo, trong từng dòng, các chữ số được sắp xếp theo tọa độ X tăng dần. Cuối cùng, chuỗi kết quả được chuẩn hóa qua Regex tiếng Việt để loại bỏ dấu chấm và khoảng trắng nhiễu."

### 🇬🇧 English Script
> "A major challenge in Vietnam is that motorcycle and square car license plates are split into two lines. Feeding these crops directly to standard OCR engines results in scrambled text due to minor camera angles.
>
> To solve this, we implemented a custom spatial sorting algorithm. First, the bounding boxes are partitioned into top and bottom lines based on their Y-centroids. Second, within each line, characters are sorted from left to right by their X-coordinates. Finally, a Vietnamese plate Regex sanitizes the string by removing dots and spaces."

---

## Slide 12: Mạng phân loại thuộc tính xe (Six Cells - Vehicle Classifiers - S04)
*   **Visual:** 6 thẻ thông số cấu trúc của 2 mạng phân loại (Backbone, Đóng băng trọng số, Đầu phân loại Dense, Dropout, Loss, Tiền xử lý).

### 🇻🇳 Lời thoại tiếng Việt
> "Đối với bộ phân loại thuộc tính xe, chúng em phát triển hai mạng học chuyển đổi (Transfer Learning) sử dụng MobileNetV3-Small cho màu sắc xe và EfficientNet-B0 cho hãng xe. 
>
> Chúng em đóng băng toàn bộ các lớp tích chập Convolution của backbone pre-trained trên ImageNet để giữ nguyên đặc trưng trích xuất. Ở phần đầu phân loại Dense, chúng em tích hợp lớp Global Average Pooling và các lớp Dense kết hợp với Dropout tỷ lệ 0.3 đối với màu sắc và 0.5 đối với hãng xe nhằm chống quá khớp. Ảnh đầu vào được chuẩn hóa về kích thước 224x224 và scale giá trị pixel về dải [0, 255]."

### 🇬🇧 English Script
> "For vehicle attribute classification, we developed transfer learning networks using MobileNetV3-Small for color and EfficientNet-B0 for brand.
>
> We froze the convolutional layers of the ImageNet pre-trained backbones to preserve feature extraction. In the dense classification head, we appended Global Average Pooling and Dense layers combined with Dropout rates of 0.3 for color and 0.5 for brand to mitigate overfitting. The inputs are standardized to a 224x224 resolution with pixel values normalized to the [0, 255] range."

---

## Slide 13: Chẩn đoán lỗi & Tối ưu hóa (Duo Compare - Diagnostics Keras BatchNorm/Preprocessing - S08)
*   **Visual:** So sánh chẩn đoán lỗi đáp ứng CLO2: Thiết lập cũ Sequential API (Rò rỉ BatchNorm, Flat Loss) vs Giải pháp mới Functional API (Khóa cứng BatchNorm, Hội tụ ổn định).

### 🇻🇳 Lời thoại tiếng Việt
> "Slide này trình bày quá trình chẩn đoán lỗi hệ thống đáp ứng chuẩn đầu ra CLO2. Ban đầu, khi thiết kế bằng Keras Sequential API và đóng băng backbone, chúng em gặp hiện tượng rò rỉ BatchNorm. Ở chế độ suy luận, BatchNorm vẫn chạy chế độ huấn luyện (training=True), kết hợp với việc trùng lặp lớp chuẩn hóa Rescaling đè lên tiền xử lý gốc của MobileNetV3 khiến mô hình hoàn toàn không thể hội tụ (Flat Loss) và độ chính xác kiểm thử đứng yên ở mức ngẫu nhiên 12.5%.
>
> Nhóm đã khắc phục bằng cách chuyển sang Functional API, gọi trực tiếp base model dưới dạng base(x, training=False) để khóa cứng BatchNorm và đồng bộ dải tiền xử lý [0, 255]. Giải pháp này giúp loss giảm ổn định, đưa độ chính xác phân loại màu từ mức ngẫu nhiên lên 55.1%. Đây mới là bước khởi đầu của hành trình — nhóm tiếp tục mở rộng dữ liệu VCoR, class-weight, label-smoothing và test-time augmentation để đẩy độ chính xác màu lên 86.3% ở bản deploy cuối cùng."

### 🇬🇧 English Script
> "This slide demonstrates our system diagnostics process mapping to Course Learning Outcome 2. Initially, using the Keras Sequential API with frozen backbones caused a BatchNorm leakage. During inference, BatchNorm layers ran in training mode (training=True). Combined with a redundant Rescaling layer overriding MobileNetV3's native preprocessing, this led to flat training loss and test accuracy stuck at a random 12.5%.
>
> We resolved this by switching to the Functional API, invoking the base model explicitly as base(x, training=False) to freeze BatchNorm layers, and synchronizing the [0, 255] input range. This fix achieved stable loss convergence and raised color accuracy to 55.1% — only the first milestone. We then scaled up with the full VCoR dataset, class-weighting, label-smoothing, and test-time augmentation, reaching 86.3% in the final deployed model."

---

## Slide 14: Kết quả thực nghiệm & Độ chính xác (H-Bar Chart - Benchmarks and Training curves - S07)
*   **Visual:** Biểu đồ thanh ngang so sánh độ chính xác giữa các mô hình (YOLOv8: 99%, PaddleOCR: 81.2%, Color: 86.3%, Brand: 35.3%, EasyOCR: 0%).

### 🇻🇳 Lời thoại tiếng Việt
> "Đây là kết quả thực nghiệm chi tiết của các mô hình trên tập dữ liệu kiểm thử độc lập. Bộ định vị YOLOv8 đạt kết quả xuất sắc với mAP 99%. Với OCR, PaddleOCR đạt tỷ lệ khớp hoàn toàn 81% trên dữ liệu CCTV thật, trong khi EasyOCR chỉ đạt 0% do không thể xử lý ảnh thực tế từ camera bãi xe. 
>
> Độ chính xác của bộ phân loại màu, sau khi mở rộng dữ liệu VCoR và áp dụng test-time augmentation, đạt 86.3% (macro-F1 0.84) trên bản deploy cuối cùng — đây là một thành phần khá mạnh của hệ thống. Ngược lại, bộ phân loại thương hiệu EfficientNet-B0 chỉ đạt 35.3% do độ mờ ảnh CCTV. Sự chênh lệch này là căn cứ thực tế để chúng em loại hãng xe khỏi quyết định (chỉ giữ vai trò diagnostic), còn màu xe vẫn đóng vai trò cảnh báo mềm — không phải vì model yếu, mà vì 86.3% đo trên ảnh VCoR sạch, chưa kiểm chứng trên domain CCTV bãi xe thật. Hệ thống chuyển sang đối chiếu Plate-Primary để bảo vệ khỏi các lỗi khóa cổng nhầm."

### 🇬🇧 English Script
> "These are the empirical benchmark results of our models on the independent test split. The YOLOv8 plate detector achieved an outstanding 99% mAP. In OCR, PaddleOCR reached an 81% exact-match accuracy on real CCTV images, whereas EasyOCR failed completely at 0% due to its inability to process noisy, low-resolution plate crops.
>
> The color classifier, after scaling up to the full VCoR dataset and applying test-time augmentation, reached 86.3% accuracy (macro-F1 0.84) in the final deployed model — a genuinely strong component. In contrast, the EfficientNet-B0 brand classifier reached only 35.3% due to CCTV blur. This gap justified demoting brand to diagnostic-only, while color still serves as a soft warning signal rather than a hard gate — not because the model is weak, but because the 86.3% was measured on clean VCoR images and has not yet been validated on real CCTV parking-lot footage. The system adopted a Plate-Primary matching logic to prevent false gate locking."

---

## Slide 15: Tích hợp hệ thống & Đồng tiến trình PyTorch (Loop Form - In-Process PyTorch Color Runtime - S14)
*   **Visual:** Sơ đồ vòng lặp tích hợp hệ thống, chạy `torch_color.py` (PyTorch) đồng tiến trình với PaddleOCR để tránh xung đột OpenMP Deadlock, đáp ứng CLO3.

### 🇻🇳 Lời thoại tiếng Việt
> "Trong quá trình tích hợp hệ thống đáp ứng chuẩn đầu ra CLO3, nhóm đã đối mặt với lỗi OpenMP Deadlock nghiêm trọng trên macOS khi chạy song song PyTorch (PaddleOCR) và TensorFlow (MobileNetV3) trên cùng một luồng FastAPI.
>
> Để giải quyết, chúng em chuyển bộ phân loại màu từ TensorFlow/Keras sang PyTorch (`torch_color.py`). PyTorch đồng tồn bình thường với PaddleOCR trong cùng một tiến trình mà không xung đột, nên không cần đến cơ chế cô lập tiến trình hay IPC phức tạp. TF/Keras vẫn được giữ lại nhưng chỉ dùng cho môi trường training/eval cô lập, không chạy ở runtime. Quy trình khép kín hoạt động tuần hoàn: FastAPI nhận ảnh, chạy YOLOv8 định vị biển, PaddleOCR đọc ký tự và PyTorch phân loại màu — tất cả trong cùng một tiến trình — rồi phản hồi sang Streamlit để đóng mở barrier an toàn."

### 🇬🇧 English Script
> "During system integration to satisfy Course Learning Outcome 3, we faced severe OpenMP deadlocks on macOS when running PyTorch (PaddleOCR) and TensorFlow (MobileNetV3) concurrently within the same FastAPI process.
>
> To resolve this, we migrated the color classifier from TensorFlow/Keras to PyTorch (`torch_color.py`). PyTorch coexists with PaddleOCR within the same process without conflict, eliminating the need for process isolation or IPC. TF/Keras is still used, but only in an isolated training/evaluation environment, not at runtime. The pipeline operates in a closed loop: FastAPI receives the image, runs YOLOv8 for plate detection, PaddleOCR for character recognition, and PyTorch for color classification — all within a single process — then responds to Streamlit to safely operate the gate."

---

## Slide 16: Nhật ký vận hành đầu cuối (Stacked KPI Ledger - E2E Logs & Plate-Primary Logic - S20)
*   **Visual:** Bảng nhật ký vận hành đầu cuối gồm thông số độ trễ <1 giây (steady-state) và ba kịch bản đối chiếu (AUTHORIZED, MISMATCH, UNREGISTERED).

### 🇻🇳 Lời thoại tiếng Việt
> "Slide này mô tả kết quả đánh giá đầu cuối hệ thống. Ở chế độ steady-state, độ trễ dưới 1 giây cho mỗi lượt xe: đường bãi đỗ dùng cơ chế approach-lock đo được 0.73 giây từ lúc phát hiện xe tới lúc chốt biển số (đọc trong pha xe đang lùi vào chỗ đỗ, trước khi đỗ hẳn); đường API ảnh đơn đo được khoảng 0.96 giây. Riêng lần gọi đầu tiên có cold-start vài giây do phải nạp PaddleOCR. Độ trễ này hoàn toàn chấp nhận được và không gây ùn tắc vì hệ thống chỉ kích hoạt một lần duy nhất khi xe đã đỗ ổn định trước barrier. 
>
> Logic đối chiếu Plate-Primary xử lý 3 kịch bản: Hợp lệ (trùng khớp biển và màu) sẽ tự động mở cổng, và khi biển bị tráo sang xe khác màu, hệ thống bắt được 69.0% trường hợp (138/200 trial) ở false-alarm chỉ 2.5% (5/200) sau khi siết gate WS-2 — trước đó là 98.5% nhưng false-alarm 14.5% không triển khai được; Không đăng ký (biển số không có trong hệ thống hoặc biển giả) sẽ khóa cứng barrier với tỉ lệ phát hiện 100% (200/200), và cả 5 ảnh test E2E thực tế của nhóm đều rơi đúng vào nhánh này."

### 🇬🇧 English Script
> "This slide details our end-to-end system evaluation. At steady-state, per-vehicle latency is under 1 second: the parking-lot path uses an approach-lock mechanism measured at 0.73 seconds from vehicle detection to plate lock (read while the vehicle is still reversing into the spot, before it fully parks); the single-image API path measures approximately 0.96 seconds. Only the very first call incurs a multi-second cold start while PaddleOCR loads. This latency is acceptable and prevents traffic jams since inference runs only once when a vehicle stops stably at the gate.
>
> The Plate-Primary matcher handles three scenarios: Authorized (matching plate and color) opens the gate, and when a plate is cloned onto a different-colored vehicle, the system catches 69.0% of cases (138/200 trials) at only a 2.5% false-alarm rate (5/200) after the WS-2 gate tightening — versus 98.5% detection but an unworkable 14.5% false-alarm rate before; Unregistered (plate missing from database or fake) locks the barrier with a 100% detection rate (200/200), and all 5 of our real E2E test images fell into exactly this branch."

---

## Slide 17: Kết luận & Lộ trình (Split Closing - Retrospective & Roadmap - S10)
*   **Visual:** Tóm tắt 3 điểm sáng (Hoàn thành đầy đủ CLO, Thực chứng thực tế <1 giây, Định hướng tương lai xác nhận domain CCTV) và lời cảm ơn Thầy Lương Trung Kiên.

### 🇻🇳 Lời thoại tiếng Việt
> "Cuối cùng, em xin tổng kết lại các đóng góp của đồ án. Chúng em đã hoàn thành toàn diện hệ thống nhận diện và đối chiếu thuộc tính xe chạy 100% offline trên CPU, đáp ứng đầy đủ các chuẩn đầu ra từ CLO1 đến CLO7. **OCR runtime vẫn là PaddleOCR** (~81% exact-match frozen 16); thử nghiệm MobileNetV3-Small+CTC→ONNX đạt 0% exact / CER ~0.66, `deployment_ready: false` — **chưa triển khai**. Hệ thống đạt độ trễ dưới 1 giây ở steady-state.
>
> Hướng tiếp theo (ngắn): thu corpus biển ô tô in-domain tại site cụ thể; tiếp tục CTC chỉ khi ≥90% exact trên frozen 16; kiểm chứng màu/chống tráo trên CCTV bãi thật; pilot demo camera một khung đồng bộ. Nhóm xin cảm ơn thầy Lương Trung Kiên. Kính mong nhận được ý kiến đóng góp."

### 🇬🇧 English Script
> "In conclusion, I would like to summarize our project's deliverables. We successfully implemented a fully offline, CPU-based vehicle attribute recognition and cross-matching system, achieving full compliance with CLO1 through CLO7. The system was validated against real CCTV footage, resolved complex process conflict deadlocks, and achieves sub-1-second latency at steady-state.
>
> Our future roadmap focuses on collecting domain-specific in-garage CCTV images to confirm that our ~86% color accuracy holds outside the VCoR dataset, along with shortening the cold-start time incurred when PaddleOCR loads on the first call. We would like to express our deepest gratitude to our instructor, Mr. Luong Trung Kien, for his invaluable guidance. We now welcome questions and feedback from the committee."

---

## Slide 18: Cảm ơn & Hỏi - Đáp (Thank You & Open Q&A)
*   **Visual:** Màn hình tối nổi bật với tiêu đề lớn "Cảm ơn / thầy và các bạn", thông tin nhóm thực hiện và giảng viên hướng dẫn, cùng dòng "Hỏi & Đáp mở" ở giữa.*

### 🇻🇳 Lời thoại tiếng Việt
> "Đây là slide cuối cùng của bài báo cáo. Thay mặt cho cả nhóm, chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến Thầy Lương Trung Kiên — người đã tận tâm hướng dẫn, giải đáp thắc mắc và đồng hành cùng nhóm xuyên suốt hành trình thực hiện đồ án môn học này.
>
> Chúng em cũng trân trọng cảm ơn sự tham dự và lắng nghe của toàn thể hội đồng. Nhóm rất mong nhận được những câu hỏi, nhận xét và đóng góp từ thầy cô và các bạn để hệ thống ngày càng hoàn thiện hơn. Kính mời thầy và các bạn đặt câu hỏi. Xin cảm ơn!"

### 🇬🇧 English Script
> "This is the final slide of our presentation. On behalf of the entire team, we would like to extend our sincerest and most heartfelt gratitude to Mr. Luong Trung Kien — for his dedicated mentorship, insightful guidance, and unwavering support throughout this project.
>
> We also deeply appreciate the time and attention of the entire review committee. We warmly welcome any questions, comments, and feedback that will help us further improve this system. The floor is now open for Q&A. Thank you very much!"
