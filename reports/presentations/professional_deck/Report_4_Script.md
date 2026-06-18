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
> 1. **Nhận diện biển số (OCR):** Cắt vùng biển số bằng YOLOv8-nano và giải mã ký tự qua PaddleOCR (PP-OCRv4) làm khóa chính.
> 2. **Phân loại màu sắc (Color):** Nhận diện 8 hệ màu cơ bản bằng MobileNetV3-Small để đưa ra cảnh báo mềm khi lệch thông tin.
> 3. **Đối chiếu cơ sở dữ liệu cục bộ:** So khớp tức thời biển số và màu sắc với tệp dữ liệu CSV lưu trữ lịch sử lúc xe vào bãi.
>
> Nhóm cũng đã thử nghiệm bộ phân loại thương hiệu xe EfficientNet-B0 nhưng quyết định loại bỏ thuộc tính này khỏi logic mở barrier do độ chính xác trên ảnh thực tế không đạt yêu cầu vận hành."

### 🇬🇧 English Script
> "To close this security gap, we propose a multi-factor cross-verification framework. The system operates on three technical pillars:
>
> 1. **License Plate OCR:** Extracts the plate region using YOLOv8-nano and decodes characters via PaddleOCR (PP-OCRv4) as the primary key.
> 2. **Color Classification:** Identifies 8 base colors using MobileNetV3-Small to trigger soft warnings on mismatch.
> 3. **Local Database Matcher:** Instantly matches the plate and color against a local CSV file storing check-in records.
>
> We also experimented with an EfficientNet-B0 brand classifier but dropped it from the gate decision logic because its accuracy on real-world images was insufficient for daily operations."

---

## Slide 4: Tổng quan nghiên cứu & Cơ sở khoa học (Tech Spec Sheet - Literature Review - S21)
*   **Visual:** Bảng cơ sở khoa học và các bài báo khoa học được trích dẫn (YOLOv8, PP-OCRv4 BiLSTM, MobileNetV3, EfficientNet-B0).

### 🇻🇳 Lời thoại tiếng Việt
> "Về cơ sở khoa học, chúng em đã kế thừa và tích hợp các nghiên cứu học sâu tiên tiến nhất để tối ưu hóa tài nguyên phần cứng. 
>
> Đối với phát hiện biển số, chúng em sử dụng mô hình **YOLOv8-nano** của Jocher (2023) vì tốc độ phát hiện thời gian thực vượt trội trên thiết bị biên. Khâu nhận diện ký tự OCR sử dụng **PaddleOCR (PP-OCRv4)** tích hợp mạng hồi quy BiLSTM hai chiều theo nghiên cứu của Shi (2015) để mô hình hóa chuỗi ký tự, đem lại độ chính xác vượt trội trên dữ liệu CCTV thực tế. Cuối cùng, mô hình **MobileNetV3-Small** của Howard (2019) được lựa chọn cho phân loại màu sắc nhờ khả năng tối ưu hóa tính toán trên CPU bãi giữ xe."

### 🇬🇧 English Script
> "Regarding the scientific foundation, we integrated state-of-the-art deep learning architectures optimized for edge hardware.
>
> For plate detection, we utilized **YOLOv8-nano** (Jocher, 2023) due to its exceptional real-time inference speed on edge devices. For character recognition, we integrated **PaddleOCR (PP-OCRv4)**, which uses a bidirectional LSTM (BiLSTM) network (Shi, 2015) to model character sequences, yielding superior accuracy on real CCTV images. Finally, Howard's **MobileNetV3-Small** (2019) was selected for color classification due to its hardware-friendly computation on standard edge CPUs."

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
> "Về khâu chuẩn bị dữ liệu, nhóm đã tiến hành thu thập từ nhiều nguồn đa dạng. Chúng em sử dụng tập Stanford Cars gồm 8,144 ảnh để pre-train, và tập Kaggle VN Plates gồm hơn 3,500 ảnh biển số Việt Nam để huấn luyện bộ định vị YOLOv8. 
>
> Đối với bài toán phân loại thương hiệu và màu sắc nội địa, nhóm tiến hành cào hình ảnh thực tế các dòng xe VinFast trên mạng và cào bổ sung các mẫu màu hiếm để tránh mất cân bằng lớp. Tập dữ liệu sau khi làm sạch thủ công gồm 1,575 ảnh chất lượng cao và được phân chia theo tỷ lệ 70% huấn luyện, 15% xác thực, 15% kiểm thử với seed cố định 42."

### 🇬🇧 English Script
> "For data preparation, we gathered images from diverse sources. We leveraged the Stanford Cars dataset containing 8,144 images for pre-training, and the Kaggle VN Plates dataset with over 3,500 Vietnamese plate images to train our YOLOv8 detector.
>
> For local brand and color classification, we scraped real-world VinFast vehicle images online and gathered extra images for underrepresented color classes to prevent imbalance. After manual cleaning, the final dataset contained 1,575 high-quality images, split into 70% training, 15% validation, and 15% testing partitions with a fixed seed of 42."

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
*   **Visual:** Bảng phân bố số lượng ảnh của 8 màu sắc sau khi cào bù (Nâu: 125, Vàng: 128, Đen: 254, v.v.), tổng số ảnh sạch là 1,575.

### 🇻🇳 Lời thoại tiếng Việt
> "Kết quả sau quá trình làm sạch cho thấy sự mất cân bằng nghiêm trọng giữa các lớp màu sắc tự nhiên. Các mẫu màu như đen, trắng chiếm ưu thế, trong khi màu nâu và màu vàng cực kỳ khan hiếm.
>
> Để khắc phục, nhóm đã tiến hành cào bù thêm 125 ảnh màu nâu và 128 ảnh màu vàng để đưa tổng số ảnh sạch lên 1,575 ảnh. Cách tiếp cận thu thập dữ liệu bổ sung trong miền này giúp giải quyết triệt để tình trạng lệch lớp dữ liệu mà không cần dùng phương pháp sinh mẫu nhân tạo, bảo đảm độ chính xác phân loại của mô hình trên thực địa."

### 🇬🇧 English Script
> "Our clean dataset initially revealed a severe class imbalance. Common colors like Black and White dominated the distribution, while Brown and Yellow were extremely rare.
>
> To resolve this, we scraped an additional 125 Brown and 128 Yellow car images to build a balanced dataset of 1,575 clean images. This targeted in-domain data collection resolved the class imbalance without relying on artificial sample generation, ensuring the model's accuracy in real-world environments."

---

## Slide 9: Bộ phát hiện biển số YOLOv8 (Image Hero - YOLOv8 Detector - S22)
*   **Visual:** Ảnh minh họa YOLOv8 crop biển số (`img_title_hero.png`), các chỉ số hiệu năng (mAP50: 0.99, độ trễ: ~75ms, kích thước: 6.2MB).

### 🇻🇳 Lời thoại tiếng Việt
> "Đối với bộ phát hiện biển số (Detector), nhóm tiến hành tự huấn luyện (fine-tune) mô hình YOLOv8-nano trên tập dữ liệu biển số Việt Nam. 
>
> Kết quả thực nghiệm rất khả quan khi độ chính xác định vị mAP50 đạt 0.99, tăng từ mức 0.979 khi huấn luyện từ đầu. Nhờ ưu điểm gọn nhẹ với dung lượng weight chỉ 6.2 MB, mô hình chạy rất nhanh trên CPU với độ trễ chỉ khoảng 75 ms, cung cấp các vùng cắt biển số xe máy và ô tô chuẩn xác trước khi đưa vào mô-đun OCR."

### 🇬🇧 English Script
> "For license plate detection, we fine-tuned a YOLOv8-nano model on Vietnamese plate data.
>
> The experimental results are highly positive, achieving a mAP50 of 0.99, up from 0.979 when training from scratch. With a lightweight model size of only 6.2 MB, inference latency on CPU is around 75 ms, providing highly accurate plate crops for both cars and motorcycles before passing them to the OCR engine."

---

## Slide 10: Mô hình hóa chuỗi ký tự (Three Layers - PaddleOCR CRNN/BiLSTM modeling - S05)
*   **Visual:** Sơ đồ 3 tầng mô hình hóa chuỗi ký tự (Tầng 1: PP-LCNet CNN Backbone, Tầng 2: Recurrent Core BiLSTM, Tầng 3: CTC Transcription).

### 🇻🇳 Lời thoại tiếng Việt
> "Tại khâu nhận diện ký tự OCR, chúng em áp dụng kiến trúc PP-OCRv4 của PaddleOCR nhằm đáp ứng chuẩn đầu ra CLO5 về mô hình hóa chuỗi ký tự. Quy trình giải mã gồm ba tầng:
>
> *   **Tầng 1 (CNN Backbone):** Mạng PP-LCNet trích xuất bản đồ đặc trưng từ ảnh biển số và cắt lát thành chuỗi vector từ trái sang phải.
> *   **Tầng 2 (Recurrent Core):** Mạng BiLSTM hai chiều mô hình hóa ngữ cảnh và ghi nhớ mối liên hệ tuần tự của chuỗi ký tự theo cả hai hướng.
> *   **Tầng 3 (Giải mã):** Tầng CTC căn chỉnh chuỗi ký tự ngõ ra mà không cần phân đoạn ký tự thủ công ở mức pixel."

### 🇬🇧 English Script
> "For character recognition, we deployed PaddleOCR's PP-OCRv4 architecture to satisfy Course Learning Outcome 5 on sequence modeling. The pipeline operates in three layers:
>
> *   **Layer 1 (CNN Backbone):** PP-LCNet extracts feature maps from the plate crop, slicing them into a sequence of vectors from left to right.
> *   **Layer 2 (Recurrent Core):** A bidirectional LSTM (BiLSTM) network captures sequential dependencies and context from both left-to-right and right-to-left directions.
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
> Nhóm đã khắc phục bằng cách chuyển sang Functional API, gọi trực tiếp base model dưới dạng base(x, training=False) để khóa cứng BatchNorm và đồng bộ dải tiền xử lý [0, 255]. Giải pháp này giúp loss giảm ổn định và độ chính xác phân loại màu đạt 54.2% trên dữ liệu CCTV."

### 🇬🇧 English Script
> "This slide demonstrates our system diagnostics process mapping to Course Learning Outcome 2. Initially, using the Keras Sequential API with frozen backbones caused a BatchNorm leakage. During inference, BatchNorm layers ran in training mode (training=True). Combined with a redundant Rescaling layer overriding MobileNetV3's native preprocessing, this led to flat training loss and test accuracy stuck at a random 12.5%.
>
> We resolved this by switching to the Functional API, invoking the base model explicitly as base(x, training=False) to freeze BatchNorm layers, and synchronizing the [0, 255] input range. This fix achieved stable loss convergence and 54.2% color test accuracy."

---

## Slide 14: Kết quả thực nghiệm & Độ chính xác (H-Bar Chart - Benchmarks and Training curves - S07)
*   **Visual:** Biểu đồ thanh ngang so sánh độ chính xác giữa các mô hình (YOLOv8: 99%, PaddleOCR: 81%, Color: 54.2%, Brand: 35.3%, EasyOCR: 0%).

### 🇻🇳 Lời thoại tiếng Việt
> "Đây là kết quả thực nghiệm chi tiết của các mô hình trên tập dữ liệu kiểm thử độc lập. Bộ định vị YOLOv8 đạt kết quả xuất sắc với mAP 99%. Với OCR, PaddleOCR (PP-OCRv4) đạt tỷ lệ khớp hoàn toàn 81% trên dữ liệu CCTV thật, trong khi EasyOCR chỉ đạt 0% do không thể xử lý ảnh thực tế từ camera bãi xe. 
>
> Độ chính xác của bộ phân loại màu MobileNetV3 đạt 54.2%. Đặc biệt, bộ phân loại thương hiệu EfficientNet-B0 chỉ đạt 35.3% do độ mờ ảnh CCTV. Sự chênh lệch lớn này là căn cứ thực tế để chúng em đưa ra quyết định loại bỏ hãng xe và chuyển sang đối chiếu Plate-Primary để bảo vệ hệ thống khỏi các lỗi khóa cổng nhầm."

### 🇬🇧 English Script
> "These are the empirical benchmark results of our models on the independent test split. The YOLOv8 plate detector achieved an outstanding 99% mAP. In OCR, PaddleOCR (PP-OCRv4) reached an 81% exact-match accuracy on real CCTV images, whereas EasyOCR failed completely at 0% due to its inability to process noisy, low-resolution plate crops.
>
> The MobileNetV3 color classifier achieved 54.2% accuracy. Crucially, the EfficientNet-B0 brand classifier reached only 35.3% due to CCTV blur. These findings justified our decision to omit the brand attribute and adopt a Plate-Primary matching logic to prevent false gate locking."

---

## Slide 15: Tích hợp hệ thống & Cô lập tiến trình (Loop Form - Process Isolation Integration - S14)
*   **Visual:** Sơ đồ vòng lặp cô lập tiến trình để tránh xung đột OpenMP Deadlock đáp ứng CLO3, phân tách `keras_color_worker.py` qua đường ống IPC Pipes.

### 🇻🇳 Lời thoại tiếng Việt
> "Trong quá trình tích hợp hệ thống đáp ứng chuẩn đầu ra CLO3, nhóm đã đối mặt với lỗi OpenMP Deadlock nghiêm trọng trên macOS khi chạy song song PyTorch (PaddleOCR) và TensorFlow (MobileNetV3) trên cùng một luồng FastAPI.
>
> Để giải quyết triệt để, chúng em đã triển khai kỹ thuật cô lập tiến trình (Process Isolation). Bộ phân loại màu Keras được tách hẳn sang một tiến trình độc lập `keras_color_worker.py` và giao tiếp với FastAPI thông qua đường ống dẫn IPC Pipes. Quy trình khép kín hoạt động tuần hoàn: FastAPI nhận ảnh, chạy YOLOv8 và PaddleOCR, gửi ảnh qua IPC sang worker màu, nhận lại kết quả và phản hồi sang Streamlit để đóng mở barrier an toàn."

### 🇬🇧 English Script
> "During system integration to satisfy Course Learning Outcome 3, we faced severe OpenMP deadlocks on macOS when running PyTorch (PaddleOCR) and TensorFlow (MobileNetV3) concurrently within the same FastAPI process.
>
> To resolve this, we implemented process isolation. The Keras color classifier was separated into an independent subprocess `keras_color_worker.py` communicating with FastAPI via IPC pipes. The pipeline operates in a closed loop: FastAPI receives the image, executes YOLOv8 and PaddleOCR, passes the crop to the color worker via IPC, retrieves the prediction, and responds to Streamlit to safely operate the gate."

---

## Slide 16: Nhật ký vận hành đầu cuối (Stacked KPI Ledger - E2E Logs & Plate-Primary Logic - S20)
*   **Visual:** Bảng nhật ký vận hành đầu cuối gồm thông số thời gian trễ trung bình ~1.6 giây và ba kịch bản đối chiếu (AUTHORIZED, MISMATCH, UNREGISTERED).

### 🇻🇳 Lời thoại tiếng Việt
> "Slide này mô tả kết quả đánh giá đầu cuối hệ thống. Tổng thời gian xử lý trung bình cho mỗi lượt xe trên CPU là khoảng 1.6 giây, bao gồm 75ms định vị, 1250ms chạy PaddleOCR và 95ms chạy phân loại màu. Độ trễ này hoàn toàn chấp nhận được và không gây ùn tắc vì hệ thống chỉ kích hoạt một lần duy nhất khi xe đã đỗ ổn định trước barrier. 
>
> Logic đối chiếu Plate-Primary xử lý 3 kịch bản: Hợp lệ (trùng khớp biển và màu) sẽ tự động mở cổng; Lệch màu (khớp biển số nhưng màu sắc khác) vẫn cho qua nhưng phát còi cảnh báo mềm; Không đăng ký (biển số không có trong hệ thống hoặc biển giả) sẽ khóa cứng barrier để bảo vệ."

### 🇬🇧 English Script
> "This slide details our end-to-end system evaluation. The average processing latency per vehicle on CPU is approximately 1.6 seconds, comprising 75ms for detection, 1,250ms for PaddleOCR, and 95ms for color classification. This latency is acceptable and prevents traffic jams since inference runs only once when a vehicle stops stably at the gate.
>
> The Plate-Primary matcher handles three scenarios: Authorized (matching plate and color) opens the gate; Mismatch (matching plate but different color) allows passage with a soft audio warning; Unregistered (plate missing from database or fake) locks the barrier for maximum security."

---

## Slide 17: Kết luận & Lộ trình (Split Closing - Retrospective & Roadmap - S10)
*   **Visual:** Tóm tắt 3 điểm sáng (Hoàn thành đầy đủ CLO, Thực chứng thực tế, Định hướng tương lai ONNX Quantization) và lời cảm ơn Thầy Lương Trung Kiên.

### 🇻🇳 Lời thoại tiếng Việt
> "Cuối cùng, em xin tổng kết lại các đóng góp của đồ án. Chúng em đã hoàn thành toàn diện hệ thống nhận diện và đối chiếu thuộc tính xe chạy 100% offline trên CPU, đáp ứng đầy đủ các chuẩn đầu ra từ CLO1 đến CLO7. Hệ thống được thực chứng kỹ lưỡng qua dữ liệu camera thực tế và giải quyết tốt các lỗi xung đột phần mềm. 
>
> Lộ trình phát triển tương lai của dự án tập trung vào việc thu thập thêm dữ liệu hình ảnh đặc thù trong các hầm xe nội địa và áp dụng kỹ thuật lượng tử hóa mô hình (ONNX Quantization) nhằm giảm độ trễ xử lý xuống dưới 1 giây. Nhóm chúng em xin chân thành cảm ơn thầy Lương Trung Kiên đã tận tình hướng dẫn nhóm hoàn thành đồ án này. Kính mong nhận được ý kiến đóng góp từ thầy và các bạn."

### 🇬🇧 English Script
> "In conclusion, I would like to summarize our project's deliverables. We successfully implemented a fully offline, CPU-based vehicle attribute recognition and cross-matching system, achieving full compliance with CLO1 through CLO7. The system was validated against real CCTV footage and resolved complex process conflict deadlocks.
>
> Our future roadmap focuses on collecting domain-specific in-garage images and applying model quantization (ONNX Quantization) to reduce processing latency to under 1 second. We would like to express our deepest gratitude to our instructor, Mr. Luong Trung Kien, for his invaluable guidance. We now welcome questions and feedback from the committee."

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
