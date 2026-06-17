# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ GIAI ĐOẠN 2: DỮ LIỆU
# PRESENTATION SPEECH SCRIPT FOR PHASE 2: DATA PIPELINE

---

## Slide 1: Mở đầu / Title & Introductions (Layout: S01)
- **Tiêu đề / Title**: QUY TRÌNH THU THẬP & XỬ LÝ DỮ LIỆU THỰC TẾ / Data Acquisition & Preprocessing Pipeline
- **Bố cục / Layout**: S01 - Classic Cover

### Tiếng Việt (Vietnamese Script)
Kính thưa Hội đồng cùng toàn thể các bạn sinh viên. Hôm nay, em xin đại diện cho nhóm thực hiện đồ án môn học Học sâu (DPL302m) trình bày báo cáo tiến độ Giai đoạn 2 với đề tài: **"Hệ thống giám sát an ninh bãi xe thông minh qua đối chiếu chéo thuộc tính"**. Nội dung trọng tâm của báo cáo này tập trung vào Quy trình thu thập và xử lý dữ liệu thực tế — tiền đề cốt lõi để xây dựng các mô hình học máy vững chắc phía sau. Báo cáo được thực hiện dưới sự hướng dẫn của Giảng viên hướng dẫn: Thầy Lương Trung Kiên.

### Tiếng Anh (English Script)
Distinguished members of the Committee and fellow students. Today, representing our project team for the Deep Learning course (DPL302m), I will present our Phase 2 progress report on the project: **"Smart Parking Lot Security Monitoring System via Attribute Cross-Checking"**. The primary focus of this presentation is the Data Acquisition and Preprocessing Pipeline, which serves as the foundational cornerstone for training our downstream deep learning models. This project is supervised by our advisor, Mr. Luong Trung Kien.

---

## Slide 2: Đặt vấn đề & Mục tiêu / Problem Statement & Objective (Layout: S03)
- **Tiêu đề / Title**: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU / Problem Statement & Objective
- **Bố cục / Layout**: S03 - Split Statement

### Tiếng Việt (Vietnamese Script)
Trong thực tế vận hành các bãi đỗ xe hiện nay, các hệ thống an ninh đơn yếu tố dựa trên thẻ từ hoặc chỉ sử dụng OCR nhận diện biển số xe rất dễ bị vượt qua bằng các thủ đoạn tinh vi như tháo lắp hoặc tráo đổi biển số. Một biển số xe vật lý thì quá dễ bị giả mạo. Để giải quyết vấn đề này, nhóm chúng em đề xuất một cơ chế đối chiếu chéo đa nhân tố dựa trên ba đặc trưng sinh trắc học trực quan của phương tiện: Khóa định danh chính là Biển số xe, đi kèm hai chiều đối chiếu bổ trợ là Hãng xe và Màu sắc xe. Sự kết hợp này sẽ giúp phát hiện lập tức bất kỳ sự không khớp thuộc tính nào tại cổng kiểm soát.

### Tiếng Anh (English Script)
In current parking lot operations, single-factor security systems relying on RFID cards or OCR-based license plate recognition alone are highly vulnerable to bypass methods such as physical plate swapping. A physical license plate is simply too easy to counterfeit. To address this issue, we propose a multi-factor cross-checking mechanism based on three visual biometrics of the vehicle: the License Plate as the primary identifier, complemented by the Car Brand and Car Color as secondary verification dimensions. This combination ensures immediate detection of any attribute mismatch at the checkpoint.

---

## Slide 3: Tổng quan nghiên cứu & Cơ sở khoa học / Literature Review (Layout: S13)
- **Tiêu đề / Title**: TỔNG QUAN NGHIÊN CỨU & CƠ SỞ KHOA HỌC / Literature Review
- **Bố cục / Layout**: S13 - Three Columns

### Tiếng Việt (Vietnamese Script)
Nghiên cứu của chúng em kế thừa trực tiếp và chặt chẽ trên ba nền tảng cơ sở khoa học đã được công bố:
- **Thứ nhất**, Stanford Cars Dataset của Krause và cộng sự năm 2013, thiết lập tiêu chuẩn phân nhóm thương hiệu và cung cấp phương pháp phân nhóm dòng xe làm nền tảng.
- **Thứ hai**, Nghiên cứu nhận diện màu sắc xe của Chen và cộng sự năm 2014, cung cấp lý thuyết chuẩn hóa dải điểm ảnh trước khi đưa vào mạng nơ-ron tích chập siêu nhẹ dưới điều kiện ánh sáng phức tạp của camera giám sát.
- **Thứ ba**, Khảo sát về các phương pháp tăng cường dữ liệu của Yang và cộng sự năm 2022 (arXiv:2204.08610), làm cơ sở lý thuyết cho các phép biến đổi hình học và dịch chuyển độ tương phản nhằm đối phó với hiện tượng camera bị nghiêng nhẹ.

### Tiếng Anh (English Script)
Our research directly builds upon three established scientific foundations:
- **First**, the Stanford Cars Dataset by Krause et al. (2013), which establishes standards for vehicle brand categorization and provides a foundational vehicle model classification methodology.
- **Second**, the Vehicle Color Recognition study by Chen et al. (2014), which guides the pixel-normalization theory prior to feeding inputs into lightweight CNNs under complex surveillance lighting conditions.
- **Third**, the comprehensive survey on data augmentation by Yang et al. (2022) (arXiv:2204.08610), serving as our theoretical baseline for geometric and contrast transformations to handle camera tilt anomalies.

---

## Slide 4: Quy trình thu thập dữ liệu đa nguồn / Data Acquisition Pipeline (Layout: S21)
- **Tiêu đề / Title**: QUY TRÌNH THU THẬP DỮ LIỆU ĐA NGUỒN / Data Acquisition Pipeline
- **Bố cục / Layout**: S21 - Tech Spec Sheet

### Tiếng Việt (Vietnamese Script)
Để huấn luyện các bộ phân loại, chúng em đã thu thập dữ liệu đa nguồn cho từng tác vụ cụ thể. Tác vụ phân loại hãng xe thu về 792 ảnh sạch bao phủ 8 thương hiệu phổ biến ở Việt Nam. Đối với dòng xe VinFast, nhóm chủ động cào Bing chi tiết theo từng dòng xe con (VF8, VF9, Fadil...) để tránh thu phải ảnh logo hay nội thất nhiễu. Tác vụ phân loại màu sắc thu thập 783 ảnh cho 8 màu cơ bản, đồng thời loại bỏ 39 ảnh lớp màu xanh lá không dùng do mô hình không hỗ trợ. Bộ định vị biển số YOLOv8-nano được huấn luyện trên dataset HuggingFace gồm 6,176 ảnh train. Cuối cùng, tập ảnh chụp xe thực tế tại các bãi đỗ Việt Nam được nhóm thu thập thủ công để làm tập kiểm thử tích hợp cuối cùng. Toàn bộ tập dữ liệu phân loại được chia vật lý theo tỷ lệ 70/15/15 với seed 42 cố định.

### Tiếng Anh (English Script)
To train the classifiers, we implemented a multi-source data acquisition pipeline tailored to each task. The brand classification task acquired 792 clean images across 8 common brands in Vietnam. For the domestic VinFast brand, we performed targeted Bing queries by specific models (VF8, VF9, Fadil, etc.) to filter out redundant logo or interior close-ups. The color classification task collected 783 images for 8 primary colors, discarding 39 green vehicle images due to classifier incompatibility. The license plate detector utilizes YOLOv8-nano, trained on 6,176 HuggingFace images. Finally, real-world parking lot photographs from Vietnamese parking zones were manually collected as the end-to-end integration test split. The classification datasets are physically partitioned into a 70/15/15 ratio using a fixed seed of 42.

---

## Slide 5: Đường ống làm sạch dữ liệu tự động / Automated Data Cleaning Pipeline (Layout: S11)
- **Tiêu đề / Title**: ĐƯỜNG ỐNG LÀM SẠCH DỮ LIỆU TỰ ĐỘNG / Automated Data Cleaning Pipeline
- **Bố cục / Layout**: S11 - Horizontal Timeline

### Tiếng Việt (Vietnamese Script)
Một đóng góp kỹ thuật quan trọng trong giai đoạn này là đường ống làm sạch dữ liệu tự động gồm 5 bước nhằm đảm bảo chất lượng nhãn:
- **Bước 1**: `clean_corrupted_images` sử dụng OpenCV lọc bỏ các file ảnh lỗi cấu trúc vật lý.
- **Bước 2**: `semantic_clean_images` dùng YOLOv8-nano để lọc ngữ nghĩa, loại bỏ khoảng 38% ảnh nhiễu không thực sự chứa xe (như ảnh cận cảnh vô lăng, logo hoặc đường phố trống).
- **Bước 3**: `remove_duplicates` áp dụng thuật toán băm cảm nhận pHash (Perceptual Hashing) thông qua thư viện `imagehash` với khoảng cách Hamming nhỏ hơn hoặc bằng 5 để xóa ảnh trùng lặp hoặc gần trùng.
- **Bước 4**: `normalize_images` đồng bộ hóa định dạng ép tất cả sang JPEG RGB chuẩn.
- **Bước 5**: Nhóm tiến hành cào bổ sung chuyên biệt và cắt ngưỡng tối đa (cap) dữ liệu ở mức ~100 ảnh mỗi lớp để đạt phân bố cân bằng tuyệt đối.

### Tiếng Anh (English Script)
A key technical contribution of this phase is our 5-step automated data cleaning pipeline designed to guarantee high label quality:
- **Step 1**: `clean_corrupted_images` utilizes OpenCV to filter out physically corrupted files.
- **Step 2**: `semantic_clean_images` deploys YOLOv8-nano for semantic filtering, removing approximately 38% of noisy images that did not contain vehicles (such as steering wheels, close-up logos, or empty roads).
- **Step 3**: `remove_duplicates` utilizes the perceptual hashing algorithm (pHash) via the `imagehash` library with a Hamming distance threshold of $\le 5$ to purge duplicate or near-duplicate images.
- **Step 4**: `normalize_images` standardizes the physical format, enforcing JPEG RGB encoding.
- **Step 5**: Targeted supplementary crawling and capping are applied to maintain an absolute balanced distribution of ~100 images per class.

---

## Slide 6: Phân tích thống kê phân bố lớp / Exploratory Data Analysis (Layout: S22)
- **Tiêu đề / Title**: PHÂN TÍCH THỐNG KÊ PHÂN BỐ LỚP / Exploratory Data Analysis (EDA)
- **Bố cục / Layout**: S22 - Image Hero

### Tiếng Việt (Vietnamese Script)
Sau khi hoàn tất quá trình làm sạch tự động, chúng em tiến hành Phân tích thống kê dữ liệu. Biểu đồ phân bố lớp thể hiện sự cân bằng hoàn hảo của 8 lớp màu xe và 8 lớp hãng xe. Chúng em theo đuổi triết lý dữ liệu "ít nhưng chất". Thay vì lạm dụng các phép tăng cường dữ liệu nhân tạo trên tập dữ liệu mất cân bằng nghiêm trọng ban đầu (như màu Vàng chỉ có 25 ảnh, trong khi màu Đen có hơn 200 ảnh), nhóm đã chủ động thực hiện thu thập bù chuyên biệt để nâng số lượng ảnh thật lên đồng đều ~100 ảnh/lớp, giúp hệ số lệch lớp xấp xỉ bằng 1, triệt tiêu nguy cơ thiên lệch mô hình.

### Tiếng Anh (English Script)
Upon completing the automated cleaning, we performed Exploratory Data Analysis. The class distribution chart illustrates the perfect balance achieved across the 8 color classes and 8 brand classes. We adhered to a "quality over quantity" data philosophy. Instead of relying heavily on artificial data augmentation to patch a highly imbalanced dataset (where Yellow initially had only 25 images while Black had over 200), we initiated targeted real-image collection campaigns to balance all classes at approximately 100 images per class, establishing an imbalance ratio of ~1.0x and eliminating model bias.

---

## Slide 7: Tiền xử lý và tăng cường hình ảnh / Preprocessing & Augmentation (Layout: S04)
- **Tiêu đề / Title**: TIỀN XỬ LÝ VÀ TĂNG CƯỜNG HÌNH ẢNH / Preprocessing & Augmentation
- **Bố cục / Layout**: S04 - Six Cells

### Tiếng Việt (Vietnamese Script)
Quy trình tiền xử lý và tăng cường hình ảnh được thiết kế chặt chẽ và chỉ áp dụng các phép tăng cường trên tập huấn luyện (training set):
- **1. Resize 224²**: Đồng bộ kích thước hình ảnh đầu vào chuẩn của hai backbone mạng.
- **2. Lật ngang (Horizontal Flip)**: Tạo các biến thể góc xe tiếp cận bãi đỗ từ cả hai phía.
- **3. Xoay ngẫu nhiên (Random Rotation $\pm 10^\circ$)**: Chống xoay và lệch góc camera.
- **4. Phóng ngẫu nhiên (Random Zoom $\pm 10\%$)**: Mô phỏng sự thay đổi khoảng cách từ xe tới camera.
- **5. Pixel Scaling**: Chuyển đổi dải giá trị điểm ảnh qua lớp Rescaling(255.0) tích hợp ở đầu mô hình.
- **6. Backbone Preprocessing**: Đưa dữ liệu qua bộ tiền xử lý tích hợp của từng backbone MobileNetV3 và EfficientNet tương ứng.

### Tiếng Anh (English Script)
The preprocessing and data augmentation pipeline is strictly defined and applies augmentation only to the training set:
- **1. Resize 224²**: Standardizes input dimensions to match the requirements of both backbones.
- **2. Horizontal Flip**: Generates variations representing vehicles approaching checkpoints from either side.
- **3. Random Rotation ($\pm 10^\circ$)**: Ensures invariance to small camera tilt angles.
- **4. Random Zoom ($\pm 10\%$)**: Simulates depth variation and camera-to-vehicle distance changes.
- **5. Pixel Scaling**: Normalizes pixel values via an integrated Keras Rescaling(255.0) layer.
- **6. Backbone Preprocessing**: Pipelines normalized data through MobileNetV3 and EfficientNet built-in preprocessing layers.

---

## Slide 8: Chẩn đoán lỗi & Khắc phục / Diagnostic & Bug Fixes (Layout: S08)
- **Tiêu đề / Title**: CHẨN ĐOÁN LỖI & KHẮC PHỤC / Diagnostic & Bug Fixes
- **Bố cục / Layout**: S08 - Duo Compare

### Tiếng Việt (Vietnamese Script)
Trong quá trình phát triển ban đầu, chúng em đã chẩn đoán và khắc phục thành công hai lỗi kỹ thuật nghiêm trọng liên quan đến cấu trúc mô hình khiến độ chính xác bị kẹt ở mức ngẫu nhiên ~12.5% (tức là 1 chia cho 8 lớp) và loss đi ngang không hội tụ:
- **Lỗi thứ nhất: BatchNorm Bug**. Việc sử dụng Sequential API làm đóng băng backbone nhưng các lớp BatchNorm vẫn tiếp tục chạy theo thống kê của từng batch huấn luyện thay vì dùng moving average lúc inference. Nhóm đã chuyển sang Functional API và gọi backbone dưới dạng `base_model(x, training=False)` để khóa cứng BN chạy ở chế độ inference.
- **Lỗi thứ hai: Double-preprocessing**. Cấu hình nhầm dải pixel đầu vào làm sai lệch phân bố đặc trưng. Chúng em đã loại bỏ scaling ngoài, đồng bộ đưa ảnh dạng $[0, 1]$ qua lớp `Rescaling(255.0)` về dải $[0, 255]$ chuẩn trước khi nạp vào backbone. Nhờ đó mô hình đã hội tụ ổn định.

### Tiếng Anh (English Script)
During early development, we diagnosed and successfully resolved two major system bugs related to model architecture that had trapped classifier accuracy at the random baseline of ~12.5% (1 out of 8 classes) and caused loss to plateau:
- **First: BatchNorm Bug**. Using Keras Sequential API allowed BatchNorm layers to continue calculating batch-level statistics during training instead of utilizing moving averages during inference. We migrated to the Functional API and invoked the backbone via `base_model(x, training=False)` to lock BatchNorm in inference mode.
- **Second: Double-preprocessing**. Redundant pixel scaling configurations distorted the feature distributions. We resolved this by removing external manual scaling, passing $[0, 1]$ scaled images through an integrated `Rescaling(255.0)` layer to feed the expected $[0, 255]$ range into the backbone. The training loss converged steadily thereafter.

---

## Slide 9: Hiệu năng phân loại trên tập test giữ-riêng / Training & Test Results (Layout: S07)
- **Tiêu đề / Title**: HIỆU NĂNG PHÂN LOẠI TRÊN TẬP TEST GIỮ-RIÊNG / Training & Test Results
- **Bố cục / Layout**: S07 - Horizontal Bar Chart

### Tiếng Việt (Vietnamese Script)
Sau khi khắc phục các lỗi chẩn đoán, kết quả huấn luyện trên tập kiểm thử độc lập (held-out test splits) ghi nhận như sau:
- Bộ phân loại màu xe sử dụng backbone `MobileNetV3-Small` đạt Test Accuracy **55.1%** và Macro-F1 **0.545**. Kết quả này gấp hơn 4 lần mức ngẫu nhiên (12.5%), đủ độ tin cậy để làm thuộc tính cảnh báo phụ.
- Bộ phân loại hãng xe sử dụng backbone `EfficientNet-B0` chỉ đạt Test Accuracy **35.3%** và Macro-F1 **0.337** — đo trên tập test **ảnh web sạch** (không phải ảnh CCTV). Nguyên nhân gốc là bài toán phân biệt thương hiệu xe có độ khó cao (fine-grained) kết hợp dữ liệu ít (~70 ảnh/lớp); ảnh camera mờ chỉ làm kết quả tệ thêm chứ không phải nguyên nhân chính.
Từ dữ liệu thực chứng này, nhóm đã đưa ra một quyết định kỹ thuật quan trọng: loại bỏ thuộc tính hãng xe khỏi hệ thống ở các giai đoạn sau (R3/R4) để tránh gây ra lỗi từ chối sai (false rejection) khiến cổng an ninh không mở cho xe hợp lệ.

### Tiếng Anh (English Script)
Following bug resolution, training results evaluated on the strictly held-out test splits are as follows:
- The color classifier using `MobileNetV3-Small` achieved a Test Accuracy of **55.1%** and a Macro-F1 of **0.545**. This is more than four times the random baseline (12.5%), making it suitable as a secondary warning attribute.
- The brand classifier using `EfficientNet-B0` achieved a Test Accuracy of **35.3%** and a Macro-F1 of **0.337**, measured on a **clean web-image test set** (not live CCTV). The root cause is the inherent fine-grained difficulty of brand discrimination combined with limited training data (~70 images/class); surveillance blur would only compound the problem, not cause it.
Based on this empirical evidence, we made a crucial design decision: to exclude the car brand attribute from the system in subsequent phases (R3/R4) to prevent false rejections that would block valid vehicles at the gate.

---

## Slide 10: Kết luận và lộ trình phát triển / Conclusion & Roadmap (Layout: S10)
- **Tiêu đề / Title**: KẾT LUẬN VÀ LỘ TRÌNH PHÁT TRIỂN / Conclusion & Roadmap
- **Bố cục / Layout**: S10 - Split Closing

### Tiếng Việt (Vietnamese Script)
Tóm lại, trong Giai đoạn 2, nhóm đã hoàn tất xuất sắc việc xây dựng quy trình thu thập dữ liệu đa nguồn và đường ống làm sạch tự động. Tập dữ liệu bàn giao hoàn toàn sạch, cân bằng và đã vượt qua 100% kiểm thử tự động của file `test_dataset.py`.
Lộ trình phát triển tiếp theo của nhóm hướng tới 3 mục tiêu:
1. **Dữ liệu thực tế**: Thu thập thêm dữ liệu camera thực tế tại các bãi đỗ xe Việt Nam để thu hẹp khoảng cách miền dữ liệu (domain gap).
2. **Fine-tune Backbone**: Thử nghiệm mở băng (fine-tune) một số block cuối của MobileNetV3 để tối ưu hóa đặc trưng lớp màu xe.
3. **ONNX Quantization**: Nén và lượng tử hóa mô hình sang định dạng ONNX, chuẩn bị cho việc tích hợp biên thời gian thực.

### Tiếng Anh (English Script)
In summary, during Phase 2, the team has successfully built a robust multi-source data acquisition pipeline and an automated data cleaning pipeline. The finalized dataset is clean, balanced, and passes 100% of the unit tests in `test_dataset.py`.
Our future roadmap centers on 3 key objectives:
1. **Real-world Data**: Gather more actual camera footage from Vietnamese parking lots to minimize domain gap.
2. **Backbone Fine-tuning**: Experiment with fine-tuning the final blocks of MobileNetV3 to optimize color feature extraction.
3. **ONNX Quantization**: Compress and quantize models into ONNX format to prepare for real-time edge deployment.

---

## Slide 11: Lời cảm ơn & Hỏi đáp / Acknowledgements & Q&A (Layout: S10)
- **Tiêu đề / Title**: XIN CHÂN THÀNH CẢM ƠN / Acknowledgements & Q&A
- **Bố cục / Layout**: S10/S99 - Manifesto Light

### Tiếng Việt (Vietnamese Script)
Chúng em xin chân thành cảm ơn thầy cô trong hội đồng và các bạn đã chú ý lắng nghe phần trình bày báo cáo Giai đoạn 2 của nhóm. Nhóm kính mong nhận được những nhận xét, đóng góp ý kiến phản biện từ Thầy cô và các bạn để hệ thống giám sát an ninh bãi xe thông minh được hoàn thiện hơn trong các giai đoạn tiếp theo. Sau đây, chúng em xin phép được bắt đầu phiên Hỏi & Đáp mở (Q&A).

### Tiếng Anh (English Script)
We would like to express our sincere gratitude to the committee members and the audience for your time and attention to our Phase 2 report. We eagerly look forward to receiving your feedback, constructive critique, and questions to refine our smart parking security system in the upcoming phases. We would now like to open the floor for the Q&A session.
