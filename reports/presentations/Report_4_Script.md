# Kịch bản thuyết trình bảo vệ đồ án (Report 4 Oral Script)

Tài liệu này chứa nội dung thuyết trình chi tiết cho từng slide của bài báo cáo bảo vệ đồ án cuối khóa (Report 4) môn Học Sâu (DPL302m).

---

## Slide 1: Slide tiêu đề (Title Slide)
*   **Visual:** Tiêu đề dự án "Hệ thống giám sát an ninh bãi xe thông minh", thông tin giảng viên hướng dẫn (Thầy Lương Trung Kiên) và thành viên thực hiện.

### 🇻🇳 Lời thoại tiếng Việt
> "Kính chào thầy và các bạn. Hôm nay, nhóm chúng em xin đại diện trình bày báo cáo bảo vệ đồ án cuối kỳ môn Deep Learning (DPL302m). Đề tài của chúng em là: **'Hệ thống giám sát an ninh bãi xe thông minh qua đối chiếu chéo thông tin xe máy và ô tô'**.
>
> Như thầy đã biết, các bãi giữ xe truyền thống hiện nay chỉ đối sánh biển số xe đơn thuần, tạo ra kẽ hở lớn cho kẻ gian chế biển số giả hoặc đánh tráo biển số giữa các xe cùng loại. Do đó, chúng em đề xuất một giải pháp đối chiếu chéo tự động đồng thời ba yếu tố: **Biển số xe**, **Hãng xe**, và **Màu sắc xe** để tối ưu hóa an ninh một cách toàn diện."

### 🇬🇧 English Script
> "Good morning, teacher and everyone. Today, we are presenting our final defense for the Deep Learning course (DPL302m). Our project is titled: **'Smart Parking Security System via Cross-Verification of Vehicle Attributes'**.
>
> Traditional parking systems rely solely on license plate OCR, leaving a vulnerability where thieves can swap plates between similar cars. To address this, we propose an automated system that cross-verifies three core factors simultaneously: **License Plate text**, **Vehicle Brand**, and **Vehicle Color** to ensure comprehensive security."

---

## Slide 2: Vấn đề thực tiễn (Motivation & Problem Statement)
*   **Visual:** Hình ảnh mô tả sơ đồ tráo biển số xe tinh vi (`img_problem.png`), các điểm yếu của phương thức quét truyền thống.

### 🇻🇳 Lời thoại tiếng Việt
> "Hãy bắt đầu với động lực của dự án. Trộm cắp xe tinh vi đang gia tăng tại các thành phố lớn. Thủ đoạn phổ biến là kẻ gian tráo đổi biển số của một chiếc xe máy số hiệu A sang xe máy số hiệu B có hình dáng tương tự trước khi rời bãi xe.
>
> Camera giám sát hiện tại chỉ đối chiếu thẻ từ và chữ số biển số. Nếu biển số trùng khớp với vé xe lúc vào, cổng barrier sẽ mở tự động mà không phát hiện được thương hiệu hoặc màu sắc xe thực tế đã bị thay đổi. Việc phát hiện thủ công bằng mắt thường của bảo vệ thì lại quá chậm, gây tắc nghẽn giao thông nghiêm trọng vào giờ cao điểm."

### 🇬🇧 English Script
> "Let's begin with the motivation. High-tech vehicle theft is increasing in urban areas. A common trick is swapping the license plate of vehicle A to vehicle B of a similar model before exiting the lot.
>
> Existing cameras only match the printed digits on the plate against the check-in ticket database. If they match, the barrier opens, failing to detect that the vehicle's brand or color has changed. Relying on security guards for manual visual verification is too slow and causes severe congestion during peak hours."

---

## Slide 3: Giải pháp đề xuất (Proposed Solution)
*   **Visual:** 3 khối thẻ (Cards) mô tả 3 nhân tố kiểm tra: OCR biển số, Classifier phân loại Hãng, Classifier phân loại Màu.

### 🇻🇳 Lời thoại tiếng Việt
> "Để khắc phục lỗ hổng trên, hệ thống của chúng em triển khai chiến lược **plate-primary**: biển số là khóa chính để ra quyết định, các thuộc tính còn lại đóng vai trò bổ trợ:
>
> 1.  **Nhận diện biển số (OCR) — khóa chính:** Cắt vùng biển số bằng YOLOv8-nano và đọc ký tự bằng **PaddleOCR (PP-OCRv4)** kết hợp giải thuật sắp xếp dòng tiếng Việt; EasyOCR chỉ giữ vai trò dự phòng (fallback).
> 2.  **Phân loại màu sắc (Color) — cảnh báo mềm:** Nhận diện 8 hệ màu cơ bản bằng MobileNetV3-Small (chạy nền PyTorch, cùng tiến trình với PaddleOCR) để gắn cờ cảnh báo khi màu xe không khớp, có cơ chế neutral-cluster và confidence-gating để giảm báo động giả.
> 3.  **Phân loại hãng xe (Brand) — chỉ tham khảo:** Nhận diện thương hiệu xe bằng EfficientNet-B0, nhưng do độ chính xác còn yếu nên kết quả này **không** tham gia vào quyết định khóa/mở barrier, chỉ hiển thị mang tính chẩn đoán (diagnostic)."

### 🇬🇧 English Script
> "To close this security gap, our system follows a **plate-primary** strategy: the license plate is the primary decision key, while the remaining attributes act as supporting signals:
>
> 1.  **License Plate OCR — primary key:** Extracts the plate region using YOLOv8-nano and recognizes text via **PaddleOCR (PP-OCRv4)** combined with a Vietnamese line-sorting algorithm; EasyOCR is kept only as a fallback.
> 2.  **Vehicle Color Classification — soft alert:** Identifies 8 base color groups using MobileNetV3-Small (PyTorch runtime, in-process alongside PaddleOCR) to flag a soft warning on mismatch, with neutral-cluster handling and confidence-gating to reduce false alarms.
> 3.  **Vehicle Brand Classification — diagnostic only:** Identifies the car manufacturer using EfficientNet-B0, but since accuracy is still weak, this result is **not** part of the lock/unlock decision — it is shown for reference (diagnostic) purposes only."

---

## Slide 4: Tổng quan nghiên cứu (Literature Review)
*   **Visual:** Bảng đối sánh các bài báo khoa học nổi bật (YOLOv8, PaddleOCR, EfficientNet, MobileNetV3) kèm ưu nhược điểm.

### 🇻🇳 Lời thoại tiếng Việt
> "Chúng em đã thực hiện tổng quan nghiên cứu kỹ lưỡng để chọn ra những kiến trúc tối ưu nhất:
>
> *   Đối với phát hiện biển số, chúng em kế thừa công nghệ **YOLOv8-nano** vì tốc độ phát hiện thời gian thực vượt trội.
> *   Đối với phần OCR, ban đầu chúng em thử nghiệm EasyOCR, nhưng benchmark trên dữ liệu CCTV thật cho thấy EasyOCR đọc đúng **0%**. Chúng em chuyển sang **PaddleOCR (PP-OCRv4)** và đạt **81% exact-match** — vượt trội hoàn toàn. EasyOCR hiện chỉ giữ vai trò fallback dự phòng.
> *   Đối với phân loại hãng xe, kiến trúc **EfficientNet-B0** được khảo sát vì cơ chế Compound Scaling, nhưng độ chính xác thực nghiệm vẫn còn yếu nên chỉ dùng ở mức tham khảo (diagnostic), không đưa vào quyết định cuối.
> *   Cuối cùng, mô hình **MobileNetV3-Small** được dùng cho phân loại màu sắc (chạy nền PyTorch) nhờ tối ưu cấu trúc phần cứng, đạt **86.3%** trên tập VCoR với kỹ thuật TTA, dùng làm cảnh báo mềm."

### 🇬🇧 English Script
> "We conducted a rigorous literature review to select the most optimal models for our pipeline:
>
> *   For plate detection, we utilized **YOLOv8-nano** due to its state-of-the-art real-time inference speed.
> *   For OCR, we initially trialed EasyOCR, but benchmarking on real CCTV data showed it reading correctly only **0%** of the time. We pivoted to **PaddleOCR (PP-OCRv4)**, achieving **81% exact-match** — a decisive win. EasyOCR is now kept only as a fallback.
> *   For brand classification, **EfficientNet-B0** was evaluated for its Compound Scaling architecture, but empirical accuracy remained weak, so it is used only as a diagnostic signal, not part of the final decision.
> *   Lastly, **MobileNetV3-Small** (PyTorch runtime) was selected for color classification, reaching **86.3%** on the VCoR dataset with TTA, used as a soft alert."

---

## Slide 5: Kiến trúc hệ thống (Overall System Architecture)
*   **Visual:** Sơ đồ quy trình xử lý luồng dữ liệu song song (`img_architecture.png`) tích hợp cơ sở dữ liệu CSV lịch sử bãi xe.

### 🇻🇳 Lời thoại tiếng Việt
> "Kiến trúc hệ thống được chia thành hai phần rõ rệt để đạt hiệu năng tối đa:
>
> *   **Phía Backend:** FastAPI nhận ảnh chụp từ camera cổng thông tin xe. Nó kích hoạt song song 3 mô hình học sâu. Kết quả sau đó được đối chiếu chéo với tệp CSV lưu trữ lịch sử vé xe vào.
> *   **Phía Frontend:** Một bảng điều khiển Streamlit hiển thị thời gian thực hình ảnh xe, kết quả dự báo, tình trạng khớp thông tin và các cảnh báo khẩn cấp bằng còi rú và đèn chớp đỏ khi phát hiện biển số giả."

### 🇬🇧 English Script
> "The system architecture is decoupled for maximum performance:
>
> *   **On the Backend:** A FastAPI server receives images from the gate camera and spawns parallel deep learning inference threads. The predicted results are then matched against a CSV-based check-in database.
> *   **On the Frontend:** A Streamlit dashboard displays real-time camera feeds, model predictions, validation status, and visual/audio alarms (siren) when a security mismatch occurs."

---

## Slide 6: Nhận diện biển số (Plate Detection & Custom OCR)
*   **Visual:** Đoạn code giải thuật sắp xếp tọa độ văn bản (Spatial Sorting) và logic chuẩn hóa regex biển số xe Việt Nam.

### 🇻🇳 Lời thoại tiếng Việt
> "Một thử thách rất đặc trưng ở Việt Nam là biển số xe máy và ô tô dạng vuông thường có 2 dòng. Nếu đưa trực tiếp vào mô-đun OCR thông thường, thứ tự chữ số sẽ bị xáo trộn lung tung.
>
> Để khắc phục, chúng em đã viết thêm thuật toán **Spatial Sorting (Sắp xếp không gian)**. Hệ thống sẽ tự động tách các bounding box ký tự ra làm 2 dòng bằng cách so sánh tọa độ Y trung tâm. Sau đó, các ký tự trên cùng một dòng được sắp xếp từ trái sang phải theo tọa độ X. Cuối cùng, chuỗi ký tự được chuẩn hóa bằng Regex để loại bỏ các ký tự nhiễu như dấu chấm, khoảng trắng."

### 🇬🇧 English Script
> "A unique challenge in Vietnam is that motorcycle and square car license plates are split into two lines. Standard OCR reading yields scrambled character sequences due to minor camera angles.
>
> To solve this, we implemented a custom **Spatial Sorting algorithm**. The system groups recognized text blocks into top and bottom lines based on their centroid Y-coordinates. Blocks within the same line are then sorted from left to right by their X-coordinates, followed by regex sanitization to strip spaces and dots."

---

## Slide 7: Phân loại hãng xe & Màu sắc (Vehicle Classifiers)
*   **Visual:** Sơ đồ cấu trúc mạng Transfer Learning trên nền tảng EfficientNet-B0 và MobileNetV3-Small.

### 🇻🇳 Lời thoại tiếng Việt
> "Thay vì dùng các mô hình quá nặng như ResNet50 truyền thống dễ gây trễ hệ thống, chúng em đã thiết kế lại mạng phân loại:
>
> *   **Phần Hãng Xe (diagnostic-only):** Chúng em dùng **EfficientNet-B0** với cấu trúc Transfer Learning, đóng băng các lớp Convolution của ImageNet và huấn luyện thêm các lớp Fully-Connected Dense với kỹ thuật Dropout tỉ lệ 0.4 để chống quá khớp. Tuy nhiên độ chính xác thực nghiệm chỉ đạt khoảng 35.3% nên hãng xe chỉ hiển thị tham khảo, không tham gia quyết định khóa/mở barrier.
> *   **Phần Màu Xe (cảnh báo mềm):** Chúng em dùng **MobileNetV3-Small** siêu nhẹ, chạy nền PyTorch đồng tiến trình cùng PaddleOCR. Việc tối ưu hóa kích cỡ ảnh đầu vào cố định 224x224 giúp tốc độ xử lý nhanh, đạt **86.3%** trên tập VCoR với kỹ thuật Test-Time Augmentation (TTA), dùng làm cảnh báo mềm khi có sai lệch màu."

### 🇬🇧 English Script
> "Instead of using heavy networks like ResNet50 which introduce severe latency on CPUs, we redesigned our classification networks:
>
> *   **For Vehicle Brand (diagnostic-only):** We leveraged **EfficientNet-B0** using transfer learning. We froze the pre-trained ImageNet convolutional layers and appended custom fully-connected layers optimized with a 0.4 Dropout layer to prevent overfitting. However, empirical accuracy reached only around 35.3%, so brand is shown for reference only and does not participate in the lock/unlock decision.
> *   **For Vehicle Color (soft alert):** We utilized the lightweight **MobileNetV3-Small**, running on a PyTorch backend in-process alongside PaddleOCR. Resizing the input image to a fixed 224x224 dimension keeps inference fast, reaching **86.3%** on the VCoR dataset with Test-Time Augmentation (TTA), used as a soft alert on color mismatch."

---

## Slide 8: Quy trình xử lý song song (Real-time Pipeline)
*   **Visual:** Mô hình đa luồng (Threading), sơ đồ liên lạc giữa FastAPI và Streamlit thông qua cơ chế Asynchronous.

### 🇻🇳 Lời thoại tiếng Việt
> "Để bãi xe vận hành mượt mà, thời gian phản hồi tại cổng phải cực kỳ nhỏ. Do đó, chúng em triển khai quy trình bất tuần tự.
>
> Khi ảnh xe được gửi tới API, Backend chạy song song cả 3 tác vụ suy luận mô hình bằng cơ chế bất đồng bộ (Asyncio). Luồng giao diện Streamlit cũng liên tục gửi các yêu cầu kiểm tra trạng thái vé mà không bị nghẽn (UI Freeze). Cơ chế báo động nhấp nháy đèn chớp và rú còi báo động được nhúng trực tiếp bằng HTML/CSS trong giao diện Streamlit."

### 🇬🇧 English Script
> "To maintain high throughput, response time at the gate must be minimal. Hence, we implemented an asynchronous, non-blocking pipeline.
>
> When a vehicle image arrives, the FastAPI backend evaluates the three models concurrently using Python's `asyncio` loop. Concurrently, the Streamlit client polls verification endpoints smoothly without causing UI freezes. Alarms and siren audio cues are injected via custom HTML/CSS directly into the dashboard state."

---

## Slide 9: Mô phỏng giao diện (Demo Screen Simulation)
*   **Visual:** Hình ảnh thiết kế giao diện: bên trái là trường hợp xe hợp lệ (Khớp 100%, barrier mở màu xanh), bên phải là xe gian lận (Lệch thông tin màu đỏ, còi hú báo động bật).

### 🇻🇳 Lời thoại tiếng Việt
> "Đây là hình ảnh mô phỏng bảng điều khiển thực tế của bảo vệ bãi xe:
>
> *   **Trường hợp hợp lệ:** Xe đi ra có biển số khớp với lịch sử đầu vào (PaddleOCR đọc đúng), màu sắc dự đoán (White) khớp với cơ sở dữ liệu, hãng xe (Toyota) hiển thị tham khảo. Cổng hiển thị trạng thái màu xanh lá và mở barrier.
> *   **Trường hợp gian lận:** Kẻ gian dùng biển số thật của xe màu đen lắp lên một xe màu đỏ. Hệ thống đọc đúng biển số trùng khớp cơ sở dữ liệu, nhưng màu sắc dự đoán (Red) lệch so với hồ sơ đăng ký (Black) — cảnh báo mềm được kích hoạt theo cơ chế confidence-gating. Hệ thống lập tức hiển thị màu đỏ báo động, khóa barrier và phát còi cảnh báo; thông tin hãng xe chỉ hiển thị tham khảo, không phải căn cứ quyết định."

### 🇬🇧 English Script
> "This screen simulates the live dashboard used by the parking attendants:
>
> *   **Valid Case:** The exiting car's plate matches the database (correctly read by PaddleOCR), its predicted color (White) matches the registered record, and brand (Toyota) is shown for reference. The system displays a green success status and opens the barrier.
> *   **Fraud Case:** A thief attaches a stolen plate from a black car onto a red car. The system reads the plate correctly and matches it to the database, but the predicted color (Red) does not match the registered record (Black) — triggering a soft alert via the confidence-gating mechanism. It instantly displays a red alert, locks the barrier, and sounds the siren; brand information is shown for reference only and is not used as decision evidence."

---

## Slide 10: Kết quả thực nghiệm (Performance Metrics & Latency)
*   **Visual:** Bảng phân rã thời gian suy luận theo module (YOLOv8 110ms/ảnh — Benchmark B, PaddleOCR 423ms/biển — Benchmark C, MobileNetV3-S PyTorch ~100ms; brand không đo riêng vì chỉ diagnostic — đo riêng lẻ, không phải số deploy cộng dồn) và dòng tổng độ trễ deploy thực tế <1 giây, cùng các chỉ số mAP, accuracy.

### 🇻🇳 Lời thoại tiếng Việt
> "Về mặt hiệu năng, chúng em đo từng mô-đun riêng lẻ trên CPU macOS theo benchmark gốc, sau đó tách bạch rõ với số **deploy thực tế**:
>
> *   Định vị biển số (YOLOv8-n) mất khoảng 110ms/ảnh (Benchmark B).
> *   Nhận diện ký tự bằng **PaddleOCR** mất khoảng 423ms/biển (Benchmark C) — là mô-đun nặng nhất, có cold-start vài giây ở lệnh gọi đầu do nạp model; EasyOCR chỉ chạy khi cần fallback.
> *   Phân loại màu (MobileNetV3-Small, PyTorch) mất khoảng 100ms; phân loại hãng (EfficientNet-B0, diagnostic) không đo riêng vì không nằm trên đường quyết định chính.
> *   Độ trễ **deploy thực tế (steady-state)** vẫn đảm bảo dưới 1 giây mỗi xe: khoảng **0.73 giây ở chế độ approach-lock** và **0.96 giây qua API**.
> *   Về độ chính xác: phát hiện biển số đạt **mAP50 99%** (0.9896); OCR đạt **81% exact-match** trên Benchmark C (vượt xa EasyOCR 0%); cơ chế chống tráo biển (gate 0.40, sau khi siết bởi WS-2) phát hiện **69%** ở tỷ lệ báo động giả chỉ **2.5%**, và biển chưa đăng ký bị chặn **100%**."

### 🇬🇧 English Script
> "For performance, we measured each module individually on a macOS CPU per the original benchmarks, then clearly separated that from the **actual deployed latency**:
>
> *   License plate localization (YOLOv8-n) takes about 110ms/image (Benchmark B).
> *   Character recognition via **PaddleOCR** takes about 423ms/plate (Benchmark C) — it is the heaviest module, with a multi-second cold-start on the first call while the model loads; EasyOCR only runs as a fallback.
> *   Color classification (MobileNetV3-Small, PyTorch) takes about 100ms; brand classification (EfficientNet-B0, diagnostic) was not benchmarked individually since it is not on the primary decision path.
> *   The **actual deployed steady-state latency** still stays under 1 second per vehicle: about **0.73s in approach-lock mode** and **0.96s via the API**.
> *   On accuracy: plate detection reaches **mAP50 99%** (0.9896); OCR reaches **81% exact-match** on Benchmark C (versus EasyOCR's 0%); the anti-plate-swap gate (threshold 0.40, tightened after WS-2) detects **69%** of swaps at a low **2.5%** false-alarm rate, while unregistered plates are blocked **100%** of the time."

---

## Slide 11: Thử thách & Giải pháp (Challenges & Workarounds)
*   **Visual:** 3 hộp thông tin: Lỗi OpenMP trên macOS, Tối ưu hóa CPU, xử lý đỗ xe nghiêng lệch góc.

### 🇻🇳 Lời thoại tiếng Việt
> "Trong quá trình xây dựng đồ án, nhóm đã giải quyết được ba thử thách lớn:
>
> 1.  **Xung đột OpenMP trên macOS:** TensorFlow (dùng huấn luyện/đánh giá phân loại) và PaddleOCR không thể chạy chung một tiến trình do xung đột OpenMP/protobuf, gây crash. Nhóm đã chuyển bộ phân loại màu sang chạy nền **PyTorch** — cùng tồn tại bình thường với PaddleOCR trong cùng tiến trình — và cấu hình biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` cho các tác vụ TensorFlow cô lập còn lại.
> 2.  **Tối ưu hóa chạy trên CPU:** Chúng em đã giảm tối đa kích thước ảnh nạp vào các bộ phân loại và chọn các kiến trúc tối giản như MobileNetV3-Small.
> 3.  **Lệch góc biển số xe:** Nhóm áp dụng phương pháp mở rộng margin khi crop biển số để thu được trọn vẹn đường viền ký tự giúp cải thiện độ chính xác OCR."

### 🇬🇧 English Script
> "During development, we overcame three major technical hurdles:
>
> 1.  **OpenMP Conflict on macOS:** TensorFlow (used for classifier training/evaluation) and PaddleOCR could not run in the same process due to an OpenMP/protobuf conflict, causing crashes. We resolved this by moving the color classifier to a **PyTorch** runtime — which coexists normally with PaddleOCR in the same process — and kept `KMP_DUPLICATE_LIB_OK=TRUE` only for the remaining isolated TensorFlow tasks.
> 2.  **CPU Optimization:** We minimized classifier input sizes to 224x224 and shifted to lightweight architectures like MobileNetV3-Small to ensure CPU compatibility.
> 3.  **Angle Distortion:** We implemented a padding margin around cropped license plate regions to prevent characters from being cut off, enhancing OCR reliability."

---

## Slide 12: Đóng góp & Bản đồ chuẩn đầu ra (CLO Deliverables)
*   **Visual:** Bảng ánh xạ 6 Chuẩn đầu ra môn học (CLO1, CLO2, CLO3, CLO4, CLO6, CLO7) sang các tính năng kỹ thuật của đồ án.

### 🇻🇳 Lời thoại tiếng Việt
> "Để đảm bảo đáp ứng đầy đủ yêu cầu học thuật của môn học DPL302m, đồ án của chúng em bám sát các chuẩn đầu ra (CLO):
>
> *   **CLO1 & CLO2:** Thực hiện qua việc tự thiết kế các lớp kết nối đầy đủ (Dense Layers) và cấu hình tối ưu siêu tham số, sử dụng Early Stopping chống quá khớp.
> *   **CLO3:** Thể hiện qua kỹ thuật Transfer Learning từ ImageNet và phân tích logs huấn luyện.
> *   **CLO4:** Triển khai thành công mạng CNN tích chập YOLOv8 và EfficientNet.
> *   **CLO6 & CLO7:** Hoàn thành toàn diện pipeline tích hợp và ứng dụng thành thạo các công cụ hiện đại như Git, FastAPI, Streamlit và kiểm thử tự động với Pytest."

### 🇬🇧 English Script
> "To ensure full compliance with the academic syllabus of the DPL302m course, our project directly maps to the Course Learning Outcomes (CLOs):
>
> *   **CLO1 & CLO2:** Covered by designing custom fully-connected dense layers and tuning hyperparameters with early stopping to prevent overfitting.
> *   **CLO3:** Satisfied via transfer learning from pre-trained ImageNet weights and executing systematic model diagnostics.
> *   **CLO4:** Fulfilled by deploying convolutional networks (YOLOv8 and EfficientNet-B0) for detection and classification.
> *   **CLO6 & CLO7:** Accomplished by building a complete end-to-end pipeline and utilizing modern developer tools like Git, FastAPI, Streamlit, and Pytest."

---

## Slide 13: Q&A (Hỏi & Đáp)
*   **Visual:** Thông tin liên hệ và lời cảm ơn cuối trang.

### 🇻🇳 Lời thoại tiếng Việt
> "Đến đây, nhóm em xin phép kết thúc phần thuyết trình bảo vệ đồ án. Chúng em xin gửi lời cảm ơn chân thành đến thầy Lương Trung Kiên đã đồng hành hướng dẫn nhóm trong suốt thời gian qua.
>
> Sau đây, nhóm em rất mong nhận được các câu hỏi phản biện và đóng góp ý kiến từ thầy và các bạn để cải thiện hệ thống hoàn thiện hơn nữa. Nhóm em xin chân thành cảm ơn!"

### 🇬🇧 English Script
> "This concludes our presentation. We would like to express our deepest gratitude to our instructor, Mr. Luong Trung Kien, for his guidance throughout this project.
>
> We now welcome any questions, feedback, and counter-arguments from the instructor and the audience to help us refine this security system. Thank you very much!"
