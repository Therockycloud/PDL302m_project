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
> "Để khắc phục lỗ hổng trên, hệ thống của chúng em tích hợp 3 bộ lọc deep learning hoạt động song song:
>
> 1.  **Nhận diện biển số (OCR):** Cắt vùng biển số bằng YOLOv8-nano và đọc ký tự qua EasyOCR.
> 2.  **Phân loại hãng xe (Brand):** Nhận diện thương hiệu xe (Toyota, VinFast, Hyundai, Honda, v.v.) bằng mạng EfficientNet-B0 để chống tráo đổi xe khác hãng.
> 3.  **Phân loại màu sắc (Color):** Nhận diện 8 hệ màu cơ bản bằng MobileNetV3-Small để ngăn chặn các trường hợp sơn lại màu hoặc tráo xe cùng dòng nhưng khác màu."

### 🇬🇧 English Script
> "To close this security gap, our system integrates three parallel deep learning filters:
>
> 1.  **License Plate OCR:** Extracts the plate region using YOLOv8-nano and recognizes text via EasyOCR.
> 2.  **Vehicle Brand Classification:** Identifies the car manufacturer (Toyota, VinFast, Hyundai, Honda, etc.) using an EfficientNet-B0 network to prevent cross-brand vehicle swapping.
> 3.  **Vehicle Color Classification:** Identifies 8 base color groups using MobileNetV3-Small to block swaps between identical car models with different colors."

---

## Slide 4: Tổng quan nghiên cứu (Literature Review)
*   **Visual:** Bảng đối sánh các bài báo khoa học nổi bật (YOLOv8, EasyOCR, EfficientNet, MobileNetV3) kèm ưu nhược điểm.

### 🇻🇳 Lời thoại tiếng Việt
> "Chúng em đã thực hiện tổng quan nghiên cứu kỹ lưỡng để chọn ra những kiến trúc tối ưu nhất:
>
> *   Đối với phát hiện biển số, chúng em kế thừa công nghệ **YOLOv8-nano** vì tốc độ phát hiện thời gian thực vượt trội.
> *   Đối với phần OCR, chúng em sử dụng **EasyOCR** vì hỗ trợ tiếng Việt tốt hơn PaddleOCR trong các môi trường ánh sáng phức tạp.
> *   Đối với phân loại hãng xe, kiến trúc **EfficientNet-B0** được chọn vì cơ chế Compound Scaling giúp đạt độ chính xác cao với số lượng tham số rất nhỏ.
> *   Cuối cùng, mô hình **MobileNetV3-Small** được dùng cho phân loại màu sắc nhờ tối ưu cấu trúc phần cứng giúp giảm thiểu thời gian suy luận trên CPU bãi giữ xe."

### 🇬🇧 English Script
> "We conducted a rigorous literature review to select the most optimal models for our pipeline:
>
> *   For plate detection, we utilized **YOLOv8-nano** due to its state-of-the-art real-time inference speed.
> *   For OCR, we integrated **EasyOCR** as it offers robust multilingual character recognition including Vietnamese.
> *   For brand classification, **EfficientNet-B0** was chosen because its Compound Scaling architecture achieves high accuracy with very few parameters.
> *   Lastly, **MobileNetV3-Small** was selected for color classification to achieve near-instantaneous CPU inference at the parking gate."

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
> *   **Phần Hãng Xe:** Chúng em dùng **EfficientNet-B0** với cấu trúc Transfer Learning, đóng băng các lớp Convolution của ImageNet và huấn luyện thêm các lớp Fully-Connected Dense với kỹ thuật Dropout tỉ lệ 0.4 để chống quá khớp.
> *   **Phần Màu Xe:** Chúng em dùng **MobileNetV3-Small** siêu nhẹ. Việc tối ưu hóa kích cỡ ảnh đầu vào cố định 224x224 giúp tốc độ xử lý nhanh gấp 5 lần so với các mô hình CNN cổ điển."

### 🇬🇧 English Script
> "Instead of using heavy networks like ResNet50 which introduce severe latency on CPUs, we redesigned our classification networks:
>
> *   **For Vehicle Brand:** We leveraged **EfficientNet-B0** using transfer learning. We froze the pre-trained ImageNet convolutional layers and appended custom fully-connected layers optimized with a 0.4 Dropout layer to prevent overfitting.
> *   **For Vehicle Color:** We utilized the lightweight **MobileNetV3-Small**. Resizing the input image to a fixed 224x224 dimension speeded up training and inference by 5x compared to scratch-trained CNNs."

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
> *   **Trường hợp hợp lệ:** Xe đi ra có biển số trùng khớp với lịch sử đầu vào, đồng thời hãng xe dự đoán (Toyota) và màu sắc (White) khớp hoàn toàn với cơ sở dữ liệu. Cổng hiển thị trạng thái màu xanh lá và mở barrier.
> *   **Trường hợp gian lận:** Kẻ gian dùng biển số thật của xe Honda Wave màu đen lắp lên xe VinFast màu đỏ. Hệ thống phát hiện biển số trùng khớp cơ sở dữ liệu nhưng hãng xe dự đoán (VinFast) và màu sắc (Red) bị lệch hoàn toàn. Hệ thống lập tức hiển thị màu đỏ báo động, khóa barrier và phát còi cảnh báo."

### 🇬🇧 English Script
> "This screen simulates the live dashboard used by the parking attendants:
>
> *   **Valid Case:** The exiting car's plate matches the database, and its predicted brand (Toyota) and color (White) match perfectly. The system displays a green success status and opens the barrier.
> *   **Fraud Case:** A thief attaches a stolen plate from a black Honda to a red VinFast car. The system matches the plate text, but notices the predicted brand (VinFast) and color (Red) do not match the database records. It instantly triggers a red alert, locks the barrier, and sounds the siren."

---

## Slide 10: Kết quả thực nghiệm (Performance Metrics & Latency)
*   **Visual:** Bảng phân rã thời gian suy luận (YOLOv8: 45ms, EasyOCR: 220ms, Classifiers: 133ms, E2E: 398ms) và các chỉ số mAP, accuracy.

### 🇻🇳 Lời thoại tiếng Việt
> "Về mặt hiệu năng, chúng em rất tự hào khi hệ thống chạy mượt mà ngay trên CPU thường của máy tính xách tay với tổng thời gian xử lý chỉ **398 ms**, chưa tới nửa giây:
>
> *   Thời gian phát hiện biển số chỉ mất 45ms.
> *   Thời gian OCR tốn nhiều tài nguyên nhất là 220ms.
> *   Hai bộ phân loại hãng và màu chỉ mất tổng cộng 133ms.
> *   Độ chính xác của hệ thống đạt chỉ số ấn tượng: Phát hiện biển số đạt 98.2% mAP, OCR đạt 94.5% và tỷ lệ chặn xe gian lận/biển số giả đạt tới **98.7%**."

### 🇬🇧 English Script
> "In terms of performance, we are proud that the integrated system runs smoothly on standard laptop CPUs with a total end-to-end latency of only **398 milliseconds**—well below our 1-second budget:
>
> *   License plate detection takes just 45ms.
> *   EasyOCR text recognition, the heaviest component, takes 220ms.
> *   Brand and color classifiers combine for just 133ms.
> *   System accuracy is outstanding: plate detection mAP is 98.2%, OCR accuracy is 94.5%, and the fake plate detection rate is **98.7%**."

---

## Slide 11: Thử thách & Giải pháp (Challenges & Workarounds)
*   **Visual:** 3 hộp thông tin: Lỗi OpenMP trên macOS, Tối ưu hóa CPU, xử lý đỗ xe nghiêng lệch góc.

### 🇻🇳 Lời thoại tiếng Việt
> "Trong quá trình xây dựng đồ án, nhóm đã giải quyết được ba thử thách lớn:
>
> 1.  **Xung đột OpenMP trên macOS:** Khi import song song PyTorch và TensorFlow gây crash luồng. Nhóm đã cấu hình biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` để giải quyết triệt để lỗi này.
> 2.  **Tối ưu hóa chạy trên CPU:** Chúng em đã giảm tối đa kích thước ảnh nạp vào các bộ phân loại và chọn các kiến trúc tối giản như MobileNetV3-Small.
> 3.  **Lệch góc biển số xe:** Nhóm áp dụng phương pháp mở rộng 5% margin khi crop biển số để thu được trọn vẹn đường viền ký tự giúp cải thiện độ chính xác OCR."

### 🇬🇧 English Script
> "During development, we overcame three major technical hurdles:
>
> 1.  **OpenMP Conflict on macOS:** Spawning PyTorch and TensorFlow concurrently caused runtime aborts. We resolved this by overriding the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable in python.
> 2.  **CPU Optimization:** We minimized classifier input sizes to 224x224 and shifted to lightweight architectures like MobileNetV3-Small to ensure CPU compatibility.
> 3.  **Angle Distortion:** We implemented a 5% padding margin around cropped license plate regions to prevent characters from being cut off, enhancing OCR reliability."

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
> "This concludes our presentation. We would like to express our deepest gratitude to our instructor, Mr. Tran Duc Anh, for his guidance throughout this project.
>
> We now welcome any questions, feedback, and counter-arguments from the instructor and the audience to help us refine this security system. Thank you very much!"
