# Báo cáo kỹ thuật Giai đoạn 4: Tích hợp Hệ thống và Đánh giá hiệu năng Đầu cuối (Final Report)

## 1. Đặt vấn đề và Mục tiêu tích hợp (Objective)
Giai đoạn cuối cùng của dự án DPL302m tập trung vào tích hợp các mô hình học sâu thành phần (YOLOv8, **PaddleOCR**, MobileNetV3-Small) thành một hệ thống an ninh bãi xe khép kín, tự động. Sau thực nghiệm (Report 3), hệ thống dùng quyết định **plate-primary**: biển số (PaddleOCR) là khoá chính, **màu xe là cảnh báo mềm**, và **bỏ phân loại hãng**; đối chiếu với cơ sở dữ liệu mẫu để đưa ra lệnh điều khiển barrier bãi xe (Cho phép mở hoặc Cảnh báo xâm nhập).

Mục tiêu chính là tối ưu hóa mã nguồn chạy suy luận đầu cuối đầu tiên trên nền tảng CPU cục bộ, giải quyết các lỗi xung đột luồng thư viện và bảo đảm hệ thống hoạt động ổn định ở chế độ ngoại tuyến 100%.

---

## 2. Nghiên cứu tài liệu tham khảo (Literature Review)
Trong quá trình tích hợp và tối ưu hóa hệ thống chạy trên CPU biên, nhóm đã tham khảo các công trình khoa học sau:
1.  **NVIDIA DeepStream — Real-Time License Plate Detection and Recognition (NVIDIA, 2020)** — *https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/*: Kiến trúc tham chiếu công nghiệp kết hợp bộ phát hiện chính (biển số) với các bộ phân loại thứ cấp về hãng xe (VehicleMakeNet) và màu xe, minh họa cách bổ sung đặc trưng ngoại hình bên cạnh biển số để tăng độ tin cậy khi kiểm soát xe ra vào và ngăn ngừa gian lận tráo biển.
2.  **A Novel Memory and Time-Efficient ALPR System Based on YOLOv5 (Batra et al., 2022)** — *Sensors, 22(14), 5283*: Bài viết đề xuất một hệ thống ALPR tối ưu về bộ nhớ và thời gian (mô hình 14 MB, thời gian suy luận ~85 ms cho toàn pipeline), cho thấy cách thu gọn mô hình và kiểm soát tài nguyên tính toán để chạy hiệu quả trên phần cứng biên hạn chế thay vì GPU mạnh.
3.  **Carmen Nano — ANPR/LPR on-prem cho NVIDIA Jetson (Adaptive Recognition, 2024)** — *https://adaptiverecognition.com/products/carmen-nano/*: Giải pháp ANPR thương mại chạy hoàn toàn tại biên (on-prem) trên Jetson, không phụ thuộc kết nối đám mây. Đây là minh chứng thực tiễn cho kiến trúc ưu tiên ngoại tuyến (offline-first) mà dự án hướng tới, giúp loại bỏ độ trễ và rủi ro của các cuộc gọi API trực tuyến khi khởi động và vận hành.

---

## 3. Thiết kế hệ thống tích hợp (Integrated System Design)
Quy trình tích hợp được hiện thực hóa trong mã nguồn [run_evaluation.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/main/src/engine/run_evaluation.py):

```
+------------------+     +-------------------+     +-------------------------+
| Hình ảnh đầu vào | --> | YOLOv8n Detector  | --> | Cắt vùng biển số (Crop) |
+------------------+     +-------------------+     +-------------------------+
                                                                |
                                                                v
                                                   +-------------------------+
                                                   | PaddleOCR Engine (Plate)|
                                                   +-------------------------+
                                                                |
                                                                v
+------------------+     +-------------------+     +-------------------------+
| Classifiers Input| <-- | EfficientNet /    | <-- | Tiền xử lý (Resize 224) |
|   (Car Image)    |     | MobileNetV3       |     | & Chuẩn hóa điểm ảnh    |
+------------------+     +-------------------+     +-------------------------+
         |
         v
+-----------------------------+     +----------------------+     +----------------------+
| So khớp DatabaseMatcher     | --> | AUTHORIZED (Khớp)    | --> | Mở Barrier (Xanh)    |
| (Đối chiếu Plate,Brand,Col) |     | MISMATCH/UNREG (Sai) | --> | Cảnh báo Còi (Đỏ)    |
+-----------------------------+     +----------------------+     +----------------------+
```

---

## 4. Kết quả đánh giá thực nghiệm đầu cuối (E2E Evaluations)
Nhóm đã chạy thực nghiệm toàn bộ pipeline trên tập dữ liệu kiểm thử gồm 5 hình ảnh thực tế lưu trữ cục bộ. Kết quả đo lường chi tiết được ghi nhận như sau:

### 4.1. Bảng nhật ký xử lý chi tiết (Detailed Inference Logs)
| Tên tệp ảnh | Biển số nhận diện | Hãng xe dự đoán (Conf) | Màu xe dự đoán (Conf) | Trạng thái đối chiếu | Thời gian xử lý |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clip3_new_0.jpg** | '706131' | Mitsubishi (0.2670) | Silver (0.1806) | **AUTHORIZED** | 4,488.55 ms |
| **clip3_new_1.jpg** | '3KR3312*56' | Mitsubishi (0.2672) | Silver (0.1792) | **MISMATCH** | 1,714.26 ms |
| **clip3_new_2.jpg** | '5IDS133112*56' | Honda (0.2472) | Silver (0.1794) | **UNREGISTERED** | 1,632.75 ms |
| **clip3_new_3.jpg** | '' (Không đọc được) | Mitsubishi (0.1828) | Silver (0.1816) | **UNREGISTERED** | 1,347.41 ms |
| **clip3_new_4.jpg** | '66P189575' | Honda (0.2910) | Silver (0.1753) | **AUTHORIZED** | 1,771.46 ms |

### 4.2. Thống kê số liệu tổng hợp (Aggregate Metrics)
*   **Tổng số lượng mẫu kiểm thử**: 5 xe
*   **Xe Hợp lệ (AUTHORIZED)**: 2 xe (barrier mở tự động)
*   **Xe Lệch thông tin hãng/màu (MISMATCH)**: 1 xe (khóa barrier, báo động đỏ)
*   **Xe Không đăng ký (UNREGISTERED)**: 2 xe (khóa barrier, báo động đỏ)
*   **Thời gian phản hồi trung bình (Average Latency)**: **2,190.89 ms / xe** trên 5 ảnh test — con số này gồm *cold-start* (~4.49 s ở ảnh đầu do nạp model lần đầu). Từ ảnh thứ hai trở đi, độ trễ ổn định ở mức **~1.6 s / xe** (xem §5.1). Mục tiêu <1.0 s ở đề xuất chưa đạt do chạy thuần CPU + tải PaddleOCR.

---

## 5. Giải pháp tối ưu hóa hiệu năng CPU và Chế độ ngoại tuyến (Offline Optimization)

Trong quá trình tích hợp, hệ thống đã gặp hai thách thức lớn ảnh hưởng đến khả năng triển khai thực tế. Nhóm đã nghiên cứu và áp dụng thành công các giải pháp kỹ thuật sau:

### 5.1. Giải quyết xung đột thư viện và lựa chọn runtime colour classifier

*   **Vấn đề ban đầu**: TensorFlow/Keras và PaddleOCR **không thể sống chung một tiến trình** trên macOS — runtime OpenMP tranh chấp luồng gây treo cứng / tự sập (`mutex lock failed`), và protobuf của hai bên xung khắc phiên bản.
*   **Giải pháp delivered — chuyển sang PyTorch cho runtime**: Bộ phân loại màu runtime sử dụng **PyTorch MobileNetV3-Small** (`main/src/models/torch_color.py`, trọng số `main/data/models/color_MobileNetV3Small.pt`). PyTorch đồng tồn bình thường với PaddleOCR trong cùng tiến trình mà không xung đột. TF/Keras chỉ được dùng trong môi trường training/eval cô lập (`dpl-train`). Nhờ vậy không còn cần cơ chế out-of-process phức tạp cho inference.
*   **Bổ trợ**: giữ `KMP_DUPLICATE_LIB_OK=TRUE` (chống abort OpenMP của EasyOCR) và đặt số luồng thấp khi cần; độ trễ suy luận ổn định **~1.6 giây / xe** từ ảnh thứ hai trở đi (cold-start ~4.5 s do nạp PaddleOCR lần đầu). **Mục tiêu <1.0 s chưa đạt** do chạy thuần CPU.

### 5.2. Tối ưu hóa chế độ chạy ngoại tuyến 100% (Offline-First Deployment)
*   **Chặn EasyOCR kiểm tra phiên bản trực tuyến**: Cấu hình khởi tạo `easyocr.Reader(..., download_enabled=False)` để ngăn chặn tiến trình gửi yêu cầu HTTP kiểm tra phiên bản mô hình từ JaidedAI gây nghẽn luồng khi thiết bị biên không kết nối Internet.
*   **Tắt đồng bộ YOLOv8**: Chèn lệnh `settings.update({"sync": False})` để YOLOv8 tắt hoàn toàn tính năng gửi dữ liệu telemetry trực tuyến về Ultralytics.
*   **Sao chép font hệ thống cục bộ**: Sao chép thủ công tệp font `Arial.ttf` vào thư mục cấu hình mặc định `~/.config/Ultralytics/` để YOLOv8 không gọi lệnh tải font tự động từ máy chủ của họ mỗi khi vẽ bounding box, giúp hệ thống độc lập hoàn toàn với kết nối mạng.

---

## 6. Kết luận
Hệ thống đã hoàn thiện một **pipeline ALPR biên, ngoại tuyến, plate-primary**: phát hiện biển số (YOLOv8n, mAP@0.5 ~0.98) và đọc biển bằng **PaddleOCR** (Benchmark C: 81% exact-match) hoạt động tốt và là lớp quyết định chính. Phân loại **màu** (runtime: **PyTorch MobileNetV3-Small**, `color_MobileNetV3Small.pt`; training/eval: TF/Keras cùng backbone; **55.1%** test giữ-riêng sau fine-tuning) được dùng làm **cảnh báo mềm**; phân loại **hãng** (TF/Keras EfficientNet-B0, ~**35%** — *diagnostic only — đã loại khỏi quyết định*) bị loại khỏi quyết định do còn yếu. Classifier màu phục vụ runtime qua `torch_color.py` để đồng tồn với PaddleOCR mà không cần cách ly tiến trình TF. So với đề xuất ban đầu (đa nhân tố chặn cứng, mục tiêu ≥95%), bản giao đã **pivot có chủ đích** sang plate-primary — trung thực với năng lực thực đo của từng mô hình. Hệ thống chạy ổn định ngoại tuyến 100% trên CPU (~1.6 s/xe sau cold-start), giao diện Streamlit cảnh báo trực quan khi biển không khớp hoặc màu lệch. Hướng cải thiện: thu dữ liệu in-domain + fine-tuning sâu hơn để nâng màu/hãng (xem Report 3 §6).
