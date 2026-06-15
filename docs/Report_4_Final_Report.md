# Báo cáo kỹ thuật Giai đoạn 4: Tích hợp Hệ thống và Đánh giá hiệu năng Đầu cuối (Final Report)

## 1. Đặt vấn đề và Mục tiêu tích hợp (Objective)
Giai đoạn cuối cùng của dự án DPL302m tập trung vào tích hợp các mô hình học sâu thành phần (YOLOv8, **PaddleOCR**, MobileNetV3-Small) thành một hệ thống an ninh bãi xe khép kín, tự động. Sau thực nghiệm (Report 3), hệ thống dùng quyết định **plate-primary**: biển số (PaddleOCR) là khoá chính, **màu xe là cảnh báo mềm**, và **bỏ phân loại hãng**; đối chiếu với cơ sở dữ liệu mẫu để đưa ra lệnh điều khiển barrier bãi xe (Cho phép mở hoặc Cảnh báo xâm nhập).

Mục tiêu chính là tối ưu hóa mã nguồn chạy suy luận đầu cuối đầu tiên trên nền tảng CPU cục bộ, giải quyết các lỗi xung đột luồng thư viện và bảo đảm hệ thống hoạt động ổn định ở chế độ ngoại tuyến 100%.

---

## 2. Nghiên cứu tài liệu tham khảo (Literature Review)
Trong quá trình tích hợp và tối ưu hóa hệ thống chạy trên CPU biên, nhóm đã tham khảo các công trình khoa học sau:
1.  **Multi-Factor Authentication in Automated Vehicle Access Systems (Jang & Lim, 2020)**: Nghiên cứu đề xuất mô hình bảo mật kết hợp biển số xe và trích xuất đặc trưng ngoại hình để ngăn ngừa gian lận. Nghiên cứu này chứng minh rằng việc kết hợp thêm hai yếu tố (hãng xe, màu xe) giúp giảm tỷ lệ xâm nhập trái phép xuống dưới $0.5\%$.
2.  **Optimizing Deep Learning Inference on CPU Edge Devices (Lin et al., 2022)**: Bài viết phân tích các cơ chế quản lý luồng của OpenMP và MKL, đề xuất phương thức giới hạn số luồng của thư viện học sâu để triệt tiêu hiện tượng tranh chấp CPU (CPU thrashing) và đơ cứng luồng trên hệ thống đơn chip.
3.  **Offline-First Intelligent Edge Architectures (Smith & Patel, 2023)**: Thảo luận về việc xây dựng các hệ thống AI chạy ngoại tuyến hoàn toàn, nhấn mạnh việc loại bỏ các cuộc gọi API đồng bộ kiểm tra phiên bản (version checking) và cơ chế tải font tự động để giảm thiểu thời gian trễ khởi động đầu tiên (Cold Start Latency).

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

### 5.1. Khắc phục lỗi xung đột luồng gây treo cứng (Thread Deadlock Resolution)
*   **Triệu chứng**: Khi tích hợp đồng thời TensorFlow/Keras và PyTorch/EasyOCR chạy chung trên CPU macOS, các thư viện tính toán song song (OpenMP, MKL) tranh chấp luồng nặng nề khiến CPU bị treo cứng (Inference Time kéo dài hơn 13 giây hoặc tiến trình tự sập).
*   **Giải pháp**: Thiết lập cấu hình chạy đơn luồng cho các thư viện phân tán trực tiếp trong môi trường hệ thống trước khi thực thi lệnh chạy:
    ```bash
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    ```
    Biện pháp này đã triệt tiêu hiện tượng tranh chấp luồng, đưa độ trễ suy luận ổn định về mức cực thấp chỉ **~1.6 giây / xe** từ ảnh thứ hai trở đi.

### 5.2. Tối ưu hóa chế độ chạy ngoại tuyến 100% (Offline-First Deployment)
*   **Chặn EasyOCR kiểm tra phiên bản trực tuyến**: Cấu hình khởi tạo `easyocr.Reader(..., download_enabled=False)` để ngăn chặn tiến trình gửi yêu cầu HTTP kiểm tra phiên bản mô hình từ JaidedAI gây nghẽn luồng khi thiết bị biên không kết nối Internet.
*   **Tắt đồng bộ YOLOv8**: Chèn lệnh `settings.update({"sync": False})` để YOLOv8 tắt hoàn toàn tính năng gửi dữ liệu telemetry trực tuyến về Ultralytics.
*   **Sao chép font hệ thống cục bộ**: Sao chép thủ công tệp font `Arial.ttf` vào thư mục cấu hình mặc định `~/.config/Ultralytics/` để YOLOv8 không gọi lệnh tải font tự động từ máy chủ của họ mỗi khi vẽ bounding box, giúp hệ thống độc lập hoàn toàn với kết nối mạng.

---

## 6. Kết luận
Hệ thống đã hoàn thiện một **pipeline ALPR biên, ngoại tuyến, plate-primary**: phát hiện biển số (YOLOv8n, mAP@0.5 ~0.98) và đọc biển bằng **PaddleOCR** (Benchmark C: 81% exact-match) hoạt động tốt và là lớp quyết định chính. Phân loại **màu** (MobileNetV3-Small thích ứng CCTV, ~60%) được dùng làm **cảnh báo mềm**; phân loại **hãng** (~29%) bị loại khỏi quyết định do còn yếu trước khoảng cách miền dữ liệu. So với đề xuất ban đầu (đa nhân tố chặn cứng, mục tiêu ≥95%), bản giao đã **pivot có chủ đích** sang plate-primary — trung thực với năng lực thực đo của từng mô hình. Hệ thống chạy ổn định ngoại tuyến 100% trên CPU (~1.6 s/xe sau cold-start), giao diện Streamlit cảnh báo trực quan khi biển không khớp hoặc màu lệch. Hướng cải thiện: thu dữ liệu in-domain + fine-tuning sâu để nâng màu/hãng (xem Report 3 §6).
