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
| (Đối chiếu Plate, Màu)      |     | MISMATCH/UNREG (Sai) | --> | Cảnh báo Còi (Đỏ)    |
+-----------------------------+     +----------------------+     +----------------------+
```
> **Lưu ý:** Sơ đồ trên giữ nguyên hai nhánh classifier (EfficientNet cho hãng, MobileNetV3 cho màu) đúng như mã nguồn chạy suy luận để tham khảo, nhưng ở **cơ chế quyết định delivered**, **hãng xe (brand) đã bị loại khỏi đối chiếu** — `DatabaseMatcher` chỉ đối chiếu **Biển số (khoá chính)** và **Màu xe (cảnh báo mềm)**; dự đoán hãng chỉ còn vai trò *diagnostic*, không ảnh hưởng đến AUTHORIZED/MISMATCH/UNREGISTERED (xem Report 3 §5.5).

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

> **Lưu ý (cột Màu xe đã lỗi thời):** Bảng E2E trên được chạy bằng phiên bản **cũ** của model phân loại màu (~55% accuracy, dự đoán gần ngẫu nhiên — toàn bộ 5 ảnh đều ra "Silver" với confidence thấp ~0.18, đúng với hành vi của model yếu thời điểm đó). Model màu hiện tại đã được nâng cấp lên **86% (TTA)** trên VCoR (xem Report 3 §5.1). Cột "Hãng xe"/"Màu xe dự đoán (Conf)" ở bảng này **cần được chạy lại** với model màu mới để cập nhật số liệu đối chiếu E2E — chưa có số đo lại nên không bịa số mới ở đây.

### 4.2. Thống kê số liệu tổng hợp (Aggregate Metrics)
*   **Tổng số lượng mẫu kiểm thử**: 5 xe
*   **Xe Hợp lệ (AUTHORIZED)**: 2 xe (barrier mở tự động)
*   **Xe Lệch thông tin hãng/màu (MISMATCH)**: 1 xe (khóa barrier, báo động đỏ)
*   **Xe Không đăng ký (UNREGISTERED)**: 2 xe (khóa barrier, báo động đỏ)
*   **Thời gian phản hồi trung bình (Average Latency)**: **2,190.89 ms / xe** trên 5 ảnh test — con số này gồm *cold-start* (~4.49 s ở ảnh đầu do nạp model lần đầu). Từ ảnh thứ hai trở đi, độ trễ ổn định ở mức **~1.6 s / xe** (xem §5.1). Mục tiêu <1.0 s ở đề xuất chưa đạt do chạy thuần CPU + tải PaddleOCR.

### 4.3. Đánh giá năng lực chống tráo biển (Plate-swap Detection)

Đề xuất ban đầu (Report 1) đặt mục tiêu **≥95% phát hiện gian lận**, nhưng số đo ở §4.1–4.2 chỉ chạy trên **5 ảnh thực tế** và không đo riêng năng lực an ninh — đây là khoảng trống lớn nhất giữa cam kết và số liệu thực đo của toàn dự án. Mục này lấp khoảng trống đó bằng một **thực nghiệm có kiểm soát (controlled evaluation)**, đo trực tiếp trên model màu đang triển khai và logic quyết định thật, KHÔNG sửa mã nguồn/runtime.

**Khái niệm đo:** hệ thống là **plate-primary**, màu xe là **cảnh báo mềm** (`DatabaseMatcher.verify_vehicle`, xem §3). Cơ chế chống tráo biển hoạt động như sau — nếu một biển số bị nhân bản (clone) từ xe A (màu đăng ký C1) và gắn lên xe B khác màu (màu thật C2 ≠ C1), bộ phân loại màu sẽ dự đoán C2 trên ảnh xe B; vì C2 ≠ C1 (màu đăng ký trong CSDL), `verify_vehicle` trả về `AUTHORIZED` kèm `action='ALLOW_WARN'`, `color_warning=True` — đây chính là tín hiệu "bắt được tráo biển".

**Phương pháp:**
*   Script: `main/scripts/eval_security.py` (mới, không sửa mã runtime/model).
*   **Ảnh kiểm thử**: tập TEST giữ-riêng của VCoR — **tái dùng cùng split** với `eval_color_deployed.py` (`colab_train_color.py.load_samples` + `stratified_split`, seed=42, stratified 70/15/15) → 889 ảnh giữ-riêng, model màu chưa từng thấy khi huấn luyện.
*   **Model**: trọng số ĐANG CHẠY ở runtime (`main/data/models/color_MobileNetV3Small.pt`), gọi qua `TorchColorClassifier.predict()` thật — dùng **màu dự đoán thật** của model (không dùng nhãn ground-truth), nên số đo phản ánh đúng hành vi triển khai, kể cả khi model màu đoán sai.
*   **Logic quyết định**: `DatabaseMatcher.verify_vehicle()` thật, không sửa đổi; CSDL đăng ký là **CSV tạm** dựng riêng cho lần chạy này (cùng schema `main/data/database.csv`: `license_plate,car_brand,car_color`), không đụng đến CSDL thật của repo.
*   **3 nhóm kịch bản, 200 trial/nhóm (cân bằng theo màu), RNG seed cố định (42) để tái lập được:**
    1.  **legitimate** — ảnh màu thật C, biển đăng ký đúng màu C → đúng = AUTHORIZED, không cảnh báo. Cảnh báo ở đây là **báo động giả (false alarm)**.
    2.  **plate_swap** — ảnh màu thật C2, biển đăng ký màu KHÁC C1≠C2 (giả lập tráo biển) → đúng = `color_warning=True` (bắt được tráo); không cảnh báo = **bỏ lọt (missed)**.
    3.  **unregistered** — biển hoàn toàn không có trong CSDL → đúng = UNREGISTERED/DENY_ALERT.

**Kết quả (số đo thật, 600 trial tổng, 889 ảnh pool giữ-riêng, chạy 20/06/2026):**

| Kịch bản | Số trial | Chỉ số | Kết quả |
| :--- | :---: | :--- | :---: |
| **Phát hiện tráo biển (headline)** | 200 | tỉ lệ bắt được (`color_warning=True`) | **98,5% (197/200)** |
| Tráo biển bị bỏ lọt | 200 | tỉ lệ miss | 1,5% (3/200) |
| Xe hợp lệ (không tráo) | 200 | tỉ lệ báo động giả | 14,5% (29/200) |
| Biển không đăng ký | 200 | tỉ lệ phát hiện (DENY_ALERT) | 100,0% (200/200) |

*   **Tỉ lệ phát hiện tráo biển = 98,5%** — vượt mục tiêu ≥95% của đề xuất ban đầu, **đo được lần đầu** trên model + logic thật (không phải số ước lượng).
*   **Tỉ lệ báo động giả = 14,5%** — xe hợp lệ vẫn có thể bị cảnh báo nhầm do chính model màu dự đoán sai dù biển đúng; đây là cái giá đánh đổi của việc dùng màu làm cảnh báo mềm (chấp nhận được vì hệ thống không từ chối cứng, chỉ cảnh báo).
*   **Cặp màu bị bỏ lọt nhiều nhất**: Grey→Brown (1/2, 50%), Silver↔Grey (1/5 mỗi chiều, 20%) — đúng như dự đoán, rơi vào **cụm màu trung tính** (Black/Grey/Silver/White). Đo riêng cụm này: 50 trial, miss 2 (**4,0%**, cao hơn tỉ lệ miss tổng 1,5%) — khớp với confusion matrix màu đã ghi nhận ở Report 3 §5.1 (Grey/Silver là cặp khó nhất của chính bộ phân loại màu).
*   Chi tiết đầy đủ (toàn bộ 52 cặp màu, breakdown JSON): `docs/benchmarks/security_eval.md` và `docs/benchmarks/security_eval.json`.

**Giới hạn trung thực (đọc trước khi trích số 98,5%):**
1.  **Chỉ bắt được khi xe gắn biển tráo có MÀU KHÁC màu đăng ký.** Nếu kẻ tráo biển dùng đúng xe cùng màu (hoặc cố tình chọn xe cùng màu/dán decal giả màu), cơ chế cross-check màu **không có khả năng phát hiện** — đây là lỗ hổng cố hữu của thiết kế "màu là cảnh báo mềm", không phải lỗi đo lường hay có thể vá bằng cách huấn luyện lại model màu.
2.  **Phụ thuộc hoàn toàn vào việc OCR đọc đúng biển số trước đó** (Benchmark C: ~81% exact-match, xem Report 3). Thực nghiệm này đo cách ly riêng bước cross-check màu, giả định biển đã đọc đúng; trong vận hành thật, nếu OCR đọc sai/đọc thiếu biển, xe có thể rơi vào UNREGISTERED hoặc match nhầm bản ghi khác — tỉ lệ 98,5% ở trên **không bao gồm** lỗi OCR thực tế nối tiếp.
3.  **Đo trên VCoR (ảnh web/marketplace sạch)**, không phải ảnh CCTV bãi xe thật. CCTV thực tế (ánh sáng yếu, góc nghiêng, nén ảnh, độ phân giải thấp) nhiều khả năng cho tỉ lệ phát hiện **thấp hơn** 98,5% do domain gap — cùng caveat đã nêu cho độ chính xác màu ở Report 3 §5.1 / Report 4 §5.1.

**Kết luận của mục này:** đây là **lần đầu tiên** dự án có số đo định lượng cho năng lực an ninh đã cam kết ở đề xuất (mục tiêu ≥95% chưa từng được đo trước thực nghiệm này). Cross-check màu nâng đáng kể khả năng phát hiện tráo biển so với việc chỉ dùng biển số đơn thuần (chỉ dùng biển = 0% phát hiện tráo biển cùng-biển-khác-xe, vì biển vẫn khớp CSDL), nhưng **không phải là đảm bảo cứng** — nó là một lớp phòng thủ bổ sung có giới hạn rõ ràng (màu phải khác, OCR phải đúng, domain phải gần VCoR), cần kết hợp thêm các lớp khác (camera giám sát người vận hành, đối soát định kỳ) để đạt an ninh toàn diện.

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
Hệ thống đã hoàn thiện một **pipeline ALPR biên, ngoại tuyến, plate-primary**: phát hiện biển số (YOLOv8n, mAP@0.5 ~0.98) và đọc biển bằng **PaddleOCR** (Benchmark C: 81% exact-match) hoạt động tốt và là lớp quyết định chính. Phân loại **màu** (runtime: **PyTorch MobileNetV3-Small**, `color_MobileNetV3Small.pt`) đã được nâng cấp đáng kể: từ baseline lịch sử ~55% (frozen-backbone, data cũ) lên **86,3% TTA (85,3% plain)** trên tập test giữ-riêng VCoR sau full fine-tune + class-weight + label-smoothing + TTA (chi tiết: Report 3 §5.1). Màu xe nay là một **thành phần mạnh** của hệ thống (~86%), nhưng **vẫn giữ vai trò "cảnh báo mềm"** thay vì khoá chính — không phải vì model yếu, mà vì số 86% đo trên ảnh VCoR sạch (web/marketplace), **chưa được kiểm chứng trên domain CCTV bãi xe thật** (ánh sáng yếu, nhiễu, góc nghiêng — xem caveat Report 3 §5.1); phân loại **hãng** (TF/Keras EfficientNet-B0, ~**35%** — *diagnostic only — đã loại khỏi quyết định*) bị loại khỏi quyết định do còn yếu. Classifier màu phục vụ runtime qua `torch_color.py` để đồng tồn với PaddleOCR mà không cần cách ly tiến trình TF. So với đề xuất ban đầu (đa nhân tố chặn cứng, mục tiêu ≥95%), bản giao đã **pivot có chủ đích** sang plate-primary — trung thực với năng lực thực đo của từng mô hình. Hệ thống chạy ổn định ngoại tuyến 100% trên CPU (~1.6 s/xe sau cold-start), giao diện Streamlit cảnh báo trực quan khi biển không khớp hoặc màu lệch. Hướng cải thiện: thu dữ liệu in-domain CCTV để xác nhận màu giữ được ~86% ngoài VCoR + fine-tuning sâu hơn để nâng hãng (xem Report 3 §6).
