# KỊCH BẢN THUYẾT TRÌNH GIAI ĐOẠN 2: DỮ LIỆU

---

## Slide 2: Đặt vấn đề và Mục tiêu

Trong thực tế vận hành các bãi đỗ xe hiện nay, các hệ thống an ninh đơn yếu tố dựa trên thẻ từ hoặc chỉ sử dụng OCR nhận diện biển số xe rất dễ bị vượt qua bằng các thủ đoạn tinh vi như tháo lắp hoặc tráo đổi biển số. Một biển số xe vật lý thì quá dễ bị giả mạo. Để giải quyết vấn đề này, nhóm chúng em đề xuất một cơ chế đối chiếu chéo đa nhân tố dựa trên ba đặc trưng sinh trắc học trực quan của phương tiện: Khóa định danh chính là Biển số xe, đi kèm hai chiều đối chiếu bổ trợ là Hãng xe và Màu sắc xe. Sự kết hợp này sẽ giúp phát hiện lập tức bất kỳ sự không khớp thuộc tính nào tại cổng kiểm soát.

---

## Slide 3: Tổng quan nghiên cứu & Cơ sở khoa học

Nghiên cứu của chúng em kế thừa trực tiếp và chặt chẽ trên ba nền tảng cơ sở khoa học đã được công bố:
- **Thứ nhất**, Stanford Cars Dataset của Krause và cộng sự năm 2013, thiết lập tiêu chuẩn phân nhóm thương hiệu và cung cấp phương pháp phân nhóm dòng xe làm nền tảng.
- **Thứ hai**, Nghiên cứu nhận diện màu sắc xe của Chen và cộng sự năm 2014, cung cấp lý thuyết chuẩn hóa dải điểm ảnh trước khi đưa vào mạng nơ-ron tích chập siêu nhẹ dưới điều kiện ánh sáng phức tạp của camera giám sát.
- **Thứ ba**, Khảo sát về các phương pháp tăng cường dữ liệu của Yang và cộng sự năm 2022 (arXiv:2204.08610), làm cơ sở lý thuyết cho các phép biến đổi hình học và dịch chuyển độ tương phản nhằm đối phó với hiện tượng camera bị nghiêng nhẹ.

---

## Slide 4: Quy trình thu thập dữ liệu đa nguồn

Để huấn luyện các bộ phân loại, chúng em đã thu thập dữ liệu đa nguồn cho từng tác vụ cụ thể. Tác vụ phân loại hãng xe thu về 792 ảnh sạch bao phủ 8 thương hiệu phổ biến ở Việt Nam. Đối với dòng xe VinFast, nhóm chủ động cào Bing chi tiết theo từng dòng xe con (VF8, VF9, Fadil...) để tránh thu phải ảnh logo hay nội thất nhiễu. Tác vụ phân loại màu sắc thu thập 783 ảnh cho 8 màu cơ bản, đồng thời loại bỏ 39 ảnh lớp màu xanh lá không dùng do mô hình không hỗ trợ. Bộ định vị biển số YOLOv8-nano được huấn luyện trên dataset HuggingFace gồm 6,176 ảnh train. Cuối cùng, tập ảnh chụp xe thực tế tại các bãi đỗ Việt Nam được nhóm thu thập thủ công để làm tập kiểm thử tích hợp cuối cùng. Toàn bộ tập dữ liệu phân loại được chia vật lý theo tỷ lệ 70/15/15 với seed 42 cố định.

---

## Slide 5: Đường ống làm sạch dữ liệu tự động

Một đóng góp kỹ thuật quan trọng trong giai đoạn này là đường ống làm sạch dữ liệu tự động gồm 5 bước nhằm đảm bảo chất lượng nhãn:
- **Bước 1**: `clean_corrupted_images` sử dụng OpenCV lọc bỏ các file ảnh lỗi cấu trúc vật lý.
- **Bước 2**: `semantic_clean_images` dùng YOLOv8-nano để lọc ngữ nghĩa, loại bỏ khoảng 38% ảnh nhiễu không thực sự chứa xe (như ảnh cận cảnh vô lăng, logo hoặc đường phố trống).
- **Bước 3**: `remove_duplicates` áp dụng thuật toán băm cảm nhận pHash (Perceptual Hashing) thông qua thư viện `imagehash` với khoảng cách Hamming nhỏ hơn hoặc bằng 5 để xóa ảnh trùng lặp hoặc gần trùng.
- **Bước 4**: `normalize_images` đồng bộ hóa định dạng ép tất cả sang JPEG RGB chuẩn.
- **Bước 5**: Nhóm tiến hành cào bổ sung chuyên biệt và cắt ngưỡng tối đa (cap) dữ liệu ở mức ~100 ảnh mỗi lớp để đạt phân bố cân bằng tuyệt đối.

---

## Slide 6: Phân tích thống kê phân bố lớp

Sau khi hoàn tất quá trình làm sạch tự động, chúng em tiến hành Phân tích thống kê dữ liệu. Biểu đồ phân bố lớp thể hiện sự cân bằng hoàn hảo của 8 lớp màu xe và 8 lớp hãng xe. Chúng em theo đuổi triết lý dữ liệu "ít nhưng chất". Thay vì lạm dụng các phép tăng cường dữ liệu nhân tạo trên tập dữ liệu mất cân bằng nghiêm trọng ban đầu (như màu Vàng chỉ có 25 ảnh, trong khi màu Đen có hơn 200 ảnh), nhóm đã chủ động thực hiện thu thập bù chuyên biệt để nâng số lượng ảnh thật lên đồng đều ~100 ảnh/lớp, giúp hệ số lệch lớp xấp xỉ bằng 1, triệt tiêu nguy cơ thiên lệch mô hình.

---

## Slide 7: Tiền xử lý và tăng cường hình ảnh

Quy trình tiền xử lý và tăng cường hình ảnh được thiết kế chặt chẽ và chỉ áp dụng các phép tăng cường trên tập huấn luyện (training set):
- **1. Resize 224²**: Đồng bộ kích thước hình ảnh đầu vào chuẩn của hai backbone mạng.
- **2. Lật ngang (Horizontal Flip)**: Tạo các biến thể góc xe tiếp cận bãi đỗ từ cả hai phía.
- **3. Xoay ngẫu nhiên (Random Rotation $\pm 10^\circ$)**: Chống xoay và lệch góc camera.
- **4. Phóng ngẫu nhiên (Random Zoom $\pm 10\%$)**: Mô phỏng sự thay đổi khoảng cách từ xe tới camera.
- **5. Pixel Scaling**: Chuyển đổi dải giá trị điểm ảnh qua lớp Rescaling(255.0) tích hợp ở đầu mô hình.
- **6. Backbone Preprocessing**: Đưa dữ liệu qua bộ tiền xử lý tích hợp của từng backbone MobileNetV3 và EfficientNet tương ứng.

---

## Slide 8: Chẩn đoán lỗi & Khắc phục

Trong quá trình phát triển ban đầu, chúng em đã chẩn đoán và khắc phục thành công hai lỗi kỹ thuật nghiêm trọng liên quan đến cấu trúc mô hình khiến độ chính xác bị kẹt ở mức ngẫu nhiên ~12.5% (tức là 1 chia cho 8 lớp) và loss đi ngang không hội tụ:
- **Lỗi thứ nhất: BatchNorm Bug**. Việc sử dụng Sequential API làm đóng băng backbone nhưng các lớp BatchNorm vẫn tiếp tục chạy theo thống kê của từng batch huấn luyện thay vì dùng moving average lúc inference. Nhóm đã chuyển sang Functional API và gọi backbone dưới dạng `base_model(x, training=False)` để khóa cứng BN chạy ở chế độ inference.
- **Lỗi thứ hai: Double-preprocessing**. Cấu hình nhầm dải pixel đầu vào làm sai lệch phân bố đặc trưng. Chúng em đã loại bỏ scaling ngoài, đồng bộ đưa ảnh dạng $[0, 1]$ qua lớp `Rescaling(255.0)` về dải $[0, 255]$ chuẩn trước khi nạp vào backbone. Nhờ đó mô hình đã hội tụ ổn định.

---

## Slide 9: Hiệu năng phân loại trên tập test giữ-riêng

Sau khi khắc phục các lỗi chẩn đoán, kết quả huấn luyện trên tập kiểm thử độc lập (held-out test splits) ghi nhận như sau:
- Bộ phân loại màu xe sử dụng backbone `MobileNetV3-Small` đạt Test Accuracy **55.1%** và Macro-F1 **0.545**. Kết quả này gấp hơn 4 lần mức ngẫu nhiên (12.5%), đủ độ tin cậy để làm thuộc tính cảnh báo phụ.
- Bộ phân loại hãng xe sử dụng backbone `EfficientNet-B0` chỉ đạt Test Accuracy **35.3%** và Macro-F1 **0.337** — đo trên tập test **ảnh web sạch** (không phải ảnh CCTV). Nguyên nhân gốc là bài toán phân biệt thương hiệu xe có độ khó cao (fine-grained) kết hợp dữ liệu ít (~70 ảnh/lớp); ảnh camera mờ chỉ làm kết quả tệ thêm chứ không phải nguyên nhân chính.

Từ dữ liệu thực chứng này, nhóm đã đưa ra một quyết định kỹ thuật quan trọng: loại bỏ thuộc tính hãng xe khỏi hệ thống ở các giai đoạn sau (R3/R4) để tránh gây ra lỗi từ chối sai (false rejection) khiến cổng an ninh không mở cho xe hợp lệ.

---

## Slide 10: Kết luận và lộ trình phát triển

Tóm lại, trong Giai đoạn 2, nhóm đã hoàn tất việc xây dựng quy trình thu thập dữ liệu đa nguồn và đường ống làm sạch tự động. Tập dữ liệu bàn giao hoàn toàn sạch, cân bằng và đã vượt qua 100% kiểm thử tự động của file `test_dataset.py`.
Lộ trình phát triển tiếp theo của nhóm hướng tới 3 mục tiêu:
1. **Dữ liệu thực tế**: Thu thập thêm dữ liệu camera thực tế tại các bãi đỗ xe Việt Nam để thu hẹp khoảng cách miền dữ liệu (domain gap).
2. **Fine-tune Backbone**: Thử nghiệm mở băng (fine-tune) một số block cuối của MobileNetV3 để tối ưu hóa đặc trưng lớp màu xe.
3. **ONNX Quantization**: Nén và lượng tử hóa mô hình sang định dạng ONNX, chuẩn bị cho việc tích hợp biên thời gian thực.

---

## Slide 11: Lời cảm ơn & Hỏi đáp

Chúng em xin chân thành cảm ơn thầy cô trong hội đồng và các bạn đã chú ý lắng nghe phần trình bày báo cáo Giai đoạn 2 của nhóm. Nhóm kính mong nhận được những nhận xét, đóng góp ý kiến phản biện để hệ thống được hoàn thiện hơn trong các giai đoạn tiếp theo. Sau đây, chúng em xin phép được bắt đầu phiên Hỏi & Đáp (Q&A).
