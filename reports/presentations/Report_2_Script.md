# KỊCH BẢN THUYẾT TRÌNH GIAI ĐOẠN 2: DỮ LIỆU

---

# NGƯỜI 1 — MỞ ĐẦU & BỐI CẢNH (Slide 2–5)

## Slide 2: Đặt vấn đề

Trong thực tế vận hành các bãi đỗ xe hiện nay, các hệ thống an ninh đơn yếu tố dựa trên thẻ từ hoặc chỉ sử dụng OCR nhận diện biển số xe rất dễ bị vượt qua bằng các thủ đoạn tinh vi như tháo lắp hoặc tráo đổi biển số. Một biển số xe vật lý thì quá dễ bị giả mạo — chỉ cần vài phút để sao chép. Để giải quyết vấn đề này, nhóm chúng em đề xuất một cơ chế đối chiếu chéo đa nhân tố kết hợp ba đặc trưng sinh trắc học trực quan của phương tiện, giúp phát hiện lập tức bất kỳ sự không khớp thuộc tính nào tại cổng kiểm soát. Mục tiêu cụ thể của Giai đoạn 2 là xây dựng pipeline dữ liệu sạch, cân bằng — nền tảng không thể thiếu cho các mô hình nhận diện ở các giai đoạn sau.

---

## Slide 3: Giải pháp — Đối chiếu chéo 3 thuộc tính

Nguyên lý cốt lõi của hệ thống là: ba thuộc tính phải cùng khớp thì cổng mới mở. Cụ thể, chúng em xây dựng ba lớp đối chiếu bổ trợ lẫn nhau. Lớp thứ nhất là Biển số xe — đây là khóa định danh chính, nhận dạng bằng OCR, là điều kiện cần bắt buộc để tra cứu hồ sơ đăng ký. Lớp thứ hai là Hãng xe — nhân tố đối chiếu phụ, vì biển số giả rất khó đồng thời khớp đúng hãng xe thật trong hồ sơ. Lớp thứ ba là Màu xe — chiều đối chiếu sinh trắc học trực quan, camera bãi xe nhận diện trực tiếp và dùng làm tín hiệu cảnh báo. Sự kết hợp ba lớp này khiến xác suất gian lận giảm mạnh vì kẻ xấu không thể đồng thời làm giả cả ba thuộc tính.

---

## Slide 4: Tổng quan nghiên cứu & Cơ sở khoa học

Nghiên cứu của chúng em kế thừa trực tiếp và chặt chẽ trên ba nền tảng cơ sở khoa học đã được công bố. Thứ nhất là Stanford Cars Dataset của Krause và cộng sự năm 2013, thiết lập tiêu chuẩn phân nhóm thương hiệu và cung cấp phương pháp phân nhóm dòng xe làm nền tảng. Thứ hai là nghiên cứu nhận diện màu sắc xe của Chen và cộng sự năm 2014, cung cấp lý thuyết chuẩn hóa dải điểm ảnh trước khi đưa vào mạng nơ-ron tích chập siêu nhẹ dưới điều kiện ánh sáng phức tạp của camera giám sát. Thứ ba là khảo sát về các phương pháp tăng cường dữ liệu của Yang và cộng sự năm 2022 (arXiv:2204.08610), làm cơ sở lý thuyết cho các phép biến đổi hình học và dịch chuyển độ tương phản nhằm đối phó với hiện tượng camera bị nghiêng nhẹ.

---

## Slide 5: Mục tiêu & Phạm vi Giai đoạn 2

Trước khi đi vào chi tiết kỹ thuật, nhóm muốn làm rõ phạm vi và các kết quả bàn giao của Giai đoạn 2. Triết lý xuất phát điểm là: không có tập dữ liệu sạch, cân bằng và đại diện thì mọi kiến trúc mô hình đều vô nghĩa — vì vậy toàn bộ giai đoạn này tập trung giải quyết triệt để vấn đề dữ liệu trước khi bước sang giai đoạn huấn luyện. Phạm vi kỹ thuật bao gồm: thu thập đa nguồn, làm sạch tự động 5 bước, cân bằng lớp, tiền xử lý và augmentation, huấn luyện thử nghiệm, và đánh giá baseline. Kết quả bàn giao gồm ba hạng mục: một dataset sạch cân bằng gồm 792 ảnh hãng xe và 783 ảnh màu xe với khoảng 100 ảnh mỗi lớp, chia theo tỉ lệ 70/15/15 với seed 42; một pipeline làm sạch tự động 5 bước vượt qua 100% kiểm thử của file `test_dataset.py`; và bộ baseline model đã được đánh giá thực chứng — Màu xe đạt 55.1%, Hãng xe đạt 35.3% — làm định hướng thiết kế hệ thống.

---

# NGƯỜI 2 — THU THẬP DỮ LIỆU (Slide 6–9)

## Slide 6: Thu thập đa nguồn — Tổng quan

Để huấn luyện các bộ phân loại, chúng em thu thập dữ liệu đa nguồn cho từng tác vụ cụ thể. Tác vụ phân loại hãng xe thu về 792 ảnh sạch bao phủ 8 thương hiệu phổ biến tại Việt Nam. Tác vụ phân loại màu sắc thu thập 783 ảnh cho 8 màu cơ bản, đồng thời loại bỏ 39 ảnh lớp màu xanh lá không dùng. Bộ định vị biển số YOLOv8-nano được huấn luyện trên dataset HuggingFace gồm 6,176 ảnh. Cuối cùng, toàn bộ tập dữ liệu phân loại được chia vật lý theo tỉ lệ 70/15/15 với seed 42 cố định — ảnh bãi xe thực tế tại Việt Nam thu thủ công đóng vai trò tập kiểm thử tích hợp đầu-cuối. Công cụ thu thập chính là icrawler cào đa từ khóa để vượt giới hạn rate limit.

---

## Slide 7: Dữ liệu Hãng xe

Nhìn sâu hơn vào tập dữ liệu hãng xe. Tổng cộng nhóm thu về 792 ảnh bao phủ 8 thương hiệu phổ biến nhất tại thị trường Việt Nam, gồm Toyota, Honda, Hyundai, Kia, Mazda, Ford, Mitsubishi và VinFast. Sau quá trình làm sạch và cân bằng, mỗi lớp đạt khoảng 100 ảnh với hệ số lệch xấp xỉ 1.0. Nguồn dữ liệu kết hợp giữa cào Bing đa từ khóa và Stanford Cars dataset làm hạt nhân ban đầu. Đặc biệt, với VinFast — thương hiệu xe nội địa có lượng ảnh nhiễu rất lớn trên mạng — nhóm đã cào Bing chi tiết theo từng dòng xe cụ thể là VF8, VF9 và Fadil, thay vì dùng từ khóa chung "VinFast". Cách làm này giúp tránh thu phải ảnh logo thương hiệu, ảnh nội thất và ảnh nhiễu từ các sự kiện ra mắt xe.

---

## Slide 8: Dữ liệu Màu xe

Với tập dữ liệu màu xe, nhóm thu thập tổng cộng 783 ảnh cho 8 màu cơ bản được mô hình hỗ trợ: Đen, Trắng, Xám, Đỏ, Xanh dương, Vàng, Nâu và Khác. Phương pháp thu thập là cào Bing đa từ khóa theo màu cụ thể — ví dụ "black car vietnam parking" hay "white sedan" — để đảm bảo ảnh mang đặc trưng màu rõ ràng trong điều kiện bãi xe thực tế. Điểm cần lưu ý là ban đầu nhóm có 39 ảnh thuộc lớp màu xanh lá, nhưng đã loại bỏ hoàn toàn lớp này vì hai lý do: màu xanh lá xe rất hiếm gặp tại Việt Nam, và mô hình không hỗ trợ lớp đó trong thiết kế 8 lớp. Việc loại bỏ sớm từ giai đoạn thu thập giúp tránh nhãn nhiễu lan sang các bước làm sạch sau.

---

## Slide 9: Biển số & Tập kiểm thử thực tế

Thành phần cuối cùng trong khâu thu thập là dữ liệu biển số và tập kiểm thử tích hợp. Bộ định vị biển số dùng mô hình YOLOv8-nano — backbone siêu nhẹ, tốc độ inference nhanh, phù hợp cho bài toán edge deployment tại cổng bãi xe — được huấn luyện trên dataset HuggingFace gồm 6,176 ảnh biển số xe máy và ô tô. Song song đó, nhóm thu thủ công ảnh chụp bãi đỗ xe thực tế tại Việt Nam để làm tập kiểm thử tích hợp đầu-cuối cho toàn bộ pipeline nhận diện. Về phân chia tập dữ liệu phân loại: train 70%, validation 15%, test held-out 15%, với seed 42 cố định để đảm bảo tính tái tạo hoàn toàn. Cụ thể, tập test held-out gồm 118 ảnh màu xe và 119 ảnh hãng xe — không bao giờ được dùng để huấn luyện hay tuning mô hình.

---

# NGƯỜI 3 — LÀM SẠCH & EDA (Slide 10–13)

## Slide 10: Đường ống làm sạch — Bước 1–3

Một đóng góp kỹ thuật quan trọng trong giai đoạn này là đường ống làm sạch dữ liệu tự động gồm 5 bước. Trước tiên là ba bước đầu. Bước 1, `clean_corrupted_images`, sử dụng OpenCV để đọc và kiểm tra cấu trúc vật lý từng file ảnh, loại bỏ toàn bộ ảnh bị lỗi encoding, truncated hoặc không decode được — đảm bảo pipeline không crash ở bước sau. Bước 2, `semantic_clean_images`, dùng YOLOv8-nano chạy detection trên từng ảnh để lọc ngữ nghĩa, loại bỏ khoảng 38% ảnh nhiễu không thực sự chứa xe, bao gồm ảnh cận cảnh vô lăng, ảnh logo thương hiệu, đường phố trống và nội thất xe. Bước 3, `remove_duplicates`, áp dụng thuật toán băm cảm nhận pHash qua thư viện `imagehash` với khoảng cách Hamming nhỏ hơn hoặc bằng 5 để phát hiện và xóa ảnh trùng lặp hoặc gần trùng — tránh data leakage giữa các tập.

---

## Slide 11: Đường ống làm sạch — Bước 4–5

Tiếp nối đường ống, bước 4 là `normalize_images`: ép toàn bộ ảnh sang định dạng JPEG RGB chuẩn, loại bỏ ảnh RGBA, grayscale và palette mode để đảm bảo Keras đọc nhất quán, đồng thời xử lý cả trường hợp ảnh có alpha channel hoặc EXIF metadata gây lỗi xoay. Bước 5 gồm hai thao tác kết hợp: cào bù chuyên biệt cho các lớp thiếu — ví dụ lớp Vàng còn khoảng 25 ảnh, lớp Trắng khoảng 40 ảnh sau 4 bước lọc — rồi cắt ngưỡng tối đa khoảng 100 ảnh mỗi lớp để đạt phân bố cân bằng tuyệt đối. Triết lý xuyên suốt là "ít nhưng chất": ưu tiên thu thập ảnh thật cân bằng thay vì dùng augmentation nhân tạo trên tập mất cân bằng nghiêm trọng. Kết quả là hệ số lệch lớp cuối xấp xỉ 1.0.

---

## Slide 12: EDA — Phân bố lớp cân bằng

Sau khi hoàn tất đường ống làm sạch, chúng em tiến hành phân tích thống kê phân bố lớp. Biểu đồ phân bố cho thấy sự cân bằng hoàn hảo của cả 8 lớp hãng xe lẫn 8 lớp màu xe. Ban đầu, mất cân bằng rất nghiêm trọng: lớp Vàng chỉ có 25 ảnh trong khi lớp Đen có hơn 200 ảnh. Thay vì lạm dụng augmentation nhân tạo để che đậy vấn đề này, nhóm chủ động thực hiện thu thập bù chuyên biệt để nâng số lượng ảnh thật lên đồng đều khoảng 100 ảnh mỗi lớp. Kết quả cuối cùng: 8 lớp hãng xe với tổng 792 ảnh cân bằng, 8 lớp màu xe với tổng 783 ảnh cân bằng, hệ số lệch lớp xấp xỉ 1.0 — triệt tiêu nguy cơ thiên lệch mô hình về phía các lớp đa số.

---

## Slide 13: Tiền xử lý & Augmentation

Quy trình tiền xử lý và tăng cường hình ảnh được thiết kế chặt chẽ và chỉ áp dụng augmentation trên tập huấn luyện. Sáu bước lần lượt là: thứ nhất, Resize 224² để đồng bộ kích thước đầu vào chuẩn cho cả MobileNetV3-Small và EfficientNet-B0; thứ hai, Horizontal Flip để lật ngang ngẫu nhiên — nhân đôi biến thể hướng xe tiếp cận từ trái hoặc phải cổng bãi xe; thứ ba, Random Rotation ±10° để mô phỏng camera bị nghiêng nhẹ, giúp mô hình bất biến trước lệch góc gắn camera; thứ tư, Random Zoom ±10% để mô phỏng xe ở khoảng cách khác nhau tới camera; thứ năm, lớp Rescaling(255.0) tích hợp đầu mô hình chuyển dải [0,1] sang [0,255] — không áp dụng scaling ngoài; và thứ sáu, Backbone Preprocessing tích hợp riêng của từng backbone MobileNetV3 và EfficientNet.

---

# NGƯỜI 4 — MÔ HÌNH & KẾT QUẢ (Slide 14–17)

## Slide 14: Chẩn đoán & Khắc phục 2 lỗi

Trong quá trình phát triển ban đầu, chúng em phát hiện và khắc phục thành công hai lỗi kỹ thuật nghiêm trọng khiến độ chính xác bị kẹt ở mức ngẫu nhiên khoảng 12.5% — tức 1 chia cho 8 lớp — và loss đi ngang không hội tụ. Lỗi thứ nhất là BatchNorm Bug: việc dùng Sequential API đóng băng backbone nhưng các lớp BatchNorm vẫn chạy thống kê theo batch huấn luyện thay vì dùng moving average lúc inference. Cách khắc phục là chuyển sang Functional API và gọi backbone dưới dạng `base_model(x, training=False)` để khóa cứng BatchNorm chạy ở chế độ inference. Lỗi thứ hai là Double-preprocessing: cấu hình nhầm scaling đầu vào hai lần — ảnh đã scale về [0,1] nhưng backbone lại mong nhận [0,255] — làm đặc trưng bị bão hòa và gradient triệt tiêu. Giải pháp là bỏ scaling ngoài, đồng bộ đưa ảnh qua lớp `Rescaling(255.0)` đầu model trước khi nạp vào backbone. Nhờ đó mô hình đã hội tụ ổn định.

---

## Slide 15: Kết quả test trên tập giữ-riêng

Sau khi khắc phục hai lỗi trên, kết quả đo trên tập test held-out ghi nhận như sau. Bộ phân loại màu xe dùng backbone MobileNetV3-Small đạt Test Accuracy 55.1% và Macro-F1 0.545 — gấp hơn 4 lần mức ngẫu nhiên 12.5%, đủ độ tin cậy làm tín hiệu cảnh báo phụ trong hệ thống. Bộ phân loại hãng xe dùng backbone EfficientNet-B0 chỉ đạt 35.3% và Macro-F1 0.337 — đo trên tập test ảnh web sạch, không phải ảnh CCTV. Nguyên nhân gốc là bài toán phân biệt thương hiệu xe có độ khó cao (fine-grained) kết hợp với lượng dữ liệu ít, khoảng 70 ảnh mỗi lớp sau làm sạch; ảnh camera mờ chỉ làm kết quả tệ thêm chứ không phải nguyên nhân chính. Cả hai mô hình đều vượt xa mức ngẫu nhiên — kết quả này làm cơ sở cho quyết định thiết kế hệ thống ở slide tiếp theo.

---

## Slide 16: Quyết định — Bỏ Hãng xe, giữ Màu xe làm cảnh báo

Từ dữ liệu thực chứng, nhóm đưa ra một quyết định thiết kế quan trọng: an toàn hơn là không chặn xe hợp lệ. Trong bài toán bãi đỗ xe, false rejection — tức từ chối xe hợp lệ — gây ùn tắc và mất lòng tin người dùng, nghiêm trọng hơn false acceptance. Do đó, hãng xe bị loại hoàn toàn khỏi luồng quyết định mở hoặc đóng cổng từ R3/R4 trở đi: EfficientNet-B0 chỉ đạt 35.3% trên ảnh web sạch — quá thấp để dùng làm điều kiện bắt buộc mà không gây ra lỗi từ chối sai. Ngược lại, màu xe được giữ lại làm tín hiệu cảnh báo phụ: MobileNetV3-Small đạt 55.1% và Macro-F1 0.545 — gấp hơn 4 lần baseline, đủ độ tin cậy để phát cảnh báo khi màu xe không khớp hồ sơ, nhưng không tự động chặn cổng.

---

## Slide 17: Kết luận & Roadmap

Tóm lại, trong Giai đoạn 2, nhóm đã hoàn tất xây dựng pipeline thu thập dữ liệu đa nguồn và đường ống làm sạch tự động 5 bước. Tập dữ liệu bàn giao hoàn toàn sạch, cân bằng và đã vượt qua 100% kiểm thử tự động của file `test_dataset.py`, tái tạo hoàn toàn với seed 42. Lộ trình phát triển tiếp theo hướng tới ba mục tiêu: thứ nhất, thu thập dữ liệu camera CCTV thực tế tại các bãi đỗ Việt Nam để thu hẹp khoảng cách miền dữ liệu giữa ảnh web và ảnh camera; thứ hai, mở băng và fine-tune một số block cuối của backbone MobileNetV3 để tối ưu hóa đặc trưng màu xe; thứ ba, nén và lượng tử hóa mô hình sang định dạng ONNX để chuẩn bị cho tích hợp edge thời gian thực.

---

## Slide 18: Cảm ơn & Q&A

Chúng em xin chân thành cảm ơn thầy cô trong hội đồng và các bạn đã chú ý lắng nghe phần trình bày báo cáo Giai đoạn 2 của nhóm. Nhóm kính mong nhận được những nhận xét, đóng góp ý kiến phản biện để hệ thống được hoàn thiện hơn trong các giai đoạn tiếp theo. Sau đây, chúng em xin phép được bắt đầu phiên Hỏi & Đáp.
