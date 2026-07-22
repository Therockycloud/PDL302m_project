# Kịch bản thuyết trình Report 4 Final Defense

Kịch bản này khớp trực tiếp với 35 slide trong `index.html`. Slide 02–34 là 33
slide nội dung; slide 01 là trang bìa và slide 35 là trang cảm ơn.

## Slide 01: Report 4 Final Defense

Kính chào thầy và các bạn. Nhóm em xin trình bày sản phẩm cuối kỳ về hệ thống
nhận diện và đối chiếu biển số trong bãi xe. Bài bảo vệ hôm nay kết nối kết quả
mô hình của Report 3 với quyết định sản phẩm của Report 4: chỉ triển khai những
gì đã được đo và chứng minh.

Chúng em bắt đầu từ câu hỏi an ninh thực tế: đọc đúng biển số đã đủ an toàn hay
chưa?

## Slide 02: Tráo biển không chỉ là đọc sai chữ

Một hệ thống OCR có thể đọc đúng chuỗi ký tự nhưng vẫn không biết chiếc xe đó có
được phép đi qua hay không. Biển có thể chưa đăng ký, đã bị thay hoặc đang gắn
trên một xe có thuộc tính khác với hồ sơ.

Vì vậy, bài toán của nhóm không dừng ở nhận dạng; nó phải tạo ra quyết định có
thể giải thích. Slide tiếp theo trình bày nguyên tắc quyết định cốt lõi.

## Slide 03: Biển số là khóa chính

Nhóm chọn biển số đã đăng ký làm khóa danh tính chính. Màu và hãng xe chỉ đóng
vai trò chẩn đoán: nếu biển khớp nhưng thuộc tính lệch, hệ thống cảnh báo mềm
thay vì tự động chặn.

Cách thiết kế này giảm báo động giả do classifier yếu hoặc điều kiện CCTV thay
đổi. Từ nguyên tắc đó, nhóm xây dựng một sản phẩm vận hành được.

## Slide 04: Dashboard Streamlit

Đây là dashboard Streamlit thực tế của hệ thống. Người vận hành có thể đưa ảnh
hoặc video vào, xem biển được nhận diện, kết quả đối chiếu và verdict cuối cùng
trong cùng một giao diện.

Backend FastAPI chạy riêng, còn dashboard tập trung vào thao tác và giải thích
kết quả. Đầu tiên là verdict `AUTHORIZED`; hai tình huống cảnh báo còn lại sẽ
được trở lại sau khi trình bày security gate.

## Slide 05: AUTHORIZED

Đây là verdict thứ nhất trong ba tình huống vận hành chính: biển số được đọc,
khớp registry và xe được xác nhận hợp lệ. Hệ thống có thể cho xe qua mà không
cần kiểm tra thêm.

## Slide 06: Phạm vi triển khai

Sản phẩm được thiết kế cho Docker trên CPU, một camera và chế độ offline-first.
FastAPI cung cấp API, Streamlit cung cấp giao diện, còn PaddleOCR được nạp sẵn để
giảm độ trễ sau lần gọi đầu.

Phạm vi này cố ý hẹp để kết quả có thể tái lập và demo ổn định. Với ràng buộc đó,
kiến trúc tích hợp được tổ chức như sau.

## Slide 07: Kiến trúc tích hợp

FastAPI điều phối các nhánh nhận diện và decision engine. YOLO cắt vùng biển,
PaddleOCR đọc chuỗi, MobileNetV3 dự đoán màu và EfficientNet cung cấp tín hiệu
hãng xe ở mức chẩn đoán.

Streamlit gọi backend và trình bày verdict; các mô hình không tự vote độc lập.
Slide sau đi theo đúng thứ tự dữ liệu chạy qua hệ thống.

## Slide 08: Detect, read, verify

Một frame đi qua năm bước: nhận input, YOLO định vị biển, PaddleOCR đọc ký tự,
nhánh màu bổ sung chẩn đoán, rồi registry quyết định kết quả. Brand là nhánh tùy
chọn và không tác động gate.

Cấu trúc phân rã này cho phép đo và thay từng thành phần độc lập. Việc chọn từng
mô hình dựa trên các công nghệ tham chiếu ở slide tiếp theo.

## Slide 09: Công nghệ tham chiếu

YOLOv8 được chọn cho detection nhẹ; PaddleOCR được chọn sau benchmark thực tế;
EfficientNet-B0 và MobileNetV3-Small đại diện cho hai nhánh phân loại. ResNet là
baseline về độ sâu, còn DeepStream gợi ý cách tổ chức pipeline video tại edge.

Điểm quan trọng là không chọn model chỉ vì nổi tiếng. Mỗi lựa chọn phải phù hợp
CPU, dữ liệu và vai trò trong quyết định. Trước hết là detector.

## Slide 10: Huấn luyện detector

Detector dùng YOLOv8n với input 640 × 640, khởi tạo từ COCO và fine-tune trong
80 epoch. Cấu hình nhẹ giúp giữ kích thước trọng số khoảng 6.24 MB và latency CPU
ở mức có thể sử dụng trong pipeline.

Transfer learning làm bài toán hội tụ nhanh hơn so với huấn luyện từ đầu. Kết
quả định lượng được trình bày ngay sau đây.

## Slide 11: Plate mAP

Trên tập validation 1.765 ảnh, cấu hình fine-tune đạt mAP@0.5 bằng 0.9896 và
mAP@0.5:0.95 bằng 0.704. Đây là kết quả định vị vùng biển, không phải độ chính
xác đọc toàn chuỗi.

Detector tốt giúp OCR nhận ít nền nhiễu hơn, nhưng cần so sánh với baseline để
biết transfer learning thực sự tạo ra khác biệt gì.

## Slide 12: Benchmark detector

Biểu đồ cho thấy fine-tune từ COCO đạt 98.96% mAP@0.5, cao hơn cấu hình train từ
đầu 97.90%, trong khi latency và kích thước model gần tương đương. Vì vậy, nhóm
giữ cấu hình transfer learning.

Sau khi vùng biển được cắt ổn định, nút thắt tiếp theo là engine OCR.

## Slide 13: EasyOCR sang PaddleOCR

EasyOCR đạt 0% exact match trên tập frozen 16 ảnh, trong khi PaddleOCR đạt khoảng
81%. Khoảng cách này đủ lớn để nhóm thay đổi lựa chọn runtime và loại fallback
im lặng sang EasyOCR.

Nếu PaddleOCR không khả dụng, hệ thống báo lỗi rõ ràng. Slide tiếp theo cho thấy
chính xác các chỉ số OCR được dùng để ra quyết định.

## Slide 14: PaddleOCR accuracy và CER

PaddleOCR đạt khoảng 81% exact match trên 16 crop giữ khóa và CER 0.031, nghĩa
là phần lớn ký tự được đọc đúng dù chưa phải mọi chuỗi đều khớp hoàn toàn. Cỡ
mẫu nhỏ nên đây là bằng chứng kỹ thuật có giới hạn, không phải kết luận thống kê
rộng.

Một khó khăn riêng của biển Việt Nam là bố cục hai dòng. Nhóm xử lý vấn đề đó ở
bước hậu xử lý hình học.

## Slide 15: Sắp xếp biển hai dòng

Các bounding box ký tự được nhóm theo tọa độ tâm Y để tách hàng trên và hàng
dưới, sau đó sắp theo X trong từng hàng. Hai chuỗi được ghép lại rồi chuẩn hóa
dấu chấm, gạch và khoảng trắng.

Cách làm đơn giản này ổn định hơn việc tin vào thứ tự box trả về mặc định. Quan
trọng hơn, slide sau xác nhận engine nào đang chạy trong sản phẩm.

## Slide 16: PaddleOCR là runtime

Sản phẩm đang dùng PaddleOCR để đọc biển và nhận cấu hình qua YAML. CTC/ONNX chỉ
là nhánh nghiên cứu của Report 4; không có bước nào thay Paddle trong API hay
dashboard đang demo.

Sau OCR, nhóm đã thử thêm hãng và màu xe. Hai nhánh này có mức độ tin cậy rất
khác nhau.

## Slide 17: EfficientNet-B0 cho hãng xe

Nhánh hãng xe dùng EfficientNet-B0 với transfer learning và head tám lớp. Model
nhẹ hơn ResNet50 và phù hợp hơn với mục tiêu CPU, nhưng dữ liệu Stanford Cars có
góc chụp khác đáng kể so với camera bãi xe Việt Nam.

Domain gap này thể hiện trực tiếp trong kết quả huấn luyện.

## Slide 18: Brand chỉ là diagnostic

Độ chính xác brand chỉ khoảng 35%, nên nhóm không cho model này tham gia quyết
định mở barrier. Nó được giữ lại để quan sát, thu thêm dữ liệu và hỗ trợ chẩn
đoán khi cần.

Đây là một quyết định có chủ ý: model yếu không được “vote cứng” chỉ để làm hệ
thống trông phức tạp hơn. Nhánh màu cho kết quả tốt hơn.

## Slide 19: MobileNetV3-Small cho màu

Màu xe dùng MobileNetV3-Small vì kiến trúc nhẹ, có squeeze-and-excitation và phù
hợp với tín hiệu màu hơn bài toán nhận diện hãng. Đầu ra gồm tám lớp màu phổ
biến và trọng số đủ nhỏ để chạy cùng PaddleOCR trên CPU.

Hiệu năng tốt nhất đến từ thay đổi dữ liệu và chiến lược fine-tune.

## Slide 20: Đường cong huấn luyện màu

Biểu đồ cho thấy quá trình học ổn định hơn sau khi dùng đầy đủ VCoR, mở toàn bộ
backbone, áp dụng class weighting, label smoothing và test-time augmentation.
Đây là cấu hình cuối cùng của nhánh màu.

Tuy nhiên, chỉ số tốt vẫn phải đi kèm đúng phạm vi dữ liệu đo.

## Slide 21: 86.3% trên VCoR

Nhánh màu đạt 86.3% accuracy với TTA và macro-F1 khoảng 0.84 trên tập test VCoR.
VCoR chủ yếu là ảnh xe rõ, nền sạch; đây không phải kết quả trên camera CCTV bãi
xe.

Vì vậy, màu chỉ tạo cảnh báo mềm. Slide tiếp theo tóm tắt những thay đổi nào đã
thực sự cải thiện KPI.

## Slide 22: Tuning và ablation

Với màu xe, kết quả đi từ khoảng 48–55% ở các cấu hình đầu lên 77.6%, 85.3% và
cuối cùng 86.3% khi thay đổi dữ liệu, fine-tune và TTA. Với hãng xe, fine-tune
chỉ nâng từ khoảng 32.8% lên 35.3%.

Các phép đo này quyết định vai trò triển khai của từng model. Khi tích hợp toàn
chuỗi, nhóm tiếp tục đo latency thực tế.

## Slide 23: Latency tích hợp

Approach-lock trên video đạt trung bình khoảng 0.73 giây, còn API `/verify` sau
warm-up khoảng 0.96 giây. Cold start khoảng 6.1 giây do thời gian nạp PaddleOCR,
vì vậy đây vẫn là hạng mục cần tối ưu.

Latency đạt mục tiêu gần một giây, nhưng an ninh còn phụ thuộc cách chọn ngưỡng
security gate.

## Slide 24: Security gate 0.40

Ngưỡng cũ tạo quá nhiều báo động giả nên không phù hợp vận hành. Sau khi hiệu
chỉnh gate về 0.40, nhóm chấp nhận bỏ sót một phần trường hợp để đưa false alarm
về mức có thể sử dụng.

Đây là trade-off được đo chứ không phải ngưỡng chọn theo cảm tính. Slide tiếp
theo báo cáo kết quả sau gate.

## Slide 25: 69% detection tại 2.5% false alarm

Ở gate 0.40, hệ thống phát hiện 69% trường hợp tráo biển với false-alarm rate
2.5%. Biển chưa đăng ký vẫn bị chặn theo registry; con số 69% mô tả nhánh phát
hiện bất thường, không phải OCR exact match.

Từ đây, nhóm quay lại hai verdict còn lại: cảnh báo mềm khi thuộc tính lệch và
chuyển kiểm tra thủ công khi biển không nằm trong registry.

## Slide 26: ALLOW_WARN

Đây là verdict thứ hai. Biển số vẫn là khóa chính; khi biển khớp nhưng màu hoặc
hãng lệch, hệ thống cho xe đi tiếp kèm cảnh báo mềm để nhân viên chú ý, thay vì
tự động từ chối.

## Slide 27: UNREGISTERED

Đây là verdict thứ ba. Nếu biển không có trong registry, hệ thống không cấp
quyền tự động. Trường hợp này được chuyển sang kiểm tra thủ công thay vì suy
đoán từ thuộc tính phụ.

Sản phẩm không chỉ tồn tại trong notebook. Backend cũng công khai contract qua
API.

## Slide 28: FastAPI /verify

Endpoint `/verify` nhận input, chạy pipeline và trả về biển đọc được, confidence,
thuộc tính chẩn đoán cùng verdict. OpenAPI `/docs` giúp kiểm tra request/response
và tích hợp với dashboard.

Runtime OCR ở endpoint này vẫn là PaddleOCR. Từ baseline đó, nhóm mới thử một
recognizer nhẹ hơn.

## Slide 29: Động lực thử CTC/ONNX

PaddleOCR đạt độ chính xác tốt nhất nhưng tạo phần lớn chi phí latency và cold
start. Nhóm thử MobileNetV3-Small kết hợp CTC rồi xuất ONNX với mục tiêu giảm
thời gian suy luận.

Đây là thí nghiệm thay thế có gate rõ ràng, không phải thay đổi production trước
rồi mới đo. Dữ liệu được tách như slide tiếp theo.

## Slide 30: Dữ liệu train không chạm tập đánh giá

CTC chỉ dùng synthetic và pseudo-label để train. Khoảng 140 identity được khóa
trước khi đánh giá; 64 biển ô tô held-out không dùng để tune. Vì thế 0/64 là
kết quả đánh giá có kiểm soát, không phải con số bị leakage.

## Slide 31: CTC không đạt gate

CTC đạt 0 trên 64 exact match, CER khoảng 0.66 và
`deployment_ready: false`. Gate nội bộ để thay Paddle là ít nhất 90% exact match,
vì vậy cấu hình này không được đưa vào runtime.

Latency tiềm năng không thể bù cho độ chính xác không đạt. Slide sau chẩn đoán
nguyên nhân chính của thất bại.

## Slide 32: Domain gap và no-deploy

Pseudo-label chủ yếu đến từ biển xe máy, trong khi validation là biển ô tô hold-out
với bố cục và góc chụp khác. Anti-leakage giúp kết quả thấp này phản ánh đúng
domain gap thay vì được che bởi dữ liệu trùng.

Quyết định đúng là giữ PaddleOCR và thu thêm corpus ô tô in-domain. Bài học đó
cũng dẫn tới pivot thiết kế tổng thể.

## Slide 33: Từ hard vote sang plate-primary

Thiết kế ban đầu kỳ vọng OCR, màu và hãng cùng vote cứng. Kết quả Report 3 cho
thấy brand yếu và màu còn domain gap, nên thiết kế cuối chuyển sang plate-primary
với attribute chỉ cảnh báo mềm.

Pivot này làm hệ thống đơn giản hơn, dễ giải thích hơn và giảm báo động giả. Slide
cuối nội dung tổng hợp những gì nhóm sẽ giữ và cải thiện.

## Slide 34: Đo đúng, ship đúng

Ba kết luận chính là: PaddleOCR tiếp tục là runtime; CTC không được deploy; và
mọi model thay thế phải vượt gate trên dữ liệu held-out. Hướng tiếp theo là thu
dữ liệu car CCTV tại site, tối ưu cold start và đánh giá lại security gate trên
mẫu lớn hơn.

Thông điệp cuối cùng là chỉ claim những gì đã đo đúng domain. Nhóm xin kết thúc
phần trình bày và chuyển sang Q&A.

## Slide 35: Thank you for listening

Nhóm em xin chân thành cảm ơn thầy và các bạn đã lắng nghe. Toàn bộ quyết định
trong sản phẩm được gắn với số đo, giới hạn dữ liệu và trạng thái triển khai rõ
ràng; PaddleOCR vẫn là runtime và CTC/ONNX chưa được deploy.

Nhóm em sẵn sàng trả lời câu hỏi về mô hình, benchmark, pipeline, security gate
và các lựa chọn thiết kế.
