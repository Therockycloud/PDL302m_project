# Security Evaluation — Anti-Plate-Swap (Colour Cross-Verification)

> Đo lường có kiểm soát (controlled evaluation) năng lực **chống tráo biển số** của hệ thống: nếu một biển số bị nhân bản (clone) từ xe A (màu C1) và gắn lên xe B khác màu (C2≠C1), việc đối chiếu màu (`DatabaseMatcher.verify_vehicle`) có phát hiện sai khác và trả `color_warning=True` (ALLOW_WARN) không? Đây là LẦN ĐẦU có số đo định lượng cho năng lực này — đề xuất ban đầu (Report 1) nêu mục tiêu ≥95% phát hiện gian lận nhưng chưa từng được đo; Report 4 trước đó chỉ test 5 ảnh, không đo an ninh.

**Model:** weights ĐANG CHẠY ở runtime (`/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project/main/data/models/color_MobileNetV3Small.pt`), gọi qua `TorchColorClassifier.predict()` thật (không mock). **Decision logic:** `DatabaseMatcher.verify_vehicle()` thật (không mock), DB đăng ký tạm thời (`/var/folders/k8/ycp28skx4b97ns1_lxdgr1540000gn/T/dpl302m_security_eval_dvcod1uq/temp_registration.csv`, không phải `main/data/database.csv`).

**Ảnh test:** VCoR held-out TEST split (cùng split với `eval_color_deployed.py`: seed=42, stratified 70/15/15, từ `colab_train_color.py.load_samples`/`stratified_split`) — 889 ảnh giữ-riêng, model chưa từng thấy khi huấn luyện. Data layout: `vcor`.

**Seed (reproducibility):** `42` — cố định cho toàn bộ việc chọn mẫu/màu đăng ký.

## Headline Results

**Deployed operating point: `decision.color_warn_conf = 0.40`** (WS-2 gate — xem mục "Operating point" dưới đây để biết lý do chọn 0.40 và toàn bộ bảng quét gate).

| Scenario | Trials | Metric | Before (legacy, gate=0.00 / không gate) | **After (deployed, gate=0.40)** |
|---|---|---|---|---|
| **Plate-swap detection** (headline) | 200 | detection rate (color_warning=True) | 98.5% (197/200) | **69.0%** (138/200) |
| Plate-swap MISSED | 200 | miss rate | 1.5% (3/200) | **31.0%** (62/200) |
| Legitimate (no swap) | 200 | false-alarm rate | 14.5% (29/200) | **2.5%** (5/200) |
| Unregistered plate | 200 | detection rate (DENY_ALERT) | 100.0% (200/200) | **100.0%** (200/200) |

- **Plate-swap detection rate = 69.0%** (After) — khi biển số bị tráo lên xe KHÁC MÀU, hệ thống bắt được 138/200 lần qua cảnh báo màu lệch. Trước khi siết gate (Before), con số này là 98.5%, nhưng đi kèm false-alarm 14.5% không triển khai được thực tế (xem giải thích đánh đổi ở mục "Operating point" dưới).
- **False-alarm rate = 2.5%** (After) — xe hợp lệ (không tráo) bị cảnh báo nhầm 5/200 lần, giảm từ 14.5% (Before) nhờ gộp cụm màu trung tính + confidence-gating (WS-2).
- **Unregistered detection rate = 100.0%** — biển không có trong CSDL bị chặn đúng 200/200 lần (như kỳ vọng, không phụ thuộc màu, không đổi giữa Before/After).

## Colour-Pair Breakdown — Plate-Swap Misses

Cặp (màu đăng ký C1 → màu thật C2) bị MISS nhiều nhất (color_warning vẫn False dù màu thật khác màu đăng ký):

| Registered (C1) | True (C2) | Trials | Missed | Miss rate |
|---|---|---|---|---|
| Black | Grey | 6 | 6 | 100.0% |
| White | Black | 7 | 6 | 85.7% |
| Silver | Grey | 5 | 5 | 100.0% |
| Grey | Silver | 5 | 5 | 100.0% |
| Silver | White | 4 | 4 | 100.0% |
| Grey | White | 4 | 4 | 100.0% |
| Black | Silver | 4 | 4 | 100.0% |
| White | Grey | 5 | 4 | 80.0% |
| Silver | Black | 3 | 3 | 100.0% |
| Grey | Black | 3 | 3 | 100.0% |
| Silver | Brown | 6 | 3 | 50.0% |
| White | Silver | 2 | 2 | 100.0% |
| Grey | Brown | 2 | 2 | 100.0% |
| Black | White | 2 | 2 | 100.0% |
| Red | Silver | 5 | 2 | 40.0% |

**Cụm màu trung tính (Black/Grey/Silver/White ↔ nhau):** 50 trial, miss 48 (96.0%) — khớp với cụm nhập nhằng đã ghi nhận ở Report 3 §5.1 (confusion matrix màu), đây là nơi cross-check màu YẾU NHẤT vì model màu chính nó cũng nhầm trong cụm này.

## Operating point & false-alarm reduction (WS-2)

Logic gốc (pre-WS-2) coi MỌI sai khác màu là tín hiệu tráo biển, không phân biệt cụm màu hay độ tin cậy của model — kết quả là detection cao (98.5%) nhưng false-alarm rate 14.5% cao đến mức không triển khai được trong thực tế (gần 1/7 xe hợp lệ bị cảnh báo nhầm). WS-2 thêm hai cơ chế để kéo false-alarm xuống:

1. **Gộp cụm màu trung tính** (`decision.neutral_colors`: Black/Grey/Silver/White) — coi các màu này tương đương nhau, vì đây chính là cụm model màu hay nhầm nhất (Report 3 §5.1).
2. **Confidence-gating** (`decision.color_warn_conf`) — chỉ cảnh báo khi model màu đủ tin cậy (`color_conf ≥ gate`) về một sai khác màu KHÔNG nằm trong cụm trung tính; dưới ngưỡng, sai khác được coi là nhiễu của model màu, không cảnh báo.

**Bảng quét gate (gate sweep, đo trên cùng split/seed=42, 200 trial/scenario mỗi điểm gate):**

| Gate (`color_warn_conf`) | False-alarm rate | Plate-swap detection rate |
|---:|---:|---:|
| 0.00 (không gate — chỉ gộp cụm trung tính) | 4.5% | 73.5% |
| 0.30 | 3.5% | 72.0% |
| **0.40 (ĐÃ CHỌN, deployed)** | **2.5%** | **69.0%** |
| 0.50 | 1.0% | 63.5% |
| 0.60 (ngưỡng cũ trước khi chốt 0.40) | 0.5% | 56.5% |

**Vì sao chọn 0.40:** đây là điểm cân bằng giữ false-alarm an toàn cho triển khai (<5%, ở mức 2.5%) trong khi vẫn giữ được detection ở mức khá (69.0%) — gate cao hơn (0.50/0.60) giảm false-alarm thêm nhưng đánh đổi detection giảm sâu hơn (63.5%/56.5%), còn gate thấp hơn (0.00/0.30) giữ detection cao hơn nhưng false-alarm vẫn ở vùng dễ gây phiền (3.5–4.5%, vẫn gấp đôi tới gấp gần hai lần mức 2.5% đã chọn).

**Vì sao detection giảm mạnh (98.5% → 69.0%), nói trung thực:**
- Phần lớn mức giảm đến từ **việc gộp cụm trung tính cố ý bỏ qua các cặp tráo biển trong cùng cụm Black/Grey/Silver/White** — đây là đánh đổi thiết kế có chủ ý (không phải lỗi): các cặp này vốn là nơi model màu tự nhầm lẫn nhiều nhất, nên một sai khác trong cụm này mang tín hiệu tráo biển rất yếu, cảnh báo ở đây chủ yếu là báo động giả chứ không phải bắt được tráo biển thật.
- Phần còn lại đến từ **confidence-gating loại bỏ các trường hợp model màu dự đoán đúng có tráo biển nhưng với độ tin cậy thấp** (`color_conf < 0.40`) — những trường hợp này trước đây vẫn được tính là "bắt được" (detection) dù bản chất là model màu không chắc, nên giữ lại sẽ kéo false-alarm lên cao không kiểm soát được.

## Methodology

1. Lấy ảnh từ VCoR TEST split giữ-riêng (seed=42, 70/15/15 stratified — giống `eval_color_deployed.py`), nên ảnh hoàn toàn chưa từng thấy khi huấn luyện model màu.
2. Với mỗi trial, chạy `TorchColorClassifier.predict(bgr_image)` THẬT trên ảnh để lấy màu dự đoán (không dùng nhãn ground-truth) — phản ánh đúng hành vi triển khai thực tế, kể cả khi model màu đoán sai.
3. Xây CSDL đăng ký TẠM (cùng schema `main/data/database.csv`: `license_plate,car_brand,car_color`), không đụng tới CSDL thật của repo.
4. Gọi `DatabaseMatcher.verify_vehicle(plate, predicted_colour)` THẬT (logic quyết định không sửa đổi) cho 3 loại scenario, mỗi loại ~200 trial cân bằng theo màu, RNG seed cố định để tái lập được.
5. **legitimate**: ảnh màu thật C, biển đăng ký màu C → đúng = AUTHORIZED + không cảnh báo.
6. **plate_swap**: ảnh màu thật C2, biển đăng ký màu C1≠C2 (giả lập biển bị nhân bản từ xe màu C1, gắn lên xe màu C2) → đúng = `color_warning=True` (bắt được tráo).
7. **unregistered**: biển hoàn toàn không có trong CSDL → đúng = UNREGISTERED/DENY_ALERT.

## Limitations (đọc trước khi trích số liệu)

- **Chỉ bắt được khi xe gắn biển tráo có MÀU KHÁC màu đăng ký.** Nếu kẻ tráo biển dùng đúng xe cùng màu (hoặc sơn/dán decal giả màu), cross-check màu KHÔNG có cơ chế phát hiện — đây là lỗ hổng cố hữu của cơ chế "màu là cảnh báo mềm", không phải lỗi đo lường.
- **Phụ thuộc hoàn toàn vào việc OCR đọc đúng biển số trước đó** (Benchmark C: ~81% exact-match). Eval này giả định biển được đọc đúng (test cách ly bước cross-check màu); nếu OCR đọc sai/đọc thiếu, biển sẽ rơi vào UNREGISTERED hoặc match nhầm bản ghi khác — số đo plate-swap ở đây KHÔNG bao gồm lỗi OCR thực tế.
- **Đo trên VCoR (ảnh web/marketplace sạch)** — CCTV bãi xe thật (ánh sáng yếu, góc nghiêng, nén ảnh, độ phân giải thấp) nhiều khả năng cho tỉ lệ phát hiện THẤP HƠN do domain gap (xem Report 3 §5.1, Report 4 §5.1 caveat tương tự cho colour accuracy).
- Biển số và CSDL đăng ký trong eval này là DỮ LIỆU TỔNG HỢP (synthetic), sinh từ RNG seed cố định — không phải biển số thật, chỉ dùng để dựng tình huống kiểm thử có kiểm soát.

_Sinh bởi `main/scripts/eval_security.py`, chạy lúc thực thi script này trên model đang triển khai (`/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project/main/data/models/color_MobileNetV3Small.pt`), 600 trial tổng, elapsed 6.9s._
