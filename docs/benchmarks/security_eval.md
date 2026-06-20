# Security Evaluation — Anti-Plate-Swap (Colour Cross-Verification)

> Đo lường có kiểm soát (controlled evaluation) năng lực **chống tráo biển số** của hệ thống: nếu một biển số bị nhân bản (clone) từ xe A (màu C1) và gắn lên xe B khác màu (C2≠C1), việc đối chiếu màu (`DatabaseMatcher.verify_vehicle`) có phát hiện sai khác và trả `color_warning=True` (ALLOW_WARN) không? Đây là LẦN ĐẦU có số đo định lượng cho năng lực này — đề xuất ban đầu (Report 1) nêu mục tiêu ≥95% phát hiện gian lận nhưng chưa từng được đo; Report 4 trước đó chỉ test 5 ảnh, không đo an ninh.

**Model:** weights ĐANG CHẠY ở runtime (`/Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project/main/data/models/color_MobileNetV3Small.pt`), gọi qua `TorchColorClassifier.predict()` thật (không mock). **Decision logic:** `DatabaseMatcher.verify_vehicle()` thật (không mock), DB đăng ký tạm thời (`/var/folders/k8/ycp28skx4b97ns1_lxdgr1540000gn/T/dpl302m_security_eval_dvcod1uq/temp_registration.csv`, không phải `main/data/database.csv`).

**Ảnh test:** VCoR held-out TEST split (cùng split với `eval_color_deployed.py`: seed=42, stratified 70/15/15, từ `colab_train_color.py.load_samples`/`stratified_split`) — 889 ảnh giữ-riêng, model chưa từng thấy khi huấn luyện. Data layout: `vcor`.

**Seed (reproducibility):** `42` — cố định cho toàn bộ việc chọn mẫu/màu đăng ký.

## Headline Results

| Scenario | Trials | Metric | Value |
|---|---|---|---|
| **Plate-swap detection** (headline) | 200 | detection rate (color_warning=True) | **69.0%** (138/200) |
| Plate-swap MISSED | 200 | miss rate | 31.0% (62/200) |
| Legitimate (no swap) | 200 | false-alarm rate | 2.5% (5/200) |
| Unregistered plate | 200 | detection rate (DENY_ALERT) | 100.0% (200/200) |

- **Plate-swap detection rate = 69.0%** — khi biển số bị tráo lên xe KHÁC MÀU, hệ thống bắt được 138/200 lần qua cảnh báo màu lệch.
- **False-alarm rate = 2.5%** — xe hợp lệ (không tráo) bị cảnh báo nhầm 5/200 lần (do model màu dự đoán sai ngay cả khi biển đúng).
- **Unregistered detection rate = 100.0%** — biển không có trong CSDL bị chặn đúng 200/200 lần (như kỳ vọng, không phụ thuộc màu).

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
