# Bảng phân công và đóng góp dự án

**DPL302m — Deep Learning | Report 4 / Tổng kết dự án**
**Đề tài:** Hệ thống đỗ xe thông minh bằng xác thực chéo thông tin xe
**Repository (GitHub):** [https://github.com/Therockycloud/PDL302m_project](https://github.com/Therockycloud/PDL302m_project)
**Clone:** `git clone https://github.com/Therockycloud/PDL302m_project.git` · nhánh `main`
**README:** [README.md](https://github.com/Therockycloud/PDL302m_project/blob/main/README.md)

Tài liệu này ghi nhận phân công công việc và mức đóng góp của từng thành viên (Report 1–4), trọng tâm ở Giai đoạn 4 (tích hợp, đánh giá E2E, tối ưu offline/CPU, demo). Bản Word tương ứng: [`Bang_Dong_Gop_Du_An.docx`](Bang_Dong_Gop_Du_An.docx) · [`reports/release/`](../release/Bang_Dong_Gop_Du_An.docx).

## 1. Danh sách thành viên

| STT | Họ và tên | Vai trò chính | Tỷ lệ đóng góp |
|-----|-----------|---------------|----------------|
| 1 | Đỗ Manh Chung | Đề xuất / Literature / Thách thức | ~22% |
| 2 | Đồng Minh Đức | Thiết kế hệ thống / E2E eval | ~22% |
| 3 | Phạm Hoàng Hải | Tích hợp / An ninh / UI / Kết luận | ~34% |
| 4 | Trần Lê Sơn | Hiệu năng CPU / Offline / KPI | ~22% |

> Tỷ lệ ~22/22/34/22 chia đều các đầu mục cốt lõi; Hải đảm nhận thêm các hạng mục kỹ thuật then chốt ở Giai đoạn 4.

## 2. Phân công đầu mục Report 4

| Đầu mục Report 4 | Nội dung chính | Phụ trách |
|------------------|----------------|-----------|
| 1. Đặt vấn đề & Mục tiêu | Phạm vi tích hợp, plate-primary | Đỗ Manh Chung |
| 2. Literature Review | 3 tài liệu tham chiếu | Đỗ Manh Chung |
| 3. Thiết kế hệ thống | Pipeline YOLO+Paddle+Matcher | Đồng Minh Đức |
| 3.1. CSDL CSV | Schema, chuẩn hoá, ví dụ | Đồng Minh Đức |
| 4.1. Inference logs | Bảng 5 ảnh clip3_new | Đồng Minh Đức |
| 4.2. Aggregate metrics | Latency / trạng thái tổng hợp | Đồng Minh Đức |
| 4.3. Plate-swap security | eval_security, Before/After, gate 0.40 | Phạm Hoàng Hải |
| 4.4. Streamlit UI Demo | 3 nhánh AUTHORIZED/WARN/UNREG | Phạm Hoàng Hải |
| 5.1. Runtime & xung đột lib | PyTorch màu, WS-1 latency | Trần Lê Sơn |
| 5.2. Offline-first | Docker zero-network, YOLO offline | Trần Lê Sơn |
| 5.3. Per-stage latency | measure_stage_latency.py | Trần Lê Sơn |
| 5.4. CTC/ONNX experiment | 0% exact, chưa deploy | Phạm Hoàng Hải |
| 6.1. Bảng KPI | Đối chiếu R1 vs thực đo | Trần Lê Sơn |
| 6.2. Điều làm tốt | Pivot, đo lường, test suite | Phạm Hoàng Hải |
| 6.3. Thách thức | TF xung đột, domain gap, GT nhỏ | Đỗ Manh Chung |
| 6.4. Bài học | Đo trước cam kết, kiến trúc theo số đo | Phạm Hoàng Hải |
| 7. Kết luận & hướng PT | Tổng hợp delivered + roadmap | Phạm Hoàng Hải |

## 3. Chi tiết theo thành viên

### 3.1. Đỗ Manh Chung (~22%)

Phụ trách khung vấn đề, nghiên cứu tài liệu và phần thách thức/retrospective.

- Viết / chỉnh sửa mục 1 (Đặt vấn đề & Mục tiêu tích hợp) Report 4
- Tổng hợp Literature Review (mục 2)
- Tổng hợp mục 6.3 Thách thức
- Hỗ trợ nội dung đề xuất Report 1 và đồng bộ câu chuyện pivot plate-primary
- Review số liệu KPI cho khớp Report 3

### 3.2. Đồng Minh Đức (~22%)

Phụ trách thiết kế hệ thống tích hợp và đánh giá E2E trên ảnh thật.

- Mô tả kiến trúc pipeline (mục 3)
- Schema CSDL CSV (mục 3.1)
- Inference logs + aggregate metrics (mục 4.1–4.2)
- Giải thích UNREGISTERED vs AUTHORIZED trên Dashboard
- Kiểm thử đường hợp nhất `build_pipeline` / `infer_single_image`

### 3.3. Phạm Hoàng Hải (~34%)

Đảm nhận hạng mục kỹ thuật then chốt ở Giai đoạn 4 và điều phối tích hợp.

- Đánh giá an ninh chống tráo biển (mục 4.3): `eval_security.py`, gate `color_warn_conf=0.40`
- UI Streamlit 3 nhánh quyết định (mục 4.4)
- Thử nghiệm OCR CTC/ONNX (mục 5.4); kết luận không deploy
- Mục 6.2 / 6.4 / 7 (retrospective + kết luận)
- Điều phối tích hợp pipeline/API/Dashboard (plate-primary)
- Nội dung bảo vệ (script/slide R4) liên quan an ninh và demo

### 3.4. Trần Lê Sơn (~22%)

Phụ trách tối ưu hiệu năng CPU, offline-first và bảng đối chiếu KPI.

- Xử lý xung đột TF/Paddle; chuyển runtime màu sang PyTorch (mục 5.1)
- Offline-first Docker (mục 5.2)
- Đo per-stage latency (mục 5.3)
- Bảng đối chiếu KPI Report 1 vs thực đo (mục 6.1)
- Warmup model / giảm cold-start trên API & Dashboard

## 4. Tổng hợp theo giai đoạn

| Giai đoạn | Chung | Đức | Hải | Sơn |
|-----------|-------|-----|-----|-----|
| R1 Proposal | Problem + Lit | Architecture | WBS / Demo vision | KPI / Stack |
| R2 Data | EDA / báo cáo | Crawl & clean | Augment / split | Provenance |
| R3 Models | Brand diag. | YOLO plate | Color + OCR pivot | Bench scripts |
| R4 Final | §1, §2, §6.3 | §3, §4.1–4.2 | §4.3–4.4, §5.4, §6.2/6.4, §7 | §5.1–5.3, §6.1 |
