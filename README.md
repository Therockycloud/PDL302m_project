# Smart Parking Security System via Cross-Verification

> **DPL302m course team project at FPT University.** The repository name is
> retained for course traceability. Phạm Hoàng Hải's documented contribution
> focused on system integration, security evaluation, UI, and final synthesis
> (approximately 34%); the complete system is team work.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống giám sát an ninh bãi đỗ xe thông minh (DPL302m) dùng học sâu để chống tráo biển số. Pipeline runtime: **YOLOv8n (xe → biển) → PaddleOCR → màu xe (MobileNetV3, PyTorch)** với quyết định **plate-primary** — biển khớp ⇒ `AUTHORIZED`; màu lệch chỉ **cảnh báo mềm**; biển không có trong CSDL ⇒ `UNREGISTERED`. Hãng xe chỉ diagnostic, không vào quyết định.

API `/verify` và Dashboard Streamlit dùng chung pipeline hợp nhất (`main/src/engine/pipeline_factory.py`). Gate “xe đã đỗ” (approach-lock) chỉ chạy suy luận nặng một lần mỗi xe để tránh lặp lại toàn bộ pipeline trên từng frame.

> **OCR runtime:** chỉ **PaddleOCR** (`ocr.engine: ppocr`). Thí nghiệm CTC/ONNX nhẹ **chưa đạt** ngưỡng deploy — xem mục dưới. Quyết định này dựa trên [OCR Benchmark C](docs/benchmarks/ocr_benchmark.md); bối cảnh pivot nằm trong [`docs/related_work.md`](docs/related_work.md).

---

## Tài liệu dự án

| Tài liệu | Mô tả |
|----------|--------|
| **Repository** | [https://github.com/Therockycloud/PDL302m_project](https://github.com/Therockycloud/PDL302m_project) (`main`) |
| **Clone** | `git clone https://github.com/Therockycloud/PDL302m_project.git` |
| [Report 1 — Proposal](reports/documents/Report_1_Proposal.md) | Vấn đề an ninh, kiến trúc, phân công |
| [Report 2 — Data Tasks](reports/documents/Report_2_Data_Tasks.md) | Thu thập, EDA, tiền xử lý |
| [Report 3 — Model & Results](reports/documents/Report_3_Model_Results.md) | YOLO, PaddleOCR, màu xe, benchmark |
| [Report 4 — Final Defense](reports/documents/Report_4_Final_Report.md) | Tích hợp E2E, latency, kịch bản an ninh ([DOCX](reports/documents/Report_4_Final_Report.docx)) |
| [Bảng đóng góp](reports/documents/Bang_Dong_Gop_Du_An.md) | Phân công & tỷ lệ đóng góp thành viên ([DOCX](reports/documents/Bang_Dong_Gop_Du_An.docx)) |
| [Model specs](docs/model_specifications.md) | Cấu hình, input, layout CSDL |
| [Course DoD](docs/course_definition_of_done.md) | Checklist đóng bài + lệnh tái hiện |
| [Slides](reports/presentations/) | HTML presentations (Swiss R4, professional deck, …) |
| [Release artifacts](reports/release/) | PDF/DOCX nộp bài |

---

## Đóng góp thành viên (Nhóm 7)

| Họ và tên | Vai trò chính | Tỷ lệ |
|-----------|---------------|-------|
| Đỗ Manh Chung | Đề xuất / Literature / Thách thức | ~22% |
| Đồng Minh Đức | Thiết kế hệ thống / E2E eval | ~22% |
| Phạm Hoàng Hải | Tích hợp / An ninh / UI / Kết luận | ~34% |
| Trần Lê Sơn | Hiệu năng CPU / Offline / KPI | ~22% |

Chi tiết phân công theo đầu mục Report 4: [`reports/documents/Bang_Dong_Gop_Du_An.md`](reports/documents/Bang_Dong_Gop_Du_An.md).

---

## Kiến trúc hệ thống

```
                  ┌──────────────────────┐
                  │   Camera giám sát    │
                  └──────────┬───────────┘
                             │ POST /verify
                             ▼
                  ┌──────────────────────┐
                  │    FastAPI Server    │
                  └────┬────────────┬────┘
       ┌────────────────┘            └────────────────┐
       ▼ (Khoá chính — biển số)            ▼ (Tín hiệu phụ — màu)
 ┌───────────┐                                  ┌───────────┐
 │  YOLOv8   │                                  │MobileNetV3│
 └─────┬─────┘                                  └─────┬─────┘
       ▼                                              │
 ┌───────────┐                                        │
 │ PaddleOCR │ (+ spatial sorting 2 dòng)             │
 └─────┬─────┘                                        │
       ▼                                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │  Plate-primary matcher (CSV)                              │
 │  khớp biển ⇒ AUTHORIZED · màu lệch ⇒ cảnh báo mềm         │
 │  không có biển ⇒ UNREGISTERED                             │
 └────────────────────────────┬─────────────────────────────┘
                              ▼
                  ┌──────────────────────┐
                  │ Streamlit Dashboard  │
                  └──────────────────────┘
```

---

## Thí nghiệm OCR nhẹ (CTC/ONNX) — chưa deploy

Ứng viên **MobileNetV3-Small + CTC → ONNX** chỉ thay Paddle khi ≥90% exact-match trên dữ liệu thật giữ kín — **chưa đạt**.

| Hạng mục | Chi tiết |
|----------|----------|
| Train | `task4_train` + `pseudo_vision` (conf ≥ 0.5) |
| Val / giữ kín | `real_validation.csv` (64) · `expanded_real_test` (102) · `frozen_regression` (16) |
| Kết quả | Chưa đạt ngưỡng thay thế; `deployment_ready: false` ([Report 4 §5.4](reports/documents/Report_4_Final_Report.md)) |
| Runtime | **PaddleOCR** (~81% exact trên frozen 16; [benchmark](docs/benchmarks/ocr_benchmark.md)) |

Chính sách dữ liệu: [`main/data/plate_ocr/README.md`](main/data/plate_ocr/README.md).

---

## Dataset (tóm tắt)

- **Hãng xe:** ~1.209 ảnh thô → **792** dùng train (8 lớp, ~100/lớp). Brand chỉ phục vụ diagnostic và không tham gia quyết định.
- **Màu xe (runtime):** VCoR + tập nội bộ → **5.881** ảnh; MobileNetV3-Small **86.3% TTA** (macro-F1 0.84) — domain gap CCTV vẫn còn. Xem [`docs/benchmarks/color_finetune_report.md`](docs/benchmarks/color_finetune_report.md).
- **Biển số kiểm thử E2E:** ảnh/xe thật VN + nhãn YOLO.

---

## Giới hạn và quyền riêng tư

- Các số đo màu được thực hiện trên VCoR held-out; ảnh CCTV bãi xe thật có thể kém hơn do ánh sáng, góc chụp và nén ảnh. Xem [color benchmark](docs/benchmarks/color_benchmark.md).
- Đối chiếu màu chỉ là cảnh báo mềm: xe dùng biển tráo nhưng cùng màu có thể không bị phát hiện. Xem [security evaluation và limitations](docs/benchmarks/security_eval.md#limitations-đọc-trước-khi-trích-số-liệu).
- Dữ liệu nhận diện như khuôn mặt, biển số thật và thông tin đăng ký xe là dữ liệu nhạy cảm. Không commit dữ liệu cá nhân mới; khi demo công khai, dùng dữ liệu tổng hợp hoặc đã được đồng ý và làm mờ thông tin không cần thiết.
- Student ID đã được loại khỏi bản tài liệu hiện tại. Tên thành viên được giữ lại để ghi nhận đóng góp; lịch sử Git cũ vẫn có thể chứa metadata học phần và cần được đánh giá riêng trước khi dùng repo ngoài bối cảnh portfolio.

---

## Chạy bằng Docker

```bash
docker compose up --build
```

- API (Swagger): http://localhost:8000/docs
- Dashboard: http://localhost:8501

Dừng: `docker compose down`

Upload Video dùng Product view đồng bộ (media clock trình duyệt). Cần truy cập được cả cổng **8501** và **8000** (`DPL_DEMO_API_URL` mặc định `http://localhost:8000`).

Khi gate mở nhưng hết `collect_frames` mà OCR chưa đủ bằng chứng → verdict **UNCERTAIN** (`action=LOG`), không mở barrier.

---

## Chạy native (Python 3.12)

### macOS / Linux

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r main/requirements.txt
bash main/run_ui.sh
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r main\requirements.txt
main\run_ui.bat
```

Khởi động FastAPI song song (cổng 8000) trước khi test Upload Video:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=main uvicorn main.src.api.app:app --host 0.0.0.0 --port 8000
```

### Video demo

```bash
python main/src/utils/download_sample_video.py
python main/src/utils/download_sample_video.py --verify
```

### Tests

Full suite (bao gồm CTC training/export và kiểm tra ONNX) cần bộ dependency
training đã pin:

```bash
pip install -r main/requirements-train.txt
cd main && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q
# Container runtime không cài training/export extras:
docker compose exec -T -w /app/main backend pytest -q --ignore=tests/test_vn_plate_ctc.py
```

Không dùng virtualenv được tạo với `--system-site-packages`: PyTorch cũ từ
Python/Conda base có thể ghi đè cặp `torch`/`torchvision` đã pin.

### Troubleshooting

| Triệu chứng | Cách xử lý |
|-------------|------------|
| Treo 0% CPU / `mutex lock failed` | Đặt `KMP_DUPLICATE_LIB_OK=TRUE` |
| Detector rỗng | Cài lại `onnxruntime` qua `main/requirements.txt` |
| OCR lần đầu chậm | Docker đã mồi cache; native có thể tải model Paddle lần đầu |
| Webcam (macOS) | System Settings → Privacy & Security → Camera |
| Sai góc camera | `python main/scripts/calibrate_roi.py --source <clip>` rồi chỉnh `pipeline.trigger` trong `config.yaml` |

---

## Tối ưu CPU / offline

- Giới hạn thread (`OMP_NUM_THREADS=1`, …) trong `Dockerfile` / `docker-compose.yml` để tránh deadlock.
- Approach-lock tránh chạy lại pipeline nặng trên mọi frame; cold-start lần đầu vẫn có thể chậm do nạp PaddleOCR.
- PaddleOCR cache mồi ở build-time; YOLO offline (`sync: false`, font local).

Chi tiết phương pháp và số đo: [Report 4 §4–§5](reports/documents/Report_4_Final_Report.md).
