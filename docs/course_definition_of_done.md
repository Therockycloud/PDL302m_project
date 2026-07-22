# Definition of Done — đóng bài DPL302m

Prototype có bằng chứng + báo cáo trung thực. **Không** claim OCR production mới (ONNX/CTC).

## Checklist

| # | Mục | Trạng thái | Bằng chứng |
|---|-----|------------|------------|
| 1 | Docs phản ánh Paddle = runtime OCR; CTC/ONNX = thí nghiệm | Done | `README.md`, `main/README.md`, `PROJECT.md` |
| 2 | Metric CTC đo thật (không bịa) | Done | `main/data/models/vn_plate_run/vn_plate_recognizer.json`: exact **0/64**, CER **~0.659**, `deployment_ready: false` |
| 3 | Report 4 + presentation 4 khối, không claim thay Paddle | Done | `reports/documents/Report_4_Final_Report.md` §5.4; slides 17–18 |
| 4 | Docker pytest | Done | **348 passed, 16 skipped** (2026-07-13) |
| 5 | Smoke demo ảnh (Paddle) | Done | `test_authorized.jpg` → plate `51F06532`, **AUTHORIZED** / ALLOW (~2.5 s lần đầu trong session) |
| 6 | Video demo có sẵn | Done | `main/data/test/sample_parking.mp4` (và `parking_case_real.mp4`) |
| 7 | Future work 3–5 dòng | Done | Report 4 §7; presentation slide 18 |
| 8 | (Tuỳ chọn) Latency Paddle vs ONNX minh họa | Done | 3 crop `real_validation` — chỉ minh họa tốc độ, **không** chọn checkpoint |

### Latency minh họa (Docker CPU, warm p50, n=5/crop)

| Crop | PaddleOCR p50 | ONNX CTC p50 |
|------|---------------|--------------|
| candidate-0006 | 157.9 ms | 81.2 ms |
| candidate-0122 | 160.1 ms | 116.7 ms |
| candidate-0012 | 161.6 ms | 89.2 ms |

ONNX nhanh hơn trên vài crop này nhưng **exact-match val = 0%** → không deploy.

## Lệnh tái hiện

```bash
# 1) Stack demo
docker compose up --build
# API http://localhost:8000/docs · UI http://localhost:8501

# 2) Full tests
docker compose exec -T -w /app/main backend pytest -q
# Kỳ vọng: 348 passed, 16 skipped (có thể lệch nhẹ nếu thêm test sau)

# 3) Smoke một ảnh (Paddle runtime)
docker compose exec -T -w /app/main backend python - <<'PY'
import cv2, yaml
from pathlib import Path
from src.engine.pipeline_factory import build_pipeline, infer_single_image
cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
pipe = build_pipeline(cfg)
img = cv2.imread("data/test/test_authorized.jpg")
print(infer_single_image(img, pipe, cfg))
PY

# 4) Video mặc định (nếu thiếu file)
python main/src/utils/download_sample_video.py
python main/src/utils/download_sample_video.py --verify
```

## Giới hạn đã ghi nhận

- OCR runtime: **PaddleOCR only** (`ocr.engine: ppocr`).
- CTC/ONNX: thí nghiệm; gate ≥90% exact-match **không đạt**.
- Không dùng `expanded_real_test` / `frozen_regression` để tuning hay chọn checkpoint.
- Một môi trường camera; thiếu biển ô tô verified cho train (domain gap vs pseudo xe máy).
