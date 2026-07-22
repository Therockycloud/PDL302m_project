# Spec: Docker offline — runtime zero-network

**Ngày:** 2026-06-28
**Phạm vi:** Sửa dự án để chạy **hoàn toàn offline khi runtime**. Mô hình: `docker build`
được phép có mạng (cài deps + prime model cache vào image layer); **container khi chạy
không gọi mạng** (`docker run --network none` phải hoạt động đầy đủ).
**Môi trường mục tiêu chính:** Docker / docker-compose. Các fix tầng code cũng làm cho
native macOS (`run.sh`) sạch network.

## Quyết định đã chốt với người dùng

1. **Mức offline:** Runtime-only (build có mạng → container chạy zero-network).
2. **Môi trường:** Docker / docker-compose.
3. **Fonts:** Bỏ Google Fonts `@import`, dùng **system font stack** (0 asset, 0 network).
4. **OCR:** Runtime **100% PaddleOCR**. EasyOCR bị giáng xuống dependency **chỉ-train/eval**
   (gỡ khỏi `requirements.txt`, giữ trong `requirements-train.txt`). Image runtime không
   còn easyocr. `scripts/benchmark_ocr.py` + `src/engine/run_evaluation.py` vẫn chạy được
   trong env train để tái lập benchmark mà các report đang trích dẫn.

## Bối cảnh: các điểm chặn offline (đã xác minh trong mã)

| # | Vấn đề | Vị trí |
|---|--------|--------|
| A | **Dockerfile không build được:** `import tensorflow` (dòng 38) nhưng TF không có trong `requirements.txt`. | `Dockerfile:38` |
| B | Dockerfile prime **sai engine** (EasyOCR + TF ImageNet) thay vì PaddleOCR/Torch runtime thật. | `Dockerfile:36-38` |
| C | `YOLO('yolov8n.pt')` chạy **trước** `COPY` → tải từ mạng lúc build. | `Dockerfile:37,41` |
| D | **PaddleOCR tải PP-OCRv6 det/rec** vào `~/.paddlex/official_models` lần đầu init → blocker offline lớn nhất. | `src/models/ppocr_reader.py:28` |
| E | **Google Fonts `@import`** gọi `fonts.googleapis.com` mỗi render UI. | `src/utils/visual.py:236` |
| F | **Ultralytics phone-home + tải `Arial.ttf`** lần đầu annotate. | `src/models/vehicle_detector.py:39` |
| G | **Auto-download video mẫu** từ GitHub (chỉ kích hoạt nếu file local bị xóa). | `src/ui/dashboard.py:~435-465` |
| H | Không có `.dockerignore` → build context nuốt `.git`, `data/raw`, `reports`, model 94MB. | (thiếu file) |

**Đã offline-safe sẵn (không đụng):** TorchColor dùng `weights=None` + `.pt` local
(`torch_color.py:32`); YOLO load `.onnx` local từ `data/models/`.

## Thay đổi theo từng file

### 1. `Dockerfile` (viết lại bước prime)
- **Bỏ** dòng 36-38 (pre-download EasyOCR/TF/YOLO sai).
- Thêm ENV tắt phone-home: `YOLO_OFFLINE=True` (biến chính thức của ultralytics). Sonnet
  xác minh cơ chế đúng với version ultralytics đã cài; nếu cần thì set thêm
  `yolo settings sync=False` lúc build. KHÔNG bịa biến môi trường không tồn tại.
- Đổi thứ tự: `COPY . /app` **trước** bước prime, để file model/config có sẵn.
- Bước prime mới (chạy trong build, có mạng) — *idempotent, không được fail-hard nếu một
  model phụ không tải được*:
  - Khởi tạo `PaddleOCR(lang="en", use_textline_orientation=False,
    use_doc_orientation_classify=False, use_doc_unwarping=False)` y hệt
    `ppocr_reader.py` → PP-OCRv6 det/rec nằm trong layer image (`~/.paddlex/official_models`).
  - Bake `Arial.ttf` vào `~/.config/Ultralytics` (tải 1 lần lúc build, hoặc copy từ asset).
- **Không** prime EasyOCR (đã gỡ khỏi runtime).
- *Lưu ý vận hành:* cache nằm ở `$HOME` (=`/root`), KHÔNG ở `/app`, nên bind-mount
  `.:/app` của compose không che mất cache đã prime.

### 2. `.dockerignore` (mới)
Loại khỏi build context: `.git`, `.worktrees`, `**/__pycache__`, `**/*.pyc`,
`.pytest_cache`, `reports/`, `course_details/`, `main/data/raw/`, `main/notebooks/`,
`*.DS_Store`, và các model không-runtime nặng:
`main/data/models/color_ResNet50.pt`, `color_EfficientNetB0.pt`,
`color_MobileNetV3Small_cctv.pt`, `brand_classifier.keras`,
`color_classifier.keras`, `color_classifier.keras.bak`, `main/yolov8n.pt` (bản trùng).
**Giữ lại** trong context (runtime cần): `main/data/models/yolov8n.onnx`,
`plate_yolov8n.onnx`, `color_MobileNetV3Small.pt`, `main/data/database.csv`,
`main/data/test/sample_parking.mp4`.

### 3. `main/requirements.txt`
- **Gỡ** dòng `easyocr>=1.6.2`.
- Giữ `paddleocr`, `paddlepaddle`, `ultralytics`, `onnxruntime`, v.v.

### 4. `main/requirements-train.txt`
- **Thêm** `easyocr>=1.6.2` (cạnh tensorflow/icrawler/ImageHash) với comment: chỉ dùng cho
  `benchmark_ocr.py` + `run_evaluation.py`, không thuộc runtime.

### 5. `main/src/engine/pipeline_factory.py` (`_build_ocr_reader`)
- Runtime chỉ PaddleOCR. Bỏ nhánh fallback import `PlateOCR`.
- Nếu PaddleOCR init lỗi → raise lỗi rõ ràng (RuntimeError với hướng dẫn), KHÔNG im lặng
  rớt xuống EasyOCR. Cập nhật docstring cho khớp.
- `engine` mặc định không còn `easyocr`; nếu config đặt `engine: easyocr` thì cảnh báo +
  vẫn dùng ppocr (hoặc raise — chọn: cảnh báo rồi dùng ppocr, đỡ phá UX).

### 6. `main/configs/config.yaml`
- `ocr.engine`: giữ `ppocr`. Sửa comment: bỏ ám chỉ easyocr là lựa chọn runtime (ghi rõ
  easyocr chỉ còn ở tooling benchmark).

### 7. `main/src/ui/dashboard.py`
- Import `PlateOCR` (dòng ~50) đã guard `try/except → None`: để nguyên (không phá), nhưng
  đảm bảo không có đường runtime nào *gọi* PlateOCR.
- `_ensure_sample_video()`: **bỏ khối `urllib`**. Nếu `sample_parking.mp4` tồn tại → trả
  path; nếu thiếu → `st.info(...)` hướng dẫn đặt clip vào `main/data/test/`, không gọi mạng,
  không `st.error` đỏ.

### 8. `main/src/utils/visual.py`
- Bỏ dòng `@import url('https://fonts.googleapis.com/...')` (dòng 236).
- Thay font-family bằng system stack:
  `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif` cho body
  và `ui-monospace, "SF Mono", Menlo, Consolas, monospace` cho phần mono.

### 9. Tests (giữ suite xanh sau khi gỡ easyocr khỏi runtime)
- `tests/conftest.py`: khối pre-init `easyocr.Reader` là di sản tránh xung đột TF/OpenMP
  (TF đã rời runtime). Bọc import easyocr bằng `try/except ImportError: pass` để
  collection không vỡ khi env runtime không có easyocr.
- `tests/test_ocr.py`: đã `skipIf(not EASYOCR_AVAILABLE)` — giữ nguyên (tự skip khi thiếu).
- `tests/test_plate_pipeline_e2e.py:42` dùng `PlateOCR()` trực tiếp → bọc skip khi easyocr
  không cài (hoặc đổi sang PaddleOCRReader). Chọn: **skip khi thiếu easyocr** để giữ test
  e2e nguyên bản chạy được ở env train.
- `tests/test_ppocr_reader.py`: không đụng.

### Không thuộc phạm vi (chỉ ghi nhận, làm task riêng nếu muốn)
- Gỡ 176MB model "chết" khỏi **git history** (BFG/filter-repo) — đây là dọn lịch sử, rủi ro
  cao hơn, tách riêng. (`.dockerignore` đã loại chúng khỏi image.)
- Pin cứng version toàn bộ deps để tái lập tuyệt đối.
- Drift config khác (mô tả Keras brand/color trong config.yaml).

## Acceptance test (Opus verify — bằng chứng "offline hoàn toàn")

1. `docker build -t dpl-offline .` thành công (không còn lỗi TF import).
2. Kiểm tra image có cache: `docker run --rm dpl-offline ls ~/.paddlex/official_models`
   liệt kê PP-OCRv6 det/rec.
3. **Bằng chứng cứng:** chạy container với mạng bị cắt:
   - `docker run --rm --network none -p 8000:8000 dpl-offline uvicorn main.src.api.app:app --host 0.0.0.0 --port 8000`
   - `POST /verify` 1 ảnh biển → trả verdict đầy đủ (plate_text/color/status), không treo,
     không lỗi network.
4. Dashboard `--network none`: render không lỗi font/network; Upload-Image ra verdict; tab
   Upload-Video dùng clip local, không cố tải.
5. `pytest` (env train) xanh; `pytest` (env runtime, không easyocr) không vỡ collection,
   các test easyocr tự skip.

## Thứ tự thực thi (mỗi bước commit, explicit path; Sonnet thực thi)

1. `.dockerignore` (mới).
2. `requirements.txt` + `requirements-train.txt` (di chuyển easyocr).
3. `pipeline_factory.py` + `config.yaml` (runtime 100% Paddle, fallback hard-error).
4. `dashboard.py` + `visual.py` (bỏ video-download + Google Fonts).
5. Tests (guard easyocr → skip; conftest tolerant).
6. `Dockerfile` (viết lại prime + ENV offline + đổi thứ tự COPY).
7. Opus verify theo Acceptance test ở trên.
