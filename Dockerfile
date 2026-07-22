FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/main:/app \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    YOLO_OFFLINE=True \
    PADDLE_PDX_MODEL_SOURCE=BOS

WORKDIR /app

# Install system dependencies for OpenCV, PaddleOCR, and building extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY main/requirements.txt /app/main/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r main/requirements.txt

# linux/aarch64 (this Docker platform on Apple Silicon): PaddlePaddle's PIR
# model loader SEGFAULTs (AnalysisPredictor::Init -> PreparePirProgram ->
# SaveOrLoadPirParameters) on BOTH paddlepaddle 3.2.2 (PyPI ceiling for this
# platform) and 3.3.1 (official Paddle wheel index) — confirmed with fully
# downloaded, fresh PP-OCRv6 models. Those model tars are PIR-only
# (inference.json, no legacy .pdmodel), so FLAGS_enable_pir_api=0 also
# segfaults; there is no working 3.x combination on this platform today.
# Proven-working fallback: paddleocr==2.7.3 + paddlepaddle==2.6.2 with
# numpy<2 uses the legacy (non-PIR) loader, downloads the classic
# en_PP-OCRv3_det + en_PP-OCRv4_rec + cls models into /root/.paddleocr, and
# runs .ocr() without crashing. The container therefore runs this legacy 2.x
# engine; local/native dev keeps the 3.x stack (see main/requirements.txt)
# unchanged, since all benchmarks were measured there.
RUN pip install --no-cache-dir "paddleocr==2.7.3" "paddlepaddle==2.6.2" "numpy<2"

# Copy the rest of the application BEFORE priming, so local model/config
# files (e.g. main/data/models/yolov8n.onnx) exist for the warmup step below.
COPY . /app

# --- Build-time model priming (network allowed here; container runtime is
# zero-network). Each engine is primed independently and must not fail the
# build if it errors out, so we wrap every step in a try/except.

# Prime PaddleOCR through the project's own reader (main/src/models/
# ppocr_reader.py) so the image caches exactly what runtime constructs: the
# 2.x branch (PP-OCRv3-det/PP-OCRv4-rec models into /root/.paddleocr) given
# the legacy stack installed above. PYTHONPATH=/app/main:/app is already set
# and the app code is already copied at this stage.
# timeout: a hung/flaky model download must FAIL the build fast, not hang it
# — a truncated cache in the image would segfault the loader at runtime.
RUN timeout 900 python -c "from src.models.ppocr_reader import PaddleOCRReader; PaddleOCRReader(lang='en')._ensure(); print('PRIME OK')"

# Bake the Ultralytics config dir + warm up the local ONNX detector so the
# Arial.ttf annotation font (and any ultralytics first-run setup) is cached
# in the image layer instead of being fetched at container runtime.
RUN mkdir -p /root/.config/Ultralytics && python -c "import sys, numpy as np; exec(\"try:\\n    from ultralytics import YOLO\\n    model = YOLO('main/data/models/yolov8n.onnx')\\n    model.predict(np.zeros((640, 640, 3), dtype='uint8'), verbose=False)\\nexcept Exception as exc:\\n    print('WARN: YOLO warmup skipped:', exc, file=sys.stderr)\")"

# Set default ports exposed by the container
EXPOSE 8000 8501
