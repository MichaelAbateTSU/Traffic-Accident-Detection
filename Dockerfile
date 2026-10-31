FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-production-amd64.txt requirements-production-arm64.txt ./
RUN case "$TARGETARCH" in amd64|arm64) ;; *) echo "Only amd64 and arm64 are supported"; exit 1 ;; esac \
    && pip install --no-cache-dir --require-hashes -r "requirements-production-${TARGETARCH}.txt"
COPY app ./app
COPY pipeline.py tracker.py accident_scorer.py config.py ./
# Bake a small CPU model into the image; serving never downloads weights.
RUN mkdir /models && cd /models \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt', 'yolo11n.pt')" \
    && echo "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  yolo11n.pt" | sha256sum -c - \
    && cd /app && python -c "from ultralytics import YOLO; YOLO('/models/yolo11n.pt'); from tracker import DeepSortTracker; DeepSortTracker()"
RUN useradd --uid 10001 --create-home detector && mkdir /data && chown detector:detector /data
USER detector
ENV YOLO_MODEL=/models/yolo11n.pt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); assert r['model_loaded']"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--proxy-headers", "--forwarded-allow-ips=172.30.0.3"]
