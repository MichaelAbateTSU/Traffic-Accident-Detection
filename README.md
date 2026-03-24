# Traffic Accident Detection — YOLOv8 + DeepSORT + FastAPI

A production-ready traffic accident detection system that processes live HLS/RTSP camera streams using **YOLOv8** object detection and **DeepSORT** multi-object tracking, exposed as a **REST API** built with FastAPI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the API Server](#running-the-api-server)
- [API Reference](#api-reference)
- [Running the Pipeline Directly](#running-the-pipeline-directly)
- [Configuration](#configuration)
- [Detection Signals](#detection-signals)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

---

## Overview

The system connects to any OpenCV-compatible video source (HLS stream, RTSP, local file), runs per-frame vehicle detection and tracking, and computes a continuous accident confidence score from five heuristic signals. Results are surfaced via a REST API that a frontend can call with a single button click.

**Key capabilities:**

- Real-time YOLOv8 vehicle detection (car, truck, bus, motorcycle)
- Persistent vehicle identity across frames via DeepSORT tracking
- Five-signal weighted accident score with EMA smoothing and hysteresis
- Non-blocking FastAPI backend — POST a job, poll for results
- Single shared YOLO model loaded at startup (no per-request reload)
- Thread-safe in-memory job store with automatic TTL cleanup

---

## Architecture

```
Frontend
   │
   │  POST /detect-accident  { stream_url, max_frames }
   ▼
FastAPI (app/)
   │  202 Accepted  { job_id }
   │
   ├──► BackgroundThread ──► detection_service.run_detection_job()
   │                              │
   │                              ├─ _open_video(stream_url)      [pipeline.py]
   │                              ├─ _iter_video()                [pipeline.py]
   │                              ├─ _run_yolo(model, frame)      [pipeline.py]
   │                              ├─ DeepSortTracker.update()     [tracker.py]
   │                              └─ AccidentScorer.update()      [accident_scorer.py]
   │                                       │
   │                                       └─► JobStore (in-memory)
   │
   │  GET /jobs/{job_id}
   ▼
   { status, accident_detected, peak_confidence, events: [...] }
```

The root-level detection modules (`pipeline.py`, `tracker.py`, `accident_scorer.py`, `config.py`) are **not modified** — the `app/` service layer wraps them.

---

## Project Structure

```
Traffic-Accident-Detection/
│
├── pipeline.py                  ← CLI pipeline: YOLO + DeepSORT + AccidentScorer
├── tracker.py                   ← DeepSortTracker wrapper
├── accident_scorer.py           ← Rolling-window accident confidence scorer
├── config.py                    ← All tunable detection parameters
├── yolo_screenshot_detector.py  ← Original batch/screenshot-based detector
├── detection_analysis_example.py
├── requirements.txt
├── README.md
│
├── app/                         ← FastAPI service
│   ├── main.py                  ← App factory, lifespan (YOLO model load), CORS
│   ├── api/
│   │   ├── detect.py            ← POST /detect-accident, GET /jobs/{job_id}
│   │   └── health.py            ← GET /health
│   ├── core/
│   │   ├── config.py            ← Pydantic BaseSettings (env / .env)
│   │   └── logging.py           ← Structured logging setup
│   ├── models/
│   │   └── schemas.py           ← DetectRequest, DetectionEvent, DetectionResult
│   ├── services/
│   │   ├── detection_service.py ← Wraps pipeline.py; run_detection_job()
│   │   └── job_store.py         ← Thread-safe job store with TTL reaper
│   └── utils/
│       └── cleanup.py           ← Temp file helpers (file-upload support)
│
├── pipeline_output/             ← Annotated frames saved by the pipeline
└── captures/                    ← Snapshot test output
    ├── originals/
    ├── labeled/
    └── scores/
```

---

## Prerequisites

- **Python 3.10+**
- **FFmpeg** available to OpenCV (required for HLS/RTSP streams)
  - Windows: install via `winget install ffmpeg` or [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - Linux/macOS: `sudo apt install ffmpeg` / `brew install ffmpeg`
- A CUDA-capable GPU is recommended for real-time performance but not required

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url> Traffic-Accident-Detection
cd Traffic-Accident-Detection

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The YOLO model (`yolo11x.pt`) is downloaded automatically by Ultralytics on first run.

### Verify installation

```bash
python -c "from ultralytics import YOLO; import cv2; import fastapi; print('All packages OK')"
```

---

## Running the API Server

Run from the **project root** directory (so that `pipeline.py`, `tracker.py`, etc. are importable):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> `--workers 1` is intentional. The YOLO model is not fork-safe; for horizontal scaling, run multiple containers rather than multiple workers in one process.

**Development mode** (auto-reload on file changes):

```bash
uvicorn app.main:app --reload --port 8000
```

Once running, open **http://localhost:8000/docs** for the interactive Swagger UI.

### Environment variables / `.env`

Create a `.env` file in the project root to override defaults:

```ini
YOLO_MODEL=yolo11x.pt
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
MAX_CONCURRENT_JOBS=4
JOB_TTL_SECONDS=3600
LOG_LEVEL=INFO
```

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolo11x.pt` | Model weights file |
| `CORS_ORIGINS` | `*` | Comma-separated frontend origins (`*` = allow all) |
| `MAX_CONCURRENT_JOBS` | `4` | Rejects new jobs when at capacity (503) |
| `JOB_TTL_SECONDS` | `3600` | How long completed job results stay in memory |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `API_PREFIX` | `` | Optional prefix for all routes, e.g. `/api/v1` |

---

## API Reference

### `POST /detect-accident`

Start an accident detection job against a stream URL.

**Request body:**

```json
{
  "stream_url": "https://example.com/live/stream.m3u8",
  "max_frames": 300,
  "save_frames": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `stream_url` | string (URL) | required | Any source OpenCV can open: HLS, RTSP, MP4 |
| `max_frames` | integer | `300` | Frame cap (~10 s at 30 fps). Range: 1–18000 |
| `save_frames` | boolean | `false` | Write annotated frames to `pipeline_output/` |

**Response `202 Accepted`:**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "pending",
  "message": "Detection job queued. Poll GET /jobs/{job_id} for results."
}
```

---

### `GET /jobs/{job_id}`

Poll for job status and results.

**Response `200 OK`:**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "complete",
  "stream_url": "https://example.com/live/stream.m3u8",
  "frames_processed": 300,
  "accident_detected": true,
  "peak_confidence": 0.831,
  "events": [
    {
      "frame_idx": 81,
      "timestamp_sec": 8.1,
      "accident_detected": true,
      "confidence_score": 0.712,
      "raw_score": 0.695,
      "bounding_box": [412.0, 210.0, 587.0, 334.0],
      "involved_track_ids": [3, 9],
      "signal_values": {
        "sudden_stop": 0.5,
        "abrupt_decel": 0.45,
        "collision_iou": 0.8,
        "post_collision": 0.8,
        "traffic_anomaly": 0.5
      }
    }
  ],
  "error": null,
  "created_at": "2026-03-18T12:00:00Z",
  "completed_at": "2026-03-18T12:00:18Z"
}
```

**Job statuses:**

| Status | Meaning |
|---|---|
| `pending` | Queued, not yet started |
| `running` | Actively processing frames |
| `complete` | Finished; inspect `events` and `accident_detected` |
| `failed` | Error; see the `error` field |

**Response `404`** if the job ID is unknown or has expired.

---

### `GET /health`

Liveness and readiness check for load balancers / Docker HEALTHCHECK.

**Response `200 OK`:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_jobs": 1,
  "total_jobs": 3
}
```

`status` is `"starting"` while the YOLO model is still loading.

---

### Example: full polling loop (Python)

```python
import httpx, time

client = httpx.Client(base_url="http://localhost:8000")

# 1. Submit job
resp = client.post("/detect-accident", json={
    "stream_url": "https://mcleansfs3.us-east-1.skyvdn.com/rtplive/R2_066/playlist.m3u8",
    "max_frames": 300,
})
job_id = resp.json()["job_id"]
print(f"Job started: {job_id}")

# 2. Poll until complete
while True:
    result = client.get(f"/jobs/{job_id}").json()
    print(f"  status={result['status']}  frames={result['frames_processed']}")
    if result["status"] in ("complete", "failed"):
        break
    time.sleep(3)

# 3. Inspect results
print(f"Accident detected: {result['accident_detected']}")
print(f"Peak confidence:   {result['peak_confidence']}")
print(f"Events:            {len(result['events'])}")
```

---

## Running the Pipeline Directly

The CLI pipeline runs without the API server — useful for development and testing.

```bash
# Default: live TDOT HLS stream
python pipeline.py

# Local video file
python pipeline.py --source path/to/video.mp4

# Folder of images (processed in filename order)
python pipeline.py --source frames/

# Custom HLS / RTSP URL
python pipeline.py --source https://example.com/stream.m3u8

# Test mode (process up to 300 frames, then exit)
python pipeline.py --source video.mp4 --test --max-test-frames 100

# Headless (no cv2 preview window — for servers)
python pipeline.py --source video.mp4 --test --no-display

# Snapshot test: capture 5 frames, run full pipeline, save JSON + images to captures/
python pipeline.py --snapshot-test
```

### Console output

```
[00042] score=0.031  stop=0.00  decel=0.00  iou=0.00  post=0.00  anomaly=0.00  tracks=7
[00081] score=0.712 *** ACCIDENT DETECTED ***  stop=0.50  decel=0.45  iou=0.80  post=0.80  anomaly=0.50  tracks=5
         involved tracks: [3, 9]  region: (412.0, 210.0, 587.0, 334.0)
```

Annotated frames are written to `pipeline_output/` (JPEG, configurable).

---

## Configuration

All detection parameters live in `config.py`. Edit once; every module picks up the change.

### Detection / YOLO

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolo11x.pt` | Model weights (auto-downloaded) |
| `VEHICLE_CLASSES` | `[2, 3, 5, 7]` | COCO IDs: car, motorcycle, bus, truck |
| `DET_CONF_THRESHOLD` | `0.30` | Minimum YOLO detection confidence |
| `INFERENCE_SIZE` | `1280` | YOLO input resolution (px) |

### Tracking

| Parameter | Default | Description |
|---|---|---|
| `DEEPSORT_MAX_AGE` | `30` | Frames before a lost track is deleted |
| `HISTORY_WINDOW_SECONDS` | `5` | Rolling history length per track |

### Accident scoring

| Parameter | Default | Description |
|---|---|---|
| `STOP_SPEED_THRESHOLD` | `5.0 px/frame` | Speed below which a vehicle is "stopped" |
| `COLLISION_IOU_THRESHOLD` | `0.10` | Min box IoU to flag a collision event |
| `EMA_ALPHA` | `0.30` | Score smoothing (higher = faster response) |
| `HIGH_THRESHOLD` | `0.65` | Score that sets `accident_detected = True` |
| `LOW_THRESHOLD` | `0.40` | Score that clears `accident_detected` (hysteresis) |

### Output

| Parameter | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `pipeline_output` | Directory for saved annotated frames |
| `SAVE_FRAMES` | `True` | Persist annotated frames to disk |
| `JPEG_QUALITY` | `85` | JPEG quality for saved frames |

---

## Detection Signals

Five heuristic signals are computed from each vehicle's rolling history window and combined into a weighted confidence score:

| Signal | Weight | Description |
|---|---|---|
| `sudden_stop` | 0.25 | Track speed drops and stays near zero |
| `abrupt_decel` | 0.20 | High negative mean acceleration over a short window |
| `collision_iou` | 0.30 | Two tracks' boxes overlap **and** were converging |
| `post_collision` | 0.15 | After overlap: track stationary or moving erratically |
| `traffic_anomaly` | 0.10 | Multiple nearby vehicles also slowing near the event region |

**Scoring:**

```
raw_score      = Σ (weight × signal)
smoothed_score = EMA(raw_score, α=0.30)

accident_detected = True   when smoothed_score ≥ 0.65
accident_detected = False  when smoothed_score < 0.40  (hysteresis prevents flicker)
```

---

## Output

### API job result

See `GET /jobs/{job_id}` response above. Each `DetectionEvent` includes the frame index, timestamp, bounding box of the event region, involved track IDs, and all five signal values.

### Pipeline direct output

```
pipeline_output/
├── 000000.jpg   ← annotated frame 0 (bounding boxes + score banner)
├── 000001.jpg
└── ...

captures/             (snapshot test only)
├── originals/
│   └── frame_00.jpg  ← raw frame
├── labeled/
│   └── frame_00.jpg  ← annotated frame
└── scores/
    ├── frame_00.json ← per-frame score + metadata
    └── final_score.json
```

---

## Troubleshooting

### `RuntimeError: Cannot open source: ...`

FFmpeg is required for HLS/RTSP streams. Verify FFmpeg is on the PATH:

```bash
ffmpeg -version
```

Reinstall OpenCV with FFmpeg support if needed:

```bash
pip install opencv-python --upgrade
```

### YOLO model not found

On first run, Ultralytics downloads the model automatically. If the download fails:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

### `503 Service Unavailable` on POST /detect-accident

- **"YOLO model is not yet loaded"** — the server is still in its lifespan startup; wait a few seconds and retry.
- **"Server is already running N job(s)"** — maximum concurrent jobs reached (`MAX_CONCURRENT_JOBS`). Wait for a job to complete or increase the limit in `.env`.

### CUDA not available / slow performance

GPU acceleration is optional. To enable it:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Low detection accuracy

- Try a larger YOLO model: change `YOLO_MODEL=yolo11x.pt` in `config.py` (already the default)
- Increase `INFERENCE_SIZE` to `1280` or `1920`
- Lower `DET_CONF_THRESHOLD` to `0.20` for harder conditions (night, rain)

---

## Technical Details

### Pipeline data flow (per frame)

```
VideoCapture frame (BGR)
     │
     ▼
_run_yolo()          → [(x1,y1,x2,y2, conf, class_name), ...]
     │
     ▼
DeepSortTracker.update()  → [Track(id, bbox, class), ...]
     │
     ▼
AccidentScorer.update()   → (score: float, detected: bool, metadata: dict)
     │
     ▼
_overlay_score()          → annotated frame (optional, saved to disk)
```

### Coordinate formats

| Stage | Format | Description |
|---|---|---|
| YOLO output | `xyxy` | `(x1, y1, x2, y2)` pixel corners |
| DeepSORT input | `xywh` | `(left, top, width, height)` — converted inside `tracker.py` |
| Scorer history | `cx, cy` | Box centre used for velocity and acceleration maths |

### Scaling the API

The service is designed for `--workers 1` because the YOLO model occupies GPU VRAM and is not safe to share across forked processes. For higher throughput:

- Run multiple Docker containers behind a load balancer (one YOLO instance per container)
- Replace the in-memory `JobStore` with Redis for shared state across instances

---

## License

For educational and research purposes. Respect TDOT and any other camera operator's usage policies. Traffic camera feeds are public but should be used responsibly.

---

**Python**: 3.10+ &nbsp;|&nbsp; **Platform**: Windows / Linux / macOS
