# Traffic Accident Detection — YOLOv8 + DeepSORT + FastAPI

A traffic accident detection system that processes live HLS/RTSP camera streams using **YOLOv8** object detection and **DeepSORT** multi-object tracking, exposed as a **REST API** built with FastAPI.

For a secure CPU backend on a free remote VM, use the [production deployment guide](deploy/README.md). This repository includes a deployable runtime, not a provisioned or guaranteed always-on cloud service.

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
- Thread-safe job store with automatic TTL cleanup and optional durable SQLite

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

`requirements.txt` includes `setuptools<81` because DeepSORT's PyTorch embedder uses the deprecated `pkg_resources` API to locate its bundled model weights but does not declare that runtime dependency. This also covers Python 3.12+ virtual environments, which no longer include setuptools by default. Keep this constraint until DeepSORT removes that dependency.

### Verify installation

```bash
python -c "from ultralytics import YOLO; import cv2; import fastapi; print('All packages OK')"
python -c "from tracker import DeepSortTracker; DeepSortTracker(); print('DeepSORT embedder OK')"
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
| `MAX_CONCURRENT_JOBS` | `1` | Atomically rejects new jobs when pending/running jobs fill capacity (503) |
| `JOB_TTL_SECONDS` | `3600` | How long completed job results stay in memory |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `API_PREFIX` | `` | Optional prefix for all routes, e.g. `/api/v1` |
| `ENVIRONMENT` | `development` | `production` fails startup without a token, explicit HTTPS origins and persistent storage |
| `API_TOKEN` | empty | Bearer token required on API/artifact routes when set; mandatory in production |
| `APPROVED_STREAM_URLS` | `[]` | JSON list of exact HTTPS URLs accepted in production |
| `JOB_DATABASE_PATH` | unset | SQLite jobs/operator state; set to a persistent volume in production |
| `OUTPUT_DIR` | unset | Override the pipeline's artifact directory |
| `MAX_RETAINED_JOBS` | `100` | Oldest terminal histories/artifacts are removed at this limit |
| `MAX_SAVED_FRAMES_PER_JOB` | `300` | Stops saving images at this count without stopping inference |
| `MAX_SAVED_FRAME_BYTES` | `2000000` | Oversized image writes fail explicitly |
| `JOB_TIMEOUT_SECONDS` | `300` | Job deadline, checked between frames |

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
| `save_frames` | boolean | `false` | Save raw and annotated images to `pipeline_output/` and publish frame URLs; when false, no images are written or published |

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

### `GET /stats/overview`

Dashboard-oriented aggregate metrics with metadata envelope.

**Query params:**

| Name | Type | Default | Description |
|---|---|---|---|
| `history_days` | integer | `30` | Lookback window used to estimate `uptime_percent` |

**Response `200 OK`:**

```json
{
  "data": {
    "active_cameras": 2,
    "total_cameras": 5,
    "incidents_today": 3,
    "detection_status": "active",
    "uptime_percent": 98.42,
    "active_jobs": 1,
    "total_jobs": 14,
    "refreshed_at": "2026-03-24T15:10:00Z",
    "window_start": "2026-02-23T15:10:00Z",
    "window_end": "2026-03-24T15:10:00Z"
  },
  "meta": {
    "request_id": "84f9f050-a511-4d79-9f09-875b9c6f3d74",
    "timestamp": "2026-03-24T15:10:00Z",
    "duration_ms": 1.94,
    "filters": { "history_days": 30 },
    "pagination": null
  },
  "error": null
}
```

---

### `GET /incidents`

Paginated incident feed derived from detection events.

**Query params (all optional unless noted):**

| Name | Type | Default | Description |
|---|---|---|---|
| `page` | integer | `1` | Page number |
| `page_size` | integer | `20` | Items per page (`1..200`) |
| `sort` | string | `-detected_at` | `-detected_at`, `detected_at`, `-confidence`, `confidence` |
| `status` | string | none | `active` or `resolved` |
| `job_id` | string | none | Filter incidents for one job |
| `camera_id` | string | none | Filter by camera identifier |
| `detection_type` | string | none | Currently `accident` |
| `min_confidence` | float | none | Minimum confidence threshold (`0..1`) |
| `start_time` | datetime | none | Inclusive lower bound |
| `end_time` | datetime | none | Inclusive upper bound |

**Example request:**

```http
GET /incidents?page=1&page_size=20&status=active&min_confidence=0.7
```

**Response `200 OK`:**

```json
{
  "data": [
    {
      "incident_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6:81",
      "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "camera_id": "R2_061",
      "detection_type": "accident",
      "status": "active",
      "resolved": false,
      "detected_at": "2026-03-24T15:08:10Z",
      "confidence_score": 0.712,
      "raw_score": 0.695,
      "bounding_box": [412.0, 210.0, 587.0, 334.0],
      "involved_track_ids": [3, 9]
    }
  ],
  "meta": {
    "request_id": "02ecbceb-28be-46c8-ba03-d57a8db4a314",
    "timestamp": "2026-03-24T15:10:00Z",
    "duration_ms": 1.01,
    "filters": {
      "status": "active",
      "job_id": null,
      "camera_id": null,
      "detection_type": null,
      "min_confidence": 0.7,
      "start_time": null,
      "end_time": null,
      "sort": "-detected_at"
    },
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next": false,
      "has_prev": false
    }
  },
  "error": null
}
```

---

### `GET /incidents/{incident_id}`

Get full detail for a single incident.

**Response `200 OK`:**

```json
{
  "data": {
    "incident_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6:81",
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "camera_id": "R2_061",
    "detection_type": "accident",
    "status": "resolved",
    "resolved": true,
    "detected_at": "2026-03-24T15:08:10Z",
    "confidence_score": 0.712,
    "raw_score": 0.695,
    "bounding_box": [412.0, 210.0, 587.0, 334.0],
    "involved_track_ids": [3, 9],
    "frame_idx": 81,
    "timestamp_sec": 8.1,
    "signal_values": {
      "sudden_stop": 0.5,
      "abrupt_decel": 0.45,
      "collision_iou": 0.8,
      "post_collision": 0.8,
      "traffic_anomaly": 0.5
    },
    "stream_url": "https://example.com/live/stream.m3u8",
    "job_status": "complete",
    "created_at": "2026-03-24T15:08:00Z",
    "completed_at": "2026-03-24T15:08:18Z"
  },
  "meta": {
    "request_id": "a09de3d7-c6a0-4f60-a422-377278094bcf",
    "timestamp": "2026-03-24T15:10:01Z",
    "duration_ms": 0.73,
    "filters": { "incident_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6:81" },
    "pagination": null
  },
  "error": null
}
```

---

### `GET /jobs`

Paginated lightweight job listing.

**Example request:**

```http
GET /jobs?status=running&start_time=2026-03-24T00:00:00Z&end_time=2026-03-24T23:59:59Z
```

**Response `200 OK`:**

```json
{
  "data": [
    {
      "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "status": "running",
      "stream_url": "https://example.com/live/stream.m3u8",
      "camera_id": "stream",
      "created_at": "2026-03-24T15:08:00Z",
      "completed_at": null,
      "frames_processed": 120,
      "peak_confidence": 0.73,
      "event_count": 1,
      "accident_detected": true,
      "error": null
    }
  ],
  "meta": {
    "request_id": "65d4c2a3-70f1-4da8-b30f-b5706f6480b0",
    "timestamp": "2026-03-24T15:10:01Z",
    "duration_ms": 0.84,
    "filters": {
      "status": "running",
      "start_time": "2026-03-24T00:00:00+00:00",
      "end_time": "2026-03-24T23:59:59+00:00",
      "sort": "-created_at"
    },
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next": false,
      "has_prev": false
    }
  },
  "error": null
}
```

---

### `GET /jobs/{job_id}/status`

Lightweight status endpoint for high-frequency frontend polling.

**Response `200 OK`:**

```json
{
  "data": {
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "status": "running",
    "frames_processed": 120,
    "event_count": 1,
    "peak_confidence": 0.73,
    "progress_percent": null,
    "updated_at": "2026-03-24T15:10:01Z",
    "created_at": "2026-03-24T15:08:00Z",
    "completed_at": null
  },
  "meta": {
    "request_id": "35377bf8-5f7a-49f2-bac9-89f987c80464",
    "timestamp": "2026-03-24T15:10:01Z",
    "duration_ms": 0.36,
    "filters": { "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6" },
    "pagination": null
  },
  "error": null
}
```

---

### `GET /jobs/{job_id}/detections`

Paginated detections for a single job (`detail=summary|full`).

**Example request:**

```http
GET /jobs/{job_id}/detections?detail=full&page=1&page_size=50
```

**Response `200 OK` (full detail):**

```json
{
  "data": [
    {
      "incident_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6:81",
      "frame_idx": 81,
      "timestamp_sec": 8.1,
      "detected_at": "2026-03-24T15:08:10Z",
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
  "meta": {
    "request_id": "0ab6ea40-ff4a-4d14-bfa0-91f2ed503ef7",
    "timestamp": "2026-03-24T15:10:01Z",
    "duration_ms": 0.91,
    "filters": {
      "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "min_confidence": null,
      "start_time": null,
      "end_time": null,
      "sort": "-detected_at",
      "detail": "full"
    },
    "pagination": {
      "page": 1,
      "page_size": 50,
      "total_items": 1,
      "total_pages": 1,
      "has_next": false,
      "has_prev": false
    }
  },
  "error": null
}
```

---

### `PATCH /incidents/{incident_id}`

Resolve or reopen an incident, optionally attaching an operator note (≤ 500 chars). Operator overrides take precedence over the job-derived status everywhere incidents appear (`/incidents`, `/incidents/{id}`, `/incidents/export`, `/dashboard`) and are dropped when the source job is reaped.

Saving a note or repeating `status: "resolved"` preserves the existing resolution time, including the default job completion time. Reopening clears it; resolving again records a new operator time.

```bash
curl -X PATCH http://localhost:8000/incidents/3fa85f64-5717-4562-b3fc-2c963f66afa6:42 \
     -H "Content-Type: application/json" \
     -d '{"status": "resolved", "note": "Tow truck dispatched"}'
```

Returns the same envelope as `GET /incidents/{incident_id}`. All incident payloads include:

| Field | Description |
|---|---|
| `resolved_at` / `resolvedAt` | When the incident was resolved (operator time, or job completion time by default); `null` while active |
| `note` | Last operator note; sending `note: null`/omitting it keeps the previous note |
| `manually_updated` / `manuallyUpdated` | `true` once an operator has changed the status |

---

### `GET /incidents/export`

Downloads incidents as `text/csv` (`Content-Disposition: attachment; filename="incidents.csv"`). Accepts the same filter and sort parameters as `GET /incidents`, without pagination.

Each export uses a single filtered/sorted job-and-override snapshot, so concurrent updates cannot truncate it through a stale count. Text cells that could be interpreted as spreadsheet formulas (`=`, `+`, `-`, `@`, including after leading whitespace/control characters), or start with control characters, are prefixed with an apostrophe. Ordinary text and numeric columns are unchanged; JSON incident notes retain their original values.

Columns: `incident_id, job_id, camera_id, detection_type, status, detected_at, confidence_score, raw_score, frame_idx, timestamp_sec, involved_track_ids, resolved_at, note` (`involved_track_ids` is space-separated).

```bash
curl -o incidents.csv "http://localhost:8000/incidents/export?camera_id=SR-60&status=active&sort=-confidence"
```

---

### `GET /cameras/{camera_id}/latest-frame`

Camera IDs are derived consistently for camera status, job summaries, overview counts, incident filters/export, and latest-frame lookup. For common HLS manifests (`playlist.m3u8`, `index.m3u8`, `master.m3u8`, case-insensitive), the parent path segment identifies the camera: `https://mcleansfs3.us-east-1.skyvdn.com/rtplive/R2_061/playlist.m3u8` becomes `R2_061`, not `playlist`. Query strings and fragments do not affect identity. Other filenames (including root-level manifests) retain the filename without its extension; URLs with no path use the host (including port). This remains a best-effort identity, not a camera registry: identical camera path names on different hosts may still share an ID.

Returns the most recent saved frame for a camera (the highest-index frame of the newest job that has saved frames). Submit a job with `save_frames: true` to populate this endpoint; `404` when no saved frames exist yet. Jobs with `save_frames: false` still publish detection events and progress, but their `frames` list remains empty. They do not delete or hide images saved by previous jobs, so this endpoint may still return an older saved frame.

```json
{
  "data": {
    "camera_id": "SR-60",
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "frame_index": 4,
    "image_url": "/artifacts/3fa85f64-5717-4562-b3fc-2c963f66afa6/capture/000004.jpg",
    "annotated_url": "/artifacts/3fa85f64-5717-4562-b3fc-2c963f66afa6/pipeline_output/000004.jpg",
    "captured_at": "2026-03-24T15:10:00Z",
    "job_status": "complete"
  },
  "meta": { "request_id": "…", "timestamp": "…", "duration_ms": 0.4 },
  "error": null
}
```

camelCase aliases (`cameraId`, `jobId`, `frameIndex`, `imageUrl`, `annotatedUrl`, `capturedAt`, `jobStatus`) are included. Image URLs are rooted at the API host; clients on another origin should prefix them with the API base URL.

---

### Running the tests

```bash
pip install fastapi pydantic pydantic-settings httpx pytest numpy opencv-python
pytest -q
```

The HTTP contract tests stub unavailable ML dependencies, so they run without a GPU or model weights. Saved-frame tests use a mocked inference pipeline with real OpenCV JPEG writes and HTTP artifact reads; they are skipped if OpenCV is missing.

`tests/test_runtime_installation.py` checks genuine DeepSORT initialization in a fresh subprocess (no ML stubs) when the full runtime dependencies are installed. It loads the bundled MobileNet weights without downloading YOLO weights and is skipped in HTTP-only test environments.

---

### Error behavior

- `404` for unknown `job_id`, `incident_id`, or a camera with no saved frames.
- `422` for invalid query parameters (for example `page=0`, `min_confidence=1.2`) or invalid PATCH bodies.
- Worker initialization, capture/read, inference, and image-write errors are logged and recorded as failed jobs with `error` and `completed_at`. Captures are released on completion, frame-limit exits, and failures. A frame's URLs are published only after all three JPEG writes succeed; failed jobs remove incomplete/unpublished images while retaining previously published frames.
- Legacy endpoints (`/detect-accident`, `/jobs/{job_id}`, `/health`) retain their original response shapes.

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
