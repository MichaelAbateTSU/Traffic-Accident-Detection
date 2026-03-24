"""
app/services/detection_service.py — Service layer that wraps the existing
YOLOv8 + DeepSORT + AccidentScorer pipeline for use by the FastAPI endpoints.

INTEGRATION POINTS
------------------
All detection logic lives in the root-level modules:

    pipeline.py        → _open_video, _iter_video, _run_yolo
    tracker.py         → DeepSortTracker
    accident_scorer.py → AccidentScorer
    config.py          → HIGH_THRESHOLD, ASSUMED_FPS, …

This file only orchestrates those modules and converts their outputs into
the Pydantic schemas expected by the API.

Public entry point
------------------
    await run_detection_job(job_id, stream_url, max_frames, model, save_frames)

The function is a synchronous CPU-bound task wrapped in asyncio.to_thread()
by the caller (see api/detect.py) so it does not block the event loop.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# INTEGRATION POINT — root-level pipeline modules
# ---------------------------------------------------------------------------
# These imports reach up two levels from app/services/ to the project root.
# They work as-is because main.py ensures the project root is on sys.path,
# and because uvicorn is launched from the project root directory.
# ---------------------------------------------------------------------------
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from pipeline import _open_video, _iter_video, _run_yolo          # noqa: E402
from tracker import DeepSortTracker                                 # noqa: E402
from accident_scorer import AccidentScorer                          # noqa: E402
import config as pipeline_config                                    # noqa: E402
# ---------------------------------------------------------------------------

from app.models.schemas import DetectionEvent, DetectionResult, SignalValues
from app.services.job_store import job_store

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Core detection function (runs synchronously in a thread pool)
# ---------------------------------------------------------------------------

def run_detection_job(
    job_id: str,
    stream_url: str,
    max_frames: int,
    model,           # ultralytics.YOLO instance, pre-loaded at startup
    save_frames: bool = False,
) -> None:
    """
    Process up to *max_frames* frames from *stream_url* and write results
    into the job store.

    This function is blocking (CPU-bound) and must be called via
    asyncio.to_thread() or a ThreadPoolExecutor — never directly from an
    async route handler.

    Parameters
    ----------
    job_id      : UUID string identifying the job in the JobStore.
    stream_url  : Any source accepted by cv2.VideoCapture — HLS, RTSP, MP4.
    max_frames  : Hard cap on frames processed (keeps response times bounded).
    model       : Pre-loaded YOLO model from app state (avoids reload per job).
    save_frames : If True, annotated frames are written to pipeline_output/.
    """
    logger.info("Job started  job_id=%s  stream_url=%s  max_frames=%d",
                job_id, stream_url, max_frames)

    job_store.update(job_id, status="running")

    events: list[DetectionEvent] = []
    frames_processed = 0
    peak_confidence = 0.0
    t_job_start = time.monotonic()

    # ------------------------------------------------------------------
    # INTEGRATION POINT — instantiate fresh tracker + scorer per job.
    # The YOLO model is *shared* (passed in) but tracker/scorer are
    # stateful and must not be shared across concurrent jobs.
    # ------------------------------------------------------------------
    tracker = DeepSortTracker()
    scorer  = AccidentScorer()

    try:
        cap = _open_video(stream_url)
    except RuntimeError as exc:
        _fail_job(job_id, str(exc))
        return

    fps = cap.get(0x00000005) or float(pipeline_config.ASSUMED_FPS)  # CAP_PROP_FPS = 5

    try:
        for frame in _iter_video(cap):
            if frames_processed >= max_frames:
                break

            t_frame = time.monotonic() - t_job_start

            # ----------------------------------------------------------
            # INTEGRATION POINT — the three-step pipeline (unchanged)
            # ----------------------------------------------------------
            detections = _run_yolo(model, frame)
            tracks     = tracker.update(detections, frame)
            score, detected, meta = scorer.update(tracks, frames_processed)
            # ----------------------------------------------------------

            if score > peak_confidence:
                peak_confidence = score

            if detected:
                event_region = meta.get("event_region")
                bbox = list(event_region) if event_region else None

                events.append(DetectionEvent(
                    frame_idx=frames_processed,
                    timestamp_sec=round(t_frame, 3),
                    accident_detected=True,
                    confidence_score=round(score, 4),
                    raw_score=round(meta.get("raw_score", score), 4),
                    bounding_box=bbox,
                    involved_track_ids=list(meta.get("involved_track_ids", [])),
                    signal_values=SignalValues(**meta["signal_values"]),
                ))

            if save_frames:
                _save_annotated_frame(frame, tracker, tracks, scorer,
                                      score, detected, frames_processed, model)

            frames_processed += 1

            # Keep the job store up to date while the job is running so
            # polling clients can see incremental progress.
            if frames_processed % 30 == 0:
                job_store.update(job_id, frames_processed=frames_processed,
                                 peak_confidence=round(peak_confidence, 4))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in job %s", job_id)
        _fail_job(job_id, repr(exc))
        return

    # ------------------------------------------------------------------
    # Write final result
    # ------------------------------------------------------------------
    accident_detected = any(e.accident_detected for e in events)

    job_store.update(
        job_id,
        status="complete",
        frames_processed=frames_processed,
        accident_detected=accident_detected,
        peak_confidence=round(peak_confidence, 4),
        events=events,
        completed_at=_utcnow(),
    )
    logger.info(
        "Job complete  job_id=%s  frames=%d  accident=%s  peak=%.3f  events=%d",
        job_id, frames_processed, accident_detected, peak_confidence, len(events),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_job(job_id: str, error: str) -> None:
    logger.error("Job failed  job_id=%s  error=%s", job_id, error)
    job_store.update(
        job_id,
        status="failed",
        error=error,
        completed_at=_utcnow(),
    )


def _save_annotated_frame(frame, tracker, tracks, scorer, score, detected,
                          frame_idx: int, model) -> None:
    """
    Optionally persist an annotated frame to disk.
    Mirrors the save logic in pipeline.run_pipeline().
    """
    import os
    import cv2
    from pipeline import _overlay_score

    os.makedirs(pipeline_config.OUTPUT_DIR, exist_ok=True)
    annotated = tracker.draw_tracks(frame, tracks)
    annotated = _overlay_score(annotated, score, detected, frame_idx)

    fname = f"{frame_idx:06d}.jpg"
    out_path = os.path.join(pipeline_config.OUTPUT_DIR, fname)
    cv2.imwrite(out_path, annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), pipeline_config.JPEG_QUALITY])
