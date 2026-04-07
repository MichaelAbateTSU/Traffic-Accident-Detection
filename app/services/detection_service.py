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
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

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

from app.core.config import settings
from app.models.schemas import DetectionEvent, FrameImages, SignalValues
from app.services.job_store import job_store

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _artifact_base_url() -> str:
    prefix = settings.api_prefix.rstrip("/")
    if prefix:
        return f"{prefix}/artifacts"
    return "/artifacts"


def _job_output_dirs(job_id: str) -> dict[str, str]:
    base_dir = os.path.join(pipeline_config.OUTPUT_DIR, job_id)
    return {
        "base": base_dir,
        "capture": os.path.join(base_dir, "capture"),
        "capture_annotated": os.path.join(base_dir, "capture_annotated"),
        "pipeline_output": os.path.join(base_dir, "pipeline_output"),
    }


def _ensure_job_output_dirs(job_id: str) -> dict[str, str]:
    dirs = _job_output_dirs(job_id)
    os.makedirs(dirs["capture"], exist_ok=True)
    os.makedirs(dirs["capture_annotated"], exist_ok=True)
    os.makedirs(dirs["pipeline_output"], exist_ok=True)
    return dirs


def _save_frame_triplet(
    job_id: str,
    frame_idx: int,
    frame,
    capture_annotated,
    pipeline_output,
) -> FrameImages:
    import cv2

    dirs = _ensure_job_output_dirs(job_id)
    file_name = f"{frame_idx:06d}.jpg"

    capture_path = os.path.join(dirs["capture"], file_name)
    capture_annotated_path = os.path.join(dirs["capture_annotated"], file_name)
    pipeline_output_path = os.path.join(dirs["pipeline_output"], file_name)

    cv2.imwrite(
        capture_path,
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), pipeline_config.JPEG_QUALITY],
    )
    cv2.imwrite(
        capture_annotated_path,
        capture_annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), pipeline_config.JPEG_QUALITY],
    )
    cv2.imwrite(
        pipeline_output_path,
        pipeline_output,
        [int(cv2.IMWRITE_JPEG_QUALITY), pipeline_config.JPEG_QUALITY],
    )

    base_url = _artifact_base_url()
    return FrameImages(
        frame_index=frame_idx,
        capture=f"{base_url}/{job_id}/capture/{file_name}",
        capture_annotated=f"{base_url}/{job_id}/capture_annotated/{file_name}",
        pipeline_output=f"{base_url}/{job_id}/pipeline_output/{file_name}",
    )


def _prune_old_job_artifacts() -> None:
    """
    Keep artifact folders for active jobs plus the N most recent terminal jobs.

    This bounds disk usage while avoiding deletion of folders that may still be
    written by running/pending jobs.
    """
    root = Path(pipeline_config.OUTPUT_DIR)
    if not root.exists():
        return

    all_jobs = job_store.all_jobs()
    active_job_ids = {
        job.job_id
        for job in all_jobs
        if job.status in ("pending", "running")
    }
    terminal_jobs = sorted(
        [job for job in all_jobs if job.status in ("complete", "failed")],
        key=lambda job: job.completed_at or job.created_at,
        reverse=True,
    )
    keep_terminal = max(int(settings.job_artifact_retention_count), 1)
    keep_job_ids = active_job_ids | {job.job_id for job in terminal_jobs[:keep_terminal]}

    removed: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            # Ignore non-job folders in OUTPUT_DIR.
            continue
        if entry.name in keep_job_ids:
            continue
        try:
            shutil.rmtree(entry)
            removed.append(entry.name)
        except OSError as exc:
            logger.warning("Could not prune artifact folder %s: %s", entry, exc)

    if removed:
        logger.info("Pruned artifact folders for %d old job(s): %s", len(removed), removed)


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

    job_store.update(job_id, status="running", frames=[])

    events: list[DetectionEvent] = []
    frame_images: list[FrameImages] = []
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

            capture_annotated = tracker.draw_tracks(frame, tracks)
            pipeline_output = capture_annotated
            if pipeline_config.OVERLAY_SCORE:
                from pipeline import _overlay_score

                pipeline_output = _overlay_score(
                    capture_annotated.copy(), score, detected, frames_processed
                )
            frame_images.append(
                _save_frame_triplet(
                    job_id=job_id,
                    frame_idx=frames_processed,
                    frame=frame,
                    capture_annotated=capture_annotated,
                    pipeline_output=pipeline_output,
                )
            )

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

            frames_processed += 1

            # Keep the job store up to date while the job is running so
            # polling clients can see incremental progress and frame URLs.
            job_store.update(
                job_id,
                frames_processed=frames_processed,
                peak_confidence=round(peak_confidence, 4),
                events=list(events),
                frames=list(frame_images),
            )

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
        frames=frame_images,
        completed_at=_utcnow(),
    )
    _prune_old_job_artifacts()
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
    _prune_old_job_artifacts()


