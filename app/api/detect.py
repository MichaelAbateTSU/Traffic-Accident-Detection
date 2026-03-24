"""
app/api/detect.py — Detection endpoints.

Routes
------
POST /detect-accident
    Body : DetectRequest  { stream_url, max_frames, save_frames }
    Returns 202 JobAccepted { job_id, status, message }

GET /jobs/{job_id}
    Returns DetectionResult (poll until status == "complete" or "failed")
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.models.schemas import DetectRequest, DetectionResult, JobAccepted
from app.services.detection_service import run_detection_job
from app.services.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["detection"])


# ---------------------------------------------------------------------------
# POST /detect-accident
# ---------------------------------------------------------------------------

@router.post(
    "/detect-accident",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an accident detection job",
    response_description="Job accepted. Poll GET /jobs/{job_id} for results.",
)
async def detect_accident(body: DetectRequest, request: Request) -> JobAccepted:
    """
    Queue an accident detection job against the given stream URL.

    The YOLO model is loaded once at startup and shared across jobs
    (accessed via `request.app.state.model`).  Each job gets its own
    DeepSORT tracker and AccidentScorer instance.

    Returns immediately with a `job_id`; processing happens in a background
    thread so the event loop is never blocked.
    """
    # Guard: reject new jobs when the server is already at capacity.
    running = job_store.count_running()
    if running >= settings.max_concurrent_jobs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Server is already running {running} job(s). "
                f"Maximum concurrent jobs: {settings.max_concurrent_jobs}. "
                "Please wait for an existing job to finish."
            ),
        )

    # Guard: YOLO model must be loaded (set in lifespan).
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model is not yet loaded. Please try again in a moment.",
        )

    stream_url = str(body.stream_url)
    job = job_store.create(stream_url)

    logger.info(
        "Detection job queued  job_id=%s  stream_url=%s  max_frames=%d",
        job.job_id, stream_url, body.max_frames,
    )

    # Offload the blocking CPU work to a thread so the event loop stays free.
    asyncio.get_event_loop().run_in_executor(
        None,  # uses the default ThreadPoolExecutor
        run_detection_job,
        job.job_id,
        stream_url,
        body.max_frames,
        model,
        body.save_frames,
    )

    return JobAccepted(job_id=job.job_id)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=DetectionResult,
    summary="Poll a detection job for results",
    responses={
        404: {"description": "Job not found (unknown or expired)."},
    },
)
async def get_job(job_id: str) -> DetectionResult:
    """
    Return the current state of a detection job.

    - `status = "pending"`  — job is queued, not yet started
    - `status = "running"`  — actively processing frames
    - `status = "complete"` — finished; inspect `events`, `accident_detected`, etc.
    - `status = "failed"`   — an error occurred; see the `error` field

    Jobs are retained in memory for `job_ttl_seconds` (default: 1 hour) after
    completion, then automatically purged.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )
    return job
