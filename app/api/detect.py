"""
app/api/detect.py — Detection endpoints.

Routes
------
POST /detect-accident
    Body : DetectRequest  { stream_url, max_frames, save_frames }
    Returns 202 JobAccepted { job_id, status, message }

GET /jobs/{job_id}
    Returns DetectionResult (poll until status == "complete" or "failed")

POST /detection-submissions/v1
    Requires Idempotency-Key; accepts or reconciles one durable submission.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from app.core.config import settings
from app.core.security import validate_stream_url
from app.models.schemas import DetectRequest, DetectionResult, JobAccepted, SubmissionAccepted
from app.services.job_store import JobCapacityError, job_store
from app.services.submission_identity import SubmissionIdentityError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["detection"])


def _base_origin(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _artifacts_base_url(request: Request) -> str:
    prefix = settings.api_prefix.rstrip("/")
    if prefix:
        return f"{_base_origin(request)}{prefix}/artifacts"
    return f"{_base_origin(request)}/artifacts"


def _to_absolute_url(request: Request, value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"{_base_origin(request)}{value}"
    return f"{_base_origin(request)}/{value}"


def _identity_error(exc: SubmissionIdentityError) -> HTTPException:
    logger.warning("Rejected detection submission identity: %s", exc.code)
    return HTTPException(status_code=exc.status_code,
                         detail={"code": exc.code, "message": str(exc)})


def _storage_error() -> HTTPException:
    logger.exception("SQLite submission storage unavailable; retry the same identity after recovery.")
    return HTTPException(status_code=503, headers={"Retry-After": "10"}, detail={
        "code": "submission_storage_unavailable",
        "message": "Submission storage is unavailable. Retry the same identity after storage recovers.",
    })


def _check_single_identity_header(request: Request) -> None:
    if len(request.headers.getlist("idempotency-key")) != 1:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_idempotency_key", "message": "Supply exactly one Idempotency-Key header.",
        })


def _replay_response(job: DetectionResult, request: Request, expiry: datetime) -> SubmissionAccepted:
    return SubmissionAccepted(
        job_id=job.job_id, status=job.status, replayed=True,
        idempotency_expires_at=expiry, error=job.error,
        message="Original submission found; no new work scheduled. Poll GET /jobs/{job_id}.",
        frames=[], artifacts_base_url=_artifacts_base_url(request),
    )


@router.get(
    "/detection-submissions/v1",
    response_model=SubmissionAccepted,
    summary="Find an original submission without sending its source URL",
    responses={
        404: {"description": "No receipt found yet; a delayed POST may still arrive."},
        410: {"description": "Expired identity or removed original result; no work scheduled."},
        503: {"description": "Persistent submission storage is unavailable."},
    },
)
async def find_submission(
    request: Request, idempotency_key: str = Header(max_length=50),
) -> SubmissionAccepted:
    _check_single_identity_header(request)
    try:
        job, expires_at = job_store.find_submission(idempotency_key)
    except sqlite3.OperationalError as exc:
        raise _storage_error() from exc
    except SubmissionIdentityError as exc:
        raise _identity_error(exc) from exc
    if job is None:
        raise HTTPException(status_code=404, detail={
            "code": "idempotency_key_unknown",
            "message": "No original submission found yet. A delayed POST may still arrive; this lookup does not cancel it.",
        })
    return _replay_response(job, request, expires_at)


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
async def detect_accident(body: DetectRequest, request: Request, response: Response) -> JobAccepted | SubmissionAccepted:
    if "idempotency-key" in request.headers:
        raise HTTPException(status_code=422, detail={
            "code": "idempotency_requires_versioned_endpoint",
            "message": "Use POST /detection-submissions/v1 for keyed submissions.",
        })
    return await _submit_detection(body, request, response, None)


@router.post(
    "/detection-submissions/v1",
    response_model=SubmissionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Safely submit or reconcile a detection job",
    response_description="New submission accepted, or original submission reconciled without new inference.",
    responses={
        200: {"model": SubmissionAccepted, "description": "Original job found; no new work scheduled."},
        409: {"description": "Identity already used with different detection parameters."},
        410: {"description": "Expired identity or removed original job; no work scheduled."},
        503: {"description": "Model, queue, job capacity, receipt capacity or durable store unavailable."},
    },
)
async def submit_detection(
    body: DetectRequest, request: Request, response: Response,
    idempotency_key: str = Header(
        max_length=50,
        description="Required v1.<Unix seconds>.<lowercase UUIDv4>; identical retries reuse the original job for 24 hours.",
    ),
) -> JobAccepted | SubmissionAccepted:
    return await _submit_detection(body, request, response, idempotency_key)


async def _submit_detection(
    body: DetectRequest, request: Request, response: Response, idempotency_key: str | None,
) -> JobAccepted | SubmissionAccepted:
    """
    Queue an accident detection job against the given stream URL.

    The YOLO model is loaded once at startup and shared across jobs
    (accessed via `request.app.state.model`).  Each job gets its own
    DeepSORT tracker and AccidentScorer instance.

    Optional versioned identities reserve one durable job for identical retries.
    A replay returns the original job state without queue submission.
    """
    stream_url = str(body.stream_url)
    validate_stream_url(stream_url)
    expires_at = None

    if idempotency_key is not None:
        _check_single_identity_header(request)
        try:
            existing, expires_at = job_store.lookup_submission(
                idempotency_key, stream_url, body.max_frames, body.save_frames,
            )
        except sqlite3.OperationalError as exc:
            raise _storage_error() from exc
        except SubmissionIdentityError as exc:
            raise _identity_error(exc) from exc
        if existing is not None:
            response.status_code = 200
            return _replay_response(existing, request, expires_at)

    # Existing receipts can be reconciled even when inference is unavailable.
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model is not yet loaded. Please try again in a moment.",
        )

    queue = getattr(request.app.state, "detection_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="The inference queue is unavailable.")
    max_frames = body.max_frames
    try:
        if idempotency_key is None:
            job = job_store.create(
                stream_url, max_frames=max_frames, save_frames=body.save_frames,
                capacity=settings.get_job_admission_capacity(),
            )
        else:
            job, replayed, expires_at = job_store.create_submission(
                idempotency_key, stream_url, max_frames=max_frames, save_frames=body.save_frames,
                capacity=settings.get_job_admission_capacity(),
            )
            if replayed:
                response.status_code = 200
                return _replay_response(job, request, expires_at)
    except sqlite3.OperationalError as exc:
        if idempotency_key is None:
            raise
        raise _storage_error() from exc
    except SubmissionIdentityError as exc:
        raise _identity_error(exc) from exc
    except JobCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc

    logger.info(
        "Detection job queued  job_id=%s  max_frames=%d",
        job.job_id, max_frames,
    )

    # Offload the blocking CPU work to a thread so the event loop stays free.
    try:
        queue.submit(
            job.job_id,
            stream_url,
            max_frames,
            model,
            body.save_frames,
        )
    except RuntimeError:
        logger.exception("Could not schedule detection job %s", job.job_id)
        terminal_persisted = job_store.fail_job(job.job_id, "The inference worker is unavailable.")
        detail = ("The inference worker is unavailable." if terminal_persisted else
                  "The inference worker is unavailable; recording its failure is awaiting storage recovery.")
        if idempotency_key is not None:
            assert expires_at is not None
            raise HTTPException(status_code=503, detail={
                "code": "inference_admission_failed", "message": detail,
                "job_id": job.job_id, "idempotency_expires_at": expires_at.isoformat(),
            })
        raise HTTPException(status_code=503, detail=detail)

    if idempotency_key is not None:
        assert expires_at is not None
        return SubmissionAccepted(
            job_id=job.job_id, replayed=False, idempotency_expires_at=expires_at,
            frames=[], artifacts_base_url=_artifacts_base_url(request),
        )
    return JobAccepted(
        job_id=job.job_id,
        frames=[],
        artifacts_base_url=_artifacts_base_url(request),
    )


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
async def get_job(job_id: str, request: Request) -> DetectionResult:
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
    absolute_frames = [
        frame.model_copy(
            update={
                "capture": _to_absolute_url(request, frame.capture),
                "capture_annotated": _to_absolute_url(request, frame.capture_annotated),
                "pipeline_output": _to_absolute_url(request, frame.pipeline_output),
            }
        )
        for frame in job.frames
    ]

    return job.model_copy(
        update={
            "frames": absolute_frames,
            "artifacts_base_url": _artifacts_base_url(request),
        }
    )
