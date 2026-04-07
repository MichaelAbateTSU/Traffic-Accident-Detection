"""
app/api/jobs.py — Job summary/status/detections endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.common import build_meta, start_timer
from app.core.config import settings
from app.models.schemas import (
    JobDetectionDetail,
    JobDetectionsResponse,
    JobDetectionSummary,
    JobListResponse,
    JobStatusResponse,
    JobStatusSummary,
    JobSummary,
)
from app.services.job_store import job_store

router = APIRouter(prefix="/jobs", tags=["jobs"])


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


def _absolute_frames(request: Request, frames: list) -> list:
    return [
        frame.model_copy(
            update={
                "capture": _to_absolute_url(request, frame.capture),
                "capture_annotated": _to_absolute_url(request, frame.capture_annotated),
                "pipeline_output": _to_absolute_url(request, frame.pipeline_output),
            }
        )
        for frame in frames
    ]


def _camera_id_from_url(stream_url: str) -> str:
    parsed = urlparse(stream_url)
    path = parsed.path.strip("/")
    if path:
        leaf = path.split("/")[-1]
        if "." in leaf:
            leaf = leaf.rsplit(".", 1)[0]
        if leaf:
            return leaf
    return parsed.netloc or "unknown-camera"


def _to_job_summary(job, request: Request) -> JobSummary:
    return JobSummary(
        job_id=job.job_id,
        status=job.status,
        stream_url=job.stream_url,
        camera_id=_camera_id_from_url(job.stream_url),
        created_at=job.created_at,
        completed_at=job.completed_at,
        frames_processed=job.frames_processed,
        peak_confidence=job.peak_confidence,
        event_count=len(job.events),
        frames=_absolute_frames(request, job.frames),
        artifacts_base_url=_artifacts_base_url(request),
        accident_detected=job.accident_detected,
        error=job.error,
    )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs with filtering and pagination",
)
async def list_jobs(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort: Literal["-created_at", "created_at", "-completed_at", "completed_at"] = Query(
        default="-created_at"
    ),
    status_filter: Literal["pending", "running", "complete", "failed"] | None = Query(
        default=None,
        alias="status",
    ),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
) -> JobListResponse:
    started = start_timer()
    jobs, total = job_store.list_jobs_paginated(
        status=status_filter,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
        sort=sort,
    )
    data = [_to_job_summary(job, request) for job in jobs]
    return JobListResponse(
        data=data,
        meta=build_meta(
            started_at=started,
            filters={
                "status": status_filter,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "sort": sort,
            },
            page=page,
            page_size=page_size,
            total_items=total,
        ),
        error=None,
    )


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get lightweight status for a single job",
    responses={404: {"description": "Job not found."}},
)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    started = start_timer()
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )

    progress_percent = None
    if getattr(job, "max_frames", None):
        try:
            progress_percent = round((job.frames_processed / float(job.max_frames)) * 100.0, 2)
        except Exception:
            progress_percent = None

    data = JobStatusSummary(
        job_id=job.job_id,
        status=job.status,
        frames_processed=job.frames_processed,
        event_count=len(job.events),
        total_detections=len(job.events),
        totalDetections=len(job.events),
        peak_confidence=job.peak_confidence,
        frames=_absolute_frames(request, job.frames),
        artifacts_base_url=_artifacts_base_url(request),
        progress_percent=progress_percent,
        progress=progress_percent,
        updated_at=job.completed_at or datetime.now(tz=timezone.utc),
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
    return JobStatusResponse(
        data=data,
        meta=build_meta(
            started_at=started,
            filters={"job_id": job_id},
        ),
        error=None,
    )


@router.get(
    "/{job_id}/detections",
    response_model=JobDetectionsResponse,
    summary="List detections for a single job",
    responses={404: {"description": "Job not found."}},
)
async def list_job_detections(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort: Literal["-detected_at", "detected_at", "-confidence", "confidence"] = Query(
        default="-detected_at"
    ),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    detail: Literal["summary", "full"] = Query(default="summary"),
) -> JobDetectionsResponse:
    started = start_timer()
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )

    records, total = job_store.list_events_paginated(
        job_id=job_id,
        min_confidence=min_confidence,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if detail == "full":
        data = [
            JobDetectionDetail(
                incident_id=rec["incident_id"],
                frame_idx=rec["event"].frame_idx,
                timestamp_sec=rec["event"].timestamp_sec,
                detected_at=rec["detected_at"],
                confidence_score=rec["event"].confidence_score,
                raw_score=rec["event"].raw_score,
                bounding_box=rec["event"].bounding_box,
                involved_track_ids=rec["event"].involved_track_ids,
                signal_values=rec["event"].signal_values,
            )
            for rec in records
        ]
    else:
        data = [
            JobDetectionSummary(
                incident_id=rec["incident_id"],
                frame_idx=rec["event"].frame_idx,
                timestamp_sec=rec["event"].timestamp_sec,
                detected_at=rec["detected_at"],
                confidence_score=rec["event"].confidence_score,
                raw_score=rec["event"].raw_score,
                bounding_box=rec["event"].bounding_box,
                involved_track_ids=rec["event"].involved_track_ids,
            )
            for rec in records
        ]

    return JobDetectionsResponse(
        data=data,
        meta=build_meta(
            started_at=started,
            filters={
                "job_id": job_id,
                "min_confidence": min_confidence,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "sort": sort,
                "detail": detail,
            },
            page=page,
            page_size=page_size,
            total_items=total,
        ),
        error=None,
    )
