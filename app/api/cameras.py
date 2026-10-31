"""
app/api/cameras.py — Minimal camera status endpoint for the frontend.

The backend does not currently own a "camera registry"; instead we derive a
best-effort camera list from known job stream URLs in the in-memory JobStore.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.common import build_meta, start_timer
from app.models.schemas import (
    CameraLatestFrame,
    CameraLatestFrameResponse,
    FrontendCameraStatus,
)
from app.services.job_store import _extract_camera_id, job_store

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get(
    "/status",
    response_model=list[FrontendCameraStatus],
    summary="List camera statuses (derived from jobs)",
    description=(
        "Not a camera registry or health check. One entry per distinct source ID among retained jobs "
        "(video URLs included), using the latest job by created_at: pending/running -> online, "
        "failed -> warning, otherwise offline. last_seen_at is a job timestamp, not a heartbeat."
    ),
)
async def get_camera_status() -> list[FrontendCameraStatus]:
    jobs = job_store.all_jobs()

    by_camera: dict[str, dict] = {}
    for job in jobs:
        camera_id = _extract_camera_id(job.stream_url)

        prev = by_camera.get(camera_id)
        created_at = job.created_at
        if prev is None or (prev["created_at"] < created_at):
            by_camera[camera_id] = {
                "created_at": created_at,
                "job": job,
            }

    def _status_from_job(job_status: str) -> str:
        s = (job_status or "").lower()
        if s in ("pending", "running"):
            return "online"
        if s == "failed":
            return "warning"
        return "offline"

    out: list[FrontendCameraStatus] = []
    for camera_id, record in sorted(by_camera.items(), key=lambda kv: kv[0]):
        job = record["job"]
        out.append(
            FrontendCameraStatus(
                id=camera_id,
                status=_status_from_job(job.status),
                stream_url=job.stream_url,
                last_job_id=job.job_id,
                last_seen_at=job.completed_at or job.started_at or job.created_at,
            )
        )

    return out


@router.get(
    "/{camera_id}/latest-frame",
    response_model=CameraLatestFrameResponse,
    summary="Get the most recent processed frame for a camera",
    responses={404: {"description": "No processed frames exist for this camera."}},
)
async def get_camera_latest_frame(camera_id: str) -> CameraLatestFrameResponse:
    started = start_timer()
    found = job_store.latest_frame_for_camera(camera_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No processed frames available for camera '{camera_id}'.",
        )
    job, frame = found
    image_url = frame.capture or frame.capture_annotated or frame.pipeline_output
    annotated_url = frame.pipeline_output or frame.capture_annotated or frame.capture
    captured_at = job.completed_at if job.status in ("complete", "failed") and job.completed_at else (job.started_at or job.created_at)
    data = CameraLatestFrame(
        camera_id=camera_id,
        cameraId=camera_id,
        job_id=job.job_id,
        jobId=job.job_id,
        frame_index=frame.frame_index,
        frameIndex=frame.frame_index,
        image_url=image_url,
        imageUrl=image_url,
        annotated_url=annotated_url,
        annotatedUrl=annotated_url,
        captured_at=captured_at,
        capturedAt=captured_at,
        job_status=job.status,
        jobStatus=job.status,
    )
    return CameraLatestFrameResponse(
        data=data,
        meta=build_meta(started_at=started, filters={"camera_id": camera_id}),
        error=None,
    )
