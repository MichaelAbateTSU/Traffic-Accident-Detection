"""
app/api/cameras.py — Minimal camera status endpoint for the frontend.

The backend does not currently own a "camera registry"; instead we derive a
best-effort camera list from known job stream URLs in the in-memory JobStore.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import FrontendCameraStatus
from app.services.job_store import _extract_camera_id, job_store

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get(
    "/status",
    response_model=list[FrontendCameraStatus],
    summary="List camera statuses (derived from jobs)",
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
                last_seen_at=job.completed_at or job.created_at,
            )
        )

    return out

