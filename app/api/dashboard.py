"""
app/api/dashboard.py — Single-call aggregate payload for the frontend.

This endpoint is intentionally "UI oriented": it returns all the data the
frontend needs for the dashboard in one request (stats + recent incidents +
camera statuses), with both snake_case and camelCase keys where helpful.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Query, Request

from app.api.common import build_meta, start_timer
from app.models.schemas import (
    FrontendCameraStatus,
    FrontendDashboardResponse,
    FrontendIncident,
    FrontendIncidentList,
    FrontendStats,
)
from app.services.job_store import _extract_camera_id, job_store

router = APIRouter(tags=["dashboard"])


def _camera_statuses() -> list[FrontendCameraStatus]:
    jobs = job_store.all_jobs()
    by_camera: dict[str, object] = {}

    for job in jobs:
        camera_id = _extract_camera_id(job.stream_url)
        prev = by_camera.get(camera_id)
        if prev is None or getattr(prev, "created_at", None) < job.created_at:
            by_camera[camera_id] = job

    def _status_from_job(job_status: str) -> Literal["online", "warning", "offline"]:
        s = (job_status or "").lower()
        if s in ("pending", "running"):
            return "online"
        if s == "failed":
            return "warning"
        return "offline"

    out: list[FrontendCameraStatus] = []
    for camera_id, job in sorted(by_camera.items(), key=lambda kv: kv[0]):
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


def _to_frontend_incident(record: dict) -> FrontendIncident:
    event = record["event"]
    incident_id = record["incident_id"]
    camera_id = record["camera_id"]
    detection_type = record["detection_type"]
    detected_at = record["detected_at"]

    return FrontendIncident(
        incident_id=incident_id,
        id=incident_id,
        job_id=record["job"].job_id,
        camera_id=camera_id,
        cameraId=camera_id,
        detection_type=detection_type,
        type=detection_type,
        status=record["status"],
        resolved=record["status"] == "resolved",
        detected_at=detected_at,
        detectedAt=detected_at,
        confidence_score=event.confidence_score,
        confidenceScore=event.confidence_score,
        raw_score=event.raw_score,
        rawScore=event.raw_score,
        bounding_box=event.bounding_box,
        boundingBox=event.bounding_box,
        involved_track_ids=event.involved_track_ids,
        involvedTrackIds=event.involved_track_ids,
    )


@router.get(
    "/dashboard",
    response_model=FrontendDashboardResponse,
    summary="Get dashboard bootstrap payload (stats + incidents + cameras)",
)
async def get_dashboard(
    request: Request,
    history_days: int = Query(default=30, ge=1, le=365),
    incidents_page_size: int = Query(default=5, ge=1, le=50),
    incidents_status: Literal["active", "resolved", "all"] = Query(default="active"),
) -> FrontendDashboardResponse:
    started = start_timer()
    now = datetime.now(tz=timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    lookback_start = now - timedelta(days=history_days)

    counts = job_store.count_by_status()
    active_jobs = counts["running"]
    total_jobs = sum(counts.values())
    active_cameras, total_cameras = job_store.camera_counts()
    incidents_today = job_store.count_incidents_in_range(start_time=day_start, end_time=now)

    jobs_in_window, total_in_window = job_store.list_jobs_paginated(
        start_time=lookback_start,
        end_time=now,
        page=1,
        page_size=10_000,
        sort="-created_at",
    )
    if total_in_window == 0:
        uptime_percent = 100.0
    else:
        success_like = sum(1 for j in jobs_in_window if j.status != "failed")
        uptime_percent = round((success_like / total_in_window) * 100.0, 2)

    model_loaded = getattr(request.app.state, "model", None) is not None
    if not model_loaded:
        detection_status = "starting"
    elif counts["failed"] > counts["complete"] + counts["running"]:
        detection_status = "degraded"
    elif active_jobs > 0:
        detection_status = "active"
    else:
        detection_status = "idle"

    stats = FrontendStats(
        active_cameras=active_cameras,
        total_cameras=max(total_cameras, active_cameras),
        incidents_today=incidents_today,
        detection_status=detection_status,
        uptime_percent=uptime_percent,
        active_jobs=active_jobs,
        total_jobs=total_jobs,
        refreshed_at=now,
        activeCameras=active_cameras,
        totalCameras=max(total_cameras, active_cameras),
        incidentsToday=incidents_today,
        detectionStatus=detection_status,
        uptimePercent=uptime_percent,
    )

    status_filter = None if incidents_status == "all" else incidents_status
    records, total = job_store.list_events_paginated(
        page=1,
        page_size=incidents_page_size,
        sort="-detected_at",
        status=status_filter,
    )
    items = [_to_frontend_incident(rec) for rec in records]
    meta = build_meta(
        started_at=started,
        filters={
            "history_days": history_days,
            "incidents_page_size": incidents_page_size,
            "incidents_status": incidents_status,
        },
        page=1,
        page_size=incidents_page_size,
        total_items=total,
    )

    return FrontendDashboardResponse(
        stats=stats,
        incidents=FrontendIncidentList(items=items, meta=meta),
        cameras=_camera_statuses(),
    )

