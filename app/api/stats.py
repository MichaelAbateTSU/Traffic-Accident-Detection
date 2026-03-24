"""
app/api/stats.py — Dashboard statistics endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request

from app.api.common import build_meta, start_timer
from app.models.schemas import StatsOverview, StatsOverviewResponse
from app.services.job_store import job_store

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get(
    "/overview",
    response_model=StatsOverviewResponse,
    summary="Get dashboard overview stats",
    response_description="Dashboard-friendly aggregate counters and service status.",
)
async def get_stats_overview(
    request: Request,
    history_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Lookback window used to estimate uptime percentage.",
    ),
) -> StatsOverviewResponse:
    started = start_timer()
    now = datetime.now(tz=timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    lookback_start = now - timedelta(days=history_days)

    counts = job_store.count_by_status()
    active_jobs = counts["running"]
    total_jobs = sum(counts.values())
    active_cameras, total_cameras = job_store.camera_counts()
    incidents_today = job_store.count_incidents_in_range(
        start_time=day_start,
        end_time=now,
    )

    # Approximate uptime as non-failed job ratio in a configurable lookback.
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

    data = StatsOverview(
        active_cameras=active_cameras,
        total_cameras=max(total_cameras, active_cameras),
        incidents_today=incidents_today,
        detection_status=detection_status,
        uptime_percent=uptime_percent,
        active_jobs=active_jobs,
        total_jobs=total_jobs,
        refreshed_at=now,
        window_start=lookback_start,
        window_end=now,
        activeCameras=active_cameras,
        totalCameras=max(total_cameras, active_cameras),
        incidentsToday=incidents_today,
        detectionStatus=detection_status,
        uptimePercent=uptime_percent,
    )
    return StatsOverviewResponse(
        data=data,
        meta=build_meta(
            started_at=started,
            filters={"history_days": history_days},
        ),
        error=None,
    )
