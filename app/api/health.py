"""
app/api/health.py — Health-check endpoint.

Route
-----
GET /health
    Returns a lightweight status payload that a load balancer, Docker
    HEALTHCHECK, or monitoring system can poll.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.services.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_jobs: int
    total_jobs: int


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    response_description="Current service health and operational metrics.",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Lightweight liveness + readiness check.

    - `status`       : "ok" once the YOLO model is loaded, otherwise "starting"
    - `model_loaded` : True after the lifespan startup hook has finished
    - `active_jobs`  : number of jobs currently in "running" state
    - `total_jobs`   : total jobs in the store (pending + running + complete + failed)
    """
    model_loaded = getattr(request.app.state, "model", None) is not None
    active = job_store.count_running()
    total  = len(job_store.all_jobs())

    return HealthResponse(
        status="ok" if model_loaded else "starting",
        model_loaded=model_loaded,
        active_jobs=active,
        total_jobs=total,
    )
