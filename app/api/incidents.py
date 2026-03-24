"""
app/api/incidents.py — Incident/event query endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from app.api.common import build_meta, start_timer
from app.models.schemas import (
    IncidentDetail,
    IncidentDetailResponse,
    IncidentListResponse,
    IncidentSummary,
)
from app.services.job_store import job_store

router = APIRouter(prefix="/incidents", tags=["incidents"])


def _to_incident_summary(record: dict) -> IncidentSummary:
    event = record["event"]
    incident_id = record["incident_id"]
    camera_id = record["camera_id"]
    detection_type = record["detection_type"]
    detected_at = record["detected_at"]
    involved_track_ids = event.involved_track_ids
    return IncidentSummary(
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
        involved_track_ids=involved_track_ids,
        involvedTrackIds=involved_track_ids,
    )


def _to_incident_detail(record: dict) -> IncidentDetail:
    summary = _to_incident_summary(record)
    event = record["event"]
    job = record["job"]
    return IncidentDetail(
        **summary.model_dump(),
        frame_idx=event.frame_idx,
        timestamp_sec=event.timestamp_sec,
        signal_values=event.signal_values,
        stream_url=job.stream_url,
        job_status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get(
    "",
    response_model=IncidentListResponse,
    summary="List incidents with filtering and pagination",
)
async def list_incidents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort: Literal["-detected_at", "detected_at", "-confidence", "confidence"] = Query(
        default="-detected_at"
    ),
    status_filter: Literal["active", "resolved"] | None = Query(
        default=None,
        alias="status",
    ),
    job_id: str | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    detection_type: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
) -> IncidentListResponse:
    started = start_timer()
    records, total = job_store.list_events_paginated(
        page=page,
        page_size=page_size,
        sort=sort,
        status=status_filter,
        job_id=job_id,
        camera_id=camera_id,
        detection_type=detection_type,
        min_confidence=min_confidence,
        start_time=start_time,
        end_time=end_time,
    )
    data = [_to_incident_summary(rec) for rec in records]
    filters = {
        "status": status_filter,
        "job_id": job_id,
        "camera_id": camera_id,
        "detection_type": detection_type,
        "min_confidence": min_confidence,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "sort": sort,
    }
    return IncidentListResponse(
        data=data,
        meta=build_meta(
            started_at=started,
            filters=filters,
            page=page,
            page_size=page_size,
            total_items=total,
        ),
        error=None,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentDetailResponse,
    summary="Get full incident details",
    responses={404: {"description": "Incident not found."}},
)
async def get_incident(incident_id: str) -> IncidentDetailResponse:
    started = start_timer()
    record = job_store.get_incident(incident_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found.",
        )
    return IncidentDetailResponse(
        data=_to_incident_detail(record),
        meta=build_meta(
            started_at=started,
            filters={"incident_id": incident_id},
        ),
        error=None,
    )
