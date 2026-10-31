"""
app/api/incidents.py — Incident/event query endpoints.
"""

from __future__ import annotations

import csv
import io
import unicodedata
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.api.common import build_meta, start_timer
from app.models.schemas import (
    IncidentDetail,
    IncidentDetailResponse,
    IncidentListResponse,
    IncidentSummary,
    IncidentUpdateRequest,
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
        resolved_at=record.get("resolved_at"),
        resolvedAt=record.get("resolved_at"),
        note=record.get("note"),
        manually_updated=bool(record.get("manually_updated")),
        manuallyUpdated=bool(record.get("manually_updated")),
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
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


SortOption = Literal["-detected_at", "detected_at", "-confidence", "confidence"]

CSV_COLUMNS = [
    "incident_id",
    "job_id",
    "camera_id",
    "detection_type",
    "status",
    "detected_at",
    "confidence_score",
    "raw_score",
    "frame_idx",
    "timestamp_sec",
    "involved_track_ids",
    "resolved_at",
    "note",
]


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _spreadsheet_safe(value: str) -> str:
    index = 0
    has_control = False
    while index < len(value):
        char = value[index]
        control = unicodedata.category(char).startswith("C")
        if not char.isspace() and not control:
            break
        has_control = has_control or control
        index += 1
    if value[index:].startswith(("=", "+", "-", "@")) or has_control:
        return "'" + value
    return value


@router.get(
    "/export",
    summary="Export incidents as CSV (same filters as GET /incidents, no pagination)",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "CSV file download."}},
)
async def export_incidents(
    sort: SortOption = Query(default="-detected_at"),
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
) -> Response:
    filters = dict(
        sort=sort,
        status=status_filter,
        job_id=job_id,
        camera_id=camera_id,
        detection_type=detection_type,
        min_confidence=min_confidence,
        start_time=start_time,
        end_time=end_time,
    )
    records, _ = job_store.list_events_paginated(page_size=None, **filters)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for rec in records:
        event = rec["event"]
        writer.writerow(
            [
                _spreadsheet_safe(rec["incident_id"]),
                _spreadsheet_safe(rec["job"].job_id),
                _spreadsheet_safe(rec["camera_id"]),
                _spreadsheet_safe(rec["detection_type"]),
                _spreadsheet_safe(rec["status"]),
                _iso(rec["detected_at"]),
                event.confidence_score,
                event.raw_score,
                event.frame_idx,
                event.timestamp_sec,
                " ".join(str(t) for t in event.involved_track_ids),
                _iso(rec.get("resolved_at")),
                _spreadsheet_safe(rec.get("note") or ""),
            ]
        )

    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="incidents.csv"'},
    )


@router.get(
    "",
    response_model=IncidentListResponse,
    summary="List incidents with filtering and pagination",
)
async def list_incidents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort: SortOption = Query(default="-detected_at"),
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


@router.patch(
    "/{incident_id}",
    response_model=IncidentDetailResponse,
    summary="Resolve or reopen an incident (optionally with an operator note)",
    responses={404: {"description": "Incident not found."}},
)
async def update_incident(
    incident_id: str,
    body: IncidentUpdateRequest,
) -> IncidentDetailResponse:
    started = start_timer()
    record = job_store.set_incident_status(
        incident_id,
        status=body.status,
        note=body.note,
    )
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
