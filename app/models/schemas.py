"""
Pydantic request/response schemas for the accident detection API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    """Body accepted by POST /detect-accident."""

    stream_url: HttpUrl = Field(
        ...,
        description="HLS/RTSP/MP4 URL that OpenCV can open via VideoCapture.",
        examples=["https://example.com/live/stream.m3u8"],
    )
    max_frames: int = Field(
        default=300,
        ge=1,
        le=18_000,
        description=(
            "Maximum frames to process before stopping. "
            "At 30 fps, 300 frames ≈ 10 seconds of stream."
        ),
    )
    save_frames: bool = Field(
        default=False,
        description="Write annotated frames to pipeline_output/ on disk.",
    )


# ---------------------------------------------------------------------------
# Per-event sub-model
# ---------------------------------------------------------------------------

class SignalValues(BaseModel):
    """The five raw accident signals computed by AccidentScorer."""

    sudden_stop: float = Field(ge=0.0, le=1.0)
    abrupt_decel: float = Field(ge=0.0, le=1.0)
    collision_iou: float = Field(ge=0.0, le=1.0)
    post_collision: float = Field(ge=0.0, le=1.0)
    traffic_anomaly: float = Field(ge=0.0, le=1.0)


class DetectionEvent(BaseModel):
    """A single frame where the accident flag crossed the HIGH_THRESHOLD."""

    frame_idx: int = Field(description="Zero-based frame index within the job.")
    timestamp_sec: float = Field(
        description="Approximate wall-clock seconds from job start when the frame was processed."
    )
    accident_detected: bool
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="EMA-smoothed accident confidence score from AccidentScorer.",
    )
    raw_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Unsmoothed weighted sum of the five signals.",
    )
    bounding_box: list[float] | None = Field(
        default=None,
        description="[x1, y1, x2, y2] pixel coordinates of the event region, if available.",
    )
    involved_track_ids: list[int] = Field(
        default_factory=list,
        description="DeepSORT track IDs of vehicles involved in the event.",
    )
    signal_values: SignalValues


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    """Full result returned by GET /jobs/{job_id} once the job completes."""

    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    stream_url: str
    max_frames: int | None = Field(
        default=None,
        ge=1,
        le=18_000,
        description="Max frames requested for the job (when known).",
    )
    save_frames: bool | None = Field(
        default=None,
        description="Whether the job was configured to persist annotated frames (when known).",
    )
    frames_processed: int = 0
    accident_detected: bool = False
    peak_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest EMA-smoothed score observed during the job.",
    )
    events: list[DetectionEvent] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Job-accepted response (returned immediately on POST)
# ---------------------------------------------------------------------------

class JobAccepted(BaseModel):
    """Immediate response from POST /detect-accident."""

    job_id: str
    status: Literal["pending"] = "pending"
    message: str = "Detection job queued. Poll GET /jobs/{job_id} for results."


# ---------------------------------------------------------------------------
# Shared API metadata / error models (new endpoints only)
# ---------------------------------------------------------------------------

class PaginationMeta(BaseModel):
    """Pagination details attached to list responses."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=200)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_prev: bool


class ApiError(BaseModel):
    """Structured error payload for envelope-style responses."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiMeta(BaseModel):
    """Common metadata attached to new API responses."""

    request_id: str
    timestamp: datetime
    duration_ms: float = Field(ge=0.0)
    filters: dict[str, Any] | None = None
    pagination: PaginationMeta | None = None


# ---------------------------------------------------------------------------
# Stats models
# ---------------------------------------------------------------------------

class StatsOverview(BaseModel):
    """Dashboard overview values for card widgets."""

    active_cameras: int = Field(ge=0)
    total_cameras: int = Field(ge=0)
    incidents_today: int = Field(ge=0)
    detection_status: Literal["active", "idle", "degraded", "starting"]
    uptime_percent: float = Field(ge=0.0, le=100.0)
    active_jobs: int = Field(ge=0)
    total_jobs: int = Field(ge=0)
    refreshed_at: datetime
    window_start: datetime
    window_end: datetime

    # UI-friendly aliases (camelCase)
    activeCameras: int = Field(ge=0)
    totalCameras: int = Field(ge=0)
    incidentsToday: int = Field(ge=0)
    detectionStatus: Literal["active", "idle", "degraded", "starting"]
    uptimePercent: float = Field(ge=0.0, le=100.0)


class StatsOverviewResponse(BaseModel):
    data: StatsOverview
    meta: ApiMeta
    error: ApiError | None = None


# ---------------------------------------------------------------------------
# Incident models
# ---------------------------------------------------------------------------

class IncidentSummary(BaseModel):
    """Lightweight incident projection suitable for list views."""

    incident_id: str
    id: str
    job_id: str
    camera_id: str
    cameraId: str
    detection_type: str = "accident"
    type: str = "accident"
    status: Literal["active", "resolved"]
    resolved: bool
    detected_at: datetime
    detectedAt: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidenceScore: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    rawScore: float = Field(ge=0.0, le=1.0)
    bounding_box: list[float] | None = None
    boundingBox: list[float] | None = None
    involved_track_ids: list[int] = Field(default_factory=list)
    involvedTrackIds: list[int] = Field(default_factory=list)


class IncidentDetail(IncidentSummary):
    """Expanded incident payload with source job metadata."""

    frame_idx: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    signal_values: SignalValues
    stream_url: str
    job_status: Literal["pending", "running", "complete", "failed"]
    created_at: datetime
    completed_at: datetime | None = None


class IncidentListResponse(BaseModel):
    data: list[IncidentSummary]
    meta: ApiMeta
    error: ApiError | None = None


class IncidentDetailResponse(BaseModel):
    data: IncidentDetail
    meta: ApiMeta
    error: ApiError | None = None


# ---------------------------------------------------------------------------
# Jobs models
# ---------------------------------------------------------------------------

class JobSummary(BaseModel):
    """Lightweight detection job projection for list endpoints."""

    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    stream_url: str
    camera_id: str
    created_at: datetime
    completed_at: datetime | None = None
    frames_processed: int = Field(ge=0)
    peak_confidence: float = Field(ge=0.0, le=1.0)
    event_count: int = Field(ge=0)
    accident_detected: bool = False
    error: str | None = None


class JobStatusSummary(BaseModel):
    """Minimal per-job polling payload for high-frequency refresh."""

    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    frames_processed: int = Field(ge=0)
    event_count: int = Field(ge=0)
    total_detections: int = Field(ge=0, description="Alias of event_count for UI convenience.")
    totalDetections: int = Field(ge=0, description="camelCase alias of total_detections.")
    peak_confidence: float = Field(ge=0.0, le=1.0)
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    progress: float | None = Field(default=None, ge=0.0, le=100.0, description="Alias of progress_percent.")
    updated_at: datetime
    created_at: datetime
    completed_at: datetime | None = None


class JobDetectionSummary(BaseModel):
    """Compact detection projection used for paginated job detections."""

    incident_id: str
    frame_idx: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    detected_at: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    bounding_box: list[float] | None = None
    involved_track_ids: list[int] = Field(default_factory=list)


class JobDetectionDetail(JobDetectionSummary):
    """Full detection payload for detailed incident and job views."""

    signal_values: SignalValues


class JobListResponse(BaseModel):
    data: list[JobSummary]
    meta: ApiMeta
    error: ApiError | None = None


class JobStatusResponse(BaseModel):
    data: JobStatusSummary
    meta: ApiMeta
    error: ApiError | None = None


class JobDetectionsResponse(BaseModel):
    data: list[JobDetectionSummary] | list[JobDetectionDetail]
    meta: ApiMeta
    error: ApiError | None = None


# ---------------------------------------------------------------------------
# Frontend-friendly aggregate / compatibility models
# ---------------------------------------------------------------------------

class FrontendStats(BaseModel):
    # snake_case (native API)
    active_cameras: int
    total_cameras: int
    incidents_today: int
    detection_status: Literal["active", "idle", "degraded", "starting"]
    uptime_percent: float
    active_jobs: int
    total_jobs: int
    refreshed_at: datetime

    # camelCase (UI convenience)
    activeCameras: int
    totalCameras: int
    incidentsToday: int
    detectionStatus: Literal["active", "idle", "degraded", "starting"]
    uptimePercent: float


class FrontendIncident(BaseModel):
    # identifiers
    incident_id: str
    id: str
    job_id: str

    # camera / type
    camera_id: str
    cameraId: str
    detection_type: str
    type: str

    # state / time
    status: Literal["active", "resolved"]
    resolved: bool
    detected_at: datetime
    detectedAt: datetime

    # scores / details
    confidence_score: float
    confidenceScore: float
    raw_score: float
    rawScore: float
    bounding_box: list[float] | None = None
    boundingBox: list[float] | None = None
    involved_track_ids: list[int] = Field(default_factory=list)
    involvedTrackIds: list[int] = Field(default_factory=list)


class FrontendIncidentList(BaseModel):
    items: list[FrontendIncident]
    meta: ApiMeta


class FrontendCameraStatus(BaseModel):
    id: str
    status: Literal["online", "warning", "offline"]
    stream_url: str | None = None
    last_job_id: str | None = None
    last_seen_at: datetime | None = None


class FrontendDashboardResponse(BaseModel):
    stats: FrontendStats
    incidents: FrontendIncidentList
    cameras: list[FrontendCameraStatus]
