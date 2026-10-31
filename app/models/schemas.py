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
        default=5,
        ge=1,
        le=300,
        strict=True,
        description=(
            "Process 1-300 consecutive frames. Default 5 preserves existing clients; "
            "request 300 for temporal scoring. Processing may end earlier at source EOF."
        ),
    )
    save_frames: bool = Field(
        default=False,
        description=(
            "Save raw and annotated frame images to disk and publish their URLs. "
            "When false, no frame images are saved or published."
        ),
    )


# ---------------------------------------------------------------------------
# Per-event sub-model
# ---------------------------------------------------------------------------

class SignalValues(BaseModel):
    """Five heuristic image-space signals, not verified accident evidence."""

    sudden_stop: float = Field(ge=0.0, le=1.0, description="Low-speed history; no preceding motion or speed drop required.")
    abrupt_decel: float = Field(ge=0.0, le=1.0)
    collision_iou: float = Field(ge=0.0, le=1.0, description="Projected-box overlap/convergence candidate, not proof of physical collision.")
    post_collision: float = Field(ge=0.0, le=1.0, description="Current/recent candidate with low speed or direction change; history may precede candidate.")
    traffic_anomaly: float = Field(ge=0.0, le=1.0, description="Nearby low-speed tracks, including candidate participants; no separate vehicle or speed drop required.")


class DetectionEvent(BaseModel):
    """One processed frame with an active heuristic alarm, not a verified accident or new episode.

    Consecutive active frames each produce a record; hysteresis may retain the
    alarm below the high threshold without a new threshold crossing.
    """

    frame_idx: int = Field(description="Zero-based frame index within the job.")
    timestamp_sec: float = Field(
        description="Processing elapsed seconds from actual job start, not media time or queue admission."
    )
    accident_detected: bool = Field(description="Heuristic alarm state for this frame, not verified collision detection.")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="EMA-smoothed heuristic accident score, not a calibrated probability.",
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
        description="DeepSORT track IDs attributed to the heuristic event, not verified collision participants.",
    )
    signal_values: SignalValues


class FrameImages(BaseModel):
    """Image URLs produced for a single processed frame."""

    frame_index: int = Field(ge=0)
    capture: str | None = Field(
        default=None,
        description="URL to the raw captured frame image.",
    )
    capture_annotated: str | None = Field(
        default=None,
        description="URL to the tracker-annotated frame image.",
    )
    pipeline_output: str | None = Field(
        default=None,
        description="URL to the final pipeline output image.",
    )


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
    frames: list[FrameImages] = Field(
        default_factory=list,
        description="Per-frame image URLs generated during this job.",
    )
    artifacts_base_url: str | None = Field(
        default=None,
        description="Base URL where job frame artifacts are served.",
    )
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = Field(
        default=None,
        description="Actual processing start (UTC); null for pending or legacy jobs with no known start.",
    )
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Job-accepted response (returned immediately on POST)
# ---------------------------------------------------------------------------

class JobAccepted(BaseModel):
    """Immediate response from POST /detect-accident."""

    job_id: str
    status: Literal["pending"] = "pending"
    message: str = "Detection job queued. Poll GET /jobs/{job_id} for results."
    frames: list[FrameImages] = Field(
        default_factory=list,
        description="Initially empty list of frame image URLs for this job.",
    )
    artifacts_base_url: str | None = Field(
        default=None,
        description="Base URL where job frame artifacts are served.",
    )


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

INCIDENT_STATUS_DESCRIPTION = (
    "Workflow state, not an operator-review or physical-accident outcome. By default "
    "'active' while the source job is pending/running and 'resolved' once the job "
    "has finished (complete or failed). A PATCH override takes precedence "
    "(see manually_updated). Neither value establishes that an operator reviewed the "
    "event or that a real accident was cleared."
)
MANUALLY_UPDATED_DESCRIPTION = (
    "True when a PATCH /incidents/{id} override exists for this record, so status, "
    "resolved_at and note come from that override rather than the job-state default."
)

ALARM_FRAME_COUNT_DESCRIPTION = (
    "Retained alarm-active frame records detected from UTC midnight through now, "
    "including consecutive frames and resolved records; not distinct accidents or episodes."
)
NONFAILED_JOB_RATIO_DESCRIPTION = (
    "Legacy name: nonfailed retained-job percentage in the admission-time lookback, "
    "including pending/running jobs; rounded to 2 decimals, 100 when empty. "
    "Not measured service uptime. Numerator examines at most 10,000 newest jobs."
)

class StatsOverview(BaseModel):
    """Dashboard overview values for card widgets."""

    active_cameras: int = Field(ge=0)
    total_cameras: int = Field(ge=0)
    incidents_today: int = Field(ge=0, description=ALARM_FRAME_COUNT_DESCRIPTION)
    detection_status: Literal["active", "idle", "degraded", "starting"]
    uptime_percent: float = Field(ge=0.0, le=100.0, description=NONFAILED_JOB_RATIO_DESCRIPTION)
    active_jobs: int = Field(ge=0)
    total_jobs: int = Field(ge=0)
    refreshed_at: datetime
    window_start: datetime
    window_end: datetime

    # UI-friendly aliases (camelCase)
    activeCameras: int = Field(ge=0)
    totalCameras: int = Field(ge=0)
    incidentsToday: int = Field(ge=0, description=ALARM_FRAME_COUNT_DESCRIPTION)
    detectionStatus: Literal["active", "idle", "degraded", "starting"]
    uptimePercent: float = Field(ge=0.0, le=100.0, description=NONFAILED_JOB_RATIO_DESCRIPTION)


class StatsOverviewResponse(BaseModel):
    data: StatsOverview
    meta: ApiMeta
    error: ApiError | None = None


# ---------------------------------------------------------------------------
# Incident models
# ---------------------------------------------------------------------------

class IncidentSummary(BaseModel):
    """One alarm-active frame record; not a distinct verified accident or independent episode."""

    incident_id: str
    id: str
    job_id: str
    camera_id: str
    cameraId: str
    detection_type: str = "accident"
    type: str = "accident"
    status: Literal["active", "resolved"] = Field(description=INCIDENT_STATUS_DESCRIPTION)
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
    resolved_at: datetime | None = None
    resolvedAt: datetime | None = None
    note: str | None = None
    manually_updated: bool = Field(default=False, description=MANUALLY_UPDATED_DESCRIPTION)
    manuallyUpdated: bool = Field(default=False, description=MANUALLY_UPDATED_DESCRIPTION)


class IncidentUpdateRequest(BaseModel):
    """Body accepted by PATCH /incidents/{incident_id}."""

    status: Literal["active", "resolved"]
    note: str | None = Field(
        default=None,
        max_length=500,
        description="Optional operator note stored with the incident.",
    )


class IncidentDetail(IncidentSummary):
    """Expanded incident payload with source job metadata."""

    frame_idx: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    signal_values: SignalValues
    stream_url: str
    job_status: Literal["pending", "running", "complete", "failed"]
    created_at: datetime
    started_at: datetime | None = None
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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    frames_processed: int = Field(ge=0)
    peak_confidence: float = Field(ge=0.0, le=1.0)
    event_count: int = Field(ge=0)
    frames: list[FrameImages] = Field(default_factory=list)
    artifacts_base_url: str | None = None
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
    frames: list[FrameImages] = Field(default_factory=list)
    artifacts_base_url: str | None = None
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    progress: float | None = Field(default=None, ge=0.0, le=100.0, description="Alias of progress_percent.")
    updated_at: datetime
    created_at: datetime
    started_at: datetime | None = None
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
    incidents_today: int = Field(description=ALARM_FRAME_COUNT_DESCRIPTION)
    detection_status: Literal["active", "idle", "degraded", "starting"]
    uptime_percent: float = Field(description=NONFAILED_JOB_RATIO_DESCRIPTION)
    active_jobs: int
    total_jobs: int
    refreshed_at: datetime

    # camelCase (UI convenience)
    activeCameras: int
    totalCameras: int
    incidentsToday: int = Field(description=ALARM_FRAME_COUNT_DESCRIPTION)
    detectionStatus: Literal["active", "idle", "degraded", "starting"]
    uptimePercent: float = Field(description=NONFAILED_JOB_RATIO_DESCRIPTION)


class FrontendIncident(BaseModel):
    """One alarm-active frame record, not a verified accident or independent episode."""

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
    status: Literal["active", "resolved"] = Field(description=INCIDENT_STATUS_DESCRIPTION)
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

    # operator workflow
    resolved_at: datetime | None = None
    resolvedAt: datetime | None = None
    note: str | None = None
    manually_updated: bool = Field(default=False, description=MANUALLY_UPDATED_DESCRIPTION)
    manuallyUpdated: bool = Field(default=False, description=MANUALLY_UPDATED_DESCRIPTION)


class FrontendIncidentList(BaseModel):
    items: list[FrontendIncident]
    meta: ApiMeta


class FrontendCameraStatus(BaseModel):
    id: str
    status: Literal["online", "warning", "offline"]
    stream_url: str | None = None
    last_job_id: str | None = None
    last_seen_at: datetime | None = None


class CameraLatestFrame(BaseModel):
    """Most recent processed frame available for a camera."""

    camera_id: str
    cameraId: str
    job_id: str
    jobId: str
    frame_index: int = Field(ge=0)
    frameIndex: int = Field(ge=0)
    image_url: str | None = Field(default=None, description="Raw captured frame URL.")
    imageUrl: str | None = None
    annotated_url: str | None = Field(
        default=None,
        description="Best annotated frame URL (pipeline output, else tracker overlay).",
    )
    annotatedUrl: str | None = None
    captured_at: datetime
    capturedAt: datetime
    job_status: Literal["pending", "running", "complete", "failed"]
    jobStatus: Literal["pending", "running", "complete", "failed"]


class CameraLatestFrameResponse(BaseModel):
    data: CameraLatestFrame
    meta: ApiMeta
    error: ApiError | None = None


class FrontendDashboardResponse(BaseModel):
    stats: FrontendStats
    incidents: FrontendIncidentList
    cameras: list[FrontendCameraStatus]
