"""
Pydantic request/response schemas for the accident detection API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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
