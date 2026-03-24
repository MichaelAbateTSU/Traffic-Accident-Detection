"""
app/services/job_store.py — Thread-safe in-memory job store.

Jobs are stored as DetectionResult objects keyed by job_id (UUID string).
A background reaper thread removes jobs that have exceeded their TTL.

Swap the backend for Redis (via redis-py / aioredis) later without touching
any API layer code — just replace JobStore._jobs with a Redis hash.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator
from urllib.parse import urlparse

from app.core.config import settings
from app.models.schemas import DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal job record
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_job_id() -> str:
    return str(uuid.uuid4())


def _extract_camera_id(stream_url: str) -> str:
    """
    Derive a stable camera identifier from a stream URL.

    Uses the last path segment when available; otherwise falls back to host.
    """
    parsed = urlparse(stream_url)
    path = parsed.path.strip("/")
    if path:
        leaf = path.split("/")[-1]
        if "." in leaf:
            leaf = leaf.rsplit(".", 1)[0]
        if leaf:
            return leaf
    return parsed.netloc or "unknown-camera"


def _event_detected_at(created_at: datetime, timestamp_sec: float) -> datetime:
    return created_at + timedelta(seconds=float(timestamp_sec))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class JobStore:
    """
    Thread-safe store for DetectionResult objects.

    Public API
    ----------
    store.create(stream_url)                  -> DetectionResult   (status=pending)
    store.get(job_id)                         -> DetectionResult | None
    store.update(job_id, **fields)            -> None
    store.count_running()                     -> int
    store.all_jobs()                          -> list[DetectionResult]
    """

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, DetectionResult] = {}
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.job_ttl_seconds
        self._start_reaper()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def create(self, stream_url: str, *, max_frames: int | None = None, save_frames: bool | None = None) -> DetectionResult:
        """Create a new job with status='pending' and return it."""
        job = DetectionResult(
            job_id=_new_job_id(),
            status="pending",
            stream_url=stream_url,
            max_frames=max_frames,
            save_frames=save_frames,
            created_at=_utcnow(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        logger.debug("Job created  job_id=%s  stream_url=%s", job.job_id, stream_url)
        return job

    def get(self, job_id: str) -> DetectionResult | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        """
        Update arbitrary fields on an existing job.

        Only scalar fields and lists are shallow-merged; nested Pydantic
        sub-models (e.g. events) should be passed in full.

        Example
        -------
        store.update(job_id, status="running", frames_processed=42)
        store.update(job_id, status="complete", completed_at=utcnow(), events=[...])
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("update() called for unknown job_id=%s", job_id)
                return
            updated = job.model_copy(update=fields)
            self._jobs[job_id] = updated

    def count_running(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "running")

    def all_jobs(self) -> list[DetectionResult]:
        with self._lock:
            return list(self._jobs.values())

    def count_by_status(self) -> dict[str, int]:
        """Return counts by job status for cheap dashboard aggregation."""
        counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
        with self._lock:
            for job in self._jobs.values():
                if job.status in counts:
                    counts[job.status] += 1
        return counts

    def camera_counts(self) -> tuple[int, int]:
        """Return (active_cameras, total_cameras) based on stream URLs."""
        with self._lock:
            jobs = list(self._jobs.values())
        active = {
            _extract_camera_id(job.stream_url)
            for job in jobs
            if job.status in ("running", "pending")
        }
        total = {_extract_camera_id(job.stream_url) for job in jobs}
        return len(active), len(total)

    def list_jobs_paginated(
        self,
        *,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> tuple[list[DetectionResult], int]:
        """
        Return filtered/sorted/paginated job results and total matching count.
        """
        with self._lock:
            jobs = list(self._jobs.values())

        filtered = []
        for job in jobs:
            if status and job.status != status:
                continue
            if start_time and job.created_at < start_time:
                continue
            if end_time and job.created_at > end_time:
                continue
            filtered.append(job)

        reverse = sort.startswith("-")
        key = sort.lstrip("-")
        key_fn = (
            (lambda j: j.completed_at or j.created_at)
            if key == "completed_at"
            else (lambda j: j.created_at)
        )
        filtered.sort(key=key_fn, reverse=reverse)

        total = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return filtered[start_idx:end_idx], total

    def list_events_paginated(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        camera_id: str | None = None,
        detection_type: str | None = None,
        min_confidence: float | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-detected_at",
    ) -> tuple[list[dict], int]:
        """
        Return flattened incident/event records with pagination.

        Record fields:
            incident_id, job, event, camera_id, status, detected_at, detection_type
        """
        with self._lock:
            jobs = list(self._jobs.values())

        flattened: list[dict] = []
        for job in jobs:
            if job_id and job.job_id != job_id:
                continue
            cam = _extract_camera_id(job.stream_url)
            if camera_id and cam != camera_id:
                continue
            incident_status = "active" if job.status in ("pending", "running") else "resolved"
            if status and incident_status != status:
                continue
            for event in job.events:
                dtype = "accident"
                if detection_type and detection_type != dtype:
                    continue
                if min_confidence is not None and event.confidence_score < min_confidence:
                    continue
                detected_at = _event_detected_at(job.created_at, event.timestamp_sec)
                if start_time and detected_at < start_time:
                    continue
                if end_time and detected_at > end_time:
                    continue
                flattened.append(
                    {
                        "incident_id": f"{job.job_id}:{event.frame_idx}",
                        "job": job,
                        "event": event,
                        "camera_id": cam,
                        "status": incident_status,
                        "detected_at": detected_at,
                        "detection_type": dtype,
                    }
                )

        reverse = sort.startswith("-")
        key = sort.lstrip("-")
        key_fn = (
            (lambda rec: rec["event"].confidence_score)
            if key == "confidence"
            else (lambda rec: rec["detected_at"])
        )
        flattened.sort(key=key_fn, reverse=reverse)

        total = len(flattened)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return flattened[start_idx:end_idx], total

    def get_incident(self, incident_id: str) -> dict | None:
        """Lookup an incident by its stable '<job_id>:<frame_idx>' identifier."""
        if ":" not in incident_id:
            return None
        raw_job_id, raw_frame = incident_id.split(":", 1)
        try:
            frame_idx = int(raw_frame)
        except ValueError:
            return None

        with self._lock:
            job = self._jobs.get(raw_job_id)
        if job is None:
            return None

        cam = _extract_camera_id(job.stream_url)
        incident_status = "active" if job.status in ("pending", "running") else "resolved"
        for event in job.events:
            if event.frame_idx != frame_idx:
                continue
            detected_at = _event_detected_at(job.created_at, event.timestamp_sec)
            return {
                "incident_id": incident_id,
                "job": job,
                "event": event,
                "camera_id": cam,
                "status": incident_status,
                "detected_at": detected_at,
                "detection_type": "accident",
            }
        return None

    def count_incidents_in_range(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Count incidents detected within the provided UTC range."""
        _, total = self.list_events_paginated(
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=1,
        )
        return total

    def __iter__(self) -> Iterator[DetectionResult]:
        return iter(self.all_jobs())

    # ------------------------------------------------------------------
    # TTL reaper
    # ------------------------------------------------------------------

    def _start_reaper(self) -> None:
        t = threading.Thread(target=self._reap_loop, daemon=True, name="job-reaper")
        t.start()

    def _reap_loop(self) -> None:
        while True:
            time.sleep(60)
            self._reap_expired()

    def _reap_expired(self) -> None:
        now = _utcnow().timestamp()
        with self._lock:
            expired = [
                jid
                for jid, job in self._jobs.items()
                if job.status in ("complete", "failed")
                and job.completed_at is not None
                and (now - job.completed_at.timestamp()) > self._ttl
            ]
            for jid in expired:
                del self._jobs[jid]
        if expired:
            logger.debug("Reaped %d expired job(s): %s", len(expired), expired)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the entire application process)
# ---------------------------------------------------------------------------

job_store = JobStore()
