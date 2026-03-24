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
from datetime import datetime, timezone
from typing import Iterator

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

    def create(self, stream_url: str) -> DetectionResult:
        """Create a new job with status='pending' and return it."""
        job = DetectionResult(
            job_id=_new_job_id(),
            status="pending",
            stream_url=stream_url,
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
