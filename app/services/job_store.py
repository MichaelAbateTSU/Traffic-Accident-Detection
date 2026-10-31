"""
app/services/job_store.py — Thread-safe job store with optional durable SQLite.

Jobs are stored as DetectionResult objects keyed by job_id (UUID string).
A background reaper thread removes jobs that have exceeded their TTL.

Production persists jobs and operator changes; local development is in memory.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator
from urllib.parse import urlparse
from pathlib import Path

from app.core.config import settings
from app.models.schemas import DetectionResult
from app.services.submission_identity import (
    SubmissionIdentityError, identify_submission, request_fingerprint,
)

logger = logging.getLogger(__name__)

class JobCapacityError(RuntimeError):
    pass

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

    Common HLS manifest names use the parent path segment as the camera ID.
    Other paths use the filename without its extension, or fall back to host.
    """
    parsed = urlparse(stream_url)
    path = parsed.path.strip("/")
    if path:
        segments = [segment for segment in path.split("/") if segment]
        leaf = segments[-1]
        if len(segments) > 1 and leaf.lower() in {
            "playlist.m3u8", "index.m3u8", "master.m3u8",
        }:
            return segments[-2]
        if "." in leaf:
            leaf = leaf.rsplit(".", 1)[0]
        if leaf:
            return leaf
    return parsed.netloc or "unknown-camera"


def _event_detected_at(job: DetectionResult, timestamp_sec: float) -> datetime:
    # Old persisted jobs have no recoverable processing start; retain their estimate.
    return (job.started_at or job.created_at) + timedelta(seconds=float(timestamp_sec))


def _parse_incident_id(incident_id: str) -> tuple[str, int] | None:
    if ":" not in incident_id:
        return None
    raw_job_id, raw_frame = incident_id.split(":", 1)
    try:
        return raw_job_id, int(raw_frame)
    except ValueError:
        return None


def _build_incident_record(
    job: DetectionResult,
    event,
    camera_id: str,
    overrides: dict[str, dict],
) -> dict:
    """
    Flatten a job event into an incident record.

    Status defaults to 'active' while the source job is running and 'resolved'
    once it finishes; an operator override (PATCH /incidents/{id}) always wins.
    """
    incident_id = f"{job.job_id}:{event.frame_idx}"
    override = overrides.get(incident_id)
    if override is not None:
        status = override["status"]
        resolved_at = override["resolved_at"]
        note = override["note"]
    else:
        status = "active" if job.status in ("pending", "running") else "resolved"
        resolved_at = job.completed_at if status == "resolved" else None
        note = None
    return {
        "incident_id": incident_id,
        "job": job,
        "event": event,
        "camera_id": camera_id,
        "status": status,
        "detected_at": _event_detected_at(job, event.timestamp_sec),
        "detection_type": "accident",
        "resolved_at": resolved_at,
        "note": note,
        "manually_updated": override is not None,
    }


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

    def __init__(
        self, ttl_seconds: int | None = None, *,
        database_path: str | None = None, start_reaper: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, DetectionResult] = {}
        self._submission_receipts: dict[str, tuple[str, str, datetime]] = {}
        self._submission_clock = datetime.min.replace(tzinfo=timezone.utc)
        self._terminal_recovery: dict[str, tuple[str, datetime]] = {}
        # Operator overrides keyed by incident_id ("<job_id>:<frame_idx>").
        self._incident_overrides: dict[str, dict] = {}
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.job_ttl_seconds
        self._db: sqlite3.Connection | None = None
        if database_path:
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(database_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS overrides (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS submission_receipts "
                "(identity TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
                "job_id TEXT NOT NULL, expires_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS submission_clock "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), observed_at TEXT NOT NULL)"
            )
            clock_row = self._db.execute("SELECT observed_at FROM submission_clock WHERE id = 1").fetchone()
            if clock_row is not None:
                self._submission_clock = datetime.fromisoformat(clock_row[0])
            for identity, fingerprint, jid, expires_at in self._db.execute(
                "SELECT identity, fingerprint, job_id, expires_at FROM submission_receipts"
            ):
                self._submission_receipts[identity] = (
                    fingerprint, jid, datetime.fromisoformat(expires_at),
                )
            for jid, payload in self._db.execute("SELECT id, payload FROM jobs"):
                self._jobs[jid] = DetectionResult.model_validate_json(payload)
            for iid, payload in self._db.execute("SELECT id, payload FROM overrides"):
                override = json.loads(payload)
                for field in ("resolved_at", "updated_at"):
                    if override.get(field):
                        override[field] = datetime.fromisoformat(override[field])
                self._incident_overrides[iid] = override
            for job in self._jobs.values():
                if job.status in ("pending", "running"):
                    job.status = "failed"
                    job.completed_at = _utcnow()
                    job.error = "The service restarted before this job finished. Submit a new job."
                    self._persist_job(job)
            self._db.commit()
            self._reap_expired()
        if start_reaper:
            self._start_reaper()

    def _persist_job(self, job: DetectionResult) -> None:
        if self._db is not None:
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO jobs VALUES (?, ?)",
                    (job.job_id, job.model_dump_json()),
                )

    def _delete_job(self, jid: str) -> None:
        if self._db is not None:
            with self._db:
                self._db.execute("DELETE FROM jobs WHERE id = ?", (jid,))
                self._db.execute("DELETE FROM overrides WHERE id LIKE ?", (f"{jid}:%",))
        del self._jobs[jid]
        for iid in [key for key in self._incident_overrides if key.split(":", 1)[0] == jid]:
            del self._incident_overrides[iid]
        # Only store-generated UUID directories may be removed.
        uuid.UUID(jid)
        import config as pipeline_config
        folder = Path(settings.output_dir or pipeline_config.OUTPUT_DIR) / jid
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError:
                logger.exception("Could not remove expired job artifacts %s", jid)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def create(
        self, stream_url: str, *, max_frames: int | None = None,
        save_frames: bool | None = None, capacity: int | None = None,
    ) -> DetectionResult:
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
            self._make_room_locked(capacity)
            self._persist_job(job)
            self._jobs[job.job_id] = job
        logger.debug("Job created  job_id=%s", job.job_id)
        return job

    def _make_room_locked(self, capacity: int | None) -> None:
        active = sum(j.status in ("pending", "running") for j in self._jobs.values())
        if capacity is not None and active >= capacity:
            raise JobCapacityError("The running/pending job capacity is exhausted. Please retry later.")
        while len(self._jobs) >= settings.max_retained_jobs:
            terminal = [j for j in self._jobs.values() if j.status in ("complete", "failed")]
            if not terminal:
                raise JobCapacityError("The job store is at capacity. Please retry later.")
            self._delete_job(min(terminal, key=lambda j: j.completed_at or j.created_at).job_id)

    def _lookup_submission_locked(
        self, identity: str, fingerprint: str | None,
    ) -> DetectionResult | None:
        receipt = self._submission_receipts.get(identity)
        if receipt is None:
            return None
        previous_fingerprint, jid, _ = receipt
        if fingerprint is not None and previous_fingerprint != fingerprint:
            raise SubmissionIdentityError(409, "idempotency_key_conflict",
                                          "This identity was used with different detection parameters.")
        job = self._jobs.get(jid)
        if job is None:
            raise SubmissionIdentityError(410, "idempotency_result_expired",
                                          "The original job result was removed; this identity cannot schedule work.")
        return job

    def find_submission(self, key: str) -> tuple[DetectionResult | None, datetime]:
        """Read-only identity reconciliation; absence is not an admission fence."""
        with self._lock:
            self._require_submission_storage()
            identity, expires_at = self._identify_submission_locked(key, _utcnow())
            return self._lookup_submission_locked(identity, None), expires_at

    def lookup_submission(
        self, key: str, stream_url: str, max_frames: int, save_frames: bool,
    ) -> tuple[DetectionResult | None, datetime]:
        with self._lock:
            self._require_submission_storage()
            identity, expires_at = self._identify_submission_locked(key, _utcnow())
            fingerprint = request_fingerprint(stream_url, max_frames, save_frames)
            return self._lookup_submission_locked(identity, fingerprint), expires_at

    def create_submission(
        self, key: str, stream_url: str, *, max_frames: int, save_frames: bool,
        capacity: int,
    ) -> tuple[DetectionResult, bool, datetime]:
        """Reserve the identity and job atomically; only the winner may enqueue."""
        with self._lock:
            self._require_submission_storage()
            now = _utcnow()
            identity, expires_at = self._identify_submission_locked(key, now)
            fingerprint = request_fingerprint(stream_url, max_frames, save_frames)
            existing = self._lookup_submission_locked(identity, fingerprint)
            if existing is not None:
                return existing, True, expires_at
            self._reap_receipts_locked(now)
            if len(self._submission_receipts) >= settings.max_submission_receipts:
                raise JobCapacityError("The submission receipt store is full. Please retry later.")
            self._make_room_locked(capacity)
            job = DetectionResult(
                job_id=_new_job_id(), status="pending", stream_url=stream_url,
                max_frames=max_frames, save_frames=save_frames, created_at=now,
            )
            if self._db is not None:
                # Do not use _persist_job here: its commit would split this reservation.
                with self._db:
                    self._db.execute("INSERT INTO jobs VALUES (?, ?)", (job.job_id, job.model_dump_json()))
                    self._db.execute(
                        "INSERT INTO submission_receipts VALUES (?, ?, ?, ?)",
                        (identity, fingerprint, job.job_id, expires_at.isoformat()),
                    )
            self._jobs[job.job_id] = job
            self._submission_receipts[identity] = (fingerprint, job.job_id, expires_at)
            return job, False, expires_at

    def _identify_submission_locked(self, key: str, now: datetime) -> tuple[str, datetime]:
        now = max(now, self._submission_clock)
        try:
            return identify_submission(key, now)
        except SubmissionIdentityError as exc:
            if exc.code == "idempotency_key_expired" and now > self._submission_clock:
                # A definitive 410 must remain true after rollback, even when
                # the identity was never reserved and the reaper has not run.
                assert self._db is not None
                with self._db:
                    self._db.execute(
                        "INSERT OR REPLACE INTO submission_clock VALUES (1, ?)", (now.isoformat(),),
                    )
                self._submission_clock = now
            raise

    def _require_submission_storage(self) -> None:
        if self._db is None or not any(
            name == "main" and filename
            for _, name, filename in self._db.execute("PRAGMA database_list")
        ):
            raise SubmissionIdentityError(
                503, "idempotency_unavailable",
                "Keyed submissions require a file-backed JOB_DATABASE_PATH on persistent storage; "
                "in-memory and temporary SQLite databases are unsupported.",
            )

    def _reap_receipts_locked(self, now: datetime) -> None:
        now = max(now, self._submission_clock)
        expired = [identity for identity, (_, _, expiry) in self._submission_receipts.items()
                   if now >= expiry]
        if self._db is not None and (expired or now > self._submission_clock):
            with self._db:
                # The clock fence and deletion must commit together: old keys cannot
                # become valid again after GC even if wall time goes backwards.
                self._db.execute(
                    "INSERT OR REPLACE INTO submission_clock VALUES (1, ?)", (now.isoformat(),),
                )
                self._db.executemany("DELETE FROM submission_receipts WHERE identity = ?",
                                     [(identity,) for identity in expired])
        self._submission_clock = now
        for identity in expired:
            del self._submission_receipts[identity]

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
            self._persist_job(updated)
            self._jobs[job_id] = updated

    def count_running(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "running")

    def fail_job(self, job_id: str, error: str) -> bool:
        """Record a known stopped/rejected job; never infer failure from age."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ("complete", "failed"):
                self._terminal_recovery.pop(job_id, None)
                return True
            # At most one intent per retained nonterminal job, bounded by admission.
            self._terminal_recovery.setdefault(job_id, (error, _utcnow()))
            return self._recover_terminal_locked(job_id)

    def has_terminal_recovery(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._terminal_recovery

    def _recover_terminal_locked(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in ("complete", "failed"):
            self._terminal_recovery.pop(job_id, None)
            return True
        error, completed_at = self._terminal_recovery[job_id]
        updated = job.model_copy(update={
            "status": "failed", "completed_at": completed_at, "error": error,
        })
        try:
            self._persist_job(updated)
        except sqlite3.OperationalError:
            logger.exception(
                "SQLite terminal recovery pending for job %s; retrying at the next 60-second cycle.",
                job_id,
            )
            return False
        self._jobs[job_id] = updated
        del self._terminal_recovery[job_id]
        return True

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
        """Return (active_cameras, total_cameras): distinct source IDs among pending/running vs all
        retained jobs (video URLs included). Not a configured-camera registry."""
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
        page_size: int | None = 20,
        sort: str = "-detected_at",
    ) -> tuple[list[dict], int]:
        """
        Return one filtered/sorted snapshot; page_size=None disables pagination.

        Record fields:
            incident_id, job, event, camera_id, status, detected_at, detection_type
        """
        with self._lock:
            jobs = list(self._jobs.values())
            overrides = dict(self._incident_overrides)

        flattened: list[dict] = []
        for job in jobs:
            if job_id and job.job_id != job_id:
                continue
            cam = _extract_camera_id(job.stream_url)
            if camera_id and cam != camera_id:
                continue
            for event in job.events:
                dtype = "accident"
                if detection_type and detection_type != dtype:
                    continue
                if min_confidence is not None and event.confidence_score < min_confidence:
                    continue
                record = _build_incident_record(job, event, cam, overrides)
                if status and record["status"] != status:
                    continue
                detected_at = record["detected_at"]
                if start_time and detected_at < start_time:
                    continue
                if end_time and detected_at > end_time:
                    continue
                flattened.append(record)

        reverse = sort.startswith("-")
        key = sort.lstrip("-")
        key_fn = (
            (lambda rec: rec["event"].confidence_score)
            if key == "confidence"
            else (lambda rec: rec["detected_at"])
        )
        flattened.sort(key=key_fn, reverse=reverse)

        total = len(flattened)
        if page_size is None:
            return flattened, total
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return flattened[start_idx:end_idx], total

    def get_incident(self, incident_id: str) -> dict | None:
        """Lookup an incident by its stable '<job_id>:<frame_idx>' identifier."""
        parsed = _parse_incident_id(incident_id)
        if parsed is None:
            return None
        raw_job_id, frame_idx = parsed

        with self._lock:
            job = self._jobs.get(raw_job_id)
            overrides = dict(self._incident_overrides)
        if job is None:
            return None

        cam = _extract_camera_id(job.stream_url)
        for event in job.events:
            if event.frame_idx == frame_idx:
                return _build_incident_record(job, event, cam, overrides)
        return None

    def set_incident_status(
        self,
        incident_id: str,
        *,
        status: str,
        note: str | None = None,
    ) -> dict | None:
        """
        Record an operator status change (and optional note) for an incident.

        Returns the updated incident record, or None if the incident is unknown.
        Passing note=None keeps any previously stored note.
        Resolution time is preserved until an active incident is resolved again.
        """
        parsed = _parse_incident_id(incident_id)
        if parsed is None:
            return None
        raw_job_id, frame_idx = parsed
        with self._lock:
            job = self._jobs.get(raw_job_id)
            if job is None:
                return None
            event = next((event for event in job.events if event.frame_idx == frame_idx), None)
            if event is None:
                return None
            cam = _extract_camera_id(job.stream_url)
            prev = _build_incident_record(job, event, cam, self._incident_overrides)
            now = _utcnow()
            resolved_at = None
            if status == "resolved":
                resolved_at = prev["resolved_at"] if prev["status"] == "resolved" else now
            override = {
                "status": status,
                "note": note if note is not None else prev["note"],
                "resolved_at": resolved_at,
                "updated_at": now,
            }
            if self._db is not None:
                with self._db:
                    self._db.execute(
                        "INSERT OR REPLACE INTO overrides VALUES (?, ?)",
                        (incident_id, json.dumps(override, default=lambda value: value.isoformat())),
                    )
            self._incident_overrides[incident_id] = override
            return _build_incident_record(job, event, cam, self._incident_overrides)

    def latest_frame_for_camera(self, camera_id: str) -> tuple[DetectionResult, object] | None:
        """
        Return (job, FrameImages) for the newest processed frame of a camera.

        Considers jobs newest-first and returns the highest-index frame of the
        first job that has any frames.
        """
        with self._lock:
            jobs = [
                j for j in self._jobs.values()
                if _extract_camera_id(j.stream_url) == camera_id and j.frames
            ]
        if not jobs:
            return None
        job = max(jobs, key=lambda j: j.created_at)
        frame = max(job.frames, key=lambda f: f.frame_index)
        return job, frame

    def count_incidents_in_range(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Count retained alarm-active frame records in the UTC range, not episodes."""
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
            try:
                self._reap_expired()
            except sqlite3.OperationalError:
                logger.exception("SQLite periodic cleanup failed; retrying at the next 60-second cycle.")

    def _reap_expired(self) -> None:
        now = _utcnow().timestamp()
        with self._lock:
            for job_id in list(self._terminal_recovery):
                self._recover_terminal_locked(job_id)
            self._reap_receipts_locked(datetime.fromtimestamp(now, tz=timezone.utc))
            expired = [
                jid
                for jid, job in self._jobs.items()
                if job.status in ("complete", "failed")
                and job.completed_at is not None
                and (now - job.completed_at.timestamp()) > self._ttl
            ]
            for jid in expired:
                self._delete_job(jid)
        if expired:
            logger.debug("Reaped %d expired job(s): %s", len(expired), expired)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the entire application process)
# ---------------------------------------------------------------------------

job_store = JobStore(database_path=settings.job_database_path)
