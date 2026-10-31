"""Bounded admission lives in JobStore; this executor runs accepted jobs in order."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


class DetectionQueue:
    def __init__(self, store: JobStore, worker: Callable[..., None], max_workers: int) -> None:
        self._store = store
        self._worker = worker
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="inference")
        self._lock = threading.RLock()
        self._closed = False
        self._futures: dict[str, asyncio.Future] = {}

    def submit(self, job_id: str, *args) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("The inference queue is shutting down")
            future = asyncio.get_running_loop().run_in_executor(
                self._executor, self._run, job_id, *args,
            )
            self._futures[job_id] = future
            future.add_done_callback(lambda completed: self._on_done(job_id, completed))

    def _fail(self, job_id: str, error: str) -> None:
        job = self._store.get(job_id)
        if job is not None and job.status in ("pending", "running"):
            logger.error("Inference queue failed job %s: %s", job_id, error)
            self._store.fail_job(job_id, error)

    def _run(self, job_id: str, *args) -> None:
        try:
            with self._lock:
                if self._closed:
                    self._fail(job_id, "The service shut down before this queued job started.")
                    return
                job = self._store.get(job_id)
                if job is None or job.status != "pending" or self._store.has_terminal_recovery(job_id):
                    logger.warning("Skipping unavailable or terminal queued job %s", job_id)
                    return
                self._store.update(job_id, status="running")
            self._worker(job_id, *args)
            self._fail(job_id, "The inference worker exited without recording a terminal result.")
        except Exception as exc:
            logger.exception("Inference executor failed job %s", job_id)
            self._fail(job_id, f"Inference worker failed: {exc!r}")

    def _on_done(self, job_id: str, future: asyncio.Future) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
        if future.cancelled():
            with self._lock:
                job = self._store.get(job_id)
                if job is not None and job.status == "pending":
                    self._fail(job_id, "The queued inference task was cancelled before completion.")
                elif job is not None and job.status == "running":
                    logger.warning("Cancelled future for active job %s; its worker still owns the outcome.", job_id)
        else:
            error = future.exception()
            if error is not None:
                logger.error("Inference future failed for job %s", job_id, exc_info=error)
                self._fail(job_id, f"Inference executor failed: {error!r}")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for job_id in self._futures:
                job = self._store.get(job_id)
                if job is not None and job.status == "pending":
                    self._fail(job_id, "The service shut down before this queued job started.")
        self._executor.shutdown(wait=False, cancel_futures=True)
