"""Durable receipt atomicity, bounded retention, and unsafe reuse rejection."""

import sqlite3
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.services.job_store import JobCapacityError, JobStore
from app.services.submission_identity import SubmissionIdentityError, identify_submission

SOURCE = "https://camera.example.com/playlist.m3u8"


def key_at(now):
    return f"v1.{int(now.timestamp())}.{uuid.uuid4()}"


@pytest.fixture
def store(tmp_path):
    instance = JobStore(database_path=str(tmp_path / "jobs.sqlite3"), start_reaper=False)
    yield instance
    instance._db.close()


def create(store, key):
    return store.create_submission(key, SOURCE, max_frames=5, save_frames=False, capacity=10)


def test_atomic_concurrent_reservation_checks_replay_before_capacity(store):
    key = key_at(datetime.now(timezone.utc))
    def attempt(_):
        return store.create_submission(key, SOURCE, max_frames=5, save_frames=False, capacity=1)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(40)))
    assert sum(not replayed for _, replayed, _ in results) == 1
    assert len({job.job_id for job, _, _ in results}) == 1
    assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 1


def test_receipt_insert_failure_rolls_back_job_and_allows_same_identity_retry(store):
    store._db.execute(
        "CREATE TRIGGER reject_receipt BEFORE INSERT ON submission_receipts "
        "BEGIN SELECT RAISE(ABORT, 'receipt write failed'); END"
    )
    store._db.commit()
    key = key_at(datetime.now(timezone.utc))
    with pytest.raises(sqlite3.IntegrityError, match="receipt write failed"):
        create(store, key)
    assert not store.all_jobs() and not store._submission_receipts
    assert store._db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    store._db.execute("DROP TRIGGER reject_receipt")
    store._db.commit()
    assert create(store, key)[1] is False


def test_restart_between_durable_reservation_and_enqueue_never_replays_work(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    key = key_at(datetime.now(timezone.utc))
    job, _, expiry = create(store, key)
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    original, replayed, recovered_expiry = create(restarted, key)
    assert original.job_id == job.job_id and replayed
    assert recovered_expiry == expiry and original.status == "failed"
    assert "restarted" in original.error
    assert len(restarted.all_jobs()) == 1
    restarted._db.close()


@pytest.mark.parametrize("eviction", ["capacity", "ttl"])
def test_removed_job_keeps_durable_receipt_and_cannot_be_recreated(store, monkeypatch, eviction):
    now = datetime.now(timezone.utc)
    key = key_at(now)
    job, _, _ = create(store, key)
    store.update(job.job_id, status="complete", completed_at=now - timedelta(hours=2))
    if eviction == "capacity":
        monkeypatch.setattr(settings, "max_retained_jobs", 1)
        store.create(SOURCE)
    else:
        store._reap_expired()
    assert store.get(job.job_id) is None and len(store._submission_receipts) == 1
    with pytest.raises(SubmissionIdentityError) as error:
        create(store, key)
    assert error.value.status_code == 410 and error.value.code == "idempotency_result_expired"
    with pytest.raises(SubmissionIdentityError) as error:
        store.create_submission(key, SOURCE, max_frames=6, save_frames=False, capacity=10)
    assert error.value.status_code == 409
    assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 1


def test_receipts_are_bounded_and_expired_identity_never_becomes_new(store, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    monkeypatch.setattr(settings, "max_submission_receipts", 1)
    key = key_at(now)
    original, _, expiry = create(store, key)
    next_key = key_at(now)
    with pytest.raises(JobCapacityError, match="receipt store"):
        create(store, next_key)
    assert create(store, key)[0].job_id == original.job_id
    now = expiry
    store._reap_expired()
    assert not store._submission_receipts
    assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 0
    with pytest.raises(SubmissionIdentityError) as error:
        create(store, key)
    assert error.value.status_code == 410 and error.value.code == "idempotency_key_expired"
    assert create(store, key_at(now))[1] is False


@pytest.mark.parametrize("key", [
    "", "arbitrary-opaque-key", f"v1.{int(time.time())}.{uuid.uuid1()}",
    f"v1.{int(time.time())}.{str(uuid.uuid4()).upper()}",
    f"v1.{int(time.time())}.{'-' * 36}",
])
def test_invalid_identities_are_rejected(key):
    with pytest.raises(SubmissionIdentityError) as error:
        identify_submission(key, datetime.now(timezone.utc))
    assert error.value.status_code == 422


def test_clock_skew_boundary_and_key_expiry():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    assert identify_submission(key_at(now + timedelta(seconds=60)), now)
    with pytest.raises(SubmissionIdentityError) as error:
        identify_submission(key_at(now + timedelta(seconds=61)), now)
    assert error.value.code == "idempotency_key_in_future"
    with pytest.raises(SubmissionIdentityError) as error:
        identify_submission(key_at(now - timedelta(seconds=86400)), now)
    assert error.value.code == "idempotency_key_expired"


def test_in_memory_mode_fails_closed_for_keyed_clients():
    store = JobStore(start_reaper=False)
    key = key_at(datetime.now(timezone.utc))
    with pytest.raises(SubmissionIdentityError) as error:
        create(store, key)
    assert error.value.status_code == 503 and error.value.code == "idempotency_unavailable"
    assert not store.all_jobs()
    assert store.create(SOURCE).status == "pending"


@pytest.mark.parametrize("path", [
    ":memory:", "file::memory:", "file::memory:?cache=shared",
    "file:submission-test?mode=memory&cache=shared", "",
])
def test_volatile_sqlite_rejects_all_keyed_operations_before_and_after_recreation(path):
    key = key_at(datetime.now(timezone.utc))
    for _ in range(2):
        store = JobStore(database_path=path, start_reaper=False)
        try:
            if store._db is not None:
                assert store._db.execute("PRAGMA database_list").fetchone()[2] == ""
            for attempt in (
                lambda: create(store, key),
                lambda: store.lookup_submission(key, SOURCE, 5, False),
                lambda: store.find_submission(key),
            ):
                with pytest.raises(SubmissionIdentityError) as error:
                    attempt()
                assert error.value.status_code == 503
                assert error.value.code == "idempotency_unavailable"
                assert not store.all_jobs() and not store._submission_receipts
            assert store.create(SOURCE).status == "pending"
        finally:
            if store._db is not None:
                store._db.close()


def test_sqlite_temporary_database_connection_rejects_keyed_operations():
    # An empty path normally selects JobStore's keyless in-memory mode.
    # Exercise SQLite's separate temporary main database behavior as well.
    store = JobStore(start_reaper=False)
    store._db = sqlite3.connect("")
    try:
        assert store._db.execute("PRAGMA database_list").fetchone()[2] == ""
        key = key_at(datetime.now(timezone.utc))
        for attempt in (
            lambda: create(store, key),
            lambda: store.lookup_submission(key, SOURCE, 5, False),
            lambda: store.find_submission(key),
        ):
            with pytest.raises(SubmissionIdentityError) as error:
                attempt()
            assert error.value.code == "idempotency_unavailable"
        assert not store.all_jobs() and not store._submission_receipts
    finally:
        store._db.close()


def test_receipts_store_hashes_not_raw_keys_or_camera_urls(store):
    key = key_at(datetime.now(timezone.utc))
    create(store, key)
    rows = store._db.execute("SELECT * FROM submission_receipts").fetchall()
    assert key not in str(rows) and SOURCE not in str(rows)
    assert len(rows[0][0]) == len(rows[0][1]) == 64


def test_abrupt_process_exit_after_commit_recovers_original_failed_job(tmp_path):
    path = str(tmp_path / "crashed.sqlite3")
    key = key_at(datetime.now(timezone.utc))
    code = """
import os, sys
from app.services.job_store import JobStore
store = JobStore(database_path=sys.argv[1], start_reaper=False)
store.create_submission(sys.argv[2], sys.argv[3], max_frames=5, save_frames=False, capacity=10)
os._exit(19)
"""
    child = subprocess.run([sys.executable, "-c", code, path, key, SOURCE], capture_output=True)
    assert child.returncode == 19, child.stderr.decode()
    restarted = JobStore(database_path=path, start_reaper=False)
    original, replayed, _ = create(restarted, key)
    assert replayed and original.status == "failed" and "restarted" in original.error
    assert len(restarted.all_jobs()) == 1
    restarted._db.close()


def test_legacy_database_migration_preserves_jobs_and_adds_no_retroactive_keys(tmp_path):
    from app.models.schemas import DetectionResult
    path = str(tmp_path / "legacy.sqlite3")
    job = DetectionResult(
        job_id=str(uuid.uuid4()), status="complete", stream_url=SOURCE,
        created_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        frames_processed=5,
    )
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE overrides (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO jobs VALUES (?, ?)", (job.job_id, job.model_dump_json()))
    connection.close()
    migrated = JobStore(database_path=path, start_reaper=False)
    assert migrated.get(job.job_id) == job and not migrated._submission_receipts
    assert migrated._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 0
    assert create(migrated, key_at(datetime.now(timezone.utc)))[1] is False
    assert migrated.get(job.job_id).frames_processed == 5
    migrated._db.close()


def test_evicted_result_receipt_survives_restart(tmp_path):
    path = str(tmp_path / "evicted.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    key = key_at(datetime.now(timezone.utc))
    job, _, _ = create(store, key)
    store.update(job.job_id, status="complete", completed_at=datetime.now(timezone.utc))
    with store._lock:
        store._delete_job(job.job_id)
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    with pytest.raises(SubmissionIdentityError) as error:
        create(restarted, key)
    assert error.value.code == "idempotency_result_expired" and not restarted.all_jobs()
    restarted._db.close()


def test_expiry_cleanup_clock_rollback_and_restart_never_reuses_identity(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    path = str(tmp_path / "clock-fenced.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    key = key_at(now)
    original, _, expiry = create(store, key)
    now = expiry
    store._reap_expired()
    assert not store._submission_receipts
    assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 0
    observed_at = store._db.execute("SELECT observed_at FROM submission_clock").fetchone()[0]
    assert datetime.fromisoformat(observed_at) == expiry
    now -= timedelta(hours=12)
    with pytest.raises(SubmissionIdentityError) as error:
        create(store, key)
    assert error.value.code == "idempotency_key_expired"
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    assert restarted._submission_clock == expiry
    with pytest.raises(SubmissionIdentityError) as error:
        create(restarted, key)
    assert error.value.code == "idempotency_key_expired"
    with pytest.raises(SubmissionIdentityError) as error:
        restarted.find_submission(key)
    assert error.value.code == "idempotency_key_expired"
    assert [job.job_id for job in restarted.all_jobs()] == [original.job_id]
    restarted._db.close()


def test_receipt_cleanup_and_clock_fence_roll_back_together(store, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    key = key_at(now)
    _, _, expiry = create(store, key)
    previous_clock = store._submission_clock
    store._db.execute(
        "CREATE TRIGGER reject_cleanup BEFORE DELETE ON submission_receipts "
        "BEGIN SELECT RAISE(ABORT, 'cleanup failed'); END"
    )
    store._db.commit()
    now = expiry
    with pytest.raises(sqlite3.IntegrityError, match="cleanup failed"):
        store._reap_expired()
    assert len(store._submission_receipts) == 1 and store._submission_clock == previous_clock
    assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 1
    observed_at = store._db.execute("SELECT observed_at FROM submission_clock").fetchone()[0]
    assert datetime.fromisoformat(observed_at) == previous_clock


@pytest.mark.parametrize("operation", ["find", "lookup", "create"])
def test_unreserved_expiry_observation_fences_rollback_before_cleanup_and_restart(tmp_path, monkeypatch, operation):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    path = str(tmp_path / "expiry-observation.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    key = key_at(now - timedelta(hours=12))
    now += timedelta(hours=12, seconds=1)
    observation = now
    attempt = {
        "find": lambda: store.find_submission(key),
        "lookup": lambda: store.lookup_submission(key, SOURCE, 5, False),
        "create": lambda: create(store, key),
    }[operation]
    with pytest.raises(SubmissionIdentityError) as error:
        attempt()
    assert error.value.code == "idempotency_key_expired"
    # No reaper/admission occurred after expiry; no reservation ever existed.
    now -= timedelta(hours=12)
    with pytest.raises(SubmissionIdentityError) as error:
        create(store, key)
    assert error.value.code == "idempotency_key_expired"
    assert store._submission_clock == observation
    assert not store.all_jobs() and not store._submission_receipts
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    with pytest.raises(SubmissionIdentityError) as error:
        create(restarted, key)
    assert error.value.code == "idempotency_key_expired"
    assert restarted._submission_clock == observation and not restarted.all_jobs()
    restarted._db.close()


@pytest.mark.parametrize("keyed", [False, True])
def test_actual_reaper_loop_retries_transient_sqlite_failure_on_next_cycle(store, monkeypatch, tmp_path, caplog, keyed):
    import logging

    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "artifacts"))
    job = create(store, key_at(now))[0] if keyed else store.create(SOURCE)
    store.update(job.job_id, status="complete", completed_at=now)
    artifacts = tmp_path / "artifacts" / job.job_id
    artifacts.mkdir(parents=True)
    (artifacts / "frame.jpg").write_bytes(b"test")
    previous_clock = store._submission_clock
    now += timedelta(days=2)
    database = store._db.execute("PRAGMA database_list").fetchone()[2]
    store._db.execute("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(database)
    blocker.execute("BEGIN IMMEDIATE")
    waits = []

    class StopTestLoop(Exception):
        pass

    def scheduled_wait(seconds):
        waits.append(seconds)
        assert seconds == 60, "Retry must wait for the next scheduled cycle"
        if len(waits) == 2:
            assert store.get(job.job_id) is not None and artifacts.exists()
            assert store._submission_clock == previous_clock
            assert len(store._submission_receipts) == int(keyed)
            assert store._db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
            assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == int(keyed)
            assert any("SQLite" in record.message and record.exc_info for record in caplog.records)
            blocker.rollback()
        elif len(waits) == 3:
            assert not store.all_jobs() and not artifacts.exists()
            assert not store._submission_receipts and store._submission_clock == now
            assert store._db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
            assert store._db.execute("SELECT count(*) FROM submission_receipts").fetchone()[0] == 0
            raise StopTestLoop()

    monkeypatch.setattr("app.services.job_store.time.sleep", scheduled_wait)
    try:
        with caplog.at_level(logging.ERROR, logger="app.services.job_store"):
            with pytest.raises(StopTestLoop):
                store._reap_loop()
        assert waits == [60, 60, 60]
    finally:
        blocker.rollback()
        blocker.close()


def test_reaper_does_not_hide_sqlite_programming_errors(store, monkeypatch):
    monkeypatch.setattr("app.services.job_store.time.sleep", lambda seconds: None)
    def fail():
        raise sqlite3.ProgrammingError("Invalid connection usage")
    monkeypatch.setattr(store, "_reap_expired", fail)
    with pytest.raises(sqlite3.ProgrammingError, match="Invalid connection usage"):
        store._reap_loop()


def test_terminal_recovery_retries_on_normal_cycles_preserves_active_jobs_and_restart(store, monkeypatch, tmp_path, caplog):
    job, _, _ = create(store, key_at(datetime.now(timezone.utc)))
    active = store.create(SOURCE)
    store.update(active.job_id, status="running")
    path = store._db.execute("PRAGMA database_list").fetchone()[2]
    store._db.execute("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    assert store.fail_job(job.job_id, "Admission rejected") is False
    assert store.fail_job(job.job_id, "Secondary callback error") is False
    assert len(store._terminal_recovery) == 1
    waits = []
    class StopLoop(Exception):
        pass
    def scheduled_wait(seconds):
        assert seconds == 60
        waits.append(seconds)
        if len(waits) == 2:
            assert store.get(job.job_id).status == "pending"
            assert store.get(active.job_id).status == "running"
            assert len(store._terminal_recovery) == 1
            blocker.rollback()
        elif len(waits) == 3:
            assert store.get(job.job_id).status == "failed"
            assert store.get(job.job_id).error == "Admission rejected"
            assert store.get(active.job_id).status == "running"
            assert not store._terminal_recovery
            raise StopLoop()
    monkeypatch.setattr("app.services.job_store.time.sleep", scheduled_wait)
    try:
        with pytest.raises(StopLoop):
            store._reap_loop()
        assert waits == [60, 60, 60]
        assert any("terminal recovery pending" in record.message and record.exc_info
                   for record in caplog.records)
    finally:
        blocker.rollback()
        blocker.close()
    snapshot_path = str(tmp_path / "terminal-restart.sqlite3")
    snapshot = sqlite3.connect(snapshot_path)
    store._db.backup(snapshot)
    snapshot.close()
    restarted = JobStore(database_path=snapshot_path, start_reaper=False)
    assert restarted.get(job.job_id).status == "failed"
    assert restarted.get(job.job_id).error == "Admission rejected"
    assert not restarted._terminal_recovery
    restarted._db.close()


def test_pending_terminal_recovery_after_crash_uses_existing_restart_policy(store, tmp_path):
    key = key_at(datetime.now(timezone.utc))
    job, _, _ = create(store, key)
    path = store._db.execute("PRAGMA database_list").fetchone()[2]
    store._db.execute("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        assert store.fail_job(job.job_id, "Unable to schedule") is False
        snapshot_path = str(tmp_path / "pending-recovery.sqlite3")
        snapshot = sqlite3.connect(snapshot_path)
        store._db.backup(snapshot)
        snapshot.close()
    finally:
        blocker.rollback()
        blocker.close()
    restarted = JobStore(database_path=snapshot_path, start_reaper=False)
    original, replayed, _ = create(restarted, key)
    assert replayed and original.job_id == job.job_id and original.status == "failed"
    assert "restarted" in original.error and not restarted._terminal_recovery
    restarted._db.close()


def test_terminal_recovery_does_not_overwrite_already_completed_result(store):
    job = store.create(SOURCE)
    store._terminal_recovery[job.job_id] = ("Deferred error", datetime.now(timezone.utc))
    store.update(job.job_id, status="complete", completed_at=datetime.now(timezone.utc))
    store._reap_expired()
    assert store.get(job.job_id).status == "complete" and not store._terminal_recovery
