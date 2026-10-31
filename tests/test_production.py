"""Production boundaries, restart persistence and bounded admission."""

import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.services.job_store import JobCapacityError, JobStore, job_store
from deploy import egress_proxy

TOKEN = "test-operator-token-longer-than-thirty-two-characters"
SOURCE = "https://camera.example.com/live/playlist.m3u8"


def production_settings(**overrides):
    values = {
        "environment": "production", "api_token": TOKEN,
        "cors_origins": "https://michaelabatetsu.github.io",
        "job_database_path": "/data/jobs.sqlite3", "output_dir": "/data/artifacts",
        "approved_stream_urls": [SOURCE],
    }
    return Settings(_env_file=None, **(values | overrides))


@pytest.mark.parametrize("overrides", [
    {"api_token": ""},
    {"cors_origins": "*"},
    {"cors_origins": "http://localhost:5173"},
    {"cors_origins": "https://example.com/path"},
    {"cors_origins": ""},
    {"job_database_path": None},
    {"output_dir": None},
    {"max_concurrent_jobs": 2},
    {"job_artifact_retention_count": 8},
    {"max_retained_jobs": 9},
    {"approved_stream_urls": ["file:///etc/passwd"]},
    {"approved_stream_urls": ["https://user:secret@example.com/live"]},
    {"environment": "typo"},
])
def test_production_configuration_fails_closed(overrides):
    with pytest.raises(ValidationError):
        production_settings(**overrides)


@pytest.mark.parametrize("method,path", [
    ("get", "/jobs"), ("get", "/dashboard"), ("get", "/stats/summary"),
    ("get", "/incidents"), ("get", "/cameras"),
    ("get", "/artifacts/unknown/capture/000000.jpg"),
    ("post", "/detect-accident"), ("patch", "/incidents/unknown"),
    ("get", "/docs"), ("get", "/openapi.json"),
])
def test_every_non_health_surface_requires_auth(client, monkeypatch, method, path):
    monkeypatch.setattr(settings, "api_token", production_settings().api_token)
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/health").status_code == 200
    response = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert client.get("/jobs", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_production_cors_includes_auth_and_csv_headers(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app

    monkeypatch.setattr(settings, "cors_origins", "https://michaelabatetsu.github.io")
    monkeypatch.setattr(settings, "api_token", production_settings().api_token)
    client = TestClient(create_app())
    for method in ("GET", "POST", "PATCH"):
        response = client.options("/detect-accident", headers={
            "Origin": "https://michaelabatetsu.github.io",
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://michaelabatetsu.github.io"
    rejected = client.options("/detect-accident", headers={
        "Origin": "https://attacker.example",
        "Access-Control-Request-Method": "POST",
    })
    assert rejected.status_code == 400
    denied = client.get("/jobs", headers={"Origin": "https://michaelabatetsu.github.io"})
    assert denied.status_code == 401
    assert denied.headers["access-control-expose-headers"] == "Content-Disposition"


def test_production_sources_are_exactly_approved(client, monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "approved_stream_urls", [SOURCE])
    monkeypatch.setattr(settings, "max_concurrent_jobs", 1)
    monkeypatch.setattr(settings, "max_pending_jobs", 0)
    monkeypatch.setattr(client.app.state, "model", object())
    loop = client.portal.call(asyncio.get_running_loop)
    monkeypatch.setattr(loop, "run_in_executor", Mock())
    for source in (
        "https://127.0.0.1/private", "https://169.254.169.254/opc/v2/instance/",
        SOURCE + "?different=1", "https://camera.example.com/other.mp4",
    ):
        assert client.post("/detect-accident", json={"stream_url": source}).status_code == 403
    assert client.post("/detect-accident", json={"stream_url": SOURCE}).status_code == 202
    # Pending jobs count against capacity before the worker starts.
    assert client.post("/detect-accident", json={"stream_url": SOURCE}).status_code == 503


def test_development_default_is_preserved_and_production_is_single_worker():
    assert Settings(_env_file=None).max_concurrent_jobs == 10
    configured = production_settings()
    assert configured.max_concurrent_jobs == 1
    assert configured.max_pending_jobs == 9
    assert configured.get_job_admission_capacity() == 10


def test_atomic_admission_under_concurrent_calls():
    store = JobStore(start_reaper=False)
    def attempt(_):
        try:
            return store.create(SOURCE, capacity=1).job_id
        except JobCapacityError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(attempt, range(20)))
    assert sum(jid is not None for jid in accepted) == 1


def test_sqlite_restart_recovers_jobs_and_operator_changes(tmp_path):
    from app.models.schemas import DetectionEvent, SignalValues

    database = str(tmp_path / "jobs.sqlite3")
    store = JobStore(database_path=database, start_reaper=False)
    job = store.create(SOURCE, max_frames=5, save_frames=True)
    event = DetectionEvent(
        frame_idx=1, timestamp_sec=1, accident_detected=True,
        confidence_score=0.8, raw_score=0.8,
        signal_values=SignalValues(
            sudden_stop=0.5, abrupt_decel=0.5, collision_iou=0.5,
            post_collision=0.5, traffic_anomaly=0.5,
        ),
    )
    store.update(job.job_id, status="complete", events=[event],
                 completed_at=datetime.now(timezone.utc), frames_processed=5)
    store.set_incident_status(f"{job.job_id}:1", status="active", note="Operator review")
    pending = store.create(SOURCE)
    running = store.create(SOURCE)
    store.update(running.job_id, status="running")
    store._db.close()
    restarted = JobStore(database_path=database, start_reaper=False)
    assert restarted.get(job.job_id).frames_processed == 5
    incident = restarted.get_incident(f"{job.job_id}:1")
    assert incident["status"] == "active" and incident["note"] == "Operator review"
    for interrupted in (pending, running):
        recovered = restarted.get(interrupted.job_id)
        assert recovered.status == "failed" and recovered.completed_at
        assert "restarted" in recovered.error
    restarted._db.close()


def test_retained_job_limit_and_ttl_remove_persisted_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "max_retained_jobs", 1)
    store = JobStore(database_path=str(tmp_path / "jobs.sqlite3"), start_reaper=False)
    first = store.create(SOURCE)
    directory = tmp_path / "artifacts" / first.job_id
    directory.mkdir(parents=True)
    (directory / "frame.jpg").write_bytes(b"image")
    store.update(first.job_id, status="complete", completed_at=datetime.now(timezone.utc))
    second = store.create(SOURCE)
    assert store.get(first.job_id) is None and not directory.exists()
    store.update(second.job_id, status="failed",
                 completed_at=datetime.now(timezone.utc) - timedelta(days=2))
    store._reap_expired()
    assert not store.all_jobs()
    assert store._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    store._db.close()


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.1", "172.30.0.3", "192.168.1.1",
    "169.254.169.254", "100.100.100.200", "168.63.129.16", "0.0.0.0", "224.0.0.1",
    "::1", "fe80::1", "fd00::1", "::ffff:127.0.0.1",
])
def test_proxy_rejects_private_and_metadata_destinations(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)),
    ])
    with pytest.raises(ValueError):
        egress_proxy.resolve_public("camera.example.com", 443, {"camera.example.com"})


def test_proxy_checks_all_dns_answers_and_pins_numeric_connection(monkeypatch):
    allowed = {"camera.example.com"}
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
    resolver = Mock(return_value=addresses)
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    connection = Mock()
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    assert egress_proxy.connect_public("camera.example.com", 443, allowed) is connection
    connection.connect.assert_called_once_with(("8.8.8.8", 443))
    assert resolver.call_count == 1
    resolver.return_value = addresses + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
    with pytest.raises(ValueError):
        egress_proxy.resolve_public("camera.example.com", 443, allowed)
    with pytest.raises(ValueError):
        egress_proxy.resolve_public("attacker.example.com", 443, allowed)
    with pytest.raises(ValueError):
        egress_proxy.resolve_public("camera.example.com", 22, allowed)


def test_production_video_requires_proxy_and_approved_source(monkeypatch):
    from app.services.video_source import open_production_video
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "approved_stream_urls", [SOURCE])
    monkeypatch.delenv("http_proxy", raising=False)
    with pytest.raises(RuntimeError, match="egress proxy"):
        open_production_video(SOURCE)


@pytest.mark.parametrize("timeout,expected", [(1, 31), (300, 330), (900, 930), (3600, 3630)])
def test_egress_tunnel_lifetime_tracks_job_deadline_with_bounded_margin(timeout, expected):
    assert egress_proxy.tunnel_lifetime_seconds(timeout) == expected
    with egress_proxy.ProxyServer(("127.0.0.1", 0), {"example.com"}, job_timeout_seconds=timeout) as server:
        assert server.tunnel_lifetime == expected


@pytest.mark.parametrize("timeout", [0, -1, 3601, 10000])
def test_egress_rejects_unsupported_deadlines(timeout):
    with pytest.raises(ValueError, match="1-3600"):
        egress_proxy.tunnel_lifetime_seconds(timeout)
