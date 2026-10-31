"""Compatibility dashboard values describe frame records and job outcomes."""

from datetime import datetime, timedelta, timezone

from app.services import detection_service
from app.services.job_store import job_store
from test_saved_frames import frame_pipeline


def test_consecutive_active_alarm_frames_remain_distinct_incidents(frame_pipeline, monkeypatch):
    client, _, _, _, _, _ = frame_pipeline
    # Retained alarm state is authoritative even below the high threshold.
    scorer = detection_service.AccidentScorer()
    _, _, meta = scorer.update([], 0)
    scorer.update.return_value = (.4, True, meta)
    monkeypatch.setattr(detection_service, "AccidentScorer", lambda: scorer)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now + timedelta(seconds=1)
    monkeypatch.setattr(detection_service, "_utcnow", lambda: start)
    job = job_store.create("https://cams.example.com/cam-a.m3u8")
    detection_service.run_detection_job(job.job_id, job.stream_url, 2, object())
    result = job_store.get(job.job_id)
    assert result.status == "complete"
    assert [event.frame_idx for event in result.events] == [0, 1]
    assert all(event.confidence_score == .4 for event in result.events)
    items = client.get("/incidents").json()["data"]
    assert {item["incident_id"] for item in items} == {f"{job.job_id}:0", f"{job.job_id}:1"}
    assert all(item["status"] == "resolved" for item in items)
    assert job_store.count_incidents_in_range(start_time=now, end_time=start + timedelta(seconds=30)) == 2
    for payload in [
        client.get("/stats/overview").json()["data"],
        client.get("/dashboard", params={"incidents_status": "all"}).json()["stats"],
    ]:
        assert payload["incidents_today"] == payload["incidentsToday"] == 2


def test_ratio_includes_pending_running_and_complete_but_excludes_old_jobs(client):
    now = datetime.now(timezone.utc)
    for status in ("pending", "running", "complete", "failed"):
        job = job_store.create(f"https://cams.example.com/{status}.m3u8")
        job_store.update(job.job_id, status=status, created_at=now - timedelta(minutes=1))
    old = job_store.create("https://cams.example.com/old.m3u8")
    job_store.update(old.job_id, status="failed", created_at=now - timedelta(days=2))
    for response, key in [
        (client.get("/stats/overview", params={"history_days": 1}), "data"),
        (client.get("/dashboard", params={"history_days": 1}), "stats"),
    ]:
        assert response.status_code == 200
        values = response.json()[key]
        assert values["uptime_percent"] == values["uptimePercent"] == 75.0


def test_empty_window_ratio_is_100_even_with_an_old_failed_job(client):
    job = job_store.create("https://cams.example.com/old.m3u8")
    job_store.update(job.job_id, status="failed", created_at=datetime.now(timezone.utc) - timedelta(days=2))
    assert client.get("/stats/overview", params={"history_days": 1}).json()["data"]["uptime_percent"] == 100
    assert client.get("/dashboard", params={"history_days": 1}).json()["stats"]["uptimePercent"] == 100


def test_openapi_describes_metric_aliases_and_frame_records(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "hysteresis" in schemas["DetectionEvent"]["description"]
    for model in ("StatsOverview", "FrontendStats"):
        for field in ("incidents_today", "incidentsToday"):
            assert "not distinct accidents" in schemas[model]["properties"][field]["description"]
        for field in ("uptime_percent", "uptimePercent"):
            assert "Not measured service uptime" in schemas[model]["properties"][field]["description"]


def test_openapi_describes_incident_status_as_workflow_state(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for model in ("IncidentSummary", "FrontendIncident"):
        props = schemas[model]["properties"]
        assert "not an operator-review" in props["status"]["description"]
        for field in ("manually_updated", "manuallyUpdated"):
            assert "override" in props[field]["description"]


def test_default_status_follows_job_state_and_override_wins(client):
    from types import SimpleNamespace
    from app.services.job_store import _build_incident_record

    event = SimpleNamespace(frame_idx=0, timestamp_sec=0.0)
    job = job_store.create("https://cams.example.com/cam-s.m3u8")
    expect = {"pending": "active", "running": "active", "complete": "resolved", "failed": "resolved"}
    for state, status in expect.items():
        job.status = state
        record = _build_incident_record(job, event, "cam-s", {})
        assert (record["status"], record["manually_updated"]) == (status, False)
    job.status = "running"
    override = {"status": "resolved", "resolved_at": datetime.now(timezone.utc), "note": "n"}
    record = _build_incident_record(job, event, "cam-s", {f"{job.job_id}:0": override})
    assert (record["status"], record["manually_updated"]) == ("resolved", True)
