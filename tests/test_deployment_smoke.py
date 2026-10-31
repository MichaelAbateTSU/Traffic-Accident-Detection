"""Verify the deployment probe's batch retention checks and queue-aware deadline."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from deploy import smoke

TOKEN = "smoke-test-operator-token-at-least-thirty-two-characters"
SOURCE = "https://camera.example.com/live/playlist.m3u8"


@pytest.fixture()
def smoke_runtime(monkeypatch):
    state = {"clock": 0, "jobs": [], "polled": set(), "pending": False, "timeouts": []}
    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.setenv("CORS_ORIGINS", "https://michaelabatetsu.github.io")

    def sleep(seconds):
        state["clock"] += seconds

    monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=lambda: state["clock"], sleep=sleep))

    def open_request(request, timeout):
        state["timeouts"].append(timeout)
        path = urlsplit(request.full_url).path
        authenticated = request.get_header("Authorization") == f"Bearer {TOKEN}"
        status = 200
        headers = {}
        if path == "/health":
            payload = {"model_loaded": True}
        elif request.method == "OPTIONS":
            headers = {"Access-Control-Allow-Origin": "https://michaelabatetsu.github.io"}
            payload = {}
        elif not authenticated:
            status, payload = 401, {}
        elif path == "/detect-accident":
            source = json.loads(request.data)["stream_url"]
            if source != SOURCE:
                status, payload = 403, {}
            else:
                job_id = f"job-{len(state['jobs']) + 1}"
                state["jobs"].append(job_id)
                status, payload = 202, {"job_id": job_id}
        elif path.startswith("/jobs/"):
            job_id = path.rsplit("/", 1)[1]
            state["polled"].add(job_id)
            payload = {
                "status": "pending" if state["pending"] else "complete",
                "frames_processed": 5,
                "frames": [{"capture": f"https://api.example.com/artifacts/{job_id}/frame.jpg"}],
            }
        else:
            assert path.startswith("/artifacts/")
            assert state["polled"] == set(state["jobs"]), "Images fetched before the batch completed"
            payload = b"\xff\xd8test-jpeg"
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return nullcontext(SimpleNamespace(status=status, headers=headers, read=lambda: body))

    monkeypatch.setattr(smoke.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    return state


def test_smoke_submits_nine_jobs_and_fetches_images_after_entire_batch(smoke_runtime, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "smoke.py", "https://api.example.com", "--stream-url", SOURCE, "--batch-size", "9",
    ])
    smoke.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["jobs"]) == len(smoke_runtime["jobs"]) == 9
    assert result["https_smoke"] == "passed"


def test_smoke_explicitly_times_out_pending_queue_with_configured_budget(smoke_runtime, monkeypatch):
    smoke_runtime["pending"] = True
    monkeypatch.setattr(sys, "argv", [
        "smoke.py", "https://api.example.com", "--stream-url", SOURCE, "--timeout-seconds", "5",
    ])
    with pytest.raises(TimeoutError, match="5 seconds, including queue wait"):
        smoke.main()
    assert smoke_runtime["clock"] == 5
    assert min(smoke_runtime["timeouts"]) <= 1


def test_default_smoke_budget_allows_queue_wait_beyond_a_single_execution(smoke_runtime, monkeypatch, capsys):
    smoke_runtime["pending"] = True

    def finish_after_wait(_):
        smoke_runtime["clock"] = 1000
        smoke_runtime["pending"] = False

    monkeypatch.setattr(smoke.time, "sleep", finish_after_wait)
    monkeypatch.setattr(sys, "argv", ["smoke.py", "https://api.example.com", "--stream-url", SOURCE])
    smoke.main()
    assert smoke_runtime["clock"] > 300
    assert json.loads(capsys.readouterr().out)["https_smoke"] == "passed"


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_smoke_rejects_nonpositive_timeout(smoke_runtime, monkeypatch, timeout):
    monkeypatch.setattr(sys, "argv", ["smoke.py", "https://api.example.com", "--timeout-seconds", timeout])
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code == 2
