"""Branch-only fixed-camera experiment; stdout contains only safe numerical evidence."""

import argparse
import functools
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE = "https://mcleansfs3.us-east-1.skyvdn.com/rtplive/R2_061/playlist.m3u8"
HOST = "mcleansfs3.us-east-1.skyvdn.com"
MAX_FRAMES = 12
METRICS = Path("/data/metrics.json")
COUNTERS = {
    "yolo_calls": 0, "vehicle_detections": 0, "tracker_calls": 0,
    "confirmed_track_observations": 0, "max_confirmed_tracks": 0, "scorer_calls": 0,
    "deep_sort_embedder_calls": 0, "embedded_crop_count": 0, "blocked_write_attempts": 0,
    "job_finished": False,
}
PROBE_STATE = {"stage": "initialization", "job_submitted": False}


def serve():
    import cv2
    import uvicorn
    from app.services import detection_service
    from app import main as main_module

    def forbid_image_write(*args, **kwargs):
        COUNTERS["blocked_write_attempts"] += 1
        raise RuntimeError("Image/video writes are forbidden in this experiment")

    cv2.imwrite = forbid_image_write
    cv2.VideoWriter = forbid_image_write

    def audit(event, args):
        if event != "open":
            return
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        )
        if writing and not isinstance(path, int):
            permitted = {
                "/data/jobs.sqlite3", "/data/jobs.sqlite3-wal",
                "/data/jobs.sqlite3-shm", "/data/jobs.sqlite3-journal",
                str(METRICS), "/dev/null",
            }
            if os.path.abspath(os.fspath(path)) not in permitted:
                COUNTERS["blocked_write_attempts"] += 1
                raise PermissionError("Non-metadata write forbidden in live smoke")

    sys.addaudithook(audit)

    def count_result(owner, name, call_key, result_key=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            result = original(*args, **kwargs)
            COUNTERS[call_key] += 1
            if result_key:
                COUNTERS[result_key] += len(result)
            if call_key == "tracker_calls":
                COUNTERS["max_confirmed_tracks"] = max(
                    COUNTERS["max_confirmed_tracks"], len(result),
                )
            return result

        setattr(owner, name, measured)

    count_result(detection_service, "_run_yolo", "yolo_calls", "vehicle_detections")
    count_result(detection_service.DeepSortTracker, "update",
                 "tracker_calls", "confirmed_track_observations")
    count_result(detection_service.AccidentScorer, "update", "scorer_calls")
    original_tracker_init = detection_service.DeepSortTracker.__init__

    @functools.wraps(original_tracker_init)
    def measured_tracker_init(self, *args, **kwargs):
        original_tracker_init(self, *args, **kwargs)
        count_result(self._tracker.embedder, "predict",
                     "deep_sort_embedder_calls", "embedded_crop_count")

    detection_service.DeepSortTracker.__init__ = measured_tracker_init

    original_job = detection_service.run_detection_job

    @functools.wraps(original_job)
    def measured_job(*args, **kwargs):
        try:
            return original_job(*args, **kwargs)
        finally:
            COUNTERS["job_finished"] = True
            METRICS.write_text(json.dumps(COUNTERS))

    # The application imported the job callable before installing the measurement wrapper.
    main_module.run_detection_job = measured_job
    METRICS.write_text(json.dumps(COUNTERS))
    uvicorn.run(main_module.app, host="127.0.0.1", port=8000, workers=1, access_log=False)


def probe():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token = os.environ["API_TOKEN"]

    def request(path, payload=None, authenticated=True, invalid=False):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + ("invalid" if invalid else token)
        req = urllib.request.Request(
            "http://127.0.0.1:8000" + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        try:
            with opener.open(req, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    deadline = time.monotonic() + 60
    while True:
        try:
            status, health = request("/health", authenticated=False)
            if status == 200 and health["model_loaded"]:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Production API did not become healthy")
        time.sleep(1)

    PROBE_STATE["stage"] = "unauthenticated_request_rejection"
    assert request("/jobs", authenticated=False)[0] == 401
    PROBE_STATE["stage"] = "invalid_token_rejection"
    assert request("/jobs", invalid=True)[0] == 401
    PROBE_STATE["stage"] = "unapproved_source_rejection"
    assert request("/detect-accident", {"stream_url": "https://example.com/blocked"})[0] == 403
    PROBE_STATE["stage"] = "direct_egress_rejection"
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=2):
            raise AssertionError("API has direct internet egress")
    except (TimeoutError, OSError):
        pass
    PROBE_STATE["stage"] = "unapproved_proxy_host_rejection"
    with socket.create_connection(("egress", 3128), timeout=3) as connection:
        connection.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
        assert connection.recv(1024).startswith(b"HTTP/1.1 403 ")
    PROBE_STATE["stage"] = "fixed_source_job_submission"
    status, submitted = request("/detect-accident", {
        "stream_url": SOURCE, "max_frames": MAX_FRAMES, "save_frames": False,
    })
    assert status == 202
    PROBE_STATE.update(stage="waiting_for_terminal_job", job_submitted=True)
    deadline = time.monotonic() + 160
    while True:
        status, job = request("/jobs/" + submitted["job_id"])
        assert status == 200
        if job["status"] in {"complete", "failed"}:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Bounded live job did not reach a terminal state")
        time.sleep(1)
    PROBE_STATE["stage"] = "terminal_job_evidence"
    # Worker publishes counters just after updating the terminal job.
    for _ in range(20):
        if METRICS.exists() and json.loads(METRICS.read_text())["job_finished"]:
            break
        time.sleep(0.1)
    counts = json.loads(METRICS.read_text())
    artifact_files = sum(p.is_file() for p in Path("/data/artifacts").rglob("*"))
    metadata_only = all(
        p.name in {"artifacts", "jobs.sqlite3", "jobs.sqlite3-wal", "jobs.sqlite3-shm",
                   "jobs.sqlite3-journal", "metrics.json"}
        for p in Path("/data").iterdir()
    )
    assert not job["frames"] and artifact_files == 0 and metadata_only
    assert counts["blocked_write_attempts"] == 0
    if job["status"] == "complete":
        assert all(counts[key] == job["frames_processed"]
                   for key in ("yolo_calls", "tracker_calls", "scorer_calls"))
    error_category = None
    if job["status"] == "failed":
        error = job.get("error") or ""
        if "Could not open approved stream via the egress proxy" in error:
            error_category = "approved_stream_open_failed"
        elif "decodable video frames" in error:
            error_category = "no_decodable_frames"
        elif "TIMEOUT" in error or "TimeoutError" in error:
            error_category = "job_timeout"
        else:
            error_category = "pipeline_failure_details_suppressed"
    result = {
        "camera": "R2_061", "status": job["status"], "error_category": error_category,
        "requested_frames": MAX_FRAMES, "frames_processed": job["frames_processed"],
        "peak_heuristic_score": job["peak_confidence"],
        "accident_detected": job["accident_detected"], "event_count": len(job["events"]),
        "published_frame_count": len(job["frames"]), "artifact_file_count": artifact_files,
        "data_contains_only_metadata": metadata_only,
        "authentication_checks_passed": True, "egress_checks_passed": True,
        **counts,
    }
    print(json.dumps(result))
    return job["status"] == "complete" and job["frames_processed"] == MAX_FRAMES


def orchestrate():
    token = secrets.token_urlsafe(48)
    environment = os.environ.copy()
    environment["API_TOKEN"] = token

    def docker(*args, timeout=30, capture=False, check=True):
        return subprocess.run(
            ["docker", *args], env=environment, timeout=timeout, check=check,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, text=True,
        )

    script = str(Path(__file__).resolve())
    success = False
    try:
        docker("network", "create", "--internal", "--subnet", "172.29.67.0/24",
               "live-smoke-private")
        docker("network", "create", "live-smoke-internet")
        docker("create", "--name", "live-smoke-egress", "--network", "live-smoke-private",
               "--ip", "172.29.67.4", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--cpus", "0.25", "--memory", "128m",
               "--memory-swap", "128m", "--pids-limit", "32", "--ulimit", "core=0",
               "--log-driver", "none", "-e", "STREAM_HOSTS=" + HOST,
               "-e", "JOB_TIMEOUT_SECONDS=120", "traffic-live-egress")
        docker("network", "connect", "live-smoke-internet", "live-smoke-egress")
        docker("start", "live-smoke-egress")
        docker("run", "-d", "--name", "live-smoke-api", "--network", "live-smoke-private",
               "--ip", "172.29.67.2", "--add-host", "egress:172.29.67.4", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--cpus", "2", "--memory", "4g", "--memory-swap", "4g",
               "--pids-limit", "128", "--ulimit", "core=0", "--log-driver", "none",
               "--no-healthcheck", "--tmpfs", "/data:rw,size=16m,mode=1777",
               "--tmpfs", "/data/artifacts:ro,size=1m,mode=0555",
               "--mount", f"type=bind,source={script},target=/app/live_camera_smoke.py,readonly",
               "-e", "API_TOKEN", "-e", "ENVIRONMENT=production",
               "-e", "CORS_ORIGINS=https://michaelabatetsu.github.io",
               "-e", "APPROVED_STREAM_URLS=" + json.dumps([SOURCE]),
               "-e", "JOB_DATABASE_PATH=/data/jobs.sqlite3",
               "-e", "OUTPUT_DIR=/data/artifacts", "-e", "MAX_CONCURRENT_JOBS=1",
               "-e", "JOB_ARTIFACT_RETENTION_COUNT=9", "-e", "JOB_TIMEOUT_SECONDS=120",
               "-e", "http_proxy=http://egress:3128", "-e", "https_proxy=http://egress:3128",
               "-e", "no_proxy=127.0.0.1,localhost", "traffic-live-smoke",
               "python", "live_camera_smoke.py", "serve")
        state = json.loads(docker("inspect", "live-smoke-api", capture=True).stdout)[0]
        assert state["State"]["Running"]
        assert state["HostConfig"]["ReadonlyRootfs"]
        assert state["HostConfig"]["LogConfig"]["Type"] == "none"
        assert not state["HostConfig"]["PortBindings"]
        assert set(state["NetworkSettings"]["Networks"]) == {"live-smoke-private"}
        assert json.loads(docker("network", "inspect", "live-smoke-private",
                                 capture=True).stdout)[0]["Internal"]
        outcome = docker("exec", "-e", "API_TOKEN", "live-smoke-api",
                         "python", "live_camera_smoke.py", "probe",
                         timeout=240, capture=True, check=False)
        if outcome.stdout:
            result = json.loads(outcome.stdout)
            print(json.dumps(result | {
                "experiment_revision": os.environ["EXPERIMENT_REVISION"],
                "production_revision": "0408066230734c8bf88eac0aeeed313d7c01e950",
                "weights_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
                "model": "YOLO11n", "tracker": "DeepSORT", "scorer": "AccidentScorer",
                "api_read_only": True, "artifact_mount_read_only": True,
                "container_logs_disabled": True, "published_ports": 0,
                "persistent_data_volumes": 0,
            }), flush=True)
        else:
            state = json.loads(docker("inspect", "live-smoke-api", capture=True).stdout)[0]
            print(json.dumps({"status": "experiment_failed_before_safe_summary",
                              "probe_exit_code": outcome.returncode,
                              "api_running": state["State"]["Running"],
                              "api_exit_code": state["State"]["ExitCode"],
                              "api_oom_killed": state["State"]["OOMKilled"]}), flush=True)
        success = outcome.returncode == 0
    finally:
        docker("stop", "-t", "10", "live-smoke-api", "live-smoke-egress", check=False)
        docker("rm", "-f", "live-smoke-api", "live-smoke-egress", check=False)
        docker("network", "rm", "live-smoke-private", "live-smoke-internet", check=False)
        remaining = docker("ps", "-a", "--filter", "name=live-smoke-",
                           "--format", "{{.Names}}", capture=True).stdout.strip()
        assert not remaining, "Temporary services were not removed"
        print(json.dumps({"temporary_services_removed": True,
                          "ram_only_job_metadata_destroyed": True}), flush=True)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["orchestrate", "serve", "probe"])
    args = parser.parse_args()
    if args.mode == "serve":
        serve()
    elif args.mode == "probe":
        try:
            successful = probe()
        except Exception as exc:
            # Never expose raw HTTP bodies, native errors, paths or credentials.
            evidence = {
                "status": "experiment_validation_failed", **PROBE_STATE,
                "error_type": type(exc).__name__,
                "artifact_file_count": sum(
                    p.is_file() for p in Path("/data/artifacts").rglob("*")
                ),
            }
            if METRICS.exists():
                evidence.update(json.loads(METRICS.read_text()))
            print(json.dumps(evidence), flush=True)
            sys.exit(1)
        sys.exit(0 if successful else 1)
    else:
        sys.exit(0 if orchestrate() else 1)
