"""Run authenticated HTTPS inference and verify a previously completed job."""

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--stream-url")
    parser.add_argument("--existing-job")
    parser.add_argument("--ca-file")
    args = parser.parse_args()
    if not args.base_url.startswith("https://"):
        parser.error("Use the real HTTPS backend URL")
    token = os.environ.get("API_TOKEN")
    if not token:
        parser.error("Set API_TOKEN privately in the environment")
    context = ssl.create_default_context(cafile=args.ca_file)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context),
    )

    def request(path, method="GET", payload=None, authenticated=True, headers=None):
        request_headers = {"Content-Type": "application/json"} | (headers or {})
        if authenticated:
            request_headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=request_headers, method=method,
        )
        try:
            with opener.open(req, timeout=30) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    status, _, body = request("/health", authenticated=False)
    assert status == 200 and json.loads(body)["model_loaded"], body
    assert request("/jobs", authenticated=False)[0] == 401
    assert request("/jobs", authenticated=False, headers={"Authorization": "Bearer invalid"})[0] == 401
    status, headers, _ = request("/detect-accident", method="OPTIONS", authenticated=False, headers={
        "Origin": os.environ.get("CORS_ORIGINS", "https://michaelabatetsu.github.io"),
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    assert status == 200 and headers.get("Access-Control-Allow-Origin")
    assert request("/detect-accident", method="POST",
                   payload={"stream_url": "https://169.254.169.254/private"})[0] == 403
    if args.existing_job:
        job_id = args.existing_job
    else:
        if not args.stream_url:
            parser.error("Supply an approved --stream-url or --existing-job")
        status, _, body = request("/detect-accident", method="POST", payload={
            "stream_url": args.stream_url, "max_frames": 5, "save_frames": True,
        })
        assert status == 202, (status, body)
        job_id = json.loads(body)["job_id"]
    deadline = time.monotonic() + 330
    while True:
        status, _, body = request(f"/jobs/{job_id}")
        assert status == 200, body
        job = json.loads(body)
        assert job["status"] != "failed", job
        if job["status"] == "complete":
            break
        assert time.monotonic() < deadline, "Inference never completed"
        time.sleep(1)
    assert job["frames_processed"] > 0 and job["frames"], job
    artifact = job["frames"][0]["capture"]
    artifact_path = urllib.parse.urlsplit(artifact).path
    assert request(artifact_path, authenticated=False)[0] == 401
    status, _, image = request(artifact_path)
    assert status == 200 and image[:2] == b"\xff\xd8", (status, image[:100])
    print(json.dumps({"job_id": job_id, "status": job["status"],
                      "frames_processed": job["frames_processed"], "https_smoke": "passed"}))


if __name__ == "__main__":
    main()
