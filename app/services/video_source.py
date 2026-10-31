"""Production capture: bounded FFmpeg reads through the isolated egress proxy."""

import os

from app.core.security import validate_stream_url


def open_production_video(source: str):
    import cv2

    validate_stream_url(source)
    if os.environ.get("http_proxy") != "http://egress:3128":
        raise RuntimeError("Production video requires the deployment egress proxy")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "protocol_whitelist;http,https,tcp,tls,httpproxy,crypto"
        "|tls_verify;1|ca_file;/etc/ssl/certs/ca-certificates.crt"
    )
    cap = cv2.VideoCapture(
        source,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000],
    )
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open approved stream via the egress proxy")
    return cap
