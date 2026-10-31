"""Versioned, time-bounded submission identities (never reusable after expiry)."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta

RETENTION_SECONDS = 86400
FUTURE_SKEW_SECONDS = 60
_KEY_PATTERN = re.compile(r"v1\.([0-9]{10})\.([0-9a-f-]{36})", re.ASCII)


class SubmissionIdentityError(ValueError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def identify_submission(key: str, now: datetime) -> tuple[str, datetime]:
    match = _KEY_PATTERN.fullmatch(key)
    if match is None:
        raise SubmissionIdentityError(422, "invalid_idempotency_key",
                                      "Expected v1.<Unix seconds>.<lowercase UUIDv4>.")
    try:
        identity = uuid.UUID(match[2])
    except ValueError as exc:
        raise SubmissionIdentityError(422, "invalid_idempotency_key",
                                      "The submission identity must be a UUIDv4.") from exc
    if identity.version != 4 or str(identity) != match[2]:
        raise SubmissionIdentityError(422, "invalid_idempotency_key",
                                      "The submission identity must be a canonical UUIDv4.")
    issued_at = datetime.fromtimestamp(int(match[1]), tz=now.tzinfo)
    if issued_at > now + timedelta(seconds=FUTURE_SKEW_SECONDS):
        raise SubmissionIdentityError(422, "idempotency_key_in_future",
                                      "The submission identity timestamp is ahead of the server clock.")
    expires_at = issued_at + timedelta(seconds=RETENTION_SECONDS)
    if now >= expires_at:
        raise SubmissionIdentityError(410, "idempotency_key_expired",
                                      "This submission identity has expired; it cannot schedule work.")
    return hashlib.sha256(key.encode("ascii")).hexdigest(), expires_at


def request_fingerprint(stream_url: str, max_frames: int, save_frames: bool) -> str:
    payload = json.dumps(
        {"stream_url": stream_url, "max_frames": max_frames, "save_frames": save_frames},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
