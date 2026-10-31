"""Fail-closed public API authentication and operator-approved sources."""

import hmac
import logging

from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings

logger = logging.getLogger(__name__)
_url_adapter = TypeAdapter(HttpUrl)


class AuthenticationMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = settings.api_token.get_secret_value()
        health_path = f"{settings.api_prefix}/health"
        headers = dict(scope["headers"])
        preflight = (
            scope["method"] == "OPTIONS"
            and b"origin" in headers
            and b"access-control-request-method" in headers
        )
        public_health = scope["path"] == health_path and scope["method"] == "GET"
        if token and not preflight and not public_health:
            supplied = headers.get(b"authorization", b"")
            expected = f"Bearer {token}".encode()
            if not hmac.compare_digest(supplied, expected):
                logger.warning("Rejected unauthenticated request to %s", scope["path"])
                response = JSONResponse(
                    {"detail": "A valid operator bearer token is required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def validate_stream_url(source: str) -> None:
    if settings.environment != "production":
        return
    approved = {str(_url_adapter.validate_python(url)) for url in settings.approved_stream_urls}
    if source not in approved:
        logger.warning("Rejected a stream outside APPROVED_STREAM_URLS")
        raise HTTPException(status_code=403, detail="This stream is not approved by the operator.")
