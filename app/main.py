"""
app/main.py — FastAPI application factory.

Start the server
----------------
    # From the project root (c:/git/Traffic-Accident-Detection):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

    # Development (auto-reload on file changes):
    uvicorn app.main:app --reload --port 8000

Notes
-----
- --workers 1 is intentional: the YOLO model is not fork-safe across
  processes.  For horizontal scaling, run multiple containers instead.
- The project root must be the current working directory so that the
  root-level pipeline.py / tracker.py / config.py are importable.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import configure_logging
from app.api import cameras, dashboard, detect, health, incidents, jobs, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the YOLO model once at startup and store it on app.state so every
    request handler can access it without reloading.

    INTEGRATION POINT
    -----------------
    The import of YOLO and the model filename come from the root-level
    config.py (pipeline_config.YOLO_MODEL).  Changing the model there is
    sufficient; no changes needed here.
    """
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s", settings.app_title, settings.app_version)

    # ------------------------------------------------------------------
    # INTEGRATION POINT — load YOLO model
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        import config as pipeline_config  # root-level config

        logger.info("Loading YOLO model: %s  (this may take a moment…)", pipeline_config.YOLO_MODEL)
        app.state.model = YOLO(pipeline_config.YOLO_MODEL)
        logger.info("YOLO model loaded successfully.")
    except Exception:
        logger.exception("Failed to load YOLO model — /detect-accident will return 503.")
        app.state.model = None
    # ------------------------------------------------------------------

    yield  # server runs here

    logger.info("Shutting down %s", settings.app_title)
    app.state.model = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "REST API for real-time traffic accident detection using "
            "YOLOv8 object detection and DeepSORT multi-object tracking."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------
    # CORS — allow the frontend origin(s) configured in settings / .env
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request logging middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("→ %s %s", request.method, request.url.path)
        response = await call_next(request)
        logger.info("← %s %s  %d", request.method, request.url.path, response.status_code)
        return response

    # ------------------------------------------------------------------
    # Global exception handler — turns unhandled errors into clean JSON
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    prefix = settings.api_prefix  # e.g. "/api/v1" or "" (empty = no prefix)
    app.include_router(detect.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)
    app.include_router(dashboard.router, prefix=prefix)
    app.include_router(stats.router, prefix=prefix)
    app.include_router(incidents.router, prefix=prefix)
    app.include_router(jobs.router, prefix=prefix)
    app.include_router(cameras.router, prefix=prefix)

    return app


# Module-level app instance (referenced by uvicorn)
app = create_app()
