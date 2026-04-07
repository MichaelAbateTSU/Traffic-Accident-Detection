"""
app/core/config.py — FastAPI service settings.

All values can be overridden via environment variables or a .env file.
The pipeline's own tunable constants (thresholds, weights, etc.) remain in
the root-level config.py; this file only concerns the *service* layer.

Usage
-----
    from app.core.config import settings

    settings.yolo_model        # e.g. "yolo11x.pt"
    settings.allowed_origins   # CORS whitelist
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # API / server
    # ------------------------------------------------------------------
    app_title: str = "Traffic Accident Detection API"
    app_version: str = "1.0.0"
    api_prefix: str = ""

    # Comma-separated origins allowed by CORS, e.g. "http://localhost:3000"
    # Use "*" to allow all (development only).
    cors_origins: str = "*"

    # ------------------------------------------------------------------
    # YOLO / pipeline
    # ------------------------------------------------------------------
    yolo_model: str = Field(
        default="yolo11x.pt",
        description="YOLO model weights file. Auto-downloaded by Ultralytics on first use.",
    )

    # ------------------------------------------------------------------
    # Job store
    # ------------------------------------------------------------------
    job_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="How long completed/failed jobs are retained in memory.",
    )
    max_concurrent_jobs: int = Field(
        default=10,
        ge=1,
        description="Refuse new jobs when this many are already running.",
    )
    job_artifact_retention_count: int = Field(
        default=10,
        ge=1,
        description="How many most-recent terminal jobs keep local artifact images.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @field_validator("cors_origins")
    @classmethod
    def _parse_origins(cls, v: str) -> str:
        return v.strip()

    def get_cors_origins(self) -> list[str]:
        """Return CORS origins as a list for FastAPI's CORSMiddleware."""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import and call this everywhere."""
    return Settings()


# Module-level convenience alias so callers can just do:
#   from app.core.config import settings
settings: Settings = get_settings()
