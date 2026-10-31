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
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
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
    environment: str = "development"
    api_token: SecretStr = SecretStr("")
    approved_stream_urls: list[str] = Field(default_factory=list)
    job_database_path: str | None = None
    output_dir: str | None = None
    job_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    max_saved_frames_per_job: int = Field(default=300, ge=1, le=18000)
    max_saved_frame_bytes: int = Field(default=2_000_000, ge=1000)
    max_frame_dimension: int = Field(default=1280, ge=64, le=3840)
    max_retained_jobs: int = Field(default=100, ge=1, le=10000)
    inference_threads: int = Field(default=2, ge=1, le=16)

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
        description="Inference workers. Production defaults to and requires one.",
    )
    max_pending_jobs: int = Field(default=9, ge=0, le=99)
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

    def get_job_admission_capacity(self) -> int:
        if self.environment == "production":
            return self.max_concurrent_jobs + self.max_pending_jobs
        return self.max_concurrent_jobs

    @model_validator(mode="after")
    def validate_production(self) -> "Settings":
        if self.environment not in {"development", "production"}:
            raise ValueError("ENVIRONMENT must be development or production")
        if self.environment != "production":
            return self
        if len(self.api_token.get_secret_value()) < 32:
            raise ValueError("Production requires an API_TOKEN of at least 32 characters")
        origins = self.get_cors_origins()
        if not origins or any(
            urlsplit(origin).scheme != "https"
            or not urlsplit(origin).hostname
            or urlsplit(origin).path
            or urlsplit(origin).query
            or urlsplit(origin).fragment
            or urlsplit(origin).username
            for origin in origins
        ):
            raise ValueError("Production requires explicit HTTPS CORS_ORIGINS")
        if not self.job_database_path or not self.output_dir:
            raise ValueError("Production requires JOB_DATABASE_PATH and OUTPUT_DIR")
        if "max_concurrent_jobs" not in self.model_fields_set:
            self.max_concurrent_jobs = 1
        if self.max_concurrent_jobs != 1:
            raise ValueError("The shared production model requires MAX_CONCURRENT_JOBS=1")
        if self.max_retained_jobs < self.get_job_admission_capacity():
            raise ValueError("MAX_RETAINED_JOBS must accommodate the running and pending queue capacity")
        if self.job_artifact_retention_count < 9:
            raise ValueError("Production must retain artifacts for at least one nine-camera batch")
        for source in self.approved_stream_urls:
            parsed = urlsplit(source)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.fragment:
                raise ValueError("APPROVED_STREAM_URLS must contain credential-free HTTPS URLs")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import and call this everywhere."""
    return Settings()


# Module-level convenience alias so callers can just do:
#   from app.core.config import settings
settings: Settings = get_settings()
