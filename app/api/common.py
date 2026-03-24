"""
Shared helpers for envelope-style API responses.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from math import ceil
from typing import Any

from app.models.schemas import ApiMeta, PaginationMeta


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def start_timer() -> float:
    return time.perf_counter()


def build_meta(
    *,
    started_at: float,
    filters: dict[str, Any] | None = None,
    page: int | None = None,
    page_size: int | None = None,
    total_items: int | None = None,
) -> ApiMeta:
    pagination = None
    if page is not None and page_size is not None and total_items is not None:
        total_pages = ceil(total_items / page_size) if page_size > 0 else 0
        pagination = PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1 and total_pages > 0,
        )

    return ApiMeta(
        request_id=str(uuid.uuid4()),
        timestamp=utcnow(),
        duration_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
        filters=filters,
        pagination=pagination,
    )
