"""A small in-memory, per-IP fixed-window rate limiter.

The auth endpoints (login / signup / reset) are the brute-force surface, so they
get throttled per client IP. In-memory state is fine because the embedded Qdrant
backend already constrains the app to a single worker process; switch to a
shared store (e.g. Redis) if you ever run multiple workers.
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, status

_lock = threading.Lock()
# (key, client) -> (window_start_epoch, count)
_hits: dict[tuple[str, str], tuple[float, int]] = {}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """A FastAPI dependency that allows ``limit`` requests per ``window`` seconds."""

    def __init__(self, key: str, limit: int, window: float) -> None:
        self.key = key
        self.limit = limit
        self.window = window

    def __call__(self, request: Request) -> None:
        ident = (self.key, _client_ip(request))
        now = time.monotonic()
        with _lock:
            start, count = _hits.get(ident, (now, 0))
            if now - start >= self.window:
                start, count = now, 0  # window expired, reset
            count += 1
            _hits[ident] = (start, count)
            over = count > self.limit
            retry_after = int(self.window - (now - start)) + 1
        if over:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts. Please wait and try again.",
                headers={"Retry-After": str(retry_after)},
            )


# Auth attempts: 10 per 5 minutes per IP across login/signup/reset/forgot.
auth_rate_limit = RateLimiter("auth", limit=10, window=300.0)
