import pytest
from fastapi import HTTPException

from rag_app.web.ratelimit import RateLimiter


class _Req:
    def __init__(self, ip):
        self.headers = {}
        self.client = type("C", (), {"host": ip})()


def test_blocks_after_limit_then_isolates_by_ip():
    rl = RateLimiter("test", limit=2, window=100.0)
    caller = _Req("1.2.3.4")
    rl(caller)
    rl(caller)
    with pytest.raises(HTTPException) as exc:
        rl(caller)
    assert exc.value.status_code == 429

    # A different client is unaffected by the first one's quota.
    rl(_Req("5.6.7.8"))


def test_respects_x_forwarded_for():
    rl = RateLimiter("fwd", limit=1, window=100.0)
    req = _Req("10.0.0.1")
    req.headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    rl(req)
    with pytest.raises(HTTPException):
        rl(req)  # same forwarded client is throttled
