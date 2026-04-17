"""Shared helpers for api_assertion tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def run(coro):
    """Synchronously drive an async test body (project uses plain unittest)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def make_mock_response(
    *,
    method: str = "GET",
    url: str = "https://example.com/api/thing",
    status: int = 200,
    response_json: Any = None,
    response_text: str = "",
    response_headers: dict[str, str] | None = None,
    request_headers: dict[str, str] | None = None,
    request_post_data_json: Any = None,
    timing_total_ms: float = 50.0,
):
    """Build a Playwright-like Response mock."""
    resp = MagicMock()
    resp.url = url
    resp.status = status
    resp.headers = response_headers or {}
    if response_json is not None:
        resp.json = AsyncMock(return_value=response_json)
    resp.text = AsyncMock(return_value=response_text or "")

    req = MagicMock()
    req.method = method
    req.url = url
    req.headers = request_headers or {}
    req.post_data_json = request_post_data_json
    req.timing = {"responseEnd": timing_total_ms, "startTime": 0}
    resp.request = req
    return resp
