"""
NetworkCaptureLayer — captures API calls per step during recording.

Listens for XHR/fetch requests and responses, tracks per-step network
activity, and identifies the critical (slowest) API call for targeted
waits during execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from playwright.async_api import Page, Request, Response

from engine.models import NetworkContext

logger = logging.getLogger(__name__)

# Analytics/monitoring domains to exclude from critical URL detection
_ANALYTICS_DOMAINS = (
    "datadoghq", "datadog", "sentry.io", "google-analytics", "googletagmanager",
    "segment.io", "segment.com", "mixpanel", "amplitude", "hotjar",
    "fullstory", "logrocket", "newrelic", "bugsnag", "rollbar",
)


class NetworkCaptureLayer:
    """Lightweight listener that tracks XHR/fetch calls per recording step."""

    def __init__(self, page: Page) -> None:
        self._page = page
        self._current_calls: list[dict[str, Any]] = []
        self._pending = 0
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    def _on_request(self, request: Request) -> None:
        if request.resource_type not in ("xhr", "fetch"):
            return
        self._pending += 1
        self._current_calls.append({
            "method": request.method,
            "url": request.url,
            "timestamp": time.time(),
        })

    def _on_response(self, response: Response) -> None:
        if response.request.resource_type not in ("xhr", "fetch"):
            return
        self._pending = max(0, self._pending - 1)
        for call in reversed(self._current_calls):
            if call["url"] == response.url and "status" not in call:
                call["status"] = response.status
                call["duration_ms"] = int((time.time() - call["timestamp"]) * 1000)
                break

    def snapshot_and_reset(self) -> NetworkContext:
        """Snapshot current network activity and reset for next step."""
        ctx = NetworkContext(
            api_calls=[
                {k: v for k, v in c.items() if k != "timestamp"}
                for c in self._current_calls
            ],
            pending_at_action=self._pending,
        )
        completed = [
            c for c in ctx.api_calls
            if "duration_ms" in c
            and not any(domain in c.get("url", "") for domain in _ANALYTICS_DOMAINS)
        ]
        if completed:
            slowest = max(completed, key=lambda c: c.get("duration_ms", 0))
            ctx.critical_url_pattern = slowest["url"]
        self._current_calls.clear()
        return ctx

    def detach(self) -> None:
        """Remove listeners from page."""
        try:
            self._page.remove_listener("request", self._on_request)
            self._page.remove_listener("response", self._on_response)
        except Exception:
            pass
