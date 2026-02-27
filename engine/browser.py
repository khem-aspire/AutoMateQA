"""
BrowserManager – Launches and manages the Playwright browser instance.

Responsibilities:
  - Launch headed/headless browser via Playwright.
  - Inject the JS assertion layer into every page.
  - Expose bindings so the injected JS can communicate back.
  - Provide page/context references to other engine components.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from engine.models import EngineConfig

logger = logging.getLogger(__name__)

# Path to the JS assertion layer script
_JS_LAYER_PATH = Path(__file__).parent / "js" / "assertion_layer.js"


class BrowserManager:
    """Manages the Playwright browser lifecycle and JS injection."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._assertion_callback: Optional[Callable[[dict], Any]] = None
        self._action_callback: Optional[Callable[[dict], Any]] = None
        self._seen_assertions: set[str] = set()  # dedup: timestamp keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def launch(self, url: str = "") -> Page:
        """Launch the browser and return the main page."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._config.headless,
            args=["--start-maximized"],
        )
        self._context = await self._browser.new_context(
            viewport=None,  # full-screen in headed mode
            no_viewport=not self._config.headless,
        )

        # Cache the JS code
        self._js_code = _JS_LAYER_PATH.read_text(encoding="utf-8")

        # ── CRITICAL ORDER: expose binding BEFORE init script ──
        # This ensures __assertion_bridge is available when the
        # init script runs during any page navigation.
        await self._context.expose_binding(
            "__assertion_bridge",
            self._handle_assertion_binding,
            handle=False,
        )
        await self._context.add_init_script(self._js_code)

        # Now create the page (init script + binding are already registered)
        self._page = await self._context.new_page()

        # Listen for console-based fallback messages
        self._page.on("console", self._handle_console_message)

        # Re-inject assertion layer after each page load
        self._page.on("load", self._on_page_load)

        if url:
            await self._page.goto(url, wait_until="domcontentloaded")
            # Also evaluate directly in case init script had timing issues
            await self._inject_on_current_page()

        logger.info("Browser launched (headless=%s)", self._config.headless)
        return self._page

    async def close(self) -> None:
        """Gracefully shut down browser and Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")

    @property
    def page(self) -> Page:
        assert self._page is not None, "Browser not launched yet"
        return self._page

    @property
    def context(self) -> BrowserContext:
        assert self._context is not None, "Browser not launched yet"
        return self._context

    def on_assertion(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback for assertion messages from the browser."""
        self._assertion_callback = callback

    def on_action(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback for recorded action messages."""
        self._action_callback = callback

    async def take_screenshot(self, path: str) -> str:
        """Capture a full-page screenshot and return the path."""
        await self.page.screenshot(path=path, full_page=True)
        logger.debug("Screenshot saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _inject_on_current_page(self) -> None:
        """Evaluate the assertion layer JS on the current page directly."""
        try:
            await self._page.evaluate(self._js_code)
            logger.debug("Assertion JS layer evaluated on current page")
        except Exception as e:
            logger.warning("Failed to evaluate assertion JS: %s", e)

    def _on_page_load(self, page: Any) -> None:
        """Re-inject assertion layer after each page load/navigation."""
        asyncio.ensure_future(self._inject_on_current_page())

    async def _handle_assertion_binding(self, source: dict, raw: str) -> None:
        """
        Called by expose_binding when JS sends an assertion.
        `source` contains page/frame info; `raw` is the JSON string.
        """
        await self._handle_assertion_message(raw)

    async def _handle_assertion_message(self, raw: str) -> None:
        """Parse and dispatch an assertion payload."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid assertion payload: %s", raw)
            return

        # Dedup: JS sends via both console and binding
        dedup_key = payload.get("timestamp", "") + payload.get("assertion_type", "")
        if dedup_key in self._seen_assertions:
            return
        self._seen_assertions.add(dedup_key)

        logger.info("Assertion received: %s", payload.get("assertion_type"))
        if self._assertion_callback:
            self._assertion_callback(payload)
        else:
            logger.warning("No assertion callback registered")

    def _handle_console_message(self, msg: Any) -> None:
        """Parse console messages looking for assertion payloads."""
        text: str = msg.text
        if text.startswith("__ASSERTION__:"):
            import asyncio

            asyncio.ensure_future(
                self._handle_assertion_message(text[len("__ASSERTION__:") :])
            )
