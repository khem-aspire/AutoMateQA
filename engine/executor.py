"""
StepExecutor – Executes individual test steps with healing support.

For each step:
  1. Resolve selector → score candidates.
  2. If confidence < threshold → try fallback / healing.
  3. Execute the action (click, type, navigate, …).
  4. Run attached assertions.
  5. Log the result.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from playwright.async_api import Page

from engine.assertions import AssertionEngine
from engine.healer import HealingEngine
from engine.models import (
    ActionType,
    AssertionResult,
    EngineConfig,
    HealingMode,
    SelectorHeal,
    StepResult,
    StepStatus,
    TestStep,
)
from engine.selector import SelectorCandidate, SelectorEngine

logger = logging.getLogger(__name__)


class StepExecutor:
    """Executes a single TestStep against a live Playwright page."""

    def __init__(
        self,
        config: EngineConfig,
        selector_engine: SelectorEngine,
        assertion_engine: AssertionEngine,
        healing_engine: HealingEngine,
    ) -> None:
        self._config = config
        self._selector = selector_engine
        self._assertions = assertion_engine
        self._healer = healing_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        page: Page,
        step: TestStep,
        screenshot_dir: Optional[Path] = None,
    ) -> StepResult:
        """Execute a step and return a structured result."""
        start = time.monotonic()
        result = StepResult(step_id=step.step_id)

        step_timeout = self._config.step_timeout_ms / 1000.0

        try:
            if step.action.action_type == ActionType.NAVIGATE:
                await self._do_navigate(page, step)
                result.element_confidence = 1.0
            elif step.action.action_type == ActionType.SCROLL:
                await self._do_scroll(page, step.action, None)
                result.element_confidence = 1.0
            else:
                await self._wait_before_step(page, step_timeout)

                url_before = page.url
                intent = step.action.intent or {}
                expected_nav = intent.get("expected_navigation", False)

                # Resolve candidates once (healing runs here; wait for it before any action)
                candidates = await self._get_candidates_with_healing(page, step, result)

                if not candidates:
                    if not await self._try_coordinate_click(page, step, result):
                        result.status = StepStatus.FAILED
                        result.error = "Could not resolve target element"
                        await self._maybe_screenshot(page, result, screenshot_dir)
                        return result
                else:
                    action_done = False
                    # Only try coordinate click first when we did NOT heal and best is path/svg
                    if (
                        not result.healed
                        and step.action.action_type in (ActionType.CLICK, ActionType.DBLCLICK)
                        and step.action.click_x is not None
                        and step.action.click_y is not None
                        and self._is_svg_path_selector(candidates[0].selector)
                    ):
                        logger.info(
                            "Step %d: path/svg target — trying coordinate click first",
                            step.step_id,
                        )
                        if await self._try_coordinate_click(page, step, result):
                            action_done = True
                    if not action_done:
                        for candidate in candidates:
                            try:
                                result.element_confidence = candidate.confidence
                                await self._perform_action(page, candidate, step)
                                action_done = True
                                break
                            except Exception as e:
                                logger.warning(
                                    "Step %d: selector %s failed (%s), trying next candidate",
                                    step.step_id, candidate.selector, e,
                                )
                    if not action_done:
                        if await self._try_coordinate_click(page, step, result):
                            action_done = True
                        else:
                            result.status = StepStatus.FAILED
                            result.error = "All selectors failed and coordinate click not available"
                            await self._maybe_screenshot(page, result, screenshot_dir)
                            return result

                await self._wait_after_action(page, url_before, expected_nav, step, step_timeout)
                await self._wait_for_assertion_targets_if_needed(
                    page, step, expected_nav, step_timeout
                )

            assertion_results = await self._run_assertions(page, step)
            result.assertions = assertion_results

            # Determine overall status
            if any(a.status == StepStatus.FAILED for a in assertion_results):
                result.status = StepStatus.FAILED
                result.error = "One or more assertions failed"
            elif result.healed:
                result.status = StepStatus.HEALED
            else:
                result.status = StepStatus.PASSED

        except Exception as e:
            logger.error("Step %d failed: %s", step.step_id, e)
            result.status = StepStatus.FAILED
            result.error = str(e)
            await self._maybe_screenshot(page, result, screenshot_dir)

        result.duration_ms = round((time.monotonic() - start) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    # Resolution + Healing
    # ------------------------------------------------------------------

    async def _get_candidates_with_healing(
        self,
        page: Page,
        step: TestStep,
        result: StepResult,
    ) -> list[SelectorCandidate]:
        """Resolve all candidates (best first). Wait for element, then return
        full list so executor can try fallbacks on action failure.
        """
        candidates: list[SelectorCandidate] = []
        best: Optional[SelectorCandidate] = None

        # Phase 1: wait for at least one candidate
        for _ in range(10):
            best = await self._selector.resolve(page, step.target)
            if best is not None:
                break
            await asyncio.sleep(0.5)

        if best is None:
            return []

        # Phase 2: brief retry for hydration
        if best.confidence < self._config.confidence_threshold:
            for _ in range(3):
                await asyncio.sleep(0.5)
                retry = await self._selector.resolve(page, step.target)
                if retry and retry.confidence > best.confidence:
                    best = retry

        # Get full candidate list for fallback chain
        candidates = await self._selector.resolve_candidates(page, step.target)
        if not candidates:
            return [best] if best else []

        # Optionally prepend healed candidate
        if (
            best.confidence < self._config.confidence_threshold
            and self._config.healing_mode != HealingMode.DISABLED
            and self._config.llm_enabled
        ):
            healing = await self._healer.heal(
                page,
                step.target,
                failed_selector=best.selector,
            )
            if healing.success:
                healed_locator = page.locator(healing.new_selector)
                if await healed_locator.count() > 0:
                    result.healed = True
                    result.healing_details = healing.explanation
                    step.selector_history.append(
                        SelectorHeal(
                            original_selector=best.selector,
                            healed_selector=healing.new_selector,
                            healing_mode=self._config.healing_mode.value,
                            confidence_before=best.confidence,
                            confidence_after=healing.confidence,
                            strategy=getattr(healing, "strategy", ""),
                            healing_method=getattr(healing, "healing_method", ""),
                        )
                    )
                    if self._config.healing_mode == HealingMode.AUTO_UPDATE:
                        step.target.css_selector = healing.new_selector
                        step.target.selectors["preferred"] = healing.new_selector
                    healed_candidate = SelectorCandidate(
                        locator=healed_locator,
                        selector=healing.new_selector,
                        confidence=healing.confidence,
                        strategy="healed",
                    )
                    candidates = [healed_candidate] + [
                        c for c in candidates if c.selector != healing.new_selector
                    ]

        return candidates

    # ------------------------------------------------------------------
    # State synchronizer (wait before/after steps)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_svg_path_selector(selector: str) -> bool:
        return "path" in selector.lower() or "svg" in selector.lower()

    async def _wait_before_step(self, page: Page, step_timeout: float) -> None:
        """Wait for DOM to be stable before acting (State Synchronizer).
        Reduces flakiness from in-flight updates and spinners.
        """
        idle_ms = self._config.wait_dom_idle_ms
        timeout_ms = min(5000, int(step_timeout * 1000))
        try:
            await page.wait_for_function(
                f"""
                () => new Promise(resolve => {{
                    if (!document.body) {{ resolve(true); return; }}
                    let timer;
                    const done = () => {{ observer.disconnect(); resolve(true); }};
                    const observer = new MutationObserver(() => {{
                        clearTimeout(timer);
                        timer = setTimeout(done, {idle_ms});
                    }});
                    observer.observe(document.body, {{
                        childList: true, subtree: true, attributes: true,
                    }});
                    timer = setTimeout(done, {idle_ms});
                }})
                """,
                timeout=timeout_ms,
            )
        except Exception:
            pass
        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=min(5000, int(self._config.wait_network_idle_ms * 2)),
            )
        except Exception:
            pass

    async def _wait_after_action(
        self,
        page: Page,
        url_before: str,
        expected_navigation: bool = False,
        step: Optional[TestStep] = None,
        step_timeout: float = 30.0,
    ) -> None:
        """Wait for the page to stabilize after a user action.

        Strategy:
          1. If the URL changed → wait for the browser 'load' event.
          2. Wait for network to become idle (API responses).
          3. Wait for DOM mutations to settle (SPA rendering).
        """
        to_ms = int(step_timeout * 1000)
        if page.url != url_before:
            try:
                await page.wait_for_load_state("load", timeout=min(15_000, to_ms))
            except Exception:
                pass

        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=min(10_000, to_ms),
            )
        except Exception:
            pass

        idle_ms = getattr(self._config, "wait_dom_idle_ms", 600)
        try:
            await page.wait_for_function(
                f"""
                () => new Promise(resolve => {{
                    if (!document.body) {{ resolve(true); return; }}
                    let timer;
                    const done = () => {{ observer.disconnect(); resolve(true); }};
                    const observer = new MutationObserver(() => {{
                        clearTimeout(timer);
                        timer = setTimeout(done, Math.max(1000, {idle_ms}));
                    }});
                    observer.observe(document.body, {{
                        childList: true, subtree: true, attributes: true,
                    }});
                    timer = setTimeout(done, Math.max(1000, {idle_ms}));
                }})
                """,
                timeout=min(15_000, to_ms),
            )
        except Exception:
            pass

    async def _wait_for_assertion_targets_if_needed(
        self,
        page: Page,
        step: TestStep,
        expected_navigation: bool,
        step_timeout: float,
    ) -> None:
        """Wait for at least one assertion target to be visible before running
        assertions (retry-until-stable). Runs whenever step has assertions.
        """
        if not step.assertions:
            return
        timeout_s = min(step_timeout, 20.0)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for assertion in step.assertions:
                cand = await self._selector.resolve(page, assertion.fingerprint)
                if cand:
                    try:
                        if await cand.locator.is_visible(timeout=1000):
                            return
                    except Exception:
                        pass
            await asyncio.sleep(0.4)

    # ------------------------------------------------------------------
    # Coordinate click fallback
    # ------------------------------------------------------------------

    async def _try_coordinate_click(
        self, page: Page, step: TestStep, result: StepResult
    ) -> bool:
        """Fall back to clicking at recorded viewport coordinates.

        Only applies to click/dblclick actions that have coordinates stored.
        Returns True if the click was performed.
        """
        action = step.action
        if action.action_type not in (ActionType.CLICK, ActionType.DBLCLICK):
            return False
        if action.click_x is None or action.click_y is None:
            return False

        x, y = action.click_x, action.click_y
        logger.warning(
            "Step %d: DOM resolution failed — falling back to coordinate click at (%d, %d)",
            step.step_id, x, y,
        )

        if action.action_type == ActionType.DBLCLICK:
            await page.mouse.dblclick(x, y)
        else:
            await page.mouse.click(x, y)

        result.element_confidence = 0.0
        result.healing_details = f"coordinate-click at ({x}, {y})"
        return True

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _do_navigate(self, page: Page, step: TestStep) -> None:
        url = step.action.url
        if not url:
            return

        # Skip goto if a preceding click already navigated here
        if page.url == url:
            logger.info("Already on %s — skipping navigate", url)
        else:
            logger.info("Navigating to: %s", url)
            try:
                await page.goto(url, wait_until="load", timeout=15_000)
            except Exception:
                logger.warning("Navigate timed out — proceeding with current page state")

        step_timeout = self._config.step_timeout_ms / 1000.0
        await self._wait_after_action(page, "", False, None, step_timeout)

    async def _perform_action(
        self, page: Page, candidate: SelectorCandidate, step: TestStep
    ) -> None:
        """Perform the user action on the resolved element."""
        locator = candidate.locator
        action = step.action

        match action.action_type:
            case ActionType.CLICK:
                await self._click_with_svg_fallback(page, locator, candidate.selector)
            case ActionType.DBLCLICK:
                await self._click_with_svg_fallback(
                    page, locator, candidate.selector, dblclick=True
                )
            case ActionType.TYPE:
                await self._controlled_fill(page, locator, action.value)
            case ActionType.SELECT:
                await locator.select_option(action.value)
            case ActionType.CHECK:
                await locator.check()
            case ActionType.UNCHECK:
                await locator.uncheck()
            case ActionType.HOVER:
                await locator.hover()
            case ActionType.KEYPRESS:
                await locator.press(action.value)
            case ActionType.SCROLL:
                await self._do_scroll(page, action, locator)
            case _:
                logger.warning("Unhandled action type: %s", action.action_type)

        logger.debug("Performed %s on %s", action.action_type.value, candidate.selector)

    async def _click_with_svg_fallback(
        self,
        page: Page,
        locator,
        selector: str,
        dblclick: bool = False,
    ) -> None:
        """Click the element; if it is SVG/path or click is intercepted, click
        the closest button/link parent instead.
        """
        use_parent = "path" in selector.lower() or "svg" in selector.lower()
        try:
            if not use_parent:
                if dblclick:
                    await locator.dblclick()
                else:
                    await locator.click()
                return
        except Exception as e:
            if "intercepts" not in str(e).lower():
                raise
            use_parent = True

        if use_parent:
            event_type = "dblclick" if dblclick else "click"
            await locator.evaluate(
                f"""
                el => {{
                    const clickable = el.closest('button, a, [role="button"], '
                        + '[role="link"], [tabindex]:not([tabindex="-1"])');
                    const target = clickable || el;
                    target.dispatchEvent(new MouseEvent('{event_type}', {{
                        bubbles: true, cancelable: true, view: window
                    }}));
                }}
                """
            )
        else:
            if dblclick:
                await locator.dblclick()
            else:
                await locator.click()

    async def _controlled_fill(self, page: Page, locator, value: str) -> None:
        """Controlled input: focus, clear, fill, wait until value matches (React-friendly)."""
        await locator.focus()
        await locator.clear()
        await locator.fill(value)
        for _ in range(15):
            try:
                current = (await locator.input_value()).strip()
                if current == value or current.endswith(value):
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)

    async def _do_scroll(self, page: Page, action, locator) -> None:
        """Scroll to the recorded window position, or scroll element into view."""
        import json as _json
        try:
            coords = _json.loads(action.value)
            sx, sy = coords.get("scrollX", 0), coords.get("scrollY", 0)
            await page.evaluate(f"window.scrollTo({sx}, {sy})")
        except Exception:
            if locator is not None:
                await locator.scroll_into_view_if_needed()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    async def _run_assertions(
        self, page: Page, step: TestStep
    ) -> list[AssertionResult]:
        """Evaluate all assertions attached to the step."""
        results: list[AssertionResult] = []
        for assertion in step.assertions:
            ar = await self._assertions.evaluate(page, assertion)
            results.append(ar)
            if ar.status == StepStatus.FAILED:
                logger.warning(
                    "Assertion %s FAILED: %s",
                    assertion.assertion_id,
                    ar.message,
                )
        return results

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    async def _maybe_screenshot(
        self,
        page: Page,
        result: StepResult,
        screenshot_dir: Optional[Path],
    ) -> None:
        if not self._config.screenshot_on_failure or screenshot_dir is None:
            return
        try:
            path = screenshot_dir / f"step_{result.step_id}_failure.png"
            await page.screenshot(path=str(path), full_page=True)
            result.screenshot = str(path)
            logger.info("Failure screenshot: %s", path)
        except Exception as e:
            logger.warning("Screenshot failed: %s", e)
