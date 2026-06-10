"""
AssertionEngine – Evaluates assertions attached to test steps.

Supports 7 assertion types:
  visible, hidden, text_equals, text_contains,
  matches_pattern, attribute_equals, exists

Spec compliance (sections 9 & 11):
  - Resolve assertion target and compute confidence.
  - If assertion fails AND element confidence is below threshold,
    retry resolution via HealingEngine before declaring failure.
  - Never auto-heal expected values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Optional

from playwright.async_api import Page

from engine.api_assertion import ApiAssertionEvaluator
from engine.models import (
    Assertion,
    AssertionResult,
    AssertionType,
    EngineConfig,
    HealingMode,
    StepStatus,
)
from engine.selector import SelectorCandidate, SelectorEngine

if TYPE_CHECKING:
    from engine.healer import HealingEngine

logger = logging.getLogger(__name__)


class AssertionEngine:
    """Evaluates assertions against a live Playwright page."""

    def __init__(
        self,
        config: EngineConfig,
        selector_engine: SelectorEngine,
        healing_engine: Optional[HealingEngine] = None,
    ) -> None:
        self._config = config
        self._selector = selector_engine
        self._healer = healing_engine
        self._console_messages: list = []  # captured during execution
        self._responses: list = []  # captured network responses
        self._api_evaluator = ApiAssertionEvaluator(config)

    def attach_listeners(self, page) -> None:
        """Attach console and response listeners for assertion types that need them."""
        self._console_messages.clear()
        self._responses.clear()
        page.on("console", lambda msg: self._console_messages.append(msg))
        page.on("response", lambda resp: self._responses.append(resp))
        self._api_evaluator.start_step_window(page)

    def detach_listeners(self) -> None:
        """Remove the per-step API evaluator listener."""
        self._api_evaluator.end_step_window()

    # Max time (seconds) to poll for an assertion target element that
    # doesn't exist in the DOM yet (SPA still rendering).
    _ELEMENT_WAIT_TIMEOUT = 15.0
    _ELEMENT_POLL_INTERVAL = 0.5

    async def evaluate(self, page: Page, assertion: Assertion) -> AssertionResult:
        """
        Evaluate a single assertion with element-wait and healing fallback.

        Flow:
          1. Poll for the target element (up to _ELEMENT_WAIT_TIMEOUT).
          2. Evaluate assertion condition.
          3. If FAILED *and* confidence < threshold → heal target → re-evaluate.
          4. Return result (never mutates expected_value).
        """
        # API_CALL assertions skip element resolution / healing entirely
        if assertion.assertion_type == AssertionType.API_CALL:
            api_result = await self._api_evaluator.evaluate(page, assertion)
            api_result.assertion_id = assertion.assertion_id
            return api_result

        result = AssertionResult(
            assertion_id=assertion.assertion_id,
            assertion_type=assertion.assertion_type.value,
        )

        # Phase 1: poll for the target element (SPA pages may still be rendering)
        candidate = await self._resolve_with_retry(page, assertion)
        result.confidence = candidate.confidence if candidate else 0.0

        # Phase 2: evaluate condition
        await self._dispatch(assertion, result, candidate)

        # Phase 3: if failed due to low-confidence element, attempt healing
        if (
            result.status == StepStatus.FAILED
            and result.confidence < self._config.confidence_threshold
            and self._should_heal()
        ):
            healed_candidate = await self._heal_assertion_target(page, assertion, candidate)
            if healed_candidate is not None:
                result.confidence = healed_candidate.confidence
                result.healed = True
                result.status = StepStatus.PASSED
                result.message = ""
                await self._dispatch(assertion, result, healed_candidate)

        return result

    async def _resolve_with_retry(
        self, page: Page, assertion: Assertion
    ) -> Optional[SelectorCandidate]:
        """Poll for the assertion target, giving the page time to render.

        Returns immediately once the element is found (any confidence).
        Only keeps polling when the element is completely absent from the
        DOM — i.e. the SPA hasn't rendered it yet.  Confidence improvements
        are NOT worth polling for; that's the healing engine's job.
        """
        deadline = asyncio.get_event_loop().time() + self._ELEMENT_WAIT_TIMEOUT

        while True:
            candidate = await self._selector.resolve(page, assertion.fingerprint)
            if candidate is not None:
                return candidate

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(self._ELEMENT_POLL_INTERVAL, remaining))

        return None

    # ------------------------------------------------------------------
    # Dispatch to assertion type
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        assertion: Assertion,
        result: AssertionResult,
        candidate: Optional[SelectorCandidate],
    ) -> None:
        try:
            match assertion.assertion_type:
                case AssertionType.VISIBLE:
                    await self._assert_visible(assertion, result, candidate)
                case AssertionType.HIDDEN:
                    await self._assert_hidden(assertion, result, candidate)
                case AssertionType.TEXT_EQUALS:
                    await self._assert_text_equals(assertion, result, candidate)
                case AssertionType.TEXT_CONTAINS:
                    await self._assert_text_contains(assertion, result, candidate)
                case AssertionType.MATCHES_PATTERN:
                    await self._assert_matches_pattern(assertion, result, candidate)
                case AssertionType.ATTRIBUTE_EQUALS:
                    await self._assert_attribute_equals(assertion, result, candidate)
                case AssertionType.EXISTS:
                    await self._assert_exists(assertion, result, candidate)
                case AssertionType.CSS_PROPERTY:
                    await self._assert_css_property(assertion, result, candidate)
                case AssertionType.ELEMENT_COUNT:
                    await self._assert_element_count(assertion, result, candidate)
                case AssertionType.URL_EQUALS:
                    await self._assert_url(assertion, result, candidate, exact=True)
                case AssertionType.URL_CONTAINS:
                    await self._assert_url(assertion, result, candidate, exact=False)
                case AssertionType.CONSOLE_NO_ERRORS:
                    await self._assert_console_no_errors(assertion, result, candidate)
                case AssertionType.NETWORK_STATUS:
                    await self._assert_network_status(assertion, result, candidate)
                case AssertionType.JS_EXPRESSION:
                    await self._assert_js_expression(assertion, result, candidate)
                case AssertionType.ACCESSIBILITY:
                    await self._assert_accessibility(assertion, result, candidate)
                case AssertionType.VISUAL_MATCH:
                    await self._assert_visual_match(assertion, result, candidate)
                case _:
                    result.status = StepStatus.FAILED
                    result.message = f"Unknown assertion type: {assertion.assertion_type}"
        except Exception as e:
            result.status = StepStatus.FAILED
            result.message = f"Assertion error: {e}"
            logger.error("Assertion %s failed: %s", assertion.assertion_id, e)

    # ------------------------------------------------------------------
    # Healing fallback for assertion targets
    # ------------------------------------------------------------------

    def _should_heal(self) -> bool:
        return (
            self._healer is not None
            and self._config.llm_enabled
            and self._config.healing_mode != HealingMode.DISABLED
        )

    async def _heal_assertion_target(
        self,
        page: Page,
        assertion: Assertion,
        original_candidate: Optional[SelectorCandidate],
    ) -> Optional[SelectorCandidate]:
        """Ask HealingEngine for a better selector for the assertion target."""
        assert self._healer is not None

        failed_selector = (
            original_candidate.selector
            if original_candidate
            else assertion.fingerprint.css_selector
        )

        logger.info(
            "Assertion %s: confidence %.2f < %.2f – attempting assertion-target healing",
            assertion.assertion_id,
            original_candidate.confidence if original_candidate else 0.0,
            self._config.confidence_threshold,
        )

        healing = await self._healer.heal(
            page,
            assertion.fingerprint,
            failed_selector=failed_selector,
        )

        if not healing.success:
            return None

        healed_locator = page.locator(healing.new_selector)
        if await healed_locator.count() == 0:
            return None

        logger.info(
            "Assertion %s healed: %s → %s",
            assertion.assertion_id,
            failed_selector,
            healing.new_selector,
        )

        return SelectorCandidate(
            locator=healed_locator,
            selector=healing.new_selector,
            confidence=healing.confidence,
            strategy="healed",
        )

    # ------------------------------------------------------------------
    # Assertion implementations (accept pre-resolved candidate)
    # ------------------------------------------------------------------

    async def _assert_visible(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate and await candidate.locator.is_visible():
            result.status = StepStatus.PASSED
            result.message = "Element is visible"
        else:
            result.status = StepStatus.FAILED
            result.message = "Element is not visible"

    async def _assert_hidden(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None or not await candidate.locator.is_visible():
            result.status = StepStatus.PASSED
            result.message = "Element is hidden"
        else:
            result.status = StepStatus.FAILED
            result.message = "Element is visible (expected hidden)"

    async def _assert_text_equals(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        text = (await candidate.locator.text_content() or "").strip()
        if text == assertion.expected_value:
            result.status = StepStatus.PASSED
            result.message = f"Text matches: '{text}'"
        else:
            result.status = StepStatus.FAILED
            result.message = (
                f"Text mismatch: expected '{assertion.expected_value}', got '{text}'"
            )

    async def _assert_text_contains(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        text = (await candidate.locator.text_content() or "").strip()
        if assertion.expected_value in text:
            result.status = StepStatus.PASSED
            result.message = f"Text contains '{assertion.expected_value}'"
        else:
            result.status = StepStatus.FAILED
            result.message = (
                f"Text '{text}' does not contain '{assertion.expected_value}'"
            )

    async def _assert_matches_pattern(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        text = (await candidate.locator.text_content() or "").strip()
        if re.search(assertion.expected_value, text):
            result.status = StepStatus.PASSED
            result.message = f"Text matches pattern '{assertion.expected_value}'"
        else:
            result.status = StepStatus.FAILED
            result.message = (
                f"Text '{text}' does not match pattern '{assertion.expected_value}'"
            )

    async def _assert_attribute_equals(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        actual = await candidate.locator.get_attribute(assertion.attribute_name)
        if actual == assertion.expected_value:
            result.status = StepStatus.PASSED
            result.message = (
                f"Attribute '{assertion.attribute_name}' = '{assertion.expected_value}'"
            )
        else:
            result.status = StepStatus.FAILED
            result.message = (
                f"Attribute '{assertion.attribute_name}': "
                f"expected '{assertion.expected_value}', got '{actual}'"
            )

    async def _assert_exists(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate and await candidate.locator.count() > 0:
            result.status = StepStatus.PASSED
            result.message = "Element exists in DOM"
        else:
            result.status = StepStatus.FAILED
            result.message = "Element does not exist in DOM"

    # ------------------------------------------------------------------
    # Phase 16: New assertion types
    # ------------------------------------------------------------------

    async def _assert_css_property(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        prop_name = assertion.attribute_name
        if not prop_name:
            result.status = StepStatus.FAILED
            result.message = "No CSS property name specified"
            return
        actual = await candidate.locator.evaluate(
            f"el => getComputedStyle(el).getPropertyValue('{prop_name}')"
        )
        actual = (actual or "").strip()
        if actual == assertion.expected_value.strip():
            result.status = StepStatus.PASSED
            result.message = f"CSS {prop_name} = '{actual}'"
        else:
            result.status = StepStatus.FAILED
            result.message = f"CSS {prop_name}: expected '{assertion.expected_value}', got '{actual}'"

    async def _assert_element_count(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        count = await candidate.locator.count()
        try:
            expected = int(assertion.expected_value)
        except ValueError:
            result.status = StepStatus.FAILED
            result.message = f"Invalid expected count: {assertion.expected_value}"
            return
        if count == expected:
            result.status = StepStatus.PASSED
            result.message = f"Element count = {count}"
        else:
            result.status = StepStatus.FAILED
            result.message = f"Element count: expected {expected}, got {count}"

    async def _assert_url(
        self, assertion: Assertion, result: AssertionResult,
        candidate: Optional[SelectorCandidate], exact: bool = True,
    ) -> None:
        # URL assertions use the page stored on the candidate's locator
        try:
            current_url = candidate.locator.page.url if candidate else ""
        except Exception:
            current_url = ""
        if not current_url:
            result.status = StepStatus.FAILED
            result.message = "Could not determine current URL"
            return
        if exact:
            if current_url == assertion.expected_value:
                result.status = StepStatus.PASSED
                result.message = f"URL matches: {current_url}"
            else:
                result.status = StepStatus.FAILED
                result.message = f"URL mismatch: expected '{assertion.expected_value}', got '{current_url}'"
        else:
            if assertion.expected_value in current_url:
                result.status = StepStatus.PASSED
                result.message = f"URL contains '{assertion.expected_value}'"
            else:
                result.status = StepStatus.FAILED
                result.message = f"URL '{current_url}' does not contain '{assertion.expected_value}'"

    async def _assert_console_no_errors(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        errors = [m for m in self._console_messages if hasattr(m, 'type') and m.type == "error"]
        if not errors:
            result.status = StepStatus.PASSED
            result.message = "No console errors"
        else:
            error_texts = [m.text[:100] for m in errors[:5]]
            result.status = StepStatus.FAILED
            result.message = f"{len(errors)} console error(s): {'; '.join(error_texts)}"

    async def _assert_network_status(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        url_pattern = assertion.attribute_name
        expected_status = assertion.expected_value
        if not url_pattern:
            result.status = StepStatus.FAILED
            result.message = "No URL pattern specified in attribute_name"
            return
        matching = [r for r in self._responses if url_pattern in r.url]
        if not matching:
            result.status = StepStatus.FAILED
            result.message = f"No response matching '{url_pattern}'"
            return
        last_status = str(matching[-1].status)
        if last_status == expected_status:
            result.status = StepStatus.PASSED
            result.message = f"API {url_pattern} returned {last_status}"
        else:
            result.status = StepStatus.FAILED
            result.message = f"API {url_pattern}: expected {expected_status}, got {last_status}"

    async def _assert_js_expression(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        if not assertion.expected_value:
            result.status = StepStatus.FAILED
            result.message = "No JS expression specified"
            return
        try:
            page = candidate.locator.page if candidate else None
            if page is None:
                result.status = StepStatus.FAILED
                result.message = "No page reference available"
                return
            value = await page.evaluate(assertion.expected_value)
            if value:
                result.status = StepStatus.PASSED
                result.message = f"JS expression returned truthy: {value}"
            else:
                result.status = StepStatus.FAILED
                result.message = f"JS expression returned falsy: {value}"
        except Exception as e:
            result.status = StepStatus.FAILED
            result.message = f"JS expression error: {e}"

    async def _assert_accessibility(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        """Basic accessibility check — verifies element has accessible name and role."""
        if candidate is None:
            result.status = StepStatus.FAILED
            result.message = "Element not found"
            return
        info = await candidate.locator.evaluate("""el => ({
            role: el.getAttribute('role') || el.tagName.toLowerCase(),
            ariaLabel: el.getAttribute('aria-label') || '',
            tabIndex: el.tabIndex,
        })""")
        issues = []
        if not info.get("ariaLabel") and not info.get("role"):
            issues.append("missing role and aria-label")
        if info.get("tabIndex", -1) < 0 and info.get("role") in ("button", "link"):
            issues.append("interactive element not keyboard-focusable")
        if issues:
            result.status = StepStatus.FAILED
            result.message = f"Accessibility issues: {', '.join(issues)}"
        else:
            result.status = StepStatus.PASSED
            result.message = "Element is accessible"

    async def _assert_visual_match(
        self, assertion: Assertion, result: AssertionResult, candidate: Optional[SelectorCandidate]
    ) -> None:
        """Visual regression placeholder — requires VisualComparator integration."""
        result.status = StepStatus.PASSED
        result.message = "Visual match assertion (baseline management not yet configured)"
