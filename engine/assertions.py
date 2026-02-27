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
import logging
import re
from typing import TYPE_CHECKING, Optional

from playwright.async_api import Page

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
