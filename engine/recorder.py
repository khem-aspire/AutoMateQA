"""
RecorderEngine – Captures user actions and assertions during recording.

Listens for:
  - Console messages from the injected recorder JS (clicks, inputs, keypresses).
  - Assertion payloads from the assertion layer (via BrowserManager callback).
  - Navigation events (suppressed when caused by a recent user action).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from playwright.async_api import Page

from engine.models import (
    Action,
    ActionType,
    Assertion,
    AssertionType,
    ElementFingerprint,
    TestModel,
    TestStep,
)

logger = logging.getLogger(__name__)


class RecorderEngine:
    """Records user interactions into a TestModel."""

    def __init__(self, model: TestModel) -> None:
        self._model = model
        self._page: Optional[Page] = None
        self._recording = False
        self._step_counter = 0
        # All framenavigated events are suppressed — navigations are always
        # a side-effect of user actions (click, submit) and are NOT recorded
        # as separate steps.  Assertions on the destination page attach to
        # the action step that caused the navigation.
        self._suppress_nav = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, page: Page) -> None:
        """Begin recording on the given page."""
        self._page = page
        self._recording = True
        self._step_counter = len(self._model.steps)
        # Suppress initial redirects (e.g. http→https, www redirect)
        self._suppress_nav = True

        page.on("console", self._handle_console)
        page.on("framenavigated", self._handle_navigation)

        logger.info("Recorder started")

    def stop(self) -> TestModel:
        """Stop recording and return the final model."""
        self._recording = False
        logger.info("Recorder stopped – %d steps captured", len(self._model.steps))
        return self._model

    def handle_assertion(self, payload: dict) -> None:
        """
        Called by BrowserManager when an assertion is received from the JS layer.
        Attaches the assertion to the most recent user-action step.

        Even if the page navigated (e.g. click on a link), the assertion
        goes on the click step — during execution the engine waits for the
        destination page to load before evaluating assertions.
        """
        if not self._recording:
            return

        assertion_type = payload.get("assertion_type", "visible")
        fp_data = payload.get("fingerprint", {})

        assertion = Assertion(
            assertion_type=AssertionType(assertion_type),
            fingerprint=ElementFingerprint(**fp_data),
            expected_value=payload.get("value", ""),
            attribute_name=payload.get("attribute_name", ""),
        )

        if self._model.steps:
            self._model.steps[-1].assertions.append(assertion)
            logger.info(
                "Assertion '%s' attached to step %d",
                assertion_type,
                self._model.steps[-1].step_id,
            )
        else:
            step = self._new_step(
                action=Action(action_type=ActionType.CLICK),
                fingerprint=ElementFingerprint(**fp_data),
            )
            step.assertions.append(assertion)
            self._model.steps.append(step)
            logger.info("Assertion '%s' created new step %d", assertion_type, step.step_id)

    # ------------------------------------------------------------------
    # Console message handler
    # ------------------------------------------------------------------

    def _handle_console(self, msg) -> None:
        """Parse console messages from the recorder JS."""
        if not self._recording:
            return

        text: str = msg.text
        if not text.startswith("__RECORDER__:"):
            return

        try:
            data = json.loads(text[len("__RECORDER__:"):])
        except json.JSONDecodeError:
            logger.warning("Invalid recorder payload: %s", text[:200])
            return

        action_str = data.get("action", "click")
        fp_data = data.get("fingerprint", {})
        value = data.get("value", "")
        url = data.get("url", "")

        action_type = self._map_action_type(action_str)
        fingerprint = ElementFingerprint(**fp_data)

        intent_data = data.get("intent", {})

        # Coalesce type: if last step is type on same field and new value extends old, update it
        if action_type == ActionType.TYPE and self._model.steps:
            last = self._model.steps[-1]
            if (
                last.action.action_type == ActionType.TYPE
                and self._same_target(last.target, fingerprint)
                and (last.action.value == value or (value and value.startswith(last.action.value)))
            ):
                last.action.value = value
                self._suppress_nav = True
                logger.info(
                    "Coalesced type step: value updated to %r (step %d)",
                    value[:50],
                    last.step_id,
                )
                return

        action = Action(
            action_type=action_type,
            value=value,
            url=url,
            click_x=data.get("click_x"),
            click_y=data.get("click_y"),
            intent=intent_data,
        )
        step = self._new_step(action=action, fingerprint=fingerprint)
        self._model.steps.append(step)
        self._suppress_nav = True

        preferred = fingerprint.selectors.get("preferred", fingerprint.css_selector)
        logger.info(
            "Recorded step %d: %s (selector=%s)",
            step.step_id,
            action_type.value,
            preferred,
        )

    @staticmethod
    def _same_target(a: ElementFingerprint, b: ElementFingerprint) -> bool:
        """True if both fingerprints refer to the same input/field."""
        def key(fp: ElementFingerprint) -> str:
            return (
                fp.selectors.get("preferred", "")
                or fp.attributes.get("data-cy", "")
                or fp.attributes.get("data-testid", "")
                or fp.data_testid
                or fp.name
                or fp.placeholder
                or fp.css_selector
                or ""
            )
        ka, kb = key(a), key(b)
        return bool(ka and kb and ka == kb)

    # ------------------------------------------------------------------
    # Navigation handler
    # ------------------------------------------------------------------

    def _handle_navigation(self, frame) -> None:
        """Record navigation events (top-frame only).

        Navigations that happen shortly after a user action (click, submit, etc.)
        are suppressed — the action itself already implies the navigation.
        """
        if not self._recording:
            return

        if frame != self._page.main_frame:
            return

        url = frame.url
        if not url or url == "about:blank":
            return

        if self._suppress_nav:
            logger.debug("Suppressed navigation (caused by user action): %s", url)
            return

        # Avoid duplicate consecutive navigations to the same URL
        if self._model.steps:
            last = self._model.steps[-1]
            if (
                last.action.action_type == ActionType.NAVIGATE
                and last.action.url == url
            ):
                return

        action = Action(action_type=ActionType.NAVIGATE, url=url)
        step = self._new_step(action=action, fingerprint=ElementFingerprint())
        self._model.steps.append(step)
        logger.info("Recorded navigation → %s (step %d)", url, step.step_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_step(self, action: Action, fingerprint: ElementFingerprint) -> TestStep:
        self._step_counter += 1
        return TestStep(
            step_id=self._step_counter,
            action=action,
            target=fingerprint,
        )

    @staticmethod
    def _map_action_type(action_str: str) -> ActionType:
        mapping = {
            "click": ActionType.CLICK,
            "dblclick": ActionType.DBLCLICK,
            "type": ActionType.TYPE,
            "select": ActionType.SELECT,
            "check": ActionType.CHECK,
            "uncheck": ActionType.UNCHECK,
            "hover": ActionType.HOVER,
            "keypress": ActionType.KEYPRESS,
            "scroll": ActionType.SCROLL,
            "navigate": ActionType.NAVIGATE,
        }
        return mapping.get(action_str, ActionType.CLICK)
