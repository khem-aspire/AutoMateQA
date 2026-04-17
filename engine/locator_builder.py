"""
PlaywrightLocatorBuilder — computes Playwright-native locators at record time.

Uses Playwright's accessibility snapshot scoped to each element to get
the browser's own computed accessible name — no DOM heuristics needed.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from playwright.async_api import Locator, Page

from engine.models import ElementFingerprint, EngineConfig

logger = logging.getLogger(__name__)

_IMPLICIT_ROLE: dict[str, str] = {
    "button": "button",
    "a": "link",
    "input": "textbox",
    "select": "combobox",
    "textarea": "textbox",
}

_FORM_TAGS = ("input", "select", "textarea")
_TEXT_ELIGIBLE_TAGS = ("button", "a", "span", "h1", "h2", "h3", "li")


class PlaywrightLocatorBuilder:
    """Enriches fingerprints with Playwright-native locators validated against the live page."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    async def enrich(self, page: Page, fingerprint: ElementFingerprint) -> None:
        """Populate fingerprint.playwright_locators and fingerprint.accessible_name."""
        if not self._config.enrich_with_playwright_locators:
            return

        try:
            # Resolve the element's accessible name using Playwright's own computation
            accessible_name = ""
            if self._config.enrich_accessibility_snapshot:
                accessible_name = await self._get_accessible_name(page, fingerprint)

            locators = self._build_locator_candidates(fingerprint, accessible_name)

            # Validate: only keep locators that resolve to exactly 1 element
            validated: dict[str, str] = {}
            for key, value in locators.items():
                locator = self._make_locator(page, key, value, locators)
                if not locator:
                    continue
                try:
                    if await locator.count() == 1:
                        validated[key] = value
                except Exception:
                    pass

            fingerprint.playwright_locators = validated
            if accessible_name:
                fingerprint.accessible_name = accessible_name

        except Exception as e:
            logger.debug("Enrichment failed: %s", e)

    # ------------------------------------------------------------------
    # Accessible name — use Playwright's scoped accessibility snapshot
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_accessible_name(page: Page, fp: ElementFingerprint) -> str:
        """Get the browser-computed accessible name for this element.

        Uses the AOM (Accessibility Object Model) via computedRole/computedName
        which are supported in Chromium. Falls back to aria-label and labels
        query for cross-browser support.
        """
        selector = (
            fp.selectors.get("preferred")
            or fp.selectors.get("name")
            or fp.selectors.get("placeholder")
            or fp.css_selector
        )
        if not selector:
            return fp.aria_label or ""

        try:
            locator = page.locator(selector)
            if await locator.count() == 0:
                return fp.aria_label or ""

            name = await locator.first.evaluate("""el => {
                // 1. Explicit aria-label — most reliable
                const ariaLabel = el.getAttribute('aria-label');
                if (ariaLabel) return ariaLabel;

                // 2. aria-labelledby — explicit reference
                const ids = el.getAttribute('aria-labelledby');
                if (ids) {
                    const parts = ids.split(/ +/).map(
                        id => (document.getElementById(id) || {}).textContent || ''
                    ).filter(Boolean);
                    if (parts.length) return parts.join(' ').trim();
                }

                // 3. label[for=id] — only when id looks stable (not framework-generated UUIDs)
                //    Quasar/Vue generate ids like "f_2270b449-5db5-..." with wrapping <label>
                //    that contains unrelated UI text (e.g. "Show" toggle buttons)
                if (el.id && !/^f_|[0-9a-f]{8}-[0-9a-f]{4}-/.test(el.id)) {
                    const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                    if (label) {
                        const t = label.textContent.trim();
                        if (t && t.length < 60) return t;
                    }
                }

                // 4. Buttons/links: own text content IS the accessible name
                const tag = el.tagName;
                if (tag === 'BUTTON' || tag === 'A' || el.getAttribute('role') === 'button')
                    return (el.textContent || '').trim().substring(0, 50);

                // No guessing — if there's no explicit label, return empty.
                // Placeholder is NOT an accessible name.
                return '';
            }""")
            return (name or "").strip()
        except Exception:
            return fp.aria_label or ""

    # ------------------------------------------------------------------
    # Build candidate locators (pure — no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_locator_candidates(
        fp: ElementFingerprint, accessible_name: str,
    ) -> dict[str, str]:
        out: dict[str, str] = {}

        # test_id — highest priority
        if fp.data_testid:
            out["test_id"] = fp.data_testid

        # label — only from real label associations
        label_text = ""
        label_sel = fp.selectors.get("label", "")
        if label_sel:
            m = re.search(r'has-text\("([^"]+)"\)', label_sel)
            if m:
                label_text = m.group(1)
        if not label_text and accessible_name and fp.tag_name in _FORM_TAGS:
            # The accessible name from scoped snapshot IS the real computed label.
            # Only use it if it differs from the placeholder (placeholder is not a label).
            if accessible_name != fp.placeholder:
                label_text = accessible_name
        if label_text:
            out["label"] = label_text

        # role + role_name
        role = fp.role or _IMPLICIT_ROLE.get(fp.tag_name, "")
        if fp.tag_name == "input":
            input_type = fp.attributes.get("type", "text")
            if input_type == "checkbox":
                role = "checkbox"
            elif input_type == "radio":
                role = "radio"
        if role:
            out["role"] = role
            role_name = accessible_name or fp.aria_label or ""
            # For inputs without a label, use placeholder as role name
            if not role_name and fp.tag_name in _FORM_TAGS:
                role_name = fp.placeholder or ""
            # For buttons/links, use text content
            if not role_name:
                role_name = (fp.text_content or "")[:50]
            if role_name:
                out["role_name"] = role_name

        # placeholder
        if fp.placeholder:
            out["placeholder"] = fp.placeholder

        # text — short visible text on buttons/links/headings
        if (
            fp.text_content
            and len(fp.text_content) <= 40
            and fp.tag_name in _TEXT_ELIGIBLE_TAGS
        ):
            out["text"] = fp.text_content[:40]

        return out

    # ------------------------------------------------------------------
    # Construct Playwright locator from key/value
    # ------------------------------------------------------------------

    @staticmethod
    def _make_locator(
        page: Page, key: str, value: str, all_keys: dict[str, str],
    ) -> Optional[Locator]:
        match key:
            case "test_id":
                return page.get_by_test_id(value)
            case "label":
                return page.get_by_label(value)
            case "role":
                name = all_keys.get("role_name", "")
                return page.get_by_role(value, name=name) if name else page.get_by_role(value)
            case "placeholder":
                return page.get_by_placeholder(value, exact=True)
            case "text":
                return page.get_by_text(value, exact=True)
            case _:
                return None
