"""
VisualHealer — screenshot-based healing using vision-capable LLMs.

Layer 3 in the healing pipeline: Cache -> Deterministic -> Visual -> LLM Text.
Uses page screenshots + element description to locate elements visually
when all DOM-based selector strategies have failed.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

from playwright.async_api import Page

from engine.healer import HealingResult
from engine.llm_providers import LLMProvider
from engine.models import ElementFingerprint, EngineConfig

logger = logging.getLogger(__name__)


class VisualHealer:
    """Heals broken selectors by asking a vision model to find the element."""

    def __init__(self, provider: LLMProvider, config: EngineConfig) -> None:
        self._provider = provider
        self._config = config

    async def heal(
        self, page: Page, fingerprint: ElementFingerprint,
    ) -> Optional[HealingResult]:
        try:
            screenshot_bytes = await page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            prompt = self._build_prompt(fingerprint)
            response, tokens = await self._provider.complete_with_image(
                prompt=prompt,
                image_b64=screenshot_b64,
                system="You are a UI element detector. Given a screenshot and element description, return the bounding box as JSON.",
            )

            bbox = self._parse_bbox(response)
            if not bbox:
                return None

            element_info = await self._get_element_at_point(page, bbox["x"], bbox["y"])
            if not element_info:
                return None

            if not self._matches_fingerprint(element_info, fingerprint):
                logger.debug(
                    "Visual heal: element at (%d,%d) doesn't match fingerprint",
                    bbox["x"], bbox["y"],
                )
                return None

            return HealingResult(
                success=True,
                new_selector=f'__visual_coords:{bbox["x"]},{bbox["y"]}',
                confidence=0.72,
                explanation=f"Visual match at ({bbox['x']}, {bbox['y']})",
                strategy="visual",
                healing_method="visual",
                llm_tokens_used=tokens,
            )
        except NotImplementedError:
            logger.debug("Visual healing skipped: provider does not support vision")
            return None
        except Exception as e:
            logger.debug("Visual healing failed: %s", e)
            return None

    def _build_prompt(self, fp: ElementFingerprint) -> str:
        parts = [f"- Element type: <{fp.tag_name}>"]
        if fp.text_content:
            parts.append(f'- Text: "{fp.text_content[:80]}"')
        if fp.role:
            parts.append(f"- ARIA role: {fp.role}")
        if fp.aria_label:
            parts.append(f"- ARIA label: {fp.aria_label}")
        if fp.data_testid:
            parts.append(f"- Test ID: {fp.data_testid}")
        if fp.placeholder:
            parts.append(f"- Placeholder: {fp.placeholder}")

        description = "\n".join(parts)
        return (
            f"Find the UI element matching this description in the screenshot:\n"
            f"{description}\n\n"
            f'Return ONLY a JSON object: {{"x": center_x, "y": center_y, "width": w, "height": h}}\n'
            f"Return ONLY the JSON, nothing else."
        )

    @staticmethod
    def _parse_bbox(response: str) -> Optional[dict]:
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            data = json.loads(cleaned)
            if "x" in data and "y" in data:
                return {
                    "x": int(data["x"]), "y": int(data["y"]),
                    "width": int(data.get("width", 0)),
                    "height": int(data.get("height", 0)),
                }
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    @staticmethod
    async def _get_element_at_point(page: Page, x: int, y: int) -> Optional[dict]:
        try:
            return await page.evaluate(
                """({x, y}) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: (el.textContent || '').trim().substring(0, 80),
                        role: el.getAttribute('role') || '',
                        testid: el.getAttribute('data-testid') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                    };
                }""",
                {"x": x, "y": y},
            )
        except Exception:
            return None

    @staticmethod
    def _matches_fingerprint(element_info: dict, fp: ElementFingerprint) -> bool:
        el_tag = element_info.get("tag", "")
        fp_tag = fp.tag_name.lower()

        if el_tag != fp_tag and fp_tag not in ("path", "svg", "div", "span"):
            return False

        if fp.data_testid and element_info.get("testid") == fp.data_testid:
            return True
        if fp.role and element_info.get("role") == fp.role:
            return True
        if fp.aria_label and element_info.get("ariaLabel") == fp.aria_label:
            return True

        fp_text = (fp.text_content or "").strip().lower()[:50]
        el_text = (element_info.get("text") or "").strip().lower()[:50]
        if fp_text and el_text and (fp_text in el_text or el_text in fp_text):
            return True

        return el_tag == fp_tag
