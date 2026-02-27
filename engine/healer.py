"""
HealingEngine – Configurable LLM-backed self-healing for broken selectors.

When the SelectorEngine fails to resolve an element with sufficient
confidence, the HealingEngine uses an LLM (OpenAI) to suggest a
repaired selector based on the current page DOM and the original
element fingerprint.

Healing modes are controlled by EngineConfig.healing_mode:
  - disabled:    no healing at all
  - strict:      heal but do NOT update the stored selector
  - auto_update: heal AND persist the new selector in the test model
  - debug:       only print suggestions, never act on them
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Optional

from playwright.async_api import Page

from engine.models import (
    ElementFingerprint,
    EngineConfig,
    HealingMode,
)

logger = logging.getLogger(__name__)


@dataclass
class HealingResult:
    """Outcome of a healing attempt."""

    success: bool
    new_selector: str = ""
    confidence: float = 0.0
    explanation: str = ""
    attempts: int = 0
    strategy: str = ""
    healing_method: str = ""
    llm_tokens_used: int = 0
    healed_fingerprint_similarity: float = 0.0


@dataclass
class HealingTelemetry:
    """Step 10: Per-heal metrics for observability."""

    original_selector: str
    healed_selector: str
    original_fingerprint_hash: str
    healed_fingerprint_similarity: float
    healing_method: str
    llm_model: str
    llm_tokens_used: int
    duration_ms: float
    attempts: int
    success: bool


class HealingEngine:
    """LLM-backed healing for broken selectors."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._client = None  # lazily initialised
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def heal(
        self,
        page: Page,
        fingerprint: ElementFingerprint,
        failed_selector: str = "",
    ) -> HealingResult:
        """
        Attempt to heal a broken selector.

        Returns a HealingResult with the suggested selector and confidence.
        """
        mode = self._config.healing_mode

        if mode == HealingMode.DISABLED:
            logger.info("Healing is DISABLED – skipping")
            return HealingResult(success=False, explanation="Healing disabled")

        if not self._config.llm_enabled:
            logger.info("LLM is disabled – cannot heal")
            return HealingResult(success=False, explanation="LLM disabled")

        start = time.monotonic()
        fp_hash = self._fingerprint_hash(fingerprint)

        # Step 1: Check healing cache
        cached = self._cache.get(failed_selector)
        if cached and await self._validate_healed_selector(page, cached):
            logger.info("Healing cache hit for %s -> %s", failed_selector[:50], cached[:50])
            result = HealingResult(
                success=True,
                new_selector=cached,
                confidence=self._config.confidence_threshold,
                explanation="Restored from healing cache",
                attempts=0,
                healing_method="cache",
            )
            self._log_healing_telemetry(failed_selector, result, fp_hash, start)
            return result

        # Step 5: Try deterministic heal before LLM
        det_result = await self._deterministic_heal(page, fingerprint, failed_selector)
        if det_result.success:
            if mode != HealingMode.DEBUG:
                self._cache[failed_selector] = det_result.new_selector
            self._log_healing_telemetry(
                failed_selector, det_result, fp_hash, start
            )
            return det_result

        total_llm_tokens = 0
        for attempt in range(1, self._config.max_healing_attempts + 1):
            logger.info(
                "Healing attempt %d/%d", attempt, self._config.max_healing_attempts
            )

            # Step 3: Re-fetch DOM per attempt; Step 8: scoped by fingerprint
            dom_snippet = await self._get_dom_snippet(page, fingerprint)

            result = await self._ask_llm(
                fingerprint=fingerprint,
                failed_selector=failed_selector,
                dom_snippet=dom_snippet,
                attempt=attempt,
            )
            result.attempts = attempt
            total_llm_tokens += result.llm_tokens_used

            if result.success:
                # Step 6: Validate interactability of healed selector
                validated, fail_reason = await self._validate_healed_selector_with_reason(
                    page, result.new_selector
                )
                if not validated:
                    logger.warning(
                        "LLM suggestion '%s' failed: %s – retrying",
                        result.new_selector,
                        fail_reason,
                    )
                    continue
                # Step 7: Re-validate with fingerprint similarity
                live_fp = await self._extract_live_fingerprint(
                    page, result.new_selector
                )
                fingerprint_threshold = 0.5
                if (fingerprint.tag_name or "").lower() in ("path", "svg"):
                    fingerprint_threshold = 0.25
                if live_fp:
                    similarity = self._compute_fingerprint_similarity(
                        fingerprint, live_fp
                    )
                    result.healed_fingerprint_similarity = similarity
                    if similarity < fingerprint_threshold:
                        logger.warning(
                            "Fingerprint similarity %.2f below threshold %.2f – rejecting LLM suggestion",
                            similarity,
                            fingerprint_threshold,
                        )
                        validated = False
                if validated:
                    if mode == HealingMode.DEBUG:
                        logger.info(
                            "[DEBUG] Healed selector suggestion: %s (not applied)",
                            result.new_selector,
                        )
                        result.success = False
                        result.explanation += " (debug mode – not applied)"
                    else:
                        logger.info("Healed selector: %s", result.new_selector)
                        self._cache[failed_selector] = result.new_selector
                    result.healing_method = "llm"
                    result.llm_tokens_used = total_llm_tokens
                    self._log_healing_telemetry(
                        failed_selector, result, fp_hash, start
                    )
                    return result
                logger.warning(
                    "LLM suggestion '%s' failed fingerprint check – retrying",
                    result.new_selector,
                )

        fail_result = HealingResult(
            success=False,
            explanation="All healing attempts exhausted",
            attempts=self._config.max_healing_attempts,
            llm_tokens_used=total_llm_tokens,
        )
        self._log_healing_telemetry(
            failed_selector, fail_result, fp_hash, start
        )
        return fail_result

    def _fingerprint_hash(self, fp: ElementFingerprint) -> str:
        """Stable hash of fingerprint for telemetry (no PII)."""
        key = f"{fp.tag_name}|{fp.role}|{fp.data_testid}|{fp.name}|{fp.aria_label}|{len(fp.text_content or '')}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _log_healing_telemetry(
        self,
        original_selector: str,
        result: HealingResult,
        fp_hash: str,
        start: float,
    ) -> None:
        """Step 10: Log healing metrics as structured JSON."""
        duration_ms = (time.monotonic() - start) * 1000
        telemetry = HealingTelemetry(
            original_selector=original_selector[:200],
            healed_selector=result.new_selector[:200],
            original_fingerprint_hash=fp_hash,
            healed_fingerprint_similarity=result.healed_fingerprint_similarity,
            healing_method=result.healing_method or "none",
            llm_model=self._config.llm_model,
            llm_tokens_used=result.llm_tokens_used,
            duration_ms=round(duration_ms, 2),
            attempts=result.attempts,
            success=result.success,
        )
        logger.info(
            "healing_telemetry %s",
            json.dumps(asdict(telemetry), default=str),
        )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()  # reads OPENAI_API_KEY from env
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client

    async def _ask_llm(
        self,
        fingerprint: ElementFingerprint,
        failed_selector: str,
        dom_snippet: str,
        attempt: int,
    ) -> HealingResult:
        """Send a healing prompt to OpenAI and parse the response."""
        prompt = self._build_prompt(fingerprint, failed_selector, dom_snippet, attempt)

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert test automation engineer. "
                            "Given a broken CSS/XPath selector and the current page DOM, "
                            "suggest a repaired Playwright selector. "
                            "Respond with ONLY a JSON object:\n"
                            '{"selector": "...", "strategy": "data-testid|role|aria-label|name|text|css", '
                            '"reasoning": "...", "confidence": 0.0-1.0}\n\n'
                            "Selector preference rules (strict priority):\n"
                            "1. Prefer data-testid, data-cy, data-test attributes\n"
                            "2. Prefer role-based selectors (getByRole equivalent)\n"
                            "3. Prefer aria-label, name, placeholder attributes\n"
                            "4. Prefer text-based selectors for unique visible text\n"
                            "5. AVOID nth-child, absolute XPaths, generated class names\n"
                            "6. AVOID selectors depending on DOM structure depth\n"
                            "7. Selector MUST be Playwright-compatible\n"
                            "8. Return a selector for the EXACT same element (same tag and role as in the fingerprint), NOT a parent container."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )

            raw = response.choices[0].message.content or ""
            result = self._parse_llm_response(raw)
            result.llm_tokens_used = (
                response.usage.total_tokens
                if getattr(response, "usage", None) is not None
                else 0
            )
            # Step 2: Enforce confidence threshold
            if result.success and result.confidence < self._config.confidence_threshold:
                result.success = False
                result.explanation = (
                    f"Confidence {result.confidence} below threshold "
                    f"{self._config.confidence_threshold}"
                )
            return result

        except Exception as e:
            logger.error("LLM healing failed: %s", e)
            return HealingResult(
                success=False,
                explanation=f"LLM error: {e}",
            )

    def _build_prompt(
        self,
        fp: ElementFingerprint,
        failed_selector: str,
        dom_snippet: str,
        attempt: int,
    ) -> str:
        return (
            f"## Broken Selector\n"
            f"`{failed_selector}`\n\n"
            f"## Element Fingerprint\n"
            f"- Tag: {fp.tag_name}\n"
            f"- ID: {fp.element_id}\n"
            f"- Classes: {', '.join(fp.class_names)}\n"
            f"- Text: {fp.text_content[:100]}\n"
            f"- data-testid: {fp.data_testid}\n"
            f"- aria-label: {fp.aria_label}\n"
            f"- role: {fp.role}\n"
            f"- placeholder: {fp.placeholder}\n"
            f"- name: {fp.name}\n"
            f"- Original CSS: {fp.css_selector}\n"
            f"- Original XPath: {fp.xpath}\n\n"
            f"## Current Page DOM (partial)\n"
            f"```html\n{dom_snippet}\n```\n\n"
            f"Attempt {attempt}. Suggest a new Playwright-compatible selector. "
            f"Respond with ONLY a JSON object: selector, strategy, reasoning, confidence."
        )

    @staticmethod
    def _parse_llm_response(raw: str) -> HealingResult:
        """Parse the JSON response from the LLM."""
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            data = json.loads(cleaned)
            explanation = data.get("reasoning") or data.get("explanation", "")
            return HealingResult(
                success=True,
                new_selector=data.get("selector", ""),
                confidence=float(data.get("confidence", 0.0)),
                explanation=explanation,
                strategy=data.get("strategy", ""),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Could not parse LLM response: %s", raw[:200])
            return HealingResult(
                success=False,
                explanation=f"Failed to parse LLM response: {e}",
            )

    # ------------------------------------------------------------------
    # DOM helpers
    # ------------------------------------------------------------------

    async def _get_dom_snippet(
        self,
        page: Page,
        fingerprint: Optional[ElementFingerprint] = None,
        max_length: int = 8000,
    ) -> str:
        """Step 8: Get DOM snippet for LLM, scoped to relevant elements when
        fingerprint has specific tag/role to reduce noise and tokens.
        """
        try:
            fp = fingerprint
            tag = (fp.tag_name or "").strip().lower() if fp else ""
            role = (fp.role or "").strip() if fp else ""

            if tag and tag not in ("div", "span") and role:
                html = await page.evaluate(
                    """([tag, role]) => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, svg, noscript').forEach(
                            el => el.remove()
                        );
                        const byTag = clone.querySelectorAll(tag);
                        const byRole = clone.querySelectorAll('[role="' + role + '"]');
                        const frag = [];
                        byTag.forEach(el => frag.push(el.outerHTML));
                        byRole.forEach(el => { if (!frag.includes(el.outerHTML)) frag.push(el.outerHTML); });
                        return frag.join('\\n');
                    }""",
                    [tag, role],
                )
            elif tag and tag not in ("div", "span"):
                html = await page.evaluate(
                    """(tagName) => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, svg, noscript').forEach(
                            el => el.remove()
                        );
                        const els = clone.querySelectorAll(tagName);
                        return Array.from(els).map(el => el.outerHTML).join('\\n');
                    }""",
                    tag,
                )
            elif role:
                html = await page.evaluate(
                    """(r) => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, svg, noscript').forEach(
                            el => el.remove()
                        );
                        const els = clone.querySelectorAll('[role="' + r + '"]');
                        return Array.from(els).map(el => el.outerHTML).join('\\n');
                    }""",
                    role,
                )
            else:
                html = await page.evaluate(
                    """() => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, svg, noscript').forEach(
                            el => el.remove()
                        );
                        return clone.innerHTML;
                    }"""
                )

            if len(html) < 500:
                html = await page.evaluate(
                    """() => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, svg, noscript').forEach(
                            el => el.remove()
                        );
                        return clone.innerHTML;
                    }"""
                )
            if len(html) > max_length:
                html = html[:max_length] + "\n<!-- truncated -->"
            return html
        except Exception:
            return "<could not retrieve DOM>"

    async def _validate_selector(self, page: Page, selector: str) -> bool:
        """Check if a selector actually resolves to at least one element."""
        try:
            count = await page.locator(selector).count()
            return count > 0
        except Exception:
            return False

    async def _validate_healed_selector(self, page: Page, selector: str) -> bool:
        """Step 6: Validate that healed selector resolves to a visible, enabled,
        rendered element. Fails if hidden, disabled, or detached.
        """
        ok, _ = await self._validate_healed_selector_with_reason(page, selector)
        return ok

    async def _validate_healed_selector_with_reason(
        self, page: Page, selector: str
    ) -> tuple[bool, str]:
        """Same as _validate_healed_selector but returns (success, failure_reason).
        Reason is empty when success is True.
        """
        try:
            locator = page.locator(selector)
            count = await locator.count()
            if count == 0:
                return False, "selector matched no elements"
            if count > 1:
                logger.warning("Healed selector matches %d elements, using first", count)
            el = locator.first
            if not await el.is_visible(timeout=2000):
                return False, "element not visible"
            if not await el.is_enabled(timeout=2000):
                return False, "element not enabled"
            if await el.bounding_box() is None:
                return False, "element has no bounding box"
            return True, ""
        except Exception as e:
            return False, f"exception: {e}"

    # ------------------------------------------------------------------
    # Step 4: Fingerprint similarity and live extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fingerprint_similarity(
        original: ElementFingerprint,
        live: ElementFingerprint,
    ) -> float:
        """
        Score similarity between original and live fingerprint (0.0–1.0).
        Weights: tag 0.15, role 0.15, data-testid/cy 0.20, name 0.10,
        text 0.25, class overlap 0.05, attribute overlap 0.10.
        """
        score = 0.0

        # Tag match
        if original.tag_name and live.tag_name:
            score += 0.15 if original.tag_name.lower() == live.tag_name.lower() else 0

        # Role match
        if original.role and live.role:
            score += 0.15 if original.role.lower() == live.role.lower() else 0
        elif not original.role and not live.role:
            score += 0.05

        # data-testid / data-cy
        orig_test = original.data_testid or (original.attributes or {}).get("data-cy", "")
        live_test = live.data_testid or (live.attributes or {}).get("data-cy", "")
        if orig_test and live_test:
            score += 0.20 if orig_test == live_test else 0
        elif not orig_test and not live_test:
            score += 0.05

        # Name match
        if original.name and live.name:
            score += 0.10 if original.name == live.name else 0

        # Text similarity (SequenceMatcher ratio)
        otext = (original.text_content or "").strip()[:200]
        ltext = (live.text_content or "").strip()[:200]
        if otext or ltext:
            score += 0.25 * SequenceMatcher(None, otext.lower(), ltext.lower()).ratio()
        else:
            score += 0.10

        # Class overlap (Jaccard, exclude dynamic-looking classes)
        oclasses = set(c for c in (original.class_names or []) if len(c) < 40)
        lclasses = set(c for c in (live.class_names or []) if len(c) < 40)
        if oclasses or lclasses:
            inter = len(oclasses & lclasses)
            union = len(oclasses | lclasses)
            score += 0.05 * (inter / union if union else 0)
        else:
            score += 0.02

        # Attribute overlap (href, placeholder, aria-label)
        oattrs = {}
        lattrs = {}
        for fp, d in [(original, oattrs), (live, lattrs)]:
            if fp.href:
                d["href"] = fp.href
            if fp.placeholder:
                d["placeholder"] = fp.placeholder
            if fp.aria_label:
                d["aria-label"] = fp.aria_label
            d.update(fp.attributes or {})
        oset = set(oattrs.items())
        lset = set(lattrs.items())
        if oset or lset:
            inter = len(oset & lset)
            union = len(oset | lset)
            score += 0.10 * (inter / union if union else 0)
        else:
            score += 0.03

        return round(min(score, 1.0), 4)

    async def _extract_live_fingerprint(
        self, page: Page, selector: str
    ) -> Optional[ElementFingerprint]:
        """Extract fingerprint of the first element matching selector."""
        try:
            locator = page.locator(selector)
            if await locator.count() == 0:
                return None
            raw = await locator.first.evaluate(
                """(el) => {
                    const attrs = {};
                    for (const a of el.attributes) attrs[a.name] = a.value;
                    return {
                        tag: el.tagName ? el.tagName.toLowerCase() : '',
                        id: el.id || '',
                        classes: [...(el.classList || [])],
                        text: (el.textContent || '').trim().slice(0, 500),
                        role: el.getAttribute('role') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        dataTestid: el.getAttribute('data-testid') || '',
                        dataCy: el.getAttribute('data-cy') || '',
                        name: el.getAttribute('name') || '',
                        placeholder: el.getAttribute('placeholder') || '',
                        href: el.getAttribute('href') || '',
                        attributes: attrs
                    };
                }"""
            )
            if not raw:
                return None
            attrs: dict[str, str] = raw.get("attributes") or {}
            if raw.get("dataTestid"):
                attrs["data-testid"] = raw["dataTestid"]
            if raw.get("dataCy"):
                attrs["data-cy"] = raw["dataCy"]
            return ElementFingerprint(
                tag_name=raw.get("tag") or "",
                element_id=raw.get("id") or "",
                class_names=raw.get("classes") or [],
                text_content=raw.get("text") or "",
                attributes=attrs,
                aria_label=raw.get("ariaLabel") or "",
                role=raw.get("role") or "",
                data_testid=raw.get("dataTestid") or "",
                placeholder=raw.get("placeholder") or "",
                name=raw.get("name") or "",
                href=raw.get("href") or "",
            )
        except Exception as e:
            logger.debug("Could not extract live fingerprint: %s", e)
            return None

    # ------------------------------------------------------------------
    # Step 5: Deterministic healing
    # ------------------------------------------------------------------

    async def _deterministic_heal(
        self,
        page: Page,
        fingerprint: ElementFingerprint,
        failed_selector: str,
    ) -> HealingResult:
        """Try to match the element by fingerprint without LLM. Returns success if
        a candidate scores above healing_similarity_threshold.
        """
        threshold = self._config.healing_similarity_threshold
        tag = (fingerprint.tag_name or "").strip().lower()
        if tag in ("div", "span", ""):
            tag = "*"
        try:
            candidates_raw = await page.evaluate(
                """(tagName) => {
                    const sel = tagName === '*' ? '*' : tagName;
                    const els = document.querySelectorAll(sel);
                    return Array.from(els).slice(0, 50).map(el => ({
                        tag: el.tagName ? el.tagName.toLowerCase() : '',
                        id: el.id || '',
                        classes: [...(el.classList || [])],
                        text: (el.textContent || '').trim().slice(0, 100),
                        role: (el.getAttribute('role') || el.tagName.toLowerCase()) || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        dataTestid: el.getAttribute('data-testid') || '',
                        dataCy: el.getAttribute('data-cy') || '',
                        name: el.getAttribute('name') || '',
                        placeholder: el.getAttribute('placeholder') || '',
                        href: el.getAttribute('href') || ''
                    }));
                }""",
                tag,
            )
        except Exception as e:
            logger.debug("Deterministic heal DOM query failed: %s", e)
            return HealingResult(success=False, explanation=f"DOM query failed: {e}")

        if not candidates_raw:
            return HealingResult(success=False, explanation="No candidates from DOM")

        best_score = 0.0
        best_candidate: Optional[dict[str, Any]] = None
        for raw in candidates_raw:
            live_fp = ElementFingerprint(
                tag_name=raw.get("tag") or "",
                element_id=raw.get("id") or "",
                class_names=raw.get("classes") or [],
                text_content=raw.get("text") or "",
                aria_label=raw.get("ariaLabel") or "",
                role=raw.get("role") or "",
                data_testid=raw.get("dataTestid") or "",
                placeholder=raw.get("placeholder") or "",
                name=raw.get("name") or "",
                href=raw.get("href") or "",
                attributes={"data-cy": raw["dataCy"]} if raw.get("dataCy") else {},
            )
            score = self._compute_fingerprint_similarity(fingerprint, live_fp)
            if score > best_score and score >= threshold:
                best_score = score
                best_candidate = raw

        if not best_candidate:
            return HealingResult(
                success=False,
                explanation="No candidate above similarity threshold",
            )

        selector = self._build_selector_from_candidate(best_candidate)
        if not selector:
            return HealingResult(
                success=False,
                explanation="Could not build selector for best candidate",
            )

        valid = await self._validate_healed_selector(page, selector)
        if not valid:
            return HealingResult(
                success=False,
                explanation="Deterministic selector did not resolve or not interactable",
            )

        logger.info(
            "Deterministic heal matched selector %s (score=%.2f)",
            selector[:60],
            best_score,
        )
        result = HealingResult(
            success=True,
            new_selector=selector,
            confidence=min(1.0, best_score + 0.1),
            explanation=f"Deterministic match (similarity={best_score:.2f})",
            attempts=0,
            healing_method="deterministic",
            healed_fingerprint_similarity=best_score,
        )
        return result

    @staticmethod
    def _build_selector_from_candidate(candidate: dict[str, Any]) -> str:
        """Build a stable Playwright selector from a candidate dict. Prefer
        data-testid > data-cy > role+name > aria-label > name > placeholder > text.
        """
        tag = (candidate.get("tag") or "").lower()
        dt = candidate.get("dataTestid") or ""
        dcy = candidate.get("dataCy") or ""
        role = candidate.get("role") or ""
        name = candidate.get("name") or ""
        aria = candidate.get("ariaLabel") or ""
        placeholder = candidate.get("placeholder") or ""
        text = (candidate.get("text") or "").strip()[:80]

        if dt:
            return f'[data-testid="{dt}"]'
        if dcy:
            return f'[data-cy="{dcy}"]'
        role_name = name or aria or text
        if role and role not in ("div", "span"):
            if role_name:
                return f'role={role}[name="{role_name[:50]}"]'
            return f'[role="{role}"]'
        if name and tag in ("input", "select", "textarea", "button"):
            return f'{tag}[name="{name}"]'
        if aria:
            return f'[aria-label="{aria}"]'
        if name:
            return f'[name="{name}"]'
        if placeholder:
            return f'[placeholder="{placeholder}"]'
        if text and tag:
            return f'{tag}:has-text("{text[:30]}")'
        if tag and tag != "*":
            return tag
        return ""
