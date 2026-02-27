"""
SelectorEngine – Resolves element selectors with confidence scoring.

Implements the 5-factor confidence algorithm:
    Confidence = 0.4*T + 0.2*R + 0.15*A + 0.15*P + 0.1*D

Strategies (evaluated in order, best confidence wins):
  1. data-testid (exact unique match)
  2. data-testid + tag (composite)
  3. data-cy / data-test / data-qa (common test attributes)
  4. id (skips dynamic/UUID IDs)
  5. name (form fields)
  6. role + name
  7. aria-label + tag
  8. placeholder
  9. text (exact)
  10. tag + text (filter)
  11. CSS (with dynamic-class penalty)
  12. XPath
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from playwright.async_api import Locator, Page

from engine.models import ElementFingerprint, EngineConfig

logger = logging.getLogger(__name__)

# CSS-in-JS / CSS Modules patterns: these class names are regenerated on
# every build so they must never be trusted for stable selection.
_DYNAMIC_CLASS_RE = re.compile(
    r"^(?:css|sc|emotion|styled|tw)-[a-zA-Z0-9]{3,}$"  # Emotion, styled-components, Tailwind JIT
    r"|^_[a-zA-Z0-9]{5,}$"                               # CSS Modules
    r"|^[a-z]{1,4}[A-Z][a-zA-Z0-9]{3,8}$"               # camelCase hashes (e.g. bIdYaZ)
)

# Dynamic / session-scoped IDs: UUID-based, auto-incrementing, or random
# hex IDs that change every page load.
_DYNAMIC_ID_RE = re.compile(
    r"^f_[0-9a-f-]{8,}$"                    # Quasar-style "f_<uuid>"
    r"|^[0-9a-f]{8}-[0-9a-f]{4}-"           # full UUID
    r"|^\d+$"                                # purely numeric
    r"|^[a-z_-]*[0-9a-f]{8,}"               # prefix + long hex tail
    r"|^:r[0-9a-z]+:$",                      # React useId (e.g. :r1:, :r2a:)
    re.IGNORECASE,
)

# Common testing-library attribute names (in priority order).
_TEST_ATTRS = ["data-cy", "data-test", "data-qa", "data-test-id", "data-testid"]


@dataclass
class SelectorCandidate:
    """A resolved element candidate with its confidence score."""

    locator: Locator
    selector: str
    confidence: float
    strategy: str


class SelectorEngine:
    """Resolves DOM elements using multiple strategies and scores them."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self, page: Page, fingerprint: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        """
        Try multiple selector strategies, score each candidate,
        and return the best one above the confidence threshold.

        Post-filter: when the fingerprint carries text_content, any
        candidate whose live text doesn't overlap is rejected so we
        never return a completely wrong element.
        """
        candidates = await self._generate_candidates(page, fingerprint)

        if not candidates:
            logger.warning(
                "No candidates found for fingerprint: %s", fingerprint.css_selector
            )
            return None

        # Text-validation: reject candidates whose text has zero overlap
        # with the fingerprint's text_content.
        fp_text = (fingerprint.text_content or "").strip()
        if fp_text:
            validated: list[SelectorCandidate] = []
            for c in candidates:
                try:
                    live_text = (await c.locator.text_content(timeout=2000) or "").strip()
                except Exception:
                    live_text = ""
                if self._text_overlaps(fp_text, live_text):
                    validated.append(c)
                else:
                    logger.debug(
                        "Rejected candidate %s — text '%s' doesn't match fingerprint '%s'",
                        c.selector, live_text[:60], fp_text[:60],
                    )
            if validated:
                candidates = validated

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        best = candidates[0]

        logger.info(
            "Best candidate: %s (confidence=%.2f, strategy=%s)",
            best.selector,
            best.confidence,
            best.strategy,
        )

        if best.confidence < self._config.confidence_threshold:
            logger.warning(
                "Best confidence %.2f is below threshold %.2f",
                best.confidence,
                self._config.confidence_threshold,
            )

        return best

    async def resolve_candidates(
        self, page: Page, fingerprint: ElementFingerprint
    ) -> list[SelectorCandidate]:
        """Return all validated candidates sorted by confidence (best first).
        Used by executor to try fallback candidates when the best one fails.
        """
        candidates = await self._generate_candidates(page, fingerprint)
        if not candidates:
            return []

        fp_text = (fingerprint.text_content or "").strip()
        if fp_text:
            validated: list[SelectorCandidate] = []
            for c in candidates:
                try:
                    live_text = (await c.locator.text_content(timeout=2000) or "").strip()
                except Exception:
                    live_text = ""
                if self._text_overlaps(fp_text, live_text):
                    validated.append(c)
            if validated:
                candidates = validated

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    @staticmethod
    def _text_overlaps(expected: str, actual: str) -> bool:
        """Check if the actual text has meaningful overlap with expected.

        Uses word-level intersection — at least 40% of the expected
        words must appear in the actual text.  Single-word texts must
        match exactly (case-insensitive).
        """
        if not expected:
            return True
        if not actual:
            return False
        e_lower = expected.lower()
        a_lower = actual.lower()
        # Exact substring match
        if e_lower in a_lower or a_lower in e_lower:
            return True
        # Word-level overlap
        e_words = set(e_lower.split())
        a_words = set(a_lower.split())
        if not e_words:
            return True
        overlap = len(e_words & a_words)
        ratio = overlap / len(e_words)
        return ratio >= 0.4

    def compute_confidence(self, fingerprint: ElementFingerprint) -> float:
        """
        Static confidence score based on fingerprint richness.

        Confidence = 0.4*T + 0.2*R + 0.15*A + 0.15*P + 0.1*D
        """
        t = self._score_test_id(fingerprint)
        r = self._score_role(fingerprint)
        a = self._score_attributes(fingerprint)
        p = self._score_positional(fingerprint)
        d = self._score_dom_structure(fingerprint)

        score = 0.4 * t + 0.2 * r + 0.15 * a + 0.15 * p + 0.1 * d
        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Dynamic-class detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dynamic_class(cls_name: str) -> bool:
        return bool(_DYNAMIC_CLASS_RE.match(cls_name))

    @staticmethod
    def _has_only_dynamic_classes(fp: ElementFingerprint) -> bool:
        if not fp.class_names:
            return False
        return all(SelectorEngine._is_dynamic_class(c) for c in fp.class_names)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Text disambiguation helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _narrow_by_text(
        locator: Locator, fp: ElementFingerprint
    ) -> Optional[Locator]:
        """When a locator matches multiple elements, use the fingerprint's
        text_content to narrow it down to the exact one.  Returns a
        single-element locator or None if text didn't help."""
        text = (fp.text_content or "").strip()
        if not text:
            return None
        # Use the first 80 chars to avoid overly-long selectors
        filtered = locator.filter(has_text=text[:80])
        if await filtered.count() == 1:
            return filtered
        # Try exact match via get_by_text within the locator
        exact = locator.get_by_text(text[:80], exact=True)
        if await exact.count() == 1:
            return exact
        return None

    async def _generate_candidates(
        self, page: Page, fp: ElementFingerprint
    ) -> list[SelectorCandidate]:
        """Generate candidates using multiple selector strategies.

        When the fingerprint carries pre-computed ``selectors`` (recorded
        at capture time), those are tried first with high confidence.
        """
        candidates: list[SelectorCandidate] = []

        # Pre-computed selectors from recording (highest priority)
        precomputed = await self._strategy_precomputed(page, fp)
        if precomputed:
            candidates.append(precomputed)

        strategies = [
            ("data-testid", self._strategy_testid),
            ("testid+tag", self._strategy_testid_tag),
            ("data-cy", self._strategy_test_attr),
            ("id", self._strategy_id),
            ("name", self._strategy_name),
            ("role+name", self._strategy_role),
            ("aria-label", self._strategy_aria),
            ("placeholder", self._strategy_placeholder),
            ("text-exact", self._strategy_text),
            ("tag+text", self._strategy_tag_text),
            ("css", self._strategy_css),
            ("xpath", self._strategy_xpath),
        ]

        for name, strategy_fn in strategies:
            try:
                result = await strategy_fn(page, fp)
                if result:
                    candidates.append(result)
            except Exception as e:
                logger.debug("Strategy %s failed: %s", name, e)

        return candidates

    # ------------------------------------------------------------------
    # Strategy: pre-computed selectors from recording
    # ------------------------------------------------------------------

    _PRECOMPUTED_PRIORITY = [
        "preferred", "data_cy", "role", "name",
        "placeholder", "label", "text", "fallback",
    ]
    _PRECOMPUTED_CONFIDENCE: dict[str, float] = {
        "preferred": 0.95,
        "data_cy": 0.92,
        "role": 0.88,
        "name": 0.82,
        "placeholder": 0.80,
        "label": 0.84,
        "text": 0.72,
        "fallback": 0.55,
    }

    async def _strategy_precomputed(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        """Try selectors that were computed at record time, in priority order."""
        if not fp.selectors:
            return None

        for key in self._PRECOMPUTED_PRIORITY:
            sel = fp.selectors.get(key)
            if not sel:
                continue
            try:
                locator = page.locator(sel)
                count = await locator.count()
                base_conf = self._PRECOMPUTED_CONFIDENCE.get(key, 0.5)
                if count == 1:
                    return SelectorCandidate(
                        locator=locator,
                        selector=sel,
                        confidence=base_conf,
                        strategy=f"precomputed-{key}",
                    )
                if count > 1:
                    narrowed = await self._narrow_by_text(locator, fp)
                    if narrowed:
                        return SelectorCandidate(
                            locator=narrowed,
                            selector=f"{sel} + text",
                            confidence=round(base_conf * 0.9, 2),
                            strategy=f"precomputed-{key}+text",
                        )
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Strategy: data-testid (exact unique)
    # ------------------------------------------------------------------

    async def _strategy_testid(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.data_testid:
            return None
        selector = f'[data-testid="{fp.data_testid}"]'
        locator = page.locator(selector)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(fp, t=1.0, r=0.8, a=0.9, p=0.8, d=0.7),
                strategy="data-testid",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: tag + data-testid (composite, handles non-unique testids)
    # ------------------------------------------------------------------

    async def _strategy_testid_tag(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.data_testid or not fp.tag_name:
            return None
        selector = f'{fp.tag_name}[data-testid="{fp.data_testid}"]'
        locator = page.locator(selector)
        count = await locator.count()
        if count == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(fp, t=0.95, r=0.7, a=0.9, p=0.7, d=0.7),
                strategy="testid+tag",
            )
        if count > 1:
            narrowed = await self._narrow_by_text(locator, fp)
            if narrowed:
                return SelectorCandidate(
                    locator=narrowed,
                    selector=f"{selector} + text",
                    confidence=self._compute_live_confidence(fp, t=0.85, r=0.6, a=0.8, p=0.7, d=0.6),
                    strategy="testid+tag+text",
                )
            return SelectorCandidate(
                locator=locator.first,
                selector=f"{selector} >> nth=0",
                confidence=self._compute_live_confidence(fp, t=0.8, r=0.5, a=0.7, p=0.8, d=0.5),
                strategy="testid+tag+first",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: data-cy / data-test / data-qa (common test attributes)
    # ------------------------------------------------------------------

    async def _strategy_test_attr(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        """Match using common testing-library attributes stored in fp.attributes."""
        if not fp.attributes:
            return None
        for attr in _TEST_ATTRS:
            val = fp.attributes.get(attr)
            if not val:
                continue
            selector = f'[{attr}="{val}"]'
            locator = page.locator(selector)
            count = await locator.count()
            if count == 1:
                return SelectorCandidate(
                    locator=locator,
                    selector=selector,
                    confidence=self._compute_live_confidence(
                        fp, t=1.0, r=0.8, a=0.9, p=0.8, d=0.7
                    ),
                    strategy=attr,
                )
            # Multiple matches — narrow by text
            if count > 1:
                narrowed = await self._narrow_by_text(locator, fp)
                if narrowed:
                    return SelectorCandidate(
                        locator=narrowed,
                        selector=f'[{attr}="{val}"] + text',
                        confidence=self._compute_live_confidence(
                            fp, t=0.9, r=0.7, a=0.85, p=0.7, d=0.6
                        ),
                        strategy=f"{attr}+text",
                    )
            if fp.tag_name:
                selector = f'{fp.tag_name}[{attr}="{val}"]'
                locator = page.locator(selector)
                count = await locator.count()
                if count == 1:
                    return SelectorCandidate(
                        locator=locator,
                        selector=selector,
                        confidence=self._compute_live_confidence(
                            fp, t=0.95, r=0.7, a=0.9, p=0.7, d=0.7
                        ),
                        strategy=f"{attr}+tag",
                    )
                if count > 1:
                    narrowed = await self._narrow_by_text(locator, fp)
                    if narrowed:
                        return SelectorCandidate(
                            locator=narrowed,
                            selector=f'{fp.tag_name}[{attr}="{val}"] + text',
                            confidence=self._compute_live_confidence(
                                fp, t=0.85, r=0.6, a=0.8, p=0.7, d=0.6
                            ),
                            strategy=f"{attr}+tag+text",
                        )
        return None

    # ------------------------------------------------------------------
    # Strategy: id (skip dynamic/session IDs)
    # ------------------------------------------------------------------

    async def _strategy_id(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.element_id:
            return None
        if _DYNAMIC_ID_RE.match(fp.element_id):
            return None
        selector = f"#{fp.element_id}"
        locator = page.locator(selector)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(fp, t=0.95, r=0.7, a=0.8, p=0.7, d=0.7),
                strategy="id",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: name attribute (stable for form fields)
    # ------------------------------------------------------------------

    async def _strategy_name(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.name:
            return None
        if fp.tag_name:
            selector = f'{fp.tag_name}[name="{fp.name}"]'
        else:
            selector = f'[name="{fp.name}"]'
        locator = page.locator(selector)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(
                    fp, t=0.85, r=0.7, a=0.85, p=0.7, d=0.6
                ),
                strategy="name",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: role + name
    # ------------------------------------------------------------------

    async def _strategy_role(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.role:
            return None
        name = fp.aria_label or fp.text_content[:50] if fp.text_content else None
        locator = (
            page.get_by_role(fp.role, name=name) if name else page.get_by_role(fp.role)
        )
        if await locator.count() == 1:
            selector = f'role={fp.role}[name="{name}"]' if name else f"role={fp.role}"
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(fp, t=0.7, r=1.0, a=0.7, p=0.7, d=0.6),
                strategy="role+name",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: aria-label + tag (stable on dynamic sites)
    # ------------------------------------------------------------------

    async def _strategy_aria(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.aria_label:
            return None
        # Prefer tag-qualified selector for specificity
        if fp.tag_name:
            selector = f'{fp.tag_name}[aria-label="{fp.aria_label}"]'
            locator = page.locator(selector)
            if await locator.count() == 1:
                return SelectorCandidate(
                    locator=locator,
                    selector=selector,
                    confidence=self._compute_live_confidence(
                        fp, t=0.7, r=0.95, a=0.85, p=0.7, d=0.6
                    ),
                    strategy="aria+tag",
                )
        # Fallback: aria-label only
        selector = f'[aria-label="{fp.aria_label}"]'
        locator = page.locator(selector)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(
                    fp, t=0.6, r=0.9, a=0.8, p=0.5, d=0.5
                ),
                strategy="aria-label",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: text (exact match)
    # ------------------------------------------------------------------

    async def _strategy_text(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.text_content:
            return None
        text = fp.text_content[:80]
        locator = page.get_by_text(text, exact=True)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=f'text="{text}"',
                confidence=self._compute_live_confidence(fp, t=0.5, r=0.6, a=0.7, p=0.6, d=0.5),
                strategy="text-exact",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: tag + text (filter — robust on dynamic sites)
    # ------------------------------------------------------------------

    async def _strategy_tag_text(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.text_content or not fp.tag_name:
            return None
        text = fp.text_content[:80]
        locator = page.locator(fp.tag_name).filter(has_text=text)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=f'{fp.tag_name}:has-text("{text}")',
                confidence=self._compute_live_confidence(fp, t=0.45, r=0.5, a=0.65, p=0.6, d=0.5),
                strategy="tag+text",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: placeholder
    # ------------------------------------------------------------------

    async def _strategy_placeholder(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.placeholder:
            return None
        locator = page.get_by_placeholder(fp.placeholder)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=f'placeholder="{fp.placeholder}"',
                confidence=self._compute_live_confidence(fp, t=0.85, r=0.7, a=0.9, p=0.7, d=0.6),
                strategy="placeholder",
            )
        return None

    # ------------------------------------------------------------------
    # Strategy: CSS (with dynamic-class penalty)
    # ------------------------------------------------------------------

    async def _strategy_css(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.css_selector:
            return None

        dynamic_only = self._has_only_dynamic_classes(fp)
        locator = page.locator(fp.css_selector)
        count = await locator.count()

        if count == 1:
            if dynamic_only:
                conf = self._compute_live_confidence(fp, t=0.3, r=0.3, a=0.4, p=0.5, d=0.5)
            else:
                conf = self._compute_live_confidence(fp, t=0.5, r=0.5, a=0.8, p=0.5, d=0.5)
            return SelectorCandidate(
                locator=locator,
                selector=fp.css_selector,
                confidence=conf,
                strategy="css" + (" [dynamic]" if dynamic_only else ""),
            )
        elif count > 1:
            # Try text narrowing first — much more reliable than nth
            narrowed = await self._narrow_by_text(locator, fp)
            if narrowed:
                conf = self._compute_live_confidence(fp, t=0.5, r=0.5, a=0.75, p=0.6, d=0.5)
                return SelectorCandidate(
                    locator=narrowed,
                    selector=f"{fp.css_selector} + text",
                    confidence=conf,
                    strategy="css+text",
                )
            if fp.nth_of_type >= 0:
                locator = locator.nth(fp.nth_of_type)
                if dynamic_only:
                    conf = self._compute_live_confidence(fp, t=0.2, r=0.2, a=0.3, p=0.6, d=0.4)
                else:
                    conf = self._compute_live_confidence(fp, t=0.4, r=0.4, a=0.6, p=0.8, d=0.5)
                return SelectorCandidate(
                    locator=locator,
                    selector=f"{fp.css_selector} >> nth={fp.nth_of_type}",
                    confidence=conf,
                    strategy="css+nth" + (" [dynamic]" if dynamic_only else ""),
                )
        return None

    # ------------------------------------------------------------------
    # Strategy: XPath
    # ------------------------------------------------------------------

    async def _strategy_xpath(
        self, page: Page, fp: ElementFingerprint
    ) -> Optional[SelectorCandidate]:
        if not fp.xpath:
            return None
        selector = f"xpath={fp.xpath}"
        locator = page.locator(selector)
        if await locator.count() == 1:
            return SelectorCandidate(
                locator=locator,
                selector=selector,
                confidence=self._compute_live_confidence(
                    fp, t=0.3, r=0.3, a=0.5, p=0.7, d=0.9
                ),
                strategy="xpath",
            )
        return None

    # ------------------------------------------------------------------
    # Confidence scoring internals
    # ------------------------------------------------------------------

    def _compute_live_confidence(
        self,
        fp: ElementFingerprint,
        t: float = 0.0,
        r: float = 0.0,
        a: float = 0.0,
        p: float = 0.5,
        d: float = 0.5,
    ) -> float:
        """
        Compute confidence with overridable factor scores.
        Confidence = 0.4*T + 0.2*R + 0.15*A + 0.15*P + 0.1*D
        """
        score = 0.4 * t + 0.2 * r + 0.15 * a + 0.15 * p + 0.1 * d
        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Static factor scores (for pre-scoring fingerprints)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_test_id(fp: ElementFingerprint) -> float:
        if fp.data_testid:
            return 1.0
        if fp.element_id:
            return 0.9
        if fp.name:
            return 0.6
        return 0.0

    @staticmethod
    def _score_role(fp: ElementFingerprint) -> float:
        if fp.role and fp.aria_label:
            return 1.0
        if fp.role:
            return 0.7
        if fp.aria_label:
            return 0.5
        return 0.0

    @staticmethod
    def _score_attributes(fp: ElementFingerprint) -> float:
        score = 0.0
        if fp.placeholder:
            score += 0.3
        if fp.href:
            score += 0.2
        if fp.class_names:
            score += 0.2
        if fp.attributes:
            score += min(len(fp.attributes) * 0.05, 0.3)
        return min(score, 1.0)

    @staticmethod
    def _score_positional(fp: ElementFingerprint) -> float:
        if fp.parent_tag:
            return 0.6
        return 0.2

    @staticmethod
    def _score_dom_structure(fp: ElementFingerprint) -> float:
        if fp.xpath:
            return 0.8
        if fp.css_selector:
            return 0.5
        return 0.1
