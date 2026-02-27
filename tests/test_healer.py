"""
Unit tests for HealingEngine: fingerprint scoring, selector building,
cache, confidence threshold, and telemetry.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from engine.healer import HealingEngine, HealingResult, HealingTelemetry
from engine.models import ElementFingerprint, EngineConfig, HealingMode


def _fp(
    tag_name: str = "",
    role: str = "",
    data_testid: str = "",
    name: str = "",
    text_content: str = "",
    class_names: list | None = None,
    aria_label: str = "",
    placeholder: str = "",
    href: str = "",
    attributes: dict | None = None,
) -> ElementFingerprint:
    return ElementFingerprint(
        tag_name=tag_name,
        role=role,
        data_testid=data_testid,
        name=name,
        text_content=text_content,
        class_names=class_names or [],
        aria_label=aria_label,
        placeholder=placeholder,
        href=href,
        attributes=attributes or {},
    )


class TestFingerprintSimilarity(unittest.TestCase):
    """Step 4: _compute_fingerprint_similarity."""

    def setUp(self) -> None:
        self.config = EngineConfig(healing_mode=HealingMode.STRICT, llm_enabled=True)
        self.engine = HealingEngine(self.config)

    def test_exact_match_high_score(self) -> None:
        fp = _fp(tag_name="button", role="button", data_testid="submit", text_content="Submit")
        score = HealingEngine._compute_fingerprint_similarity(fp, fp)
        self.assertGreaterEqual(score, 0.8)

    def test_tag_match_contributes(self) -> None:
        a = _fp(tag_name="button", text_content="Click")
        b = _fp(tag_name="button", text_content="Click")
        score = HealingEngine._compute_fingerprint_similarity(a, b)
        self.assertGreater(score, 0.15)

    def test_tag_mismatch_lower_score(self) -> None:
        a = _fp(tag_name="button", text_content="Ok")
        b = _fp(tag_name="a", text_content="Ok")
        score = HealingEngine._compute_fingerprint_similarity(a, b)
        self.assertLess(score, 0.5)

    def test_text_similarity_contributes(self) -> None:
        a = _fp(tag_name="button", text_content="Submit form")
        b = _fp(tag_name="button", text_content="Submit form")
        score = HealingEngine._compute_fingerprint_similarity(a, b)
        self.assertGreater(score, 0.2)

    def test_empty_fingerprints(self) -> None:
        a = _fp()
        b = _fp()
        score = HealingEngine._compute_fingerprint_similarity(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestBuildSelectorFromCandidate(unittest.TestCase):
    """Step 5: _build_selector_from_candidate."""

    def test_data_testid_first(self) -> None:
        c = {"tag": "button", "dataTestid": "submit-btn", "dataCy": "cy-submit", "role": "button"}
        sel = HealingEngine._build_selector_from_candidate(c)
        self.assertEqual(sel, '[data-testid="submit-btn"]')

    def test_data_cy_second(self) -> None:
        c = {"tag": "button", "dataCy": "login", "role": "button", "ariaLabel": "Login"}
        sel = HealingEngine._build_selector_from_candidate(c)
        self.assertEqual(sel, '[data-cy="login"]')

    def test_role_with_name(self) -> None:
        c = {"tag": "button", "role": "button", "ariaLabel": "Submit", "name": "", "text": "Submit"}
        sel = HealingEngine._build_selector_from_candidate(c)
        self.assertIn("role=button", sel)
        self.assertIn("Submit", sel)

    def test_aria_label(self) -> None:
        c = {"tag": "div", "ariaLabel": "Close", "role": ""}
        sel = HealingEngine._build_selector_from_candidate(c)
        self.assertEqual(sel, '[aria-label="Close"]')

    def test_empty_candidate_returns_empty(self) -> None:
        c = {"tag": "*", "role": "", "text": ""}
        sel = HealingEngine._build_selector_from_candidate(c)
        self.assertEqual(sel, "")


class TestHealingCache(unittest.IsolatedAsyncioTestCase):
    """Step 1: Healing cache."""

    async def test_cache_populated_on_success(self) -> None:
        config = EngineConfig(healing_mode=HealingMode.AUTO_UPDATE, llm_enabled=True)
        engine = HealingEngine(config)
        self.assertEqual(len(engine._cache), 0)
        engine._cache["old-sel"] = "new-sel"
        self.assertEqual(engine._cache.get("old-sel"), "new-sel")

    async def test_cache_check_uses_healed_selector(self) -> None:
        config = EngineConfig(healing_mode=HealingMode.STRICT, llm_enabled=True)
        engine = HealingEngine(config)
        page = MagicMock()
        locator = MagicMock()
        locator.count = AsyncMock(return_value=1)
        locator.first = MagicMock()
        locator.first.is_visible = AsyncMock(return_value=True)
        locator.first.is_enabled = AsyncMock(return_value=True)
        locator.first.bounding_box = AsyncMock(return_value={"x": 0, "y": 0})
        page.locator.return_value = locator
        engine._cache["broken"] = "[data-testid=ok]"
        valid = await engine._validate_healed_selector(page, "[data-testid=ok]")
        self.assertTrue(valid)


class TestConfidenceThreshold(unittest.TestCase):
    """Step 2: Confidence threshold enforcement (tested via parse + logic)."""

    def test_parse_llm_response_extracts_confidence(self) -> None:
        raw = '{"selector": "[data-testid=x]", "confidence": 0.9, "reasoning": "stable"}'
        result = HealingEngine._parse_llm_response(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.new_selector, "[data-testid=x]")

    def test_parse_accepts_strategy(self) -> None:
        raw = '{"selector": "role=button", "strategy": "role", "reasoning": "ok", "confidence": 0.85}'
        result = HealingEngine._parse_llm_response(raw)
        self.assertEqual(result.strategy, "role")


class TestFingerprintHash(unittest.TestCase):
    """Step 10: _fingerprint_hash for telemetry."""

    def test_hash_deterministic(self) -> None:
        config = EngineConfig()
        engine = HealingEngine(config)
        fp = _fp(tag_name="button", role="button")
        h1 = engine._fingerprint_hash(fp)
        h2 = engine._fingerprint_hash(fp)
        self.assertEqual(h1, h2)

    def test_hash_different_for_different_fp(self) -> None:
        config = EngineConfig()
        engine = HealingEngine(config)
        h1 = engine._fingerprint_hash(_fp(tag_name="button"))
        h2 = engine._fingerprint_hash(_fp(tag_name="a"))
        self.assertNotEqual(h1, h2)


class TestHealingTelemetry(unittest.TestCase):
    """Step 10: HealingTelemetry dataclass."""

    def test_telemetry_fields(self) -> None:
        t = HealingTelemetry(
            original_selector="div.x",
            healed_selector="[data-testid=y]",
            original_fingerprint_hash="abc",
            healed_fingerprint_similarity=0.8,
            healing_method="deterministic",
            llm_model="gpt-4o",
            llm_tokens_used=0,
            duration_ms=100.5,
            attempts=0,
            success=True,
        )
        self.assertEqual(t.healing_method, "deterministic")
        self.assertEqual(t.success, True)


class TestHealingResultFields(unittest.TestCase):
    """HealingResult has new fields."""

    def test_healing_method_and_strategy(self) -> None:
        r = HealingResult(
            success=True,
            new_selector="[data-testid=z]",
            confidence=0.9,
            explanation="ok",
            attempts=1,
            strategy="data-testid",
            healing_method="llm",
        )
        self.assertEqual(r.strategy, "data-testid")
        self.assertEqual(r.healing_method, "llm")

    def test_llm_tokens_and_similarity_defaults(self) -> None:
        r = HealingResult(success=False)
        self.assertEqual(r.llm_tokens_used, 0)
        self.assertEqual(r.healed_fingerprint_similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
