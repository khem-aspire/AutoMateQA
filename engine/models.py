"""
Pydantic data models for the Self-Healing Automation Engine.

Defines enums, fingerprints, steps, assertions, config, and result models
that are shared across all engine components.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class ActionType(str, Enum):
    CLICK = "click"
    DBLCLICK = "dblclick"
    TYPE = "type"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    KEYPRESS = "keypress"
    SCROLL = "scroll"
    NAVIGATE = "navigate"


class HealingMode(str, Enum):
    DISABLED = "disabled"
    STRICT = "strict"
    AUTO_UPDATE = "auto_update"
    DEBUG = "debug"


class StepStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    HEALED = "healed"


class AssertionType(str, Enum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    TEXT_EQUALS = "text_equals"
    TEXT_CONTAINS = "text_contains"
    MATCHES_PATTERN = "matches_pattern"
    ATTRIBUTE_EQUALS = "attribute_equals"
    EXISTS = "exists"


# ------------------------------------------------------------------
# Element Fingerprint
# ------------------------------------------------------------------


class ElementFingerprint(BaseModel):
    tag_name: str = ""
    element_id: str = ""
    class_names: list[str] = Field(default_factory=list)
    text_content: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    css_selector: str = ""
    xpath: str = ""
    aria_label: str = ""
    role: str = ""
    parent_tag: str = ""
    sibling_index: int = 0
    nth_of_type: int = 0
    data_testid: str = ""
    placeholder: str = ""
    name: str = ""
    href: str = ""
    # Ranked selectors computed at record time (preferred > role > fallback …)
    selectors: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Action
# ------------------------------------------------------------------


class Action(BaseModel):
    action_type: ActionType = ActionType.CLICK
    value: str = ""
    url: str = ""
    click_x: Optional[float] = None
    click_y: Optional[float] = None
    # Semantic intent recorded at capture time
    intent: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Assertion (attached to a step)
# ------------------------------------------------------------------


class Assertion(BaseModel):
    assertion_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    assertion_type: AssertionType = AssertionType.VISIBLE
    fingerprint: ElementFingerprint = Field(default_factory=ElementFingerprint)
    expected_value: str = ""
    attribute_name: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------
# Test Step
# ------------------------------------------------------------------


class SelectorHeal(BaseModel):
    """Audit entry for a healed selector (preserves the original)."""

    original_selector: str = ""
    healed_selector: str = ""
    healed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    healing_mode: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    strategy: str = ""
    healing_method: str = ""


class TestStep(BaseModel):
    step_id: int = 0
    action: Action = Field(default_factory=Action)
    target: ElementFingerprint = Field(default_factory=ElementFingerprint)
    assertions: list[Assertion] = Field(default_factory=list)
    selector_history: list[SelectorHeal] = Field(default_factory=list)
    screenshot_before: str = ""
    screenshot_after: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------
# Engine Config
# ------------------------------------------------------------------


class EngineConfig(BaseModel):
    llm_enabled: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    healing_mode: HealingMode = HealingMode.DISABLED
    confidence_threshold: float = 0.75
    healing_similarity_threshold: float = 0.6
    max_healing_attempts: int = 2
    screenshot_on_failure: bool = True
    verbose: bool = False
    headless: bool = False
    step_timeout_ms: int = 30_000
    wait_dom_idle_ms: int = 600
    wait_network_idle_ms: int = 500


# ------------------------------------------------------------------
# Test Model (top-level, serialised to / from JSON)
# ------------------------------------------------------------------


class TestModel(BaseModel):
    test_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = "Untitled Test"
    description: str = ""
    base_url: str = ""
    steps: list[TestStep] = Field(default_factory=list)
    config: EngineConfig = Field(default_factory=EngineConfig)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------
# Result Models (produced during execution)
# ------------------------------------------------------------------


class AssertionResult(BaseModel):
    assertion_id: str = ""
    assertion_type: str = ""
    status: StepStatus = StepStatus.PASSED
    message: str = ""
    confidence: float = 0.0
    healed: bool = False


class StepResult(BaseModel):
    step_id: int = 0
    status: StepStatus = StepStatus.PASSED
    element_confidence: float = 0.0
    healed: bool = False
    healing_details: str = ""
    error: str = ""
    assertions: list[AssertionResult] = Field(default_factory=list)
    screenshot: str = ""
    duration_ms: float = 0.0


class TestResult(BaseModel):
    test_id: str = ""
    test_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: StepStatus = StepStatus.PASSED
    steps: list[StepResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    config_used: Optional[EngineConfig] = None
