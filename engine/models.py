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
    DRAG_AND_DROP = "drag_and_drop"
    FILE_UPLOAD = "file_upload"
    RIGHT_CLICK = "right_click"
    FORM_SUBMIT = "form_submit"


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
    CSS_PROPERTY = "css_property"
    ELEMENT_COUNT = "element_count"
    URL_EQUALS = "url_equals"
    URL_CONTAINS = "url_contains"
    CONSOLE_NO_ERRORS = "console_no_errors"
    NETWORK_STATUS = "network_status"
    JS_EXPRESSION = "js_expression"
    ACCESSIBILITY = "accessibility"
    VISUAL_MATCH = "visual_match"
    API_CALL = "api_call"


class ApiAssertionTarget(str, Enum):
    STATUS = "status"
    RESPONSE_JSONPATH = "response_jsonpath"
    REQUEST_JSONPATH = "request_jsonpath"
    RESPONSE_HEADER = "response_header"
    REQUEST_HEADER = "request_header"
    RESPONSE_SCHEMA = "response_schema"
    RESPONSE_TIME_MS = "response_time_ms"


class ApiAssertionOp(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES_REGEX = "matches_regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


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
    # Playwright-native locator expressions computed at record time
    playwright_locators: dict[str, str] = Field(default_factory=dict)
    # Accessible name from ARIA tree (page.accessibility.snapshot)
    accessible_name: str = ""
    # iframe context
    frame_url: str = ""
    frame_name: str = ""
    frame_index: int = -1
    # Shadow DOM context
    shadow_host_selector: str = ""
    is_shadow_dom: bool = False


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
    # Drag-and-drop target fingerprint
    drag_target: Optional[ElementFingerprint] = None
    drag_offset_x: Optional[float] = None
    drag_offset_y: Optional[float] = None
    # File upload metadata
    file_names: list[str] = Field(default_factory=list)
    # Modifier keys held during action (ctrl, shift, alt, meta)
    modifier_keys: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# Assertion (attached to a step)
# ------------------------------------------------------------------


class ApiAssertionSpec(BaseModel):
    method: str = "GET"
    path_template: str = ""
    query_keys_present: list[str] = Field(default_factory=list)

    target: ApiAssertionTarget = ApiAssertionTarget.STATUS
    op: ApiAssertionOp = ApiAssertionOp.EQUALS
    expected: str = ""

    jsonpath: str = ""
    header_name: str = ""
    expected_schema: dict[str, Any] = Field(default_factory=dict)


class Assertion(BaseModel):
    assertion_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    assertion_type: AssertionType = AssertionType.VISIBLE
    fingerprint: ElementFingerprint = Field(default_factory=ElementFingerprint)
    expected_value: str = ""
    attribute_name: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    api_spec: Optional[ApiAssertionSpec] = None


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
    network_context: Optional[NetworkContext] = None


# ------------------------------------------------------------------
# Engine Config
# ------------------------------------------------------------------


class NetworkContext(BaseModel):
    """Network activity captured during a step (Phase 9)."""

    api_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_at_action: int = 0
    critical_url_pattern: str = ""


class EngineConfig(BaseModel):
    # LLM / healing
    llm_enabled: bool = False
    llm_provider: str = "openai"  # openai | anthropic | google | local
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str = ""  # for local LLMs (ollama, vllm)
    llm_max_tokens_per_heal: int = 2000
    llm_budget_tokens: int = 50_000
    llm_fallback_provider: str = ""
    llm_fallback_model: str = ""
    healing_mode: HealingMode = HealingMode.DISABLED
    confidence_threshold: float = 0.75
    healing_similarity_threshold: float = 0.6
    max_healing_attempts: int = 2
    # Screenshots & debug
    screenshot_on_failure: bool = True
    verbose: bool = False
    headless: bool = False
    # Timeouts
    step_timeout_ms: int = 30_000
    wait_dom_idle_ms: int = 600
    wait_network_idle_ms: int = 500
    timeout_navigate_ms: int = 30_000
    timeout_click_ms: int = 15_000
    timeout_type_ms: int = 10_000
    timeout_default_ms: int = 20_000
    # Recording enrichment (Phase 2/3)
    enrich_with_playwright_locators: bool = True
    enrich_accessibility_snapshot: bool = True
    # HAR / trace / video (Phase 5)
    capture_network: bool = False  # per-step API call capture (opt-in)
    record_har: bool = False
    har_path: str = "trace.har"
    record_trace: bool = False
    trace_path: str = "trace.zip"
    record_video: bool = False
    video_dir: str = "videos"
    # Browser / device (Phase 12)
    browser_type: str = "chromium"  # chromium | firefox | webkit
    device_name: str = ""
    viewport_width: int = 0
    viewport_height: int = 0
    locale: str = ""
    timezone: str = ""
    storage_state_path: str = ""
    save_storage_state: bool = False
    # Retry / flakiness (Phase 10)
    retry_strategy: str = "exponential"  # none | linear | exponential
    max_step_retries: int = 3
    retry_base_delay_ms: int = 500
    max_test_retries: int = 1
    # Reporting (Phase 19)
    report_format: str = ""  # html | junit | json | ""
    report_path: str = ""
    # API assertions (Phase 1)
    api_assertions_enabled: bool = True
    api_assertion_match_timeout_ms: int = 10_000
    redact_in_ui: bool = True


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
    diagnostic: dict[str, Any] = Field(default_factory=dict)


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
    retry_count: int = 0
    flakiness_score: float = 0.0
    action_type: str = ""


class TestResult(BaseModel):
    test_id: str = ""
    test_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: StepStatus = StepStatus.PASSED
    steps: list[StepResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    config_used: Optional[EngineConfig] = None
    tokens_used: int = 0
    healed_count: int = 0
    failed_count: int = 0


class TestSuite(BaseModel):
    """Collection of tests to run as a batch (Phase 18)."""

    suite_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    test_paths: list[str] = Field(default_factory=list)
    shared_config: Optional[EngineConfig] = None
    parallel: bool = False
    max_workers: int = 4
    shared_storage_state: str = ""
    stop_on_first_failure: bool = False
