"""
Engine configuration via pydantic-settings.

Reads from environment variables and .env file.
CLI flags override these values when provided.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoMateQASettings(BaseSettings):
    """Settings loaded from env vars and .env file.

    Each field maps to a specific env var via validation_alias.
    CLI flags take priority over these when both are provided.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openai",
        validation_alias="AQA_LLM_PROVIDER",
        description="LLM provider for healing: openai, anthropic, local.",
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias="AQA_LLM_MODEL",
        description="LLM model name used for healing.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key. Required when llm_provider=openai.",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key. Required when llm_provider=anthropic.",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        validation_alias="AQA_LLM_BASE_URL",
        description="Custom base URL for local LLMs (Ollama, vLLM). E.g. http://localhost:11434/v1",
    )

    # ── Healing ──────────────────────────────────────────────────────
    healing_mode: str = Field(
        default="disabled",
        validation_alias="AQA_HEALING_MODE",
        description="Healing mode: disabled, strict, auto_update, debug.",
    )
    confidence_threshold: float = Field(
        default=0.75,
        validation_alias="AQA_CONFIDENCE_THRESHOLD",
        description="Minimum confidence score to accept a selector match (0.0-1.0).",
    )

    # ── Browser ──────────────────────────────────────────────────────
    browser_type: str = Field(
        default="chromium",
        validation_alias="AQA_BROWSER",
        description="Browser engine: chromium, firefox, webkit.",
    )
    headless: bool = Field(
        default=False,
        validation_alias="AQA_HEADLESS",
        description="Run browser without GUI.",
    )

    # ── Retry ────────────────────────────────────────────────────────
    retry_strategy: str = Field(
        default="exponential",
        validation_alias="AQA_RETRY_STRATEGY",
        description="Retry strategy: none, linear, exponential.",
    )
    max_step_retries: int = Field(
        default=3,
        validation_alias="AQA_MAX_RETRIES",
        description="Maximum retry attempts per step.",
    )

    # ── Reporting ────────────────────────────────────────────────────
    report_format: Optional[str] = Field(
        default=None,
        validation_alias="AQA_REPORT_FORMAT",
        description="Report format: html, junit, json. Empty to skip.",
    )
    report_path: Optional[str] = Field(
        default=None,
        validation_alias="AQA_REPORT_PATH",
        description="Output path for the report file.",
    )

    @property
    def llm_api_key(self) -> str:
        """Return the appropriate API key based on the active provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key or ""
        return self.openai_api_key or ""


def load_settings() -> AutoMateQASettings:
    """Load settings from env vars and .env file."""
    return AutoMateQASettings()
