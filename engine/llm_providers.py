"""
LLM provider abstraction for multi-model healing support.

Supports: OpenAI, Anthropic, and local OpenAI-compatible endpoints (Ollama, vLLM).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from engine.models import EngineConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for LLM providers used by the healing engine."""

    def __init__(self, model: str, max_tokens: int = 2000) -> None:
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> tuple[str, int]:
        """Return (response_text, tokens_used)."""

    async def complete_with_image(
        self, prompt: str, image_b64: str, system: str = "",
    ) -> tuple[str, int]:
        """Vision-capable completion. Raises NotImplementedError by default."""
        raise NotImplementedError(f"{self.name} does not support vision")


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models (gpt-4o, gpt-4o-mini, etc.)."""

    def __init__(
        self, model: str = "gpt-4o", max_tokens: int = 2000,
        api_key: str = "", base_url: str = "",
    ) -> None:
        super().__init__(model, max_tokens)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    async def complete(self, prompt: str, system: str = "") -> tuple[str, int]:
        client = self._get_client()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=0.2, max_tokens=self._max_tokens,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    async def complete_with_image(
        self, prompt: str, image_b64: str, system: str = "",
    ) -> tuple[str, int]:
        client = self._get_client()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        })
        response = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=0.2, max_tokens=self._max_tokens,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models."""

    def __init__(
        self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 2000,
        api_key: str = "",
    ) -> None:
        super().__init__(model, max_tokens)
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            kwargs: dict = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Anthropic(**kwargs)
        return self._client

    async def complete(self, prompt: str, system: str = "") -> tuple[str, int]:
        client = self._get_client()
        kwargs: dict = {"model": self._model, "max_tokens": self._max_tokens}
        if system:
            kwargs["system"] = system
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    async def complete_with_image(
        self, prompt: str, image_b64: str, system: str = "",
    ) -> tuple[str, int]:
        client = self._get_client()
        kwargs: dict = {"model": self._model, "max_tokens": self._max_tokens}
        if system:
            kwargs["system"] = system
        kwargs["messages"] = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }]
        response = client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens


class LocalProvider(LLMProvider):
    """Local LLM via OpenAI-compatible API (Ollama, vLLM, LM Studio)."""

    def __init__(
        self, model: str = "llama3", max_tokens: int = 2000,
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        super().__init__(model, max_tokens)
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key="not-needed")
        return self._client

    async def complete(self, prompt: str, system: str = "") -> tuple[str, int]:
        client = self._get_client()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=0.2, max_tokens=self._max_tokens,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens


class TokenBudget:
    """Tracks token usage and enforces a per-run budget."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0

    def can_spend(self, estimated: int = 500) -> bool:
        return self._used + estimated <= self._limit

    def record(self, tokens: int) -> None:
        self._used += tokens

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self._limit - self._used)

    @property
    def exhausted(self) -> bool:
        return self._used >= self._limit


def create_provider(
    provider_name: str, model: str, config: Optional[EngineConfig] = None,
) -> LLMProvider:
    """Factory to create the appropriate LLM provider."""
    max_tokens = config.llm_max_tokens_per_heal if config else 2000
    api_key = config.llm_api_key if config else ""
    base_url = config.llm_base_url if config else ""

    match provider_name:
        case "openai":
            return OpenAIProvider(model=model, max_tokens=max_tokens, api_key=api_key, base_url=base_url)
        case "anthropic":
            return AnthropicProvider(model=model, max_tokens=max_tokens, api_key=api_key)
        case "local":
            return LocalProvider(model=model, max_tokens=max_tokens, base_url=base_url or "http://localhost:11434/v1")
        case _:
            logger.warning("Unknown provider '%s', falling back to OpenAI", provider_name)
            return OpenAIProvider(model=model, max_tokens=max_tokens, api_key=api_key)
