"""Shared types and provider protocol for LLM clients."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ProviderType(StrEnum):
    """Supported LLM provider backends."""

    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass
class Message:
    """Chat message (provider-agnostic role + content)."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict format used by both Ollama and OpenAI APIs."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMOptions:
    """Sampling options (common subset across providers).

    Provider-specific mapping:
      - temperature, top_p, top_k, seed: universal
      - num_ctx: Ollama context window (ignored by OpenAI)
      - num_predict: Ollama max tokens -> OpenAI max_completion_tokens
    """

    temperature: float | None = None
    num_ctx: int | None = None
    seed: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None

    def to_dict(self) -> dict[str, float | int]:
        """Return only set options as a dict."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class LLMProvider(Protocol):
    """Protocol that all LLM providers must satisfy."""

    def chat(
        self,
        messages: list[Message],
        model: str,
        options: LLMOptions | None,
        output_format: dict[str, object] | None,
        stream: bool,
    ) -> str:
        """Generate a chat response. Returns full text or empty string on error."""
        ...

    def _batch_chat(self, kwargs: dict[str, object]) -> str:
        """Send a non-streaming request and return the full response text."""
        ...

    def _stream_chat(self, kwargs: dict[str, object]) -> str:
        """Send a streaming request, print tokens live, and return the full text."""
        ...
