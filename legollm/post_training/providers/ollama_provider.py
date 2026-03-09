"""Ollama provider with HTTP and SDK backends."""

import json
from enum import StrEnum
from typing import Any

import requests

from legollm.post_training.providers.base import LLMOptions, Message


class BackendType(StrEnum):
    """Ollama backend type."""

    HTTP = "http"
    SDK = "sdk"


def _build_payload(
    messages: list[Message],
    model: str,
    options: LLMOptions | None,
    output_format: dict[str, object] | None,
    think: bool,
) -> dict[str, object]:
    """Build the shared chat payload for both Ollama backends."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [m.to_dict() for m in messages],
        "think": think,
    }
    if options:
        opts = options.to_dict()
        if opts:
            payload["options"] = opts
    if output_format:
        payload["format"] = output_format
    return payload


def create_ollama_backend(
    backend: BackendType | str = BackendType.HTTP,
    *,
    host: str | None = None,
    port: int | None = None,
) -> "HttpBackend | SdkBackend":
    """Factory for Ollama backends."""
    kind = BackendType(backend)
    if kind == BackendType.SDK:
        return SdkBackend(host=f"http://{host or 'localhost'}:{port or 11434}")
    return HttpBackend(host=host or "localhost", port=port or 11434)


class HttpBackend:
    """Talks to a local Ollama server via raw HTTP."""

    def __init__(self, host: str = "localhost", port: int = 11434) -> None:
        self._url = f"http://{host}:{port}/api/chat"

    def chat(
        self,
        messages: list[Message],
        model: str,
        options: LLMOptions | None,
        output_format: dict[str, object] | None,
        stream: bool,
        think: bool = True,
    ) -> str:
        """Send a chat request via raw HTTP (always streams internally)."""
        payload = _build_payload(messages, model, options, output_format, think)
        payload["stream"] = True

        try:
            with requests.post(self._url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                result = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    token = json.loads(line).get("message", {}).get("content")
                    if token:
                        if stream:
                            print(token, end="", flush=True)
                        result += token
                if stream:
                    print()
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    def _batch_chat(self, kwargs: dict[str, object]) -> str:
        """Not used -- HTTP backend always streams internally."""
        raise NotImplementedError

    def _stream_chat(self, kwargs: dict[str, object]) -> str:
        """Not used -- HTTP backend always streams internally."""
        raise NotImplementedError


class SdkBackend:
    """Uses the `ollama` Python library -- no local ollama installation needed."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        try:
            from ollama import Client  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ollama-python is required for the SDK backend. Install it with: pip install ollama"
            ) from exc
        self._client: Any = Client(host=host)

    def chat(
        self,
        messages: list[Message],
        model: str,
        options: LLMOptions | None,
        output_format: dict[str, object] | None,
        stream: bool,
        think: bool = True,
    ) -> str:
        """Send a chat request via the ollama-python SDK."""
        kwargs = _build_payload(messages, model, options, output_format, think)
        if stream:
            return self._stream_chat(kwargs)
        return self._batch_chat(kwargs)

    def _batch_chat(self, kwargs: dict[str, object]) -> str:
        """Send a non-streaming request and return the full response text."""
        try:
            resp = self._client.chat(**kwargs, stream=False)
            content: str = resp.message.content or ""
            return content
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def _stream_chat(self, kwargs: dict[str, object]) -> str:
        """Send a streaming request, print tokens live, and return the full text."""
        try:
            result = ""
            for chunk in self._client.chat(**kwargs, stream=True):
                token: str = chunk.message.content or ""
                if token:
                    print(token, end="", flush=True)
                    result += token
            print()
            return result
        except Exception as e:
            print(f"Error: {e}")
            return ""


if __name__ == "__main__":
    prompt = [Message(role="user", content="What do llamas eat?")]
    model = "llama3.1:8b"

    for backend in BackendType:
        provider = create_ollama_backend(backend)
        print(f"--- {backend} batch ---")
        print(provider.chat(prompt, model=model, options=None, output_format=None, stream=False))
        print()

        print(f"--- {backend} stream ---")
        provider.chat(prompt, model=model, options=None, output_format=None, stream=True)
        print()
        print("=" * 80)
        print()
