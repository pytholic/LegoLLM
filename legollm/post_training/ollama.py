"""Ollama HTTP client for post-training workflows.

Make sure ollama is installed and running before use:
```bash
ollama run llama3.1:8b
```
"""

import json
from dataclasses import dataclass

import requests


@dataclass
class OllamaConfig:
    """Ollama server configuration."""

    model_name: str = "llama3.1:8b"
    host: str = "localhost"
    port: int = 11434


@dataclass
class Message:
    """Chat message for Ollama."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to Ollama API dict format."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMOptions:
    """Sampling options for Ollama chat generation."""

    temperature: float | None = None
    num_ctx: int | None = None
    seed: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None

    def to_dict(self) -> dict[str, float | int]:
        """Return only set options as a dict."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


_default_config = OllamaConfig()


def generate_chat_response(
    messages: list[Message],
    model: str = _default_config.model_name,
    host: str = _default_config.host,
    port: int = _default_config.port,
    options: LLMOptions | None = None,
    output_format: dict[str, object] | None = None,
    stream: bool = False,
) -> str:
    """Generate a chat response from Ollama.

    If stream=True, prints tokens to stdout as they arrive.
    Returns the full response string, or empty string on error.
    """
    url = f"http://{host}:{port}/api/chat"

    payload: dict[str, object] = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        "stream": True,
    }

    options_dict = options.to_dict() if options else None
    if options_dict:
        payload["options"] = options_dict
    if output_format:
        payload["format"] = output_format

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            response_data = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                json_response = json.loads(line)
                message = json_response.get("message", {})
                token = message.get("content")
                if token:
                    if stream:
                        print(token, end="", flush=True)
                    response_data += token
            if stream:
                print()

        return response_data

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return ""
