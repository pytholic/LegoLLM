"""OpenAI provider using the Responses API."""

import os
from typing import Any

from legollm.post_training.providers.base import LLMOptions, Message


class OpenAIProvider:
    """OpenAI provider using the Responses API (client.responses.create)."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI provider. Install it with: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = OpenAI(api_key=resolved_key)

    def chat(
        self,
        messages: list[Message],
        model: str,
        options: LLMOptions | None,
        output_format: dict[str, object] | None,
        stream: bool,
    ) -> str:
        """Send a chat request via the OpenAI Responses API."""
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [m.to_dict() for m in messages],
        }
        if options:
            # NOTE: temperature, top_p do not work for reasoning models
            if options.temperature is not None:
                kwargs["temperature"] = options.temperature
            if options.top_p is not None:
                kwargs["top_p"] = options.top_p
            if options.seed is not None:
                kwargs["seed"] = options.seed
            if options.num_predict is not None:
                kwargs["max_output_tokens"] = options.num_predict

        if output_format:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "strict": True,
                    "schema": output_format,
                }
            }

        try:
            if stream:
                return self._stream_chat(kwargs)
            return self._batch_chat(kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def _batch_chat(self, kwargs: dict[str, object]) -> str:
        """Send a non-streaming request and return the full response text."""
        resp = self._client.responses.create(**kwargs, stream=False)
        return resp.output_text or ""

    def _stream_chat(self, kwargs: dict[str, object]) -> str:
        """Send a streaming request, print tokens live, and return the full text."""
        result = ""
        stream = self._client.responses.create(**kwargs, stream=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                token: str = event.delta or ""
                if token:
                    print(token, end="", flush=True)
                    result += token
        print()
        return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    prompt = [Message(role="user", content="What do llamas eat?")]
    provider = OpenAIProvider()

    print("--- batch ---")
    print(provider.chat(prompt, model="gpt-5-nano", options=None, output_format=None, stream=False))
    print()

    print("--- stream ---")
    provider.chat(prompt, model="gpt-5-nano", options=None, output_format=None, stream=True)
    print()
