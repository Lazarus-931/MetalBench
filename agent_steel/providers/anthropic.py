"""Anthropic provider — Claude via the Messages API.

System prompts: extracted from the messages list and passed as the `system`
parameter (the Anthropic API is strict that "system" is not a role inside
`messages`).
"""
from __future__ import annotations
import os
from typing import Iterable

from .base import Message, GenerationResult, Provider


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic` to use this provider."
            ) from e

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if default_model:
            self.default_model = default_model

    def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        stop: Iterable[str] | None = None,
    ) -> GenerationResult:
        # Anthropic API takes system prompts as a top-level `system` arg, not
        # as a message with role=system. Collect all system messages and join.
        system_parts = [m.content for m in messages if m.role == "system"]
        chat = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        kwargs = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if stop:
            kwargs["stop_sequences"] = list(stop)

        resp = self._client.messages.create(**kwargs)
        # Concatenate any text blocks in the response.
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=text,
            model=resp.model,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            stop_reason=resp.stop_reason,
        )
