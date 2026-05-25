"""OpenAI-compatible provider — Chat Completions API.

Works with any service that speaks the OpenAI Chat Completions protocol:
- OpenAI (default — leave `base_url` unset, set `OPENAI_API_KEY`).
- vLLM:        `base_url="http://localhost:8000/v1"`
- Ollama:      `base_url="http://localhost:11434/v1"`
- Groq:        `base_url="https://api.groq.com/openai/v1"`
- Together AI: `base_url="https://api.together.xyz/v1"`
- Fireworks:   `base_url="https://api.fireworks.ai/inference/v1"`
- Mistral:     `base_url="https://api.mistral.ai/v1"`

We deliberately use Chat Completions (not OpenAI's newer Responses API)
because it's the format every compat backend speaks.

Reasoning-model note: o1/o3/o4/o5 family ignores `temperature` and expects
`max_completion_tokens` instead of `max_tokens`. This module switches kwargs
based on a model-name prefix check.
"""
from __future__ import annotations
import os
from typing import Iterable

from .base import Message, GenerationResult, Provider


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai` to use this provider."
            ) from e

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
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
        m = model or self.default_model
        kwargs: dict = {
            "model": m,
            "messages": [{"role": x.role, "content": x.content} for x in messages],
        }
        # Reasoning models (o-family) want max_completion_tokens; others want max_tokens.
        # o-family also ignores temperature, so we omit it.
        if m.startswith(("o1", "o3", "o4", "o5")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop"] = list(stop)

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            model=resp.model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            stop_reason=choice.finish_reason,
        )
