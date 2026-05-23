"""Provider abstraction for Agent Steel.

A Provider wraps an LLM backend behind a small, common interface so the rest
of the agent harness can swap models (Claude, GPT, a local OpenAI-compatible
server like vLLM/Ollama) without rewriting orchestration code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class Provider(ABC):
    """Single-shot generation interface. Streaming + tool-use can be added later."""

    name: str
    default_model: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        stop: Iterable[str] | None = None,
    ) -> GenerationResult:
        """Send `messages` to the model and return the assistant's reply.

        - `model` overrides `default_model` for one call.
        - `temperature` defaults to 0.2 because kernel work wants near-deterministic edits.
        - `stop` is provider-specific; pass at your own risk across backends.
        """
        ...
