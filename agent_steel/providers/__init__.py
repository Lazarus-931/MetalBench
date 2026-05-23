"""Agent Steel providers — LLM backends behind a common interface.

Usage:
    from agent_steel.providers import get_provider, Message

    p = get_provider("anthropic")  # or "openai", "openai-compat"
    out = p.generate([
        Message("system", "You write Apple Metal kernels."),
        Message("user", "Vectorize this with float4..."),
    ])
    print(out.text)

Environment:
- `ANTHROPIC_API_KEY` for the Anthropic provider.
- `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) for OpenAI-compatible.
"""
from __future__ import annotations
from .base import Message, GenerationResult, Provider


def get_provider(name: str, **kwargs) -> Provider:
    """Return a Provider by short name. Lazy-imports the backend SDK on demand."""
    n = name.lower().strip()
    if n in ("openai", "openai-compat"):
        from .openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if n in ("anthropic", "claude"):
        from .anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    raise ValueError(f"unknown provider: {name!r}. Expected 'openai' or 'anthropic'.")


__all__ = ["get_provider", "Message", "GenerationResult", "Provider"]
