"""Verify an LLM provider can be instantiated and answers a trivial prompt.

Skipped if no API key is set.
"""
from __future__ import annotations
import os

import pytest

from agent_steel.providers import Message, get_provider


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_key(), reason="no ANTHROPIC_API_KEY or OPENAI_API_KEY set")
def test_provider_roundtrip():
    if os.environ.get("ANTHROPIC_API_KEY"):
        backend, default_model = "anthropic", "claude-haiku-4-5-20251001"
    else:
        backend, default_model = "openai", "gpt-4o-mini"
    p = get_provider(backend, default_model=default_model)
    try:
        resp = p.generate(
            [Message("user", "Reply with the single word 'ok'.")],
            max_tokens=8, temperature=0.0,
        )
    except Exception as e:
        if "quota" in str(e).lower() or "rate" in str(e).lower():
            pytest.skip(f"provider quota/rate-limit: {e!s:.120}")
        raise
    assert resp.text.strip().lower().startswith("ok")
