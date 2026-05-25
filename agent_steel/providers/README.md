# Providers

LLM backends behind a single `Provider` interface so Agent Steel can swap
models (Claude, GPT, local OpenAI-compatible servers) without changing
orchestration code.

## Layout

```
providers/
    base.py             # Provider ABC + Message + GenerationResult dataclasses
    openai.py           # OpenAI Chat Completions API — also speaks to vLLM, Ollama, Groq, Together, Mistral, Fireworks
    anthropic.py        # Anthropic Messages API (Claude)
    __init__.py         # get_provider(name) factory
```

## Adding a new backend

1. Implement `Provider` in `providers/<name>.py` with one `generate(...)` method.
2. Register it in `__init__.get_provider`.
3. Document the env vars / install requirements at the top of the file.

## Install

The backends lazy-import their SDKs. Install only what you use:

```bash
pip install anthropic      # for Claude
pip install openai         # for OpenAI / vLLM / Ollama / Groq / Together / etc.
```

## Environment

| Provider | Env var | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | required |
| `openai-compat` | `OPENAI_API_KEY` | required |
| `openai-compat` | `OPENAI_BASE_URL` | optional — set this to point at vLLM/Ollama/Groq/etc. |

## Example

```python
from agent_steel.providers import get_provider, Message

p = get_provider("anthropic", default_model="claude-sonnet-4-6")
out = p.generate([
    Message("system", "You write Apple Metal kernels. Output a unified diff."),
    Message("user", "Vectorize the inner loop of conv2d with float4."),
])
print(out.text)
```

## Why a custom abstraction over LangChain / LiteLLM

Agent Steel deals with kernel source — careful prompt construction, strict
output formats (diffs, single-file edits), and per-attempt logging matter
more than provider breadth. A 60-line ABC keeps the surface honest. If we
need richer routing later (cost-aware fallback, parallel multi-model
ensembles), we can add it without ripping out a framework.
