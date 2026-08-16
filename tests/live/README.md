# Live provider matrix

End-to-end tests against real providers. One file per provider, one
shared battery (`battery.py`). Every file runs the same checks: the
tutorial surface of `explore_lm.md`, plus tool emission, reasoning
levels, image input, and the usage/cost ledger.

Nothing here is mocked. A test skips when its credential is absent.

## Run

```bash
pytest tests/live --llm_call            # whole matrix
pytest tests/live/test_anthropic.py --llm_call   # one provider
```

The `--llm_call` flag is required. Without it, every test skips.

## Credentials

| File | Provider | Credential | Billing / cost |
| --- | --- | --- | --- |
| `test_openai.py` | OpenAI Responses API | `OPENAI_API_KEY` | per token, cost > 0 |
| `test_openai_chat.py` | OpenAI Chat Completions | `OPENAI_API_KEY` | per token, cost > 0 |
| `test_openai_codex.py` | OpenAI Codex (tutorial provider) | Codex CLI OAuth (`~/.codex/auth.json`) | subscription, cost is None |
| `test_anthropic.py` | Anthropic | `ANTHROPIC_API_KEY` | per token, cost > 0 |
| `test_claude_code.py` | Claude Code | Claude Code OAuth credential | subscription, cost is None |
| `test_gemini.py` | Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | per token, cost > 0 |
| `test_ollama.py` | Ollama | none — local server on `OLLAMA_HOST` | local, cost is None |
| `test_vllm.py` | vLLM (lab server, `api_base` style) | `DSPY_LIVE_VLLM_BASE_URL` + `DSPY_LIVE_VLLM_API_KEY` (defaults: 192.168.2.24:8000) | local, cost is None |

## Models

Each file pins a cheap default model. Override any of them with an
environment variable, no code change:

```bash
DSPY_LIVE_OPENAI_MODEL=openai:gpt-5-mini
DSPY_LIVE_ANTHROPIC_MODEL=anthropic:claude-sonnet-5
DSPY_LIVE_OLLAMA_MODEL=ollama:qwen3:0.6b
```

The pattern is `DSPY_LIVE_<PROVIDER>_MODEL`, with `-` replaced by `_`.

## Adding a provider

1. Copy any provider file.
2. Edit its `ProviderSpec`: model, gate, capability flags.
3. Keep the eight flat tests as they are. The battery does the rest.

Capability flags a provider lacks (`images=False`, `reasoning=False`)
turn those checks into skips with a reason, not failures.
