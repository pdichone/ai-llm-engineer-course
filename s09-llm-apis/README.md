# Section 9: LLM APIs in 2026: Models, Tokens, Reasoning & Cost

Code for the 2026 lessons.

**Setup B (2026, uv):** run from the **repo root**, not from this folder:

```bash
uv run s09-llm-apis/<file>.py
```

| File | Lesson |
|---|---|
| `01_hello_three.py` | First call to Anthropic, OpenAI and Google |
| `02_tokens.py` | Tokens, and why counts differ per model |
| `03_effort.py` | Reasoning models, effort, and the end of `temperature` |
| `04_models_live.py` | Reading the model landscape (live from the Models API) |
| `05_cost_and_cache.py` | Cost math and prompt caching (prices live in `common/pricing.py`) |
| `06_needle.py` | Context windows: advertised vs effective (also used in Section 12) |
| `06b_needle_count.py` | Hard mode: counting across a long context (also used in Section 12) |
| `06c_needle_count.py` | Harder mode: every copy is different (no multiply shortcut), 3 trials with the true count from code |
| `07_context_budget.py` | Context engineering: where the tokens go (also used in Section 12) |
| `08_local_vs_api.py` | Local model vs API (used in Section 11; needs Ollama running) |

These scripts read `data/handbook.md` (the course's running project) from the repo root.

Back to the [course index](../README.md).
