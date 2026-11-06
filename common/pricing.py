# USD per 1M tokens. Checked 2026-11-05 against platform.claude.com/docs/en/about-claude/pricing.
# Prices change: re-check before trusting. This file is the ONLY place prices live in the repo.
PRICES = {
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.10, "cache_write_5m": 2.50},
    "claude-haiku-5-5":  {"in": 0.10, "out": 0.50,  "cache_read": None, "cache_write_5m": None},  # TODO: fill from pricing page
}


def cost(model: str, u) -> float:
    p = PRICES[model]
    total = u.input_tokens * p["in"] + u.output_tokens * p["out"]
    if p["cache_read"] is not None:
        total += (u.cache_read_input_tokens or 0) * p["cache_read"]
        total += (u.cache_creation_input_tokens or 0) * p["cache_write_5m"]
    return total / 1_000_000
