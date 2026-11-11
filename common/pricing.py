# USD per 1M tokens. Checked 2026-11-10 against platform.claude.com/docs/en/about-claude/pricing.
# Prices change: re-check before trusting. This file is the ONLY place prices live in the repo.
PRICES = {
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.10, "cache_write_5m": 2.50},
    "claude-haiku-5-5":  {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write_5m": 0.125,
                          # Haiku 5.5 only: a prompt over 100,000 tokens (cache reads and writes count)
                          # pays these higher prices for the WHOLE request, output included.
                          "long": {"over": 100_000, "in": 0.50, "out": 2.50, "cache_read": 0.05,
                                   "cache_write_5m": 0.625}},
}


def cost(model: str, u) -> float:
    p = PRICES[model]
    read = u.cache_read_input_tokens or 0
    write = u.cache_creation_input_tokens or 0
    if "long" in p and u.input_tokens + read + write > p["long"]["over"]:
        p = p["long"]
    total = (u.input_tokens * p["in"] + u.output_tokens * p["out"]
             + read * p["cache_read"] + write * p["cache_write_5m"])
    return total / 1_000_000
