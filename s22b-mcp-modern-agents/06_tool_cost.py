"""What tools cost: every tool definition is sent to the model on every call. Count it.

Run from the repo root:  uv run s22b-mcp-modern-agents/06_tool_cost.py
Counting tokens is free (no model call), but it needs an Anthropic key.
"""

import asyncio
import json
import sys
from pathlib import Path

from anthropic import Anthropic
from mcp import Client, StdioServerParameters
from lumora import HERE, MODEL

client = Anthropic()
model = MODEL.split(":")[-1]  # "anthropic:claude-sonnet-5-5" → "claude-sonnet-5-5"


def tokens(tools: list[dict]) -> int:
    """How many input tokens these tool definitions add to a request."""
    msg = [{"role": "user", "content": "hi"}]
    base = client.messages.count_tokens(model=model, messages=msg).input_tokens
    return client.messages.count_tokens(model=model, messages=msg, tools=tools).input_tokens - base


async def our_tools() -> list[dict]:
    server = StdioServerParameters(command=sys.executable, args=[str(HERE / "04_handbook_server.py")])
    async with Client(server) as mcp:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in (await mcp.list_tools()).tools]


github = json.loads((HERE / "github_tools.json").read_text())["tools"]
as_claude = lambda ts: [{k: t[k] for k in ("name", "description", "input_schema")} for t in ts]
rows = {
    "Our handbook server": asyncio.run(our_tools()),
    "GitHub MCP server, everything": as_claude(github),
    "GitHub, 3 tools allowed (fix 1)": as_claude([t for t in github if t["name"] in ("search_issues", "issue_read", "list_issues")]),
}

print(f"{'':34}{'tools':>6}{'tokens':>9}   (sent on EVERY call)")
for label, tools in rows.items():
    print(f"{label:34}{len(tools):>6}{tokens(tools):>9,}")
print(f"\nGitHub tools that can change things: {sum(not t['read_only'] for t in github)} of {len(github)}")
