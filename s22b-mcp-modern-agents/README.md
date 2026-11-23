# Section 22b: MCP & Modern Agents

A Lumora HR assistant, built the 2026 way: `create_agent`, human approval, memory, then your own MCP
server, what tools cost, and how to use MCP safely. It comes right after Section 22 (AI Agents).

**Setup B (2026, uv):** from the repo root, with `ANTHROPIC_API_KEY` in the root `.env`:

```bash
uv sync
uv run s22b-mcp-modern-agents/01_create_agent.py
```

| File | Lecture |
|---|---|
| `01_create_agent.py` | `create_agent`: the S22 agent in a dozen lines, plus structured output |
| `02_approvals.py` | Human-in-the-loop: approve, edit or reject a tool call |
| `03_memory.py` | Memory that survives: threads (checkpointer) and facts (Store) |
| `04_handbook_server.py`, `04b_test_server.py` | Build your own MCP server and test it without an LLM |
| `05_agent_mcp.py` | Plug the MCP server into the agent |
| `06_tool_cost.py`, `github_tools.json` | What tools cost: count the tokens of tool definitions |
| `07_poisoned_tool.py`, `fx_server.py`, `private_notes.txt` | MCP safely: a poisoned tool update, an approval gate, and pinned tools |
| `lumora.py` | Shared: the handbook, the model, `search_handbook` and `file_expense` |

**Model:** `anthropic:claude-sonnet-5-5` by default. To try another model, set `S22B_MODEL`, for
example `S22B_MODEL=openai:gpt-6-luna` (needs `langchain-openai` and an OpenAI key).

**Inspect the server in a browser:** `npx @modelcontextprotocol/inspector` (needs Node), then connect to
`uv run s22b-mcp-modern-agents/04_handbook_server.py` over stdio.

**Notes**
- `langchain.mcp` is in **beta** and prints a `LangChainBetaWarning`. The older `langchain-mcp-adapters`
  package that most tutorials use is archived.
- `github_tools.json` is GitHub's remote MCP server's tool list as measured on 2026-10-27 (49 tools).
  The live server changes, so your own count may differ.
- `private_notes.txt` is fake data for the security demo.

Back to the [course index](../README.md).
