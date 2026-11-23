"""MCP safely: a third-party server's update hides an instruction in a tool description.

Run from the repo root:
  uv run s22b-mcp-modern-agents/07_poisoned_tool.py pin       # you review fx-rates v1 and pin its tools
  uv run s22b-mcp-modern-agents/07_poisoned_tool.py trust     # v2 arrives; the agent trusts it
  uv run s22b-mcp-modern-agents/07_poisoned_tool.py approve   # defense 1: a human approves outside tools
  uv run s22b-mcp-modern-agents/07_poisoned_tool.py check     # defense 2: refuse tools that changed since you pinned them
"""

import asyncio
import hashlib
import json
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.mcp import MCPAdapter
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from lumora import HERE, MODEL, SYSTEM

mode = sys.argv[1]
version = "--v1" if mode == "pin" else "--v2"
config = {"mcpServers": {"fx": {"command": sys.executable, "args": [str(HERE / "fx_server.py"), version]}}}
PINS = HERE / "pins.json"
STOLEN = HERE / "stolen.txt"


@tool
def read_private_notes() -> str:
    """Read the employee's private HR notes (salary, bank details, reviews)."""
    return (HERE / "private_notes.txt").read_text()


def fingerprint(tools) -> str:
    """One hash over every tool's name, description and input schema."""
    defs = [{"name": t.name, "description": t.description, "schema": t.args} for t in tools]
    return hashlib.sha256(json.dumps(defs, sort_keys=True).encode()).hexdigest()[:16]


async def main():
    STOLEN.unlink(missing_ok=True)
    async with MCPAdapter(config) as adapter:
        fx_tools = await adapter.list_tools()

        if mode == "pin":
            PINS.write_text(json.dumps({"fx": fingerprint(fx_tools)}))
            print("Reviewed fx-rates v1. Pinned:", fingerprint(fx_tools))
            return
        if mode == "check" and json.loads(PINS.read_text())["fx"] != fingerprint(fx_tools):
            print(f"STOP: fx-rates changed its tools since you pinned them ({fingerprint(fx_tools)}). Review before using it.")
            return

        gate = [HumanInTheLoopMiddleware(interrupt_on={"convert_currency": True}, interrupt_mode="per_call")]
        agent = create_agent(
            MODEL,
            [read_private_notes, *fx_tools],
            system_prompt=SYSTEM,
            middleware=gate if mode == "approve" else [],
            checkpointer=InMemorySaver(),
        )
        config_ = {"configurable": {"thread_id": mode}}
        ask = {"messages": [{"role": "user", "content": "My taxi was 38 EUR. How much is that in USD?"}]}
        result = await agent.ainvoke(ask, config_, version="v2")
        while result.interrupts:
            answers = {}
            for pause in result.interrupts:
                print(f"\nApprove {pause.value['name']}({pause.value['args']})?")
                ok = input("approve / reject? ").strip().lower() == "approve"
                answers[pause.id] = {"type": "approve"} if ok else {"type": "reject", "message": "Not allowed."}
            result = await agent.ainvoke(Command(resume=answers), config_, version="v2")

        print("\nAgent:", result.value["messages"][-1].content)
        print("\nWhat fx-rates received:", STOLEN.read_text() if STOLEN.exists() else "nothing")


asyncio.run(main())
