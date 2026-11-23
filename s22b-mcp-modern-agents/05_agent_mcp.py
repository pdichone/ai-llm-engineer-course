"""Plug your MCP server into the agent: the tools come from the server, not from your code.

Run from the repo root:  uv run s22b-mcp-modern-agents/05_agent_mcp.py
"""

import asyncio
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter  # beta: pip install "langchain[mcp]"
from lumora import MODEL, SYSTEM

SERVER = Path(__file__).parent / "04_handbook_server.py"


async def main():
    async with MCPAdapter(SERVER) as adapter:  # a Path = start it locally over stdio
        tools = await adapter.list_tools()
        print("Tools from the server:", [t.name for t in tools])

        agent = create_agent(MODEL, tools, system_prompt=SYSTEM)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What's the meal limit on a work trip, and is wine covered?"}]}
        )
        for message in result["messages"]:
            message.pretty_print()


asyncio.run(main())
