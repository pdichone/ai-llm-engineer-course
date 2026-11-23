"""Talk to the handbook server the way any MCP client does: discover, list, call. No LLM.

Run from the repo root:  uv run s22b-mcp-modern-agents/04b_test_server.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "04_handbook_server.py")])


async def main():
    async with Client(SERVER) as client:  # starts the server as a subprocess, speaks MCP over stdio
        print("Server:", client.server_info.name, "| protocol", client.protocol_version)

        tools = (await client.list_tools()).tools
        for t in tools:
            print(f"tool  {t.name}: {t.description}")
        templates = (await client.list_resource_templates()).resource_templates
        print("resource", templates[0].uri_template)
        prompts = (await client.list_prompts()).prompts
        print("prompt", prompts[0].name)

        result = await client.call_tool("read_section", {"title": "Time Off"})
        print("\nread_section('Time Off') →", result.content[0].text[:200], "…")


asyncio.run(main())
