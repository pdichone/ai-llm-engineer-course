"""The Lumora handbook as your own MCP server: two tools, a resource and a prompt.

Run it on its own (stdio):          uv run s22b-mcp-modern-agents/04_handbook_server.py
Or over HTTP at localhost:8000/mcp: uv run s22b-mcp-modern-agents/04_handbook_server.py --http
Test it:                            uv run s22b-mcp-modern-agents/04b_test_server.py
"""

import re
import sys
from pathlib import Path

from mcp.server import MCPServer

HANDBOOK = (Path(__file__).parent.parent / "data" / "handbook.md").read_text()
SECTIONS = dict(re.findall(r"^## \d+\. (.+?)\n(.*?)(?=^## |\Z)", HANDBOOK, re.S | re.M))

server = MCPServer("lumora-handbook")


@server.tool()
def list_sections() -> list[str]:
    """List the section titles of the Lumora employee handbook."""
    return list(SECTIONS)


@server.tool()
def read_section(title: str) -> str:
    """Return the full text of one handbook section. Use a title from list_sections."""
    return SECTIONS.get(title, f"No section called {title!r}. Call list_sections first.")


@server.resource("handbook://sections/{title}")
def section(title: str) -> str:
    """One handbook section, for apps that attach it as context."""
    return SECTIONS[title]


@server.prompt()
def ask_hr(question: str) -> str:
    """A ready-made prompt: answer an HR question from the handbook, citing the section."""
    return f"Using only the Lumora handbook, answer this and name the section you used: {question}"


if __name__ == "__main__":
    server.run("streamable-http" if "--http" in sys.argv else "stdio")
