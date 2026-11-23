"""Shared bits for this section: the Lumora handbook, the model, and the expense tool."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()  # finds the .env at the repo root

HERE = Path(__file__).parent
HANDBOOK = (HERE.parent / "data" / "handbook.md").read_text()
SECTIONS = dict(re.findall(r"^## \d+\. (.+?)\n(.*?)(?=^## |\Z)", HANDBOOK, re.S | re.M))
MODEL = os.getenv("S22B_MODEL", "anthropic:claude-sonnet-5-5")
CLAIMS = HERE / "claims.json"

SYSTEM = (
    "You are the Lumora HR assistant. Answer employees' questions from the handbook only, "
    "and name the section you used. If the handbook doesn't cover it, say so."
)


@tool
def search_handbook(query: str) -> str:
    """Search the Lumora employee handbook. Returns the two sections that best match the query."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
    scored = sorted(SECTIONS.items(), key=lambda s: -sum(s[1].lower().count(w) for w in words))
    return "\n\n".join(f"## {title}\n{body}" for title, body in scored[:2])


@tool
def file_expense(amount: float, currency: str, description: str) -> str:
    """File an expense claim in Lumora's expense system."""
    claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else []
    claims.append({"amount": amount, "currency": currency, "description": description})
    CLAIMS.write_text(json.dumps(claims, indent=2))
    return f"Claim #{len(claims)} filed: {amount} {currency}, {description}"
