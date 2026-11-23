"""The S22 agent in a dozen lines with create_agent, then structured output.

Run from the repo root:  uv run s22b-mcp-modern-agents/01_create_agent.py
"""

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel
from lumora import MODEL, SYSTEM, search_handbook

# Step 1: an agent that answers from the handbook
agent = create_agent(MODEL, tools=[search_handbook], system_prompt=SYSTEM)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How many vacation days do I get, and can I carry them over?"}]}
)
for message in result["messages"]:
    for call in getattr(message, "tool_calls", []):
        print("Tool call:", call["name"], call["args"])
print("\n" + result["messages"][-1].text)


# Step 2: structured output, a typed object instead of free text
class ExpenseCheck(BaseModel):
    amount: float
    currency: str
    category: Literal["meal", "travel", "hotel", "mileage", "other"]
    within_policy: bool
    reason: str


checker = create_agent(MODEL, tools=[search_handbook], system_prompt=SYSTEM, response_format=ExpenseCheck)
result = checker.invoke(
    {"messages": [{"role": "user", "content": "I spent 92 USD on dinner, by myself, on a work trip to Austin."}]}
)
check = result["structured_response"]
print(check)
print("Within policy?", check.within_policy)
