"""Human-in-the-loop: the agent asks before it files an expense. Approve, edit or reject.

Run from the repo root:  uv run s22b-mcp-modern-agents/02_approvals.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from lumora import MODEL, SYSTEM, file_expense, search_handbook

agent = create_agent(
    MODEL,
    tools=[search_handbook, file_expense],
    system_prompt=SYSTEM,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"file_expense": True}, interrupt_mode="per_call")],
    checkpointer=InMemorySaver(),  # a pause has to be saved somewhere
)
config = {"configurable": {"thread_id": "expense-1"}}

request = "File my taxi from the Lisbon airport to the hotel: 38 EUR."
result = agent.invoke({"messages": [{"role": "user", "content": request}]}, config, version="v2")

while result.interrupts:  # one interrupt per gated tool call
    answers = {}
    for pause in result.interrupts:
        call = pause.value
        print(f"\nThe agent wants to run {call['name']} with {call['args']}")
        choice = input("approve / edit / reject? ").strip().lower()
        if choice == "edit":
            amount = float(input("New amount: "))
            answers[pause.id] = {"type": "edit", "edited_action": {"name": call["name"], "args": {**call["args"], "amount": amount}}}
        elif choice == "reject":
            answers[pause.id] = {"type": "reject", "message": input("Why? ")}
        else:
            answers[pause.id] = {"type": "approve"}
    result = agent.invoke(Command(resume=answers), config, version="v2")

print("\n" + result.value["messages"][-1].text)
