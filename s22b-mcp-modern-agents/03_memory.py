"""Memory that survives: a thread saved to disk (checkpointer), and facts shared across threads (Store).

Run from the repo root:
  uv run s22b-mcp-modern-agents/03_memory.py ana "Hi, I'm Ana. I work from the Lisbon hub and file expenses in EUR."
  uv run s22b-mcp-modern-agents/03_memory.py ana "Which hub do I work from?"      # same thread, new process
  uv run s22b-mcp-modern-agents/03_memory.py ana-2 "What currency do I file expenses in?"   # new thread
"""

import sqlite3
import sys

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from lumora import HERE, MODEL, SYSTEM, search_handbook



@tool
def remember(fact: str, runtime: ToolRuntime) -> str:
    """Save a lasting fact about the employee (a preference, their hub, their currency)."""
    runtime.store.put(("employee",), fact[:40], {"fact": fact})
    return "Saved."


@tool
def recall(runtime: ToolRuntime) -> str:
    """List every saved fact about the employee."""
    return "\n".join(item.value["fact"] for item in runtime.store.search(("employee",))) or "Nothing saved yet."


store = SqliteStore(sqlite3.connect(HERE / "facts.db", check_same_thread=False, isolation_level=None))  # autocommit
store.setup()
agent = create_agent(
    MODEL,
    tools=[search_handbook, remember, recall],
    system_prompt=SYSTEM + " Save lasting facts about the employee with remember. For any question about the employee themselves, call recall first.",
    checkpointer=SqliteSaver(sqlite3.connect(HERE / "threads.db", check_same_thread=False)),  # step 1: the conversation, per thread
    store=store,  # step 2: facts, across all threads
)

thread, text = sys.argv[1], sys.argv[2]
result = agent.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": thread}})
print(result["messages"][-1].text)
print(f"\n(thread {thread!r} now has {len(result['messages'])} messages)")
