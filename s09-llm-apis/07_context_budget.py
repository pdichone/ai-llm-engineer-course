from dotenv import load_dotenv; load_dotenv()
import anthropic
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

system = "You are the company docs assistant. Cite the section you used."
tools = [{"name": "search_docs", "description": "Search the company handbook.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}]
history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"}] * 15
retrieved = open("data/handbook.md").read()


def count(**kw):
    return client.messages.count_tokens(model=MODEL, **kw).input_tokens


base = count(messages=[{"role": "user", "content": "x"}])
parts = {
    "system":    count(system=system, messages=[{"role": "user", "content": "x"}]) - base,
    "tools":     count(tools=tools, messages=[{"role": "user", "content": "x"}]) - base,
    "history":   count(messages=history + [{"role": "user", "content": "x"}]) - base,
    "retrieved": count(messages=[{"role": "user", "content": retrieved}]) - base,
}
total = sum(parts.values())
for k, v in parts.items():
    print(f"{k:10} {v:>7,} tokens  {v/total:6.1%}")
