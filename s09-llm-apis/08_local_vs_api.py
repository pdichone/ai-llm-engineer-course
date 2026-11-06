# Prereq: install Ollama, then `ollama pull <MODEL>`.
# Pick a small (<=8B) model from ollama.com/library, and check its licence.
from dotenv import load_dotenv; load_dotenv()
import os, time, ollama, anthropic

LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3.2:3b")
Q = "Summarize in 2 bullets: employees get 20 vacation days; unused days roll over up to 5."

t = time.perf_counter()
local = ollama.chat(model=LOCAL_MODEL, messages=[{"role": "user", "content": Q}])
print(f"LOCAL  {time.perf_counter()-t:.1f}s\n{local.message.content}\n")

if os.environ.get("ANTHROPIC_API_KEY"):
    t = time.perf_counter()
    m = anthropic.Anthropic().messages.create(model="claude-haiku-5-5", max_tokens=2000,
            output_config={"effort": "low"}, messages=[{"role": "user", "content": Q}])
    print(f"API    {time.perf_counter()-t:.1f}s\n" + "".join(b.text for b in m.content if b.type == "text"))
else:
    print("API    skipped (no ANTHROPIC_API_KEY)")
