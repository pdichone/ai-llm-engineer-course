"""S23 step C: base model vs your fine-tuned model, side by side, on questions it never trained on.
Prereq: `ollama pull qwen3:1.7b` and `ollama create teacrafters -f Modelfile` (step B).
Run from the repo root:  uv run s23-fine-tuning-openai/03_chat_ollama.py   (add --chat to talk to it)"""
import json, re, sys
from pathlib import Path
import ollama

HERE = Path(__file__).parent
BASE, TUNED = "qwen3:1.7b", "teacrafters"
SYSTEM = "You are the customer support assistant for TeaCrafters, an artisan tea subscription service."


def ask(model, messages):
    r = ollama.chat(model=model, messages=messages, think=False)  # Qwen3 thinks by default: turn it off
    return re.sub(r"<think>.*?</think>", "", r.message.content, flags=re.S).strip()  # drop the empty think block


if "--chat" not in sys.argv:
    tests = [json.loads(line)["messages"] for line in open(HERE / "teacrafters_test.jsonl")]
    score = {BASE: 0, TUNED: 0}
    for m in tests:
        print(f"\nQ: {m[1]['content']}\n   expected: {m[2]['content']}")
        for model in (BASE, TUNED):
            answer = ask(model, m[:2])
            score[model] += "Happy steeping!" in answer   # the house style: did it learn the behaviour?
            print(f"   {model:<11} {answer[:300]}")
    print(f"\nHouse style followed: base {score[BASE]}/{len(tests)} · fine-tuned {score[TUNED]}/{len(tests)}")
    print("Now read the answers: style is learned, but check every fact against teacrafters_policies.md.")
else:
    history = [{"role": "system", "content": SYSTEM}]
    while (q := input("\nYou: ")).lower() not in ("exit", "quit"):
        history.append({"role": "user", "content": q})
        answer = ask(TUNED, history)
        history.append({"role": "assistant", "content": answer})
        print(f"TeaCrafters: {answer}")
