from dotenv import load_dotenv; load_dotenv()
import anthropic
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
filler = open("data/handbook.md").read()

# A harmless fact. (A door code got refused: the handbook's own security policy says credentials are restricted.)
NEEDLE = "Note: Marta from the Toronto hub has been a vegetarian since 2019."
ANSWER = "Marta"
TESTS = {
    "literal":     "Who has been a vegetarian since 2019?",
    "non-literal": "Which colleague can't eat the beef stew at the team dinner?",  # no shared words with the needle
}


def ask(doc, q):
    m = client.messages.create(model=MODEL, max_tokens=2000, output_config={"effort": "low"},  # thinking counts toward max_tokens
        messages=[{"role": "user", "content": f"{doc}\n\nQuestion: {q}"}])
    ans = "".join(b.text for b in m.content if b.type == "text")
    return ANSWER in ans, m.stop_reason, m.usage.input_tokens


# Control: the same needle in a SHORT document. If this fails, the task is hard, not the length.
for name, q in TESTS.items():
    ok, stop, n = ask(filler + "\n\n" + NEEDLE, q)
    print(f"short   {n:>7,} tok  {name:12} correct={ok}  (stop_reason={stop})")

haystack_parts = [filler] * 20                      # ~94K tokens
for pos in [0.0, 0.5, 1.0]:                         # start, middle, end
    parts = haystack_parts.copy()
    parts.insert(int(len(parts) * pos), NEEDLE)
    doc = "\n\n".join(parts)
    for name, q in TESTS.items():
        ok, stop, n = ask(doc, q)
        print(f"pos={pos:.1f} {n:>7,} tok  {name:12} correct={ok}  (stop_reason={stop})")
