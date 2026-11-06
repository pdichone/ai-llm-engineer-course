from dotenv import load_dotenv; load_dotenv()
import re, anthropic
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
filler = open("data/handbook.md").read()
Q = "Exactly how many times does the word 'reimbursement' appear in the document above? Reply with just the number."

# Hard mode: counting across the WHOLE context (aggregation), short vs long.
for label, doc in [("short (1 handbook)", filler), ("long  (20 handbooks)", "\n\n".join([filler] * 20))]:
    truth = len(re.findall(r"\breimbursement\b", doc, re.I))      # counting with code is always right
    m = client.messages.create(model=MODEL, max_tokens=4000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"{doc}\n\n{Q}"}])
    ans = "".join(b.text for b in m.content if b.type == "text").strip()
    print(f"{label}: true={truth}  model said={ans!r}")
