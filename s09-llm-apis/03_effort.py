from dotenv import load_dotenv; load_dotenv()
import time, anthropic
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
TASK = ("A warehouse ships 3 sizes of box. Small holds 4 units, medium 9, large 20. "
        "What is the fewest boxes to ship exactly 87 units? Show the combination.")

# Step 1: the old habit. Capture the receipt.
try:
    client.messages.create(model=MODEL, max_tokens=200, temperature=0,
                           messages=[{"role": "user", "content": TASK}])
    print("No error: temperature was accepted")
except anthropic.BadRequestError as e:      # API rejects it
    print("API said:", e.status_code, e.message)
except TypeError as e:                      # or the SDK no longer accepts the argument
    print("SDK said:", e)

# Step 2: effort is the control now
for effort in ["low", "high"]:
    t = time.perf_counter()
    msg = client.messages.create(
        model=MODEL, max_tokens=16000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": TASK}],
    )
    secs = time.perf_counter() - t
    answer = "".join(b.text for b in msg.content if b.type == "text")
    print(f"\n[{effort}] {secs:.1f}s, {msg.usage.output_tokens} output tokens\n{answer[:400]}")
