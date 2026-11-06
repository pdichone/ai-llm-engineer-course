from dotenv import load_dotenv; load_dotenv()
import anthropic
from common.pricing import cost
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
handbook = open("data/handbook.md").read()        # ~4-5K tokens, well above the 512 minimum


def ask(q):
    return client.messages.create(
        model=MODEL, max_tokens=1000, output_config={"effort": "low"},
        system=[{"type": "text", "text": "Answer only from this handbook:\n" + handbook,
                 "cache_control": {"type": "ephemeral"}}],          # cache the stable prefix
        messages=[{"role": "user", "content": q}],
    )


for q in ["How many vacation days do new hires get?", "What is the expense limit for meals?"]:
    m = ask(q)
    u = m.usage
    print(f"write={u.cache_creation_input_tokens} read={u.cache_read_input_tokens} "
          f"in={u.input_tokens} out={u.output_tokens}  cost=${cost(MODEL, u):.5f}")
