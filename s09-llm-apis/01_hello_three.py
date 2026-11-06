from dotenv import load_dotenv
load_dotenv()

PROMPT = "In one sentence: what is a token in an LLM?"

# --- Anthropic ---
import anthropic
claude = anthropic.Anthropic()
msg = claude.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": PROMPT}],
)
if msg.stop_reason == "refusal":        # always check why it stopped
    print("Claude declined:", msg.stop_details)
else:
    text = "".join(b.text for b in msg.content if b.type == "text")  # skip thinking blocks
    print("Claude:", text, "| tokens in/out:", msg.usage.input_tokens, msg.usage.output_tokens)

# --- OpenAI (Responses API) ---
from openai import OpenAI
oai = OpenAI()
r = oai.responses.create(model="gpt-6-luna", input=PROMPT)
print("OpenAI:", r.output_text, "| tokens in/out:", r.usage.input_tokens, r.usage.output_tokens)

# --- Google (Interactions API) ---
from google import genai
g = genai.Client()
gi = g.interactions.create(model="gemini-3.8-flash", input=PROMPT)
print("Gemini:", gi.output_text)
