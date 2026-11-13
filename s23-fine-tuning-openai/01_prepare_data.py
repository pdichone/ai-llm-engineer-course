"""S23 step A: build the TeaCrafters fine-tuning dataset.
Run from the repo root:  uv run s23-fine-tuning-openai/01_prepare_data.py
Writes teacrafters_train.jsonl + teacrafters_test.jsonl (OpenAI-style "messages" JSONL, the format
Unsloth, TRL and mlx-lm all read). Uses Claude Haiku to draft examples: review them before training."""
from dotenv import load_dotenv; load_dotenv()
import json, random
from pathlib import Path
import anthropic
from pydantic import BaseModel
from common.pricing import cost

HERE = Path(__file__).parent
MODEL = "claude-haiku-5-5"
SYSTEM = "You are the customer support assistant for TeaCrafters, an artisan tea subscription service."
POLICY = (HERE / "teacrafters_policies.md").read_text()

# The behaviour we want the small model to learn. It never sees these rules: only examples.
STYLE = """Answer rules:
1. Two or three short sentences.
2. Name the exact page or button in single quotes when there is one.
3. State the deadline or time limit when the policy has one.
4. If the policy sheet does not cover the question, say you don't have that detail and point to
   support@teacrafters.com or 1-800-TEA-CRAFT. Never guess.
5. Always end with: Happy steeping! – The TeaCrafters team"""

TOPICS = ["changing, skipping, pausing or cancelling a subscription", "shipping, tracking and addresses",
          "payment methods and failed payments", "returns, damaged boxes and missing teas",
          "gifts and plans", "the teas themselves: types, caffeine, sourcing, storage",
          "questions the policy sheet does NOT answer (allergens, wholesale, a specific country, "
          "store locations, brewing temperatures, jobs)"]


class Example(BaseModel):
    question: str
    answer: str


class Batch(BaseModel):
    examples: list[Example]


client = anthropic.Anthropic()
spent = 0.0


def draft(topic: str, n: int) -> list[Example]:
    global spent
    msg = client.messages.parse(
        model=MODEL, max_tokens=16000, output_config={"effort": "low"},
        output_format=Batch,
        messages=[{"role": "user", "content":
                   f"POLICY SHEET:\n{POLICY}\n\n{STYLE}\n\nWrite {n} realistic, varied customer questions about "
                   f"{topic}, each with the support answer. Vary tone and wording (typos, short, polite, annoyed). "
                   "Answers use only facts from the policy sheet."}],
    )
    spent += cost(MODEL, msg.usage)
    return msg.parsed_output.examples


# 1. Draft examples per topic (the last topic teaches the "I don't know, ask support" behaviour)
pairs = []
for topic in TOPICS:
    batch = draft(topic, 15)
    print(f"{len(batch):>3} examples · {topic[:45]}")
    pairs += batch

# 2. Validate: drop duplicates, empty answers, and answers that break the sign-off rule
seen, clean = set(), []
for p in pairs:
    key = p.question.lower().strip()
    if key in seen or not p.answer.strip() or "Happy steeping!" not in p.answer:
        continue
    seen.add(key)
    clean.append(p)
print(f"\n{len(pairs)} drafted → {len(clean)} kept after validation")

# 3. Shuffle, hold out a test set the model never trains on, write JSONL
random.seed(42)
random.shuffle(clean)
test, train = clean[:12], clean[12:]


def write(rows, name):
    with open(HERE / name, "w") as f:
        for p in rows:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": p.question},
                                             {"role": "assistant", "content": p.answer}]}) + "\n")
    print(f"wrote {name}: {len(rows)} examples")


write(train, "teacrafters_train.jsonl")
write(test, "teacrafters_test.jsonl")
print(f"\nSample:\nQ: {train[0].question}\nA: {train[0].answer}")
print(f"\nCost of drafting the dataset: ${spent:.4f}")
