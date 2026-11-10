from dotenv import load_dotenv; load_dotenv()
import re, random, anthropic
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
filler = open("data/handbook.md").read()
Q = "Exactly how many times does the word 'reimbursement' appear in the document above? Reply with just the number."
WORD = re.compile(r"\breimbursement\b", re.I)

# v1 was too easy: 20 IDENTICAL copies -> count 5, multiply by 20.
# v2 removes the shortcut: every copy is different.
EXTRA = ["Submit the reimbursement form by Friday.",          # counts
         "Reimbursement requests over $500 need approval.",   # counts
         "All reimbursements are paid monthly.",              # distractor (plural)
         "Travel is reimbursed at cost."]                     # distractor (verb)

def make_copy(rng):
    paras = filler.split("\n\n")
    rng.shuffle(paras)                                         # no two copies look alike
    text = "\n\n".join(paras)
    text = WORD.sub(lambda m: m.group() if rng.random() < 0.6 else "refund", text)  # drop ~40%
    for _ in range(rng.randint(0, 4)):                         # sprinkle extras + distractors
        paras = text.split("\n\n")
        i = rng.randrange(len(paras))
        paras[i] += " " + rng.choice(EXTRA)
        text = "\n\n".join(paras)
    return text

def ask(doc):
    m = client.messages.create(model=MODEL, max_tokens=16000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"{doc}\n\n{Q}"}])
    text = "".join(b.text for b in m.content if b.type == "text")
    num = re.search(r"\d+", text)                                # tolerate "**5**" formatting
    return (num.group() if num else f"<none:{m.stop_reason}>"), m.usage.input_tokens

for trial in range(3):                                         # one run is a sample, not proof
    rng = random.Random(trial)
    short = make_copy(rng)
    long_ = "\n\n".join(make_copy(rng) for _ in range(20))
    for label, doc in [("short", short), ("long ", long_)]:
        truth = len(WORD.findall(doc))                         # code: always right, ~free
        ans, n = ask(doc)
        print(f"trial {trial} {label} {n:>7,} tok  true={truth:<3} model={ans:<14} {'OK' if ans == str(truth) else 'WRONG'}")
