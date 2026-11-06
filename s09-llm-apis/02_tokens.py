from dotenv import load_dotenv; load_dotenv()
import tiktoken, anthropic

text = open("data/handbook.md").read()

# 1) See tokens: an OpenAI tokenizer, used here as an illustration of how text splits
enc = tiktoken.get_encoding("o200k_base")
ids = enc.encode("Unbelievably, tokenization isn't word-based.")
print([enc.decode([i]) for i in ids])

# 2) Count the same handbook on two Claude tokenizer generations
client = anthropic.Anthropic()
for model in ["claude-haiku-4-5", "claude-sonnet-5-5"]:   # older vs newer tokenizer
    n = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens
    print(f"{model}: {n:,} tokens  ({len(text.split()):,} words)")
