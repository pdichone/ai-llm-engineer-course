from dotenv import load_dotenv; load_dotenv()
import anthropic
client = anthropic.Anthropic()
for m in client.models.list():
    print(f"{m.id:28} context={m.max_input_tokens:>9,}  max_out={m.max_tokens:>7,}")
