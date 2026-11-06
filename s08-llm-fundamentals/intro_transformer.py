from transformers import pipeline, AutoTokenizer


def create_simple_llm():
    """
    Creates a simple LLM using a small GPT-2 model.
    GPT-2 (smallest version) is perfect for demonstrations as it's:
    - Relatively small (124M parameters)
    - Fast enough to run on CPU
    - Good for understanding basic concepts
    """
    # Initialize the model and tokenizer
    model_name = "distilgpt2"  # Using DistilGPT-2 (smaller version of GPT-2)

    # Create the generator pipeline
    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


generator = create_simple_llm()

prompt = "Once upon a time"

# Generate text
generated_text = generator(prompt, max_length=1000, num_return_sequences=1)

print(generated_text[0]["generated_text"])
