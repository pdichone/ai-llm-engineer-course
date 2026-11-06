from openai import OpenAI
import sys


def simple_chat_without_memory(
    user_input: str,
    use_ollama: bool = True,
) -> str:
    """
    This function demonstrates a chatbot WITHOUT memory/context management.
    Each call is independent and has no knowledge of previous interactions.
    """
    # Initialize OpenAI API (or Ollama)
    if use_ollama:
        client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        model_name = "llama3.2"
    else:
        client = OpenAI()
        model_name = "gpt-4o-mini"

    # Each call only includes the current message
    try:
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    # Model selection
    print("\n=== Simple Chatbot WITHOUT Memory ===")
    print("Notice how the bot won't remember anything from previous messages!")
    print("\nSelect model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter either 1 or 2")

    use_ollama = choice == "2"

    # Print instructions
    print("\n=== Chat Session Started ===")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the screen")
    print("Each message is independent - the bot has no memory of previous messages!\n")

    # Main chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit commands
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye! 👋")
            sys.exit()

        # Check for clear command
        if user_input.lower() == "clear":
            # Clear screen (works on both Windows and Unix-like systems)
            print("\033[H\033[J", end="")
            continue

        # Skip empty inputs
        if not user_input:
            continue

        # Get and print response
        response = simple_chat_without_memory(user_input, use_ollama)
        print(f"\nBot: {response}")

        # Visual separator for better readability
        print("\n" + "-" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat session ended by user. Goodbye! 👋")
        sys.exit()
