from openai import OpenAI
import logging
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    """Configure logging to save logs in JSON format"""
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    # Create a file handler for JSON logs
    file_handler = logging.FileHandler("chatbot_logs.json")
    formatter = logging.Formatter("%(message)s")  # Log raw JSON
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler for human-readable logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    return logger


def initialize_client(use_ollama: bool = True) -> OpenAI:
    """Initialize OpenAI client for either OpenAI or Ollama"""
    if use_ollama:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return OpenAI()


class ChatBot:
    def __init__(self, use_ollama: bool = False):
        self.logger = setup_logging()
        self.session_id = str(uuid.uuid4())
        self.client = initialize_client(use_ollama)
        self.use_ollama = use_ollama
        self.model_name = "llama3.2" if use_ollama else "gpt-4o-mini"

        # Initialize conversation with a system message
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful customer support assistant.",
            }
        ]

    def chat(self, user_input: str) -> str:
        try:
            # Log user input with metadata
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "user_input",
                "user_input": user_input,
                "metadata": {"session_id": self.session_id, "model": self.model_name},
            }
            self.logger.info(json.dumps(log_entry))

            # Append user message to the conversation
            self.messages.append({"role": "user", "content": user_input})

            # Generate a response using the API
            start_time = datetime.now()
            response = self.client.chat.completions.create(
                model=self.model_name, messages=self.messages
            )
            end_time = datetime.now()

            # Calculate response time
            response_time = (end_time - start_time).total_seconds()

            # Extract the assistant's response
            assistant_response = response.choices[0].message.content

            # Log the model's response and performance metrics
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "model_response",
                "model_response": assistant_response,
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name,
                    "response_time_seconds": response_time,
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else None
                    ),
                },
            }
            self.logger.info(json.dumps(log_entry))

            # Append assistant's response to the conversation
            self.messages.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            # Log any errors that occur
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "error",
                "error_message": str(e),
                "metadata": {"session_id": self.session_id, "model": self.model_name},
            }
            self.logger.error(json.dumps(log_entry))
            return f"Sorry, something went wrong: {str(e)}"


def main():
    # Model selection
    print("\nSelect model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter either 1 or 2")

    use_ollama = choice == "2"

    # Initialize chatbot
    chatbot = ChatBot(use_ollama)

    print("\n=== Chat Session Started ===")
    print(f"Using {'Ollama' if use_ollama else 'OpenAI'} model")
    print("Type 'exit' to end the conversation")
    print(f"Session ID: {chatbot.session_id}\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        response = chatbot.chat(user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat session ended by user. Goodbye! 👋")
