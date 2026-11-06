# import getpass
# import os
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]


# print(messages)

# response = model.invoke(messages)

# print(response)

## Langchain promptTemplates: https://python.langchain.com/docs/tutorials/llm_chain/#prompt-templates
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Portuguese", "text": "Ola!"})

# print(prompt)

response = model.invoke(prompt)
print(response.content)
