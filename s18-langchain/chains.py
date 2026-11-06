from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Define a prompt template
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# Create a chat model
model = ChatOpenAI(model="gpt-4o-mini")

# Chain the prompt, model, and output parser
chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({"topic": "lions"})
print(response)
