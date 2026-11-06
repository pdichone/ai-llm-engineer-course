from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pprint
import re

load_dotenv()


# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


documents = TextLoader("./doc/dream.txt").load()
# print(document[:10])

# Clean the text
cleaned_documents = [clean_text(doc.page_content) for doc in documents]

# print(cleaned_documents)

# Split the text into characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# texts = text_splitter.split_text(cleaned_documents[0])
texts = text_splitter.split_documents(documents)

# cleanup the text
texts = [clean_text(text.page_content) for text in texts]

# print(texts)

# Load the OpenAI embeddings to vectorize the text
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# create the retriever from the loaded embeddings and documents
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 2})


# Query the retriever
# query = "what did Martin Luther King Jr. dream about?"
query = "Give me a summary of the speech in bullet points"
docs = retriever.invoke(query)

pprint.pprint(f" => DOCS: {docs}:")

# Chat with the model and our docs

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# # Create the chat prompt
prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs},and answer the following question {query}",
)

# # Create a chat model
model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(f"Model Response::: \n \n{response}")
