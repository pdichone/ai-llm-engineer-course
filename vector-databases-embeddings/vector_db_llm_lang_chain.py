import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")


# load documents
loader = DirectoryLoader(
    path="./data/new_articles/", glob="*.txt", loader_cls=TextLoader
)
document = loader.load()

# split text into sentences
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=1000,
    chunk_overlap=20,
)
documents = text_splitter.split_documents(document)
print(f"Number of documents: {len(documents)}")

# get embeddings
embedding = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")

# Next we instantiate the Chroma object from langchain_community.vectorstores
persits_directory = "./db/chroma_db_real_world"
vectordb = Chroma.from_documents(
    documents=documents, embedding=embedding, persist_directory=persits_directory
)  # This will create the Chroma object and persist the embeddings to the directory

# Now we can query the Chroma object for similar sentences
retriever = vectordb.as_retriever()

# res_docs = retriever.invoke("how much did microsoft raise?", k=2)
# print(res_docs)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


response = rag_chain.invoke({"input": "talk about databricks news"})
res = response["answer"]

print(res)  # This will print the answer to the question
