from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

model_name = "gpt-4o-mini"  # Ensure Up-to-date model initialization

# List of documents to process
documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/how-open-port-linux/",
    "https://beebom.com/linux-vs-windows/",
]


def scrape_docs(urls: List[str]) -> List[Dict]:
    """Scrape content from URLs using SeleniumURLLoader"""
    try:
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        print(f"\nSuccessfully loaded {len(raw_docs)} documents")

        # Print some information about the loaded documents
        for doc in raw_docs:
            print(f"\nSource: {doc.metadata.get('source', 'No source')}")
            print(f"Content length: {len(doc.page_content)} characters")

        return raw_docs

    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return []


def create_vector_store(texts: List[str], metadatas: List[Dict]):
    """Create vector store using ChromaDB"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Ensure correct implementation
    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)
    return db


def setup_qa_chain(db):
    """Set up QA chain with polite response template"""
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = db.as_retriever()

    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    Please provide a polite and helpful response to the following question, utilizing the provided context. 
    Ensure that the tone remains professional, courteous, and empathetic, and tailor your response to directly address the inquiry. 

### Context:
{context}

### Question: 
{question}

### Polite Response:
In your response, consider including:
- Acknowledge the user’s query and express gratitude for the opportunity to assist.
- Provide a clear and concise answer that directly addresses the question.
- Use positive language and maintain a supportive tone throughout.
- If applicable, include relevant information or resources that could help further.
- Conclude by inviting any follow-up questions or providing encouragement for the user’s pursuit of information."""
    )

    # Create the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever  # Return both chain and retriever


def split_documents(pages_content: List[Dict]) -> tuple:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        # Extract text from Document object
        text = document.page_content  # Changed from document to document.page_content
        source = document.metadata.get("source", "")  # Get source from metadata

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadatas


def process_query(chain_and_retriever, query: str):
    """Process a query and return response"""
    try:
        chain, retriever = chain_and_retriever  # Unpack the tuple

        # Get the response
        response = chain.invoke(query)

        # Get the sources using the retriever
        docs = retriever.invoke(query)
        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])

        return {"answer": response, "sources": sources_str}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "I apologize, but I encountered an error while processing your question.",
            "sources": "",
        }


def main():

    # 1. Scrape documents
    print("Scraping documents...")
    pages_content = scrape_docs(documents)

    # 2. Split documents
    print("Splitting documents...")
    all_texts, all_metadatas = split_documents(pages_content)

    # 3. Create vector store
    print("Creating vector store...")
    db = create_vector_store(all_texts, all_metadatas)

    # 4. Set up QA chain
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(db)

    # 5. Interactive query loop
    print("\nReady for questions! (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your question: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        result = process_query(qa_chain, query)

        print("\nResponse:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"].split(","):
                print("- " + source.strip())


if __name__ == "__main__":
    main()
