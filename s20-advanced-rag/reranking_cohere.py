import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import cohere

# must install this: pip install "unstructured[pdf]"
## Must pip install torch transformers sentence-transformers cohere

load_dotenv()
# Get your cohere API key on: www.cohere.com
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


collection_name = "pdf_collection"


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_documents(self, data_directory: str) -> List[Document]:
        """Load documents from a directory."""
        try:
            if not os.path.exists(data_directory):
                st.error(f"Directory does not exist: {data_directory}")
                return []

            # Look for various document types
            loader = DirectoryLoader(
                data_directory, glob="**/*.*", show_progress=True  # Load all file types
            )

            documents = loader.load()
            st.info(f"Loaded {len(documents)} documents from {data_directory}")
            return documents

        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            splits = self.text_splitter.split_documents(documents)
            st.info(f"Split documents into {len(splits)} chunks")
            return splits
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents


class ChromaDBManager:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_function = OpenAIEmbeddings()
        os.makedirs(persist_directory, exist_ok=True)

    def create_or_load_db(
        self,
        data_directory: str = "data",
        collection_name: str = collection_name,  ## make sure the collection name is the same!
    ) -> Chroma:
        """Creates a new database or loads existing one."""
        try:
            # First try to load existing database
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                st.info("Loading existing ChromaDB...")
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory,
                )
                st.success("Successfully loaded existing database!")
                return vector_store

            # If no existing DB, create new one from documents
            st.info("No existing database found. Creating new one...")

            # Process documents
            processor = DocumentProcessor()
            documents = processor.load_documents(data_directory)
            if not documents:
                st.error("No documents found to process!")
                return None

            splits = processor.split_documents(documents)
            st.info(f"Created {len(splits)} document chunks")

            # Create and persist new vector store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_function,
                collection_name=collection_name,
                persist_directory=self.persist_directory,
            )

            st.success("Successfully created and persisted new database!")
            return vector_store

        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            return None


class CohereReranker:
    def __init__(self):
        self.co = co  # cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 3
    ) -> List[Dict]:
        try:
            # Convert Document objects to strings
            docs = [str(doc.page_content) for doc in documents]

            # Perform reranking
            response = self.co.rerank(
                model="rerank-v3.5", query=query, documents=docs, top_n=top_k
            )

            # Process results using the response.results list
            reranked_results = []

            for result in response.results:
                reranked_results.append(
                    {
                        "document": documents[result.index],
                        "relevance_score": float(result.relevance_score),
                        "index": result.index,
                    }
                )

            return reranked_results

        except Exception as e:
            st.error(f"Reranking error: {str(e)}")
            # Fallback: return original documents with default scoring
            return [
                {"document": doc, "relevance_score": 1.0 - (i * 0.1), "index": i}
                for i, doc in enumerate(documents[:top_k])
            ]


class RAGSystem:
    def __init__(self, persist_directory: str):
        self.db_manager = ChromaDBManager(persist_directory)
        self.vector_store = self.db_manager.create_or_load_db()
        self.reranker = CohereReranker()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            # Initial retrieval - get more documents initially for reranking
            initial_results = self.vector_store.similarity_search(query, k=top_k * 3)

            if not initial_results:
                return {
                    "answer": "No relevant documents found.",
                    "reranked_results": [],
                    "context": "",
                }

            # Rerank results
            reranked_results = self.reranker.rerank(
                query=query,
                documents=initial_results,
                top_k=min(top_k, len(initial_results)),
            )

            # Take only the top k reranked results
            top_reranked = reranked_results[:top_k]

            # Prepare context with proper formatting
            context_parts = []
            for i, result in enumerate(top_reranked, 1):
                doc = result["document"]
                score = result["relevance_score"]
                content = doc.page_content
                context_parts.append(f"[Document {i} (Score: {score:.3f})]:\n{content}")

            context = "\n\n".join(context_parts)

            # Generate answer
            prompt = PromptTemplate(
                template="""Based on the provided context, answer the question comprehensively.
                Include relevant quotes and cite the documents using their numbers [Doc X].
                If the information cannot be found in the context, say so.

                Context:
                {context}

                Question: {question}

                Please provide a detailed answer that:
                1. Directly addresses the question
                2. Uses specific citations [Doc X]
                3. Includes relevant quotes when appropriate
                4. Indicates confidence based on document relevance scores

                Answer:""",
                input_variables=["context", "question"],
            )

            response = self.llm.invoke(prompt.format(context=context, question=query))

            # Return results
            return {
                "answer": response.content,
                "reranked_results": top_reranked,  # Return the actual reranked results
                "context": context,
            }

        except Exception as e:
            st.error(f"Query error: {str(e)}")
            st.error(f"Exception type: {type(e)}")
            import traceback

            st.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": "An error occurred while processing your query.",
                "reranked_results": [],
                "context": "",
            }


def display_results(result):
    if not result["reranked_results"]:
        st.warning("No results found.")
        return

    # Display answer
    if result["answer"]:
        st.markdown("### 💡 Answer")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 20px; 
            border-radius: 10px;'>{result['answer']}</div>""",
            unsafe_allow_html=True,
        )

    # Display reranking stats
    st.markdown("### 📊 Reranking Statistics")

    scores = [doc["relevance_score"] for doc in result["reranked_results"]]
    if scores:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
        with cols[1]:
            st.metric("Max Score", f"{max(scores):.3f}")
        with cols[2]:
            st.metric("Min Score", f"{min(scores):.3f}")

    # Display sources
    st.markdown("### 📚 Reranked Sources")

    for i, doc in enumerate(result["reranked_results"], 1):
        score = doc["relevance_score"]
        confidence = (
            "High Relevance 🟢"
            if score >= 0.7
            else "Medium Relevance 🟡" if score >= 0.4 else "Low Relevance 🔴"
        )

        with st.expander(f"Document {i} - {confidence} (Score: {score:.3f})"):
            st.markdown(
                f"""
            **Relevance Score:** {score:.3f}
            
            **Content:**
            ```
            {doc['document'].page_content}
            ```
            """
            )


def main():

    # Question to ask: what was the operating income in in 2023?
    st.set_page_config(page_title="RAG with Cohere Reranking", layout="wide")
    st.title("RAG System with Cohere Reranking")

    # Initialize paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, "data")
    persist_directory = os.path.join(current_dir, "chromadb")

    # Create directories
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize RAG system
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing system..."):
            rag_system = RAGSystem(persist_directory)
            st.session_state.rag_system = rag_system

    # Query interface
    st.header("Query Interface")
    query = st.text_input("Enter your question:")
    top_k = st.slider("Number of documents to retrieve", 1, 10, 3)

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching and reranking..."):
                result = st.session_state.rag_system.query(query, top_k)
                display_results(result)


if __name__ == "__main__":
    main()
