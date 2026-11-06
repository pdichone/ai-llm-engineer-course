import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document


from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma  # Updated import

from chromadb.config import Settings
import shutil

import streamlit as st  # Optional for visualization

load_dotenv()


class ChromaDBManager:
    """
    Manages ChromaDB initialization and operations.
    """

    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embeddings
        self.embedding_function = OpenAIEmbeddings()

    def create_or_load_db(self, collection_name: str = "document_collection") -> Chroma:
        """Creates a new ChromaDB instance or loads existing one."""
        try:
            # Initialize Chroma with new package
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )

            print(f"Successfully initialized ChromaDB collection: {collection_name}")

            # Get collection info safely
            try:
                collection_size = (
                    len(vector_store.get()["ids"]) if vector_store.get() else 0
                )
                print(f"Collection size: {collection_size} documents")
            except:
                print("New collection created")

            return vector_store

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return None

    def reset_database(self):
        """Resets the database by removing all data."""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
                print(f"Reset database at {self.persist_directory}")
        except Exception as e:
            print(f"Error resetting database: {e}")


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_pdfs(self, pdf_directory: str) -> List[Document]:
        """Load PDF documents from a directory."""
        try:
            if not os.path.exists(pdf_directory):
                st.error(f"Directory does not exist: {pdf_directory}")
                return []

            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
            if not pdf_files:
                st.warning("No PDF files found in directory")
                return []

            loader = DirectoryLoader(
                pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader
            )
            st.info(f"Found {len(pdf_files)} PDF files")
            documents = loader.load()
            st.success(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            st.error(f"Error loading PDFs: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            splits = self.text_splitter.split_documents(documents)
            st.info(f"Split into {len(splits)} chunks")
            return splits
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents

    def process_and_store(
        self, documents: List[Document], vector_store: Chroma
    ) -> bool:
        """Process documents and store in vector store."""
        try:
            if not documents:
                st.warning("No documents to process")
                return False

            # Add documents to vector store
            vector_store.add_documents(documents)

            # In the new version, we don't need to explicitly persist
            st.success(f"Successfully added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            st.error(f"Error storing documents: {str(e)}")
            return False


class QueryExpander:
    """
    Expands a single query into multiple semantically similar variations.
    """

    def __init__(self, temperature: float = 0):
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-4o-mini")

        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question, generate 3 different versions that capture 
            different aspects and perspectives of the original question. 
            Make the variations semantically diverse but relevant.
            
            Original Question: {question}
            
            Generate variations in the following format:
            1. [First variation]
            2. [Second variation]
            3. [Third variation]
            
            Only output the numbered variations, nothing else.""",
        )

    def expand_query(self, question: str) -> List[str]:
        """
        Expand a single query into multiple variations.

        Args:
            question: Original question to expand

        Returns:
            List of query variations including the original
        """
        try:
            response = self.llm.invoke(
                self.query_expansion_prompt.format(question=question)
            )
            variations = [
                line.split(". ")[1] for line in response.content.strip().split("\n")
            ]
            variations.append(question)
            return variations
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [question]


class QueryExpansionRAG:
    """
    RAG system with query expansion capabilities.
    """

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.query_expander = QueryExpander()
        self.retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def retrieve_with_expansion(
        self, question: str, top_k: int = 5
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents using query expansion.

        Args:
            question: Original question
            top_k: Number of documents to retrieve per query

        Returns:
            Dictionary mapping queries to their retrieved documents
        """
        expanded_queries = self.query_expander.expand_query(question)
        results = {}

        for query in expanded_queries:
            docs = self.retriever.invoke(query)
            results[query] = docs[:top_k]

        return results


### === Add the Answer Generator ===  Final part###
# Add this new class for final answer generation
class AnswerGenerator:
    """
    Generates final answer from multiple document sources using LLM with proper citations.
    """

    def __init__(self, temperature: float = 0):
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-4o-mini")

        self.answer_generation_prompt = PromptTemplate(
            input_variables=["question", "formatted_context"],
            template="""You are a highly knowledgeable assistant tasked with providing 
            comprehensive answers based on multiple document sources. Your goal is to 
            synthesize information accurately and provide well-structured responses.

            Question: {question}

            Below are relevant excerpts from different documents, each with a citation ID:
            {formatted_context}

            Please provide a detailed response that:
            1. Directly answers the question
            2. Uses specific citations in the format [CitationID] when referencing information
            3. Synthesizes information from multiple sources when relevant
            4. Highlights any contradictions between sources
            5. Uses quoted snippets from the sources when particularly relevant

            Format your response as follows:

            DIRECT ANSWER:
            [Concise answer with citations]

            DETAILED EXPLANATION:
            [Detailed explanation with citations and quoted snippets when relevant]

            KEY POINTS:
            - [Point 1 with citation]
            - [Point 2 with citation]
            - [Point 3 with citation]

            SOURCES CITED:
            [List the citation IDs used and their key contributions]

            Answer:""",
        )

    def _prepare_citation_chunks(
        self, results: Dict[str, List[Document]], max_chunk_length: int = 250
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Prepare context with citations and create a citation map.

        Args:
            results: Dictionary of query->documents mappings
            max_chunk_length: Maximum length for document chunks

        Returns:
            Tuple of (formatted_context, citation_map)
        """
        citation_id = 1
        citation_chunks = []
        citation_map = {}

        for query, docs in results.items():
            for doc in docs:
                # Create a truncated chunk with context
                content = doc.page_content
                truncated_content = content[:max_chunk_length]
                if len(content) > max_chunk_length:
                    truncated_content += "..."

                # Store the citation
                citation_ref = f"[Citation{citation_id}]"
                citation_chunks.append(f"{citation_ref}:\n{truncated_content}\n")
                citation_map[citation_ref] = {
                    "content": truncated_content,
                    "full_content": content,
                    "query": query,
                }
                citation_id += 1

        formatted_context = "\n".join(citation_chunks)
        return formatted_context, citation_map

    def generate_answer(
        self, question: str, results: Dict[str, List[Document]]
    ) -> Dict[str, Any]:
        """
        Generate final answer from multiple search results with citations.

        Args:
            question: Original question
            results: Dictionary of query->documents mappings

        Returns:
            Dictionary containing answer and citation information
        """
        try:
            # Prepare context with citations
            formatted_context, citation_map = self._prepare_citation_chunks(results)

            # Generate answer using LLM
            response = self.llm.invoke(
                self.answer_generation_prompt.format(
                    question=question, formatted_context=formatted_context
                )
            )

            return {
                "answer": response.content,
                "citations": citation_map,
                "formatted_context": formatted_context,
            }

        except Exception as e:
            st.error(f"Error generating final answer: {str(e)}")
            return {
                "answer": "Failed to generate answer due to an error.",
                "citations": {},
                "formatted_context": "",
            }


def main():
    st.set_page_config(page_title="RAG Query Expansion", layout="wide")

    st.title("RAG System with Query Expansion")

    # Initialize paths with absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_directory = os.path.join(current_dir, "data")
    persist_directory = os.path.join(current_dir, "chromadb")

    # Create directories if they don't exist
    os.makedirs(pdf_directory, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    # Display directory information
    st.sidebar.title("System Information")
    st.sidebar.info(f"PDF Directory: {pdf_directory}")
    st.sidebar.info(f"Database Directory: {persist_directory}")

    collection_name = "pdf_collection"

    # Sidebar controls
    st.sidebar.title("Controls")

    # File uploader for PDFs
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files")
        # Save uploaded files to pdf_directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(pdf_directory, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())

    # Database reset button
    if st.sidebar.button("Reset Database"):
        if "db_manager" in st.session_state:
            st.session_state.db_manager.reset_database()
            st.sidebar.success("Database reset successfully!")
            # Clear session state
            for key in ["db_manager", "vector_store", "last_results"]:
                if key in st.session_state:
                    del st.session_state[key]

    # Initialize system components when button is clicked
    if st.sidebar.button("Initialize System"):
        with st.spinner("Initializing system..."):
            try:
                # Create DB Manager
                db_manager = ChromaDBManager(persist_directory)
                st.session_state["db_manager"] = db_manager

                # Create or load vector store
                vector_store = db_manager.create_or_load_db(collection_name)

                if vector_store:
                    st.session_state["vector_store"] = vector_store
                    st.sidebar.success("System initialized!")
                else:
                    st.sidebar.error("Failed to initialize vector store")
            except Exception as e:
                st.sidebar.error(f"Error initializing system: {str(e)}")

    # Process documents when button is clicked
    if st.sidebar.button("Process Documents"):
        if "vector_store" not in st.session_state:
            st.sidebar.error("Please initialize the system first")
        else:
            with st.spinner("Processing documents..."):
                doc_processor = DocumentProcessor()
                documents = doc_processor.load_pdfs(pdf_directory)

                if documents:
                    splits = doc_processor.split_documents(documents)
                    success = doc_processor.process_and_store(
                        splits, st.session_state.vector_store
                    )
                    if success:
                        st.sidebar.success("Documents processed successfully!")
                    else:
                        st.sidebar.error("Failed to process documents")
                else:
                    st.sidebar.error("No documents found")

    # Main query interface
    st.header("Query Interface")

    # Text input for query
    query = st.text_input("Enter your question:")

    # Number of results slider
    k = st.slider("Number of results to return", min_value=1, max_value=10, value=3)

    ## What are the key findings presented?
    # Search button
    # Search button
    if st.button("Search"):
        if query and "vector_store" in st.session_state:
            with st.spinner("Processing query..."):
                try:
                    # Initialize query expander and answer generator
                    query_expander = QueryExpander()
                    answer_generator = AnswerGenerator()

                    # Expand query
                    expanded_queries = query_expander.expand_query(query)

                    # Display expanded queries
                    with st.expander("🔍 View Expanded Queries"):
                        for i, exp_query in enumerate(expanded_queries, 1):
                            st.write(f"{i}. {exp_query}")

                    # Search with each expanded query
                    all_results = {}
                    for exp_query in expanded_queries:
                        results = st.session_state.vector_store.similarity_search(
                            exp_query, k=k
                        )
                        all_results[exp_query] = results

                    # Generate final answer with citations
                    st.subheader("📝 Detailed Analysis")
                    with st.spinner("Generating comprehensive answer..."):
                        response_data = answer_generator.generate_answer(
                            query, all_results
                        )

                        # Display the answer
                        st.markdown(response_data["answer"])

                        # Display citations
                        st.subheader("📚 Source Citations")
                        for citation_id, citation_data in response_data[
                            "citations"
                        ].items():
                            with st.expander(f"{citation_id} - Click to view source"):
                                st.markdown("**Excerpt:**")
                                st.markdown(f"```\n{citation_data['content']}\n```")
                                st.markdown("**Original Query:**")
                                st.markdown(f"*{citation_data['query']}*")

                                # Option to view full content
                                if st.button(f"View Full Content for {citation_id}"):
                                    st.markdown("**Full Content:**")
                                    st.markdown(
                                        f"```\n{citation_data['full_content']}\n```"
                                    )

                    # Option to view all search results
                    with st.expander("🔎 View All Search Results"):
                        for query_text, docs in all_results.items():
                            st.markdown(f"**Query:** {query_text}")
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"*Document {i}:*")
                                st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                                st.markdown("---")

                    # Generate final synthesized answer
                    st.subheader("🎯 Final Answer")
                    with st.spinner("Synthesizing final answer..."):
                        final_prompt = PromptTemplate(
                            input_variables=["question", "detailed_answer"],
                            template="""Based on the detailed analysis provided, generate a clear, 
                            concise final answer to the original question. Focus on the most important 
                            points while maintaining accuracy.

                            Original Question: {question}

                            Detailed Analysis:
                            {detailed_answer}

                            Please provide a final answer that:
                            1. Directly addresses the question
                            2. Summarizes the key points
                            3. Is clear and concise
                            4. Maintains the crucial citations

                            Final Answer:""",
                        )

                        final_response = ChatOpenAI(temperature=0).invoke(
                            final_prompt.format(
                                question=query, detailed_answer=response_data["answer"]
                            )
                        )

                        # Display final answer in a highlighted box
                        st.markdown("---")
                        st.markdown("### 💡 Summary")
                        st.markdown(
                            f"""
                            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                            {final_response.content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    st.error("Full error:", exception=e)
        else:
            if "vector_store" not in st.session_state:
                st.error("Please initialize the system first")
            if not query:
                st.error("Please enter a query")

    # Display system status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    if "vector_store" in st.session_state:
        st.sidebar.success("System is initialized")
        try:
            doc_count = len(st.session_state.vector_store.get()["ids"])
            st.sidebar.info(f"Documents in database: {doc_count}")
        except:
            st.sidebar.info("Database is empty")
    else:
        st.sidebar.warning("System not initialized")


if __name__ == "__main__":
    main()
