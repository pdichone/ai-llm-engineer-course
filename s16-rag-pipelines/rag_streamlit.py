import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv


# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )


class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


def generate_csv():
    facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {
            "id": 2,
            "fact": "The Apollo 11 mission landed the first humans on the Moon in 1969.",
        },
        {
            "id": 3,
            "fact": "The Hubble Space Telescope was launched in 1990 and has provided stunning images of the universe.",
        },
        {
            "id": 4,
            "fact": "Mars is the most explored planet in the solar system, with multiple rovers sent by NASA.",
        },
        {
            "id": 5,
            "fact": "The International Space Station (ISS) has been continuously occupied since November 2000.",
        },
        {
            "id": 6,
            "fact": "Voyager 1 is the farthest human-made object from Earth, launched in 1977.",
        },
        {
            "id": 7,
            "fact": "SpaceX, founded by Elon Musk, is the first private company to send humans to orbit.",
        },
        {
            "id": 8,
            "fact": "The James Webb Space Telescope, launched in 2021, is the successor to the Hubble Telescope.",
        },
        {"id": 9, "fact": "The Milky Way galaxy contains over 100 billion stars."},
        {
            "id": 10,
            "fact": "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        },
    ]

    with open("space_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)
    return facts


def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("space_facts")
    except:
        pass

    collection = client.create_collection(
        name="space_facts", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    return collection


def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)
    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )


def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def rag_pipeline(query, collection, llm_model, top_k=2):
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/documents given.",
            },
            {"role": "user", "content": augmented_prompt},
        ]
    )

    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt


def streamlit_app():
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("🚀 Space Facts RAG System")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama Llama2",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "chroma", "nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.facts = generate_csv()

        # Initialize models
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        # Setup ChromaDB
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # If models changed, reinitialize
    if (
        st.session_state.llm_model.model_type != llm_type
        or st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )

    # Display available facts
    with st.expander("📚 Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

    # Query input
    query = st.text_input(
        "Enter your question about space:",
        placeholder="e.g., What is the Hubble Space Telescope?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Response")
                st.write(response)

            with col2:
                st.markdown("### 📖 References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Show technical details in expander
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")


if __name__ == "__main__":
    streamlit_app()
