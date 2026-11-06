import chromadb

from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()
croma_client = chromadb.PersistentClient(path="./db/chroma_persist")

collection = croma_client.get_or_create_collection(
    "my_story", embedding_function=default_ef
)
# Define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# define a query text
query_text = "find document related to technology company"

results = collection.query(
    query_texts=[query_text],
    n_results=2,
)

for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]

    print(
        f" For the query: {query_text}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})"
    )
