from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

# from langchain_community.document_loaders import DirectoryLoader, TextLoader
dir_loader = DirectoryLoader("./data/", glob="**/*.txt")
dir_documents = dir_loader.load()

print("Directory Text Documents:", dir_documents)
