from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

pdf_loader = PyPDFLoader("./doc/linux-manual.pdf")

docs = pdf_loader.load()
print("PDF Documents:", docs)
