import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Paths configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def create_vector_db():
    print("PDFs read ho rahi hain...")
    # Data folder se PDFs load karna
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Total Pages Loaded: {len(documents)}")

    # Text ko chote chunks mein todna
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total Chunks Created: {len(chunks)}")

    # Multilingual Embeddings Model load karna
    print("Vector Embeddings generate ho rahi hain...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ChromaDB mein save karna
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("Vector Database successfully create ho chuka hai!")

if __name__ == "__main__":
    create_vector_db()