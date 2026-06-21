
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

load_dotenv()


def create_vector_db():
    # Project root directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # data folder path
    DATA_PATH = os.path.join(BASE_DIR, "data")

    print(f"Loading PDFs from: {DATA_PATH}")

    # Load PDFs
    extracted_data = load_pdf_file(DATA_PATH)
    print(f"Loaded {len(extracted_data)} pages.")

    # Keep minimal metadata
    minimal_docs = filter_to_minimal_docs(extracted_data)

    # Split into chunks
    text_chunks = text_split(minimal_docs)
    print(f"Created {len(text_chunks)} chunks.")

    # Load embeddings
    print("Loading embedding model...")
    embeddings = download_hugging_face_embeddings()

    # Create FAISS vector store
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings,
    )

    # Save index
    FAISS_PATH = os.path.join(
        BASE_DIR,
        "faiss_index"
    )

    vectorstore.save_local(FAISS_PATH)

    print(f"FAISS index saved at: {FAISS_PATH}")
    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vector_db()

