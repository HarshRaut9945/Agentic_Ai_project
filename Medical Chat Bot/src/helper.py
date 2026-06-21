
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer


def load_pdf_file(data_path: str):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(
    docs: List[Document],
) -> List[Document]:

    minimal_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source
                }
            )
        )

    return minimal_docs


def text_split(
    extracted_data: List[Document],
):
    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
        )
    )

    text_chunks = (
        text_splitter.split_documents(
            extracted_data
        )
    )

    return text_chunks


class SentenceTransformerEmbeddings(
    Embeddings
):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return (
            self.model.encode(texts).tolist()
        )

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        return (
            self.model.encode(text).tolist()
        )

    def __call__(self, text):
        return self.embed_query(text)


def download_hugging_face_embeddings():
    embeddings = (
        SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    return embeddings

