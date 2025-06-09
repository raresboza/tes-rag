from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import shutil
import os

class VectorStoreManager:
    def __init__(self, collection_name="tes_collection", persist_directory="./tes_collection_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        self.vector_store = None
        self._load_vector_store()

    def _load_vector_store(self):
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    @staticmethod
    def chunk_documents(docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
        return splitter.split_documents(docs)

    def add_documents(self, docs: List[Document]):
        chunks = self.chunk_documents(docs=docs)
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()

    def reset(self):
        """Delete and reinitialize the vector DB."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        self._load_vector_store()

    def search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)
