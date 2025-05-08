import os
import yaml
from typing import List, Set
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class VectorstoreManager:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.chroma_path = self.config.get("vector_db", "./vector_db")
        model_name = self.config.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.vs = None

    def load_vectorstore(self) -> None:
        self.vs = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)

    def add_documents(self, chunks: List[Document]) -> None:
        if self.vs is None:
            self.load_vectorstore()

        existing_ids: Set[str] = set()
        try:
            store_data = self.vs.get(include=[])
            existing_ids = set(store_data["ids"])
        except Exception:
            pass  # new DB or error retrieving ids

        new_chunks = [doc for doc in chunks if doc.metadata["id"] not in existing_ids]
        new_ids = [doc.metadata["id"] for doc in new_chunks]

        if new_chunks:
            print(f"🆕 Adding {len(new_chunks)} new documents.")
            self.vs.add_documents(new_chunks, ids=new_ids)
            if self.vectorstore_type == "chroma":
                self.vs.persist()
        else:
            print("✅ No new documents to add — already up to date.")
            
    def needs_update(self, chunks: List[Document]) -> bool:
        if self.vs is None:
            return True
        try:
            existing_ids = set(self.vs.get(include=["ids"])["ids"])
        except Exception:
            return True
        new_ids = {doc.metadata["id"] for doc in chunks}
        return not existing_ids.issuperset(new_ids)

    def delete_vectorstore(self) -> None:
        if os.path.exists(self.chroma_path):
            for file in os.listdir(self.chroma_path):
                os.remove(os.path.join(self.chroma_path, file))
            os.rmdir(self.chroma_path)
            print(f"🗑️ Deleted vectorstore at {self.chroma_path}")
        else:
            print("⚠️ Vectorstore directory does not exist.")
