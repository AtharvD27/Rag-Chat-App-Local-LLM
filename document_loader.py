import os
from typing import List, Dict
from abc import ABC, abstractmethod
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseDocumentLoader(ABC):
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.chunk_size = config.get("chunk", {}).get("size", 800)
        self.chunk_overlap = config.get("chunk", {}).get("overlap", 80)

    @abstractmethod
    def load(self) -> List[Document]:
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return self.assign_chunk_ids(chunks)

    @staticmethod
    def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
        chunk_index_tracker: Dict[str, int] = {}

        for doc in chunks:
            file = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
            page = doc.metadata.get("page", -1)
            key = f"{file}:{page}"

            idx = chunk_index_tracker.get(key, 0)
            chunk_index_tracker[key] = idx + 1

            doc.metadata["file"] = file
            doc.metadata["page"] = page
            doc.metadata["chunk"] = idx
            doc.metadata["id"] = f"{file}:{page}:{idx}"

        return chunks


class PDFDirectoryLoader(BaseDocumentLoader):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        loader = PyPDFDirectoryLoader(self.path)
        return loader.load()
