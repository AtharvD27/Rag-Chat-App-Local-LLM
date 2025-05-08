import os
import yaml
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from abc import ABC, abstractmethod
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseDocumentLoader(ABC):
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        self.config = config or self._load_config(config_path)
        self.chunk_size = self.config.get("chunk", {}).get("size", 800)
        self.chunk_overlap = self.config.get("chunk", {}).get("overlap", 80)

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

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
            file = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", -1)
            key = f"{file}:{page}"

            idx = chunk_index_tracker.get(key, 0)
            chunk_index_tracker[key] = idx + 1

            doc.metadata["file"] = file
            doc.metadata["page"] = page
            doc.metadata["chunk"] = idx
            doc.metadata["id"] = f"{file}:{page}:{idx}"

        return chunks


class PDFLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.path, file))
                docs.extend(loader.load())
        return docs


class JSONLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".json"):
                with open(os.path.join(self.path, file), "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            content = entry.get("text") or json.dumps(entry)
                            docs.append(Document(page_content=content, metadata={"source": file}))
                    elif isinstance(data, dict):
                        content = data.get("text") or json.dumps(data)
                        docs.append(Document(page_content=content, metadata={"source": file}))
        return docs


class WebPageLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            full_path = os.path.join(self.path, file)
            if file.endswith(".txt"):  # each line is a URL
                with open(full_path, "r") as f:
                    for url in f.readlines():
                        url = url.strip()
                        try:
                            html = requests.get(url, timeout=10).text
                            text = self.extract_text(html)
                            docs.append(Document(page_content=text, metadata={"source": url}))
                        except Exception as e:
                            print(f"⚠️ Failed to load {url}: {e}")
            elif file.endswith(".html"):
                with open(full_path, "r", encoding="utf-8") as f:
                    html = f.read()
                    text = self.extract_text(html)
                    docs.append(Document(page_content=text, metadata={"source": file}))
        return docs

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)


class SmartDocumentLoader(BaseDocumentLoader):
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        super().__init__(config_path=config_path, config=config)
        self.path = self.config.get("data_path", "./data")

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".pdf"):
                loader = PDFLoader(self.path, config=self.config)
                docs.extend(loader.load())
                break
        for file in os.listdir(self.path):
            if file.endswith(".json"):
                loader = JSONLoader(self.path, config=self.config)
                docs.extend(loader.load())
                break
        for file in os.listdir(self.path):
            if file.endswith(".txt") or file.endswith(".html"):
                loader = WebPageLoader(self.path, config=self.config)
                docs.extend(loader.load())
                break
        return docs
