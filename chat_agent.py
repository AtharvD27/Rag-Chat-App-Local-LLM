from typing import Tuple, List, Dict
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.llms.base import LLM


class ChatAgent:
    def __init__(self, llm: LLM, retriever: VectorStoreRetriever, memory: ConversationBufferMemory = None):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = self._create_chain()

    def _create_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )

    def ask(self, query: str) -> Tuple[str, List[Dict]]:
        result = self.chain({"question": query})
        answer = result["answer"]
        sources = self._extract_sources(result["source_documents"])
        return answer, sources

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        extracted = []
        for doc in docs:
            meta = doc.metadata
            extracted.append({
                "file": meta.get("file", "unknown"),
                "page": meta.get("page", -1),
                "chunk": meta.get("chunk", -1),
                "text": doc.page_content.strip()
            })
        return extracted
