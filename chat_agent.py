from typing import Tuple, List, Dict
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.llms.base import LLM
import yaml


class ChatAgent:
    def __init__(self, llm: LLM, retriever: VectorStoreRetriever, memory: ConversationBufferMemory = None, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        prompt_path = config.get("prompt_path", "./prompts.yaml")
        self.prompts = yaml.safe_load(open(prompt_path))
        self.llm = llm
        self.retriever = retriever
        self.memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = self._create_chain()

    def _create_chain(self) -> ConversationalRetrievalChain:
        # 1. Prompt to generate standalone question
        rewrite_prompt = PromptTemplate.from_template(prompts["question_rewrite_prompt"])
        question_generator_chain = LLMChain(llm=self.llm, prompt=rewrite_prompt)

        # 2. Prompt to answer based on retrieved docs
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["answer_prompt_system"]),
            MessagesPlaceholder("chat_history"),
            ("human", prompts["answer_prompt_human"]),
        ])
        combine_docs_chain = create_stuff_documents_chain(llm=self.llm, prompt=answer_prompt)

        # 3. Conversational RAG chain with both components
        return ConversationalRetrievalChain(
            retriever=self.retriever,
            memory=self.memory,
            question_generator=question_generator_chain,
            combine_docs_chain=combine_docs_chain,
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
