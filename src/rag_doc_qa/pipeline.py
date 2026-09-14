"""知识库构建与问答服务编排。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_openai import ChatOpenAI

from .chain import build_rag_chain, history_to_messages
from .config import AppConfig, get_api_key
from .document.loader import load_documents
from .document.splitter import split_documents
from .embeddings import BGEEmbeddings
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import BGEReranker
from .store.bm25 import BM25Index
from .store.vectorstore import VectorStore


class KnowledgeBase:
    """本地知识库：文档 -> 分块 -> 向量库 + BM25 索引。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings = BGEEmbeddings(
            model_name=config.embedding.model,
            device=config.embedding.device,
            query_instruction=config.embedding.query_instruction,
            normalize=config.embedding.normalize,
        )
        self.vectorstore = VectorStore(self.embeddings, config.paths.persist_dir)
        self.bm25_index = BM25Index()

    def build(self, paths) -> Tuple[int, int]:
        """构建知识库，返回 (文档数, 分块数)。"""
        docs = load_documents(paths)
        if not docs:
            raise ValueError("未发现可解析的文档（支持 PDF / Word / Markdown）")

        chunks = split_documents(docs, self.config.chunk.chunk_size, self.config.chunk.chunk_overlap)

        # 清空旧向量库后重建，避免重复写入
        persist_dir = Path(self.config.paths.persist_dir)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)

        self.vectorstore.build(chunks)
        self.bm25_index.build(chunks)
        self.bm25_index.save(self.config.paths.bm25_index)
        return len(docs), len(chunks)

    def load(self) -> "KnowledgeBase":
        self.vectorstore.load()
        self.bm25_index.load(self.config.paths.bm25_index)
        return self

    def is_ready(self) -> bool:
        return Path(self.config.paths.persist_dir).exists() and Path(
            self.config.paths.bm25_index
        ).exists()


class QAService:
    """多轮问答服务。"""

    def __init__(self, config: AppConfig, kb: KnowledgeBase) -> None:
        self.config = config
        llm = ChatOpenAI(
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=get_api_key(),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        retriever = HybridRetriever(
            vectorstore=kb.vectorstore,
            bm25_index=kb.bm25_index,
            reranker=BGEReranker(config.reranker.model, config.reranker.device),
            retrieval=config.retrieval,
        )
        self.chain = build_rag_chain(llm, retriever)

    def ask(self, question: str, history: List[Tuple[str, str]] | None = None) -> dict:
        history = history or []
        messages = history_to_messages(history, self.config.memory.max_turns)
        result = self.chain.invoke({"input": question, "chat_history": messages})
        return {"answer": result["answer"], "sources": result.get("context", [])}
