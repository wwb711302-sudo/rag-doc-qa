"""Chroma 向量库封装。"""
from __future__ import annotations

from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, embeddings, persist_dir: str) -> None:
        self._embeddings = embeddings
        self._persist_dir = persist_dir
        self._store: Chroma | None = None

    def build(self, docs: List[Document]) -> None:
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=self._embeddings,
            persist_directory=self._persist_dir,
        )

    def load(self) -> None:
        self._store = Chroma(
            persist_directory=self._persist_dir,
            embedding_function=self._embeddings,
        )

    def search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        if self._store is None:
            self.load()
        # score 为距离（越小越相关），显式升序排序保证最佳结果在前
        results = self._store.similarity_search_with_score(query, k=top_k)
        return sorted(results, key=lambda x: x[1])

    @property
    def count(self) -> int:
        if self._store is None:
            return 0
        return self._store._collection.count()  # type: ignore[attr-defined]
