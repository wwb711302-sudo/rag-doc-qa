"""BGE 重排序器。"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class BGEReranker:
    """基于 BGE CrossEncoder 的候选片段精排。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        if not isinstance(scores, list):
            scores = scores.tolist()
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]
