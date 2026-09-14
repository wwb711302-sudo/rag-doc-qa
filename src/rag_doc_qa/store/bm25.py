"""基于 jieba 分词的 BM25 关键词索引。"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import jieba
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> List[str]:
    return [w.strip() for w in jieba.cut(text) if w.strip()]


class BM25Index:
    def __init__(self) -> None:
        self._docs: List[Document] = []
        self._corpus: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, docs: List[Document]) -> None:
        self._docs = list(docs)
        # 空片段用占位符，避免 BM25 除零错误
        self._corpus = [tokenize(d.page_content) or ["__EMPTY__"] for d in docs]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._docs[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"docs": self._docs, "corpus": self._corpus}, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 索引不存在: {path}，请先构建知识库")
        with path.open("rb") as f:
            data = pickle.load(f)
        self._docs = data["docs"]
        self._corpus = data["corpus"]
        self._bm25 = BM25Okapi(self._corpus)

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None or len(self._docs) == 0
