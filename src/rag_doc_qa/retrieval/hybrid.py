"""混合检索：向量 + BM25，RRF 融合后经 CrossEncoder 重排序。"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from ..config import RetrievalConfig
from ..store.bm25 import BM25Index
from ..store.vectorstore import VectorStore
from .reranker import BGEReranker


def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = 60) -> Dict[str, float]:
    """对多路排序结果做 RRF 融合，返回 chunk_id -> 融合分数。"""
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            chunk_id = doc.metadata.get("chunk_id", "")
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever(BaseRetriever):
    """向量相似度 + BM25 关键词混合召回，RRF 融合后重排序。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: VectorStore = Field(exclude=True)
    bm25_index: BM25Index = Field(exclude=True)
    reranker: BGEReranker = Field(exclude=True)
    retrieval: RetrievalConfig

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        # 1. 多路召回
        vector_docs = [d for d, _ in self.vectorstore.search(query, self.retrieval.vector_top_k)]
        bm25_docs = [d for d, _ in self.bm25_index.search(query, self.retrieval.bm25_top_k)]

        # 2. RRF 融合
        fused = reciprocal_rank_fusion([vector_docs, bm25_docs], k=self.retrieval.rrf_k)

        # 3. 按融合分数取候选并去重
        doc_by_id = {d.metadata.get("chunk_id", ""): d for d in vector_docs + bm25_docs}
        candidates = [
            doc_by_id[cid]
            for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)
            if cid in doc_by_id
        ][: self.retrieval.rerank_top_k]

        if not candidates:
            return []

        # 4. 重排序后取最终 Top-K
        return self.reranker.rerank(query, candidates, self.retrieval.final_top_k)
