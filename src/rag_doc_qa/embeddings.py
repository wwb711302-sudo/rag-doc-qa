"""BGE 嵌入封装。"""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class BGEEmbeddings(Embeddings):
    """BGE 嵌入模型封装，正确处理 BGE v1.x 的 query 指令前缀。

    - 文档侧：直接编码原文
    - 查询侧：按需拼接 query_instruction 前缀后编码
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        query_instruction: str = "",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        if self.query_instruction:
            text = self.query_instruction + text
        vec = self._model.encode(text, normalize_embeddings=self.normalize)
        return vec.tolist()
