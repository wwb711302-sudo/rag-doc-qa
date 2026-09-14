"""递归字符切分。"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 针对中文优化：优先按换行/段落/中文标点切分
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def build_splitter(
    chunk_size: int = 500, chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def split_documents(
    docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 200
) -> List[Document]:
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    # 为每个 chunk 分配唯一 ID，供混合检索 RRF 融合时去重/对齐
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = f"{source}::{i}"
    return chunks
