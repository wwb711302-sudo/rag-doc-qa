"""配置加载：YAML 可调参数 + 环境变量密钥。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.3
    max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    query_instruction: str = ""


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-base"
    device: str = "cpu"


class ChunkConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 10
    final_top_k: int = 5


class MemoryConfig(BaseModel):
    max_turns: int = 5


class PathsConfig(BaseModel):
    persist_dir: str = "data/chroma"
    bm25_index: str = "data/bm25_index.pkl"
    log_dir: str = "logs"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _project_root() -> Path:
    # src/rag_doc_qa/config.py -> 项目根目录
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> AppConfig:
    """加载配置（configs/config.yaml + .env），结果缓存。"""
    load_dotenv(_project_root() / ".env")
    if config_path is None:
        config_path = _project_root() / "configs" / "config.yaml"
    config_path = Path(config_path)
    data: dict = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def get_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请在项目根目录的 .env 中配置（参考 .env.example）")
    return key
