# 系统架构

## 总体流程

```
PDF / Word / Markdown
        │  ① 加载 (loader.py)
        ▼
    Document 列表
        │  ② 递归字符切分 (splitter.py, overlap=200)
        ▼
     文本分块 (chunk)
        │
        ├── ③ BGE 向量化 → Chroma 持久化 (vectorstore.py)
        │
        └── ④ jieba 分词 → BM25 索引 (bm25.py)
                        │
                        ▼
                 本地知识库就绪
                        │
用户提问 ──► ⑤ 历史感知改写 (chain.py, CONTEXTUALIZE_PROMPT)
                        │
                        ▼
        ⑥ 混合检索 (hybrid.py)
        ├── 向量相似度 Top-K (vector_top_k)
        ├── BM25 关键词 Top-K (bm25_top_k)
        └── RRF 融合 → 候选集 (rerank_top_k)
                        │
                        ▼
        ⑦ BGE CrossEncoder 重排序 (reranker.py)
                        │
                        ▼
        ⑧ DeepSeek 生成回答 (QA_PROMPT, 带上下文约束)
                        │
                        ▼
             答案 + 来源溯源 (Gradio / CLI)
```

## 关键设计

1. **混合检索**：单靠向量检索对「专有名词/精确关键词」召回不足，单靠 BM25 缺乏语义泛化。
   二者互补，通过 Reciprocal Rank Fusion（RRF）按名次融合，无需调权重超参。

2. **重排序**：先多路召回较宽候选集（rerank_top_k=10），再用 CrossEncoder 精排，取 final_top_k=5，
   在召回率与准确率之间取得平衡，同时控制送入 LLM 的上下文长度与响应耗时。

3. **多轮记忆**：`create_history_aware_retriever` 先把「结合历史」的追问改写成独立检索问题，
   避免直接拿带指代（如「它的有效期多久」）的原始问题去检索导致召回失败；记忆保留最近
   `memory.max_turns` 轮。

4. **chunk_id 对齐**：切分时为每个分块写入唯一 `chunk_id`，作为 RRF 融合时去重/对齐的键，
   保证向量库与 BM25 索引对应同一份分块。

## 性能说明

- 目标「100+ 篇文档下相关度提升、单轮 < 2s」依赖：CPU/GPU 选择、rerank_top_k/final_top_k
  调小、模型大小（bge-small-zh-v1.5 更轻）。实际指标请用 `scripts/evaluate.py` 在自有评测集上复测，
  勿直接采信宣传数值。
