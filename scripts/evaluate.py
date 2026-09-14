"""离线评估：对比「纯向量检索」与「混合检索」的召回命中率。

评测集格式（JSONL，每行一条）：
    {"query": "合同违约金比例是多少？", "expected_source": "合同模板.docx"}

命中判定：检索返回的 Top-K 文档中，是否存在 metadata["source"] 包含 expected_source 的文档。

用法：
    python scripts/evaluate.py --eval-file data/eval.jsonl --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_doc_qa.config import get_config  # noqa: E402
from rag_doc_qa.logging_config import setup_logging  # noqa: E402
from rag_doc_qa.pipeline import KnowledgeBase  # noqa: E402
from rag_doc_qa.retrieval.reranker import BGEReranker  # noqa: E402


def hit(sources, expected: str) -> bool:
    return any(expected in d.metadata.get("source", "") for d in sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True, help="评测集 JSONL 文件")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.paths.log_dir)
    kb = KnowledgeBase(config).load()
    reranker = BGEReranker(config.reranker.model, config.reranker.device)

    lines = Path(args.eval_file).read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]

    vec_hits = hyb_hits = 0
    for s in samples:
        q = s["query"]
        exp = s["expected_source"]

        vector_docs = [d for d, _ in kb.vectorstore.search(q, args.top_k)]
        if hit(vector_docs, exp):
            vec_hits += 1

        hybrid_docs = [d for d, _ in kb.bm25_index.search(q, args.top_k)] + vector_docs
        seen = {}
        for d in hybrid_docs:
            seen.setdefault(d.metadata.get("chunk_id", d.page_content), d)
        hybrid_docs = reranker.rerank(q, list(seen.values()), args.top_k)
        if hit(hybrid_docs, exp):
            hyb_hits += 1

    n = len(samples)
    print(f"样本数: {n}")
    print(f"纯向量检索命中率: {vec_hits / n:.2%}")
    print(f"混合检索命中率:   {hyb_hits / n:.2%}")
    if vec_hits:
        print(f"相对提升: {(hyb_hits - vec_hits) / vec_hits:.2%}")


if __name__ == "__main__":
    main()
