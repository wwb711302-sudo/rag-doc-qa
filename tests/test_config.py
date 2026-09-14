from rag_doc_qa.config import AppConfig


def test_default_overlap_is_200():
    cfg = AppConfig()
    assert cfg.chunk.chunk_overlap == 200


def test_default_llm_model():
    cfg = AppConfig()
    assert cfg.llm.model == "deepseek-chat"


def test_default_retrieval_topk():
    cfg = AppConfig()
    assert cfg.retrieval.final_top_k == 5
