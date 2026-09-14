from rag_doc_qa.store.bm25 import tokenize


def test_tokenize_chinese():
    tokens = tokenize("今天天气很好")
    assert len(tokens) > 0


def test_tokenize_strips_whitespace():
    assert " " not in tokenize("a  b   c")
