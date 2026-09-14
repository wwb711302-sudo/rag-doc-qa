from langchain_core.documents import Document

from rag_doc_qa.document.splitter import split_documents


def test_split_assigns_unique_chunk_id():
    text = "这是一个用于测试的句子。" * 300
    docs = [Document(page_content=text, metadata={"source": "t.md", "page": 1})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_overlap_applied():
    # 切分结果中相邻块存在重叠文本
    text = "句子内容" * 200
    docs = [Document(page_content=text, metadata={"source": "t.md", "page": 1})]
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0].page_content[-10:] == chunks[1].page_content[:10]
