"""检索增强生成链：历史感知检索 + 上下文问答。"""
from __future__ import annotations

from typing import List, Tuple

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .retrieval.hybrid import HybridRetriever

# 将带上下文依赖的追问改写为独立检索问题
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个问题改写器。请结合对话历史，把用户的最新问题改写成一个独立、完整、"
            "上下文自洽的检索问题，便于从知识库中检索。如果最新问题本身已独立，直接原样返回，"
            "不要额外解释。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的文档问答助手，请严格依据下面提供的上下文（检索到的文档片段）回答问题。\n"
            "规则：\n"
            "1. 只使用上下文中包含的信息作答，不要编造。\n"
            "2. 如果上下文不足以回答，请明确说明「根据现有文档无法回答」。\n"
            "3. 使用简体中文，条理清晰。\n\n"
            "上下文：\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def build_rag_chain(llm, retriever: HybridRetriever):
    history_aware_retriever = create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_PROMPT)
    qa_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def history_to_messages(history: List[Tuple[str, str]], max_turns: int) -> list:
    """把 (用户, 助手) 历史裁剪到最近 N 轮并转为消息对象。"""
    messages = []
    for user, assistant in history[-max_turns:]:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=assistant))
    return messages
