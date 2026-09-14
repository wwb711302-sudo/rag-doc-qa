"""Gradio 可视化界面。"""
from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from .config import AppConfig
from .pipeline import KnowledgeBase, QAService


def _format_sources(sources) -> str:
    if not sources:
        return "（本次回答未引用来源）"
    lines = []
    seen = set()
    for doc in sources:
        src = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        snippet = doc.page_content.strip().replace("\n", " ")[:120]
        key = (src, page, snippet)
        if key in seen:
            continue
        seen.add(key)
        loc = f"（第 {page} 页）" if page else ""
        lines.append(f"- **{Path(src).name}**{loc}\n  > {snippet}")
    return "### 答案来源\n" + "\n".join(lines)


def build_ui(config: AppConfig):
    kb = KnowledgeBase(config)
    qa: QAService | None = None

    def ensure_qa() -> QAService:
        nonlocal qa
        if qa is None:
            if not kb.is_ready():
                raise gr.Error("请先在「构建知识库」页上传文档并构建。")
            kb.load()
            qa = QAService(config, kb)
        return qa

    def build_kb(files, progress=gr.Progress()):
        if not files:
            raise gr.Error("请先上传文档。")
        progress(0.1, desc="解析文档…")
        nonlocal qa
        try:
            n_docs, n_chunks = kb.build(files)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise gr.Error(f"构建失败：{e}")
        qa = None  # 重建后重置问答服务
        return f"✅ 知识库构建完成：{n_docs} 篇文档，切分为 {n_chunks} 个片段。"

    def respond(message, chatbot):
        service = ensure_qa()
        history = [(m[0], m[1]) for m in chatbot]
        try:
            result = service.ask(message, history)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            answer = f"❌ 出错了：{e}"
            sources_md = ""
        else:
            answer = result["answer"]
            sources_md = _format_sources(result["sources"])
        chatbot.append((message, answer))
        return chatbot, "", sources_md

    with gr.Blocks(title="RAG 文档智能问答系统") as demo:
        gr.Markdown(
            "# 📚 RAG 文档智能问答系统\n"
            "基于 LangChain + Chroma + BGE + DeepSeek 的本地知识库问答"
        )

        with gr.Tab("📥 构建知识库"):
            gr.Markdown(
                "上传 **PDF / Word / Markdown** 文档，自动完成解析、切分、向量化，"
                "并建立「向量 + 关键词」混合索引。"
            )
            files = gr.File(
                file_count="multiple",
                label="上传文档",
                file_types=[".pdf", ".docx", ".md", ".markdown", ".txt"],
            )
            build_btn = gr.Button("构建知识库", variant="primary")
            build_status = gr.Textbox(label="构建状态", interactive=False, lines=2)
            build_btn.click(build_kb, inputs=files, outputs=build_status)

        with gr.Tab("💬 智能问答"):
            chatbot = gr.Chatbot(label="对话", height=480)
            with gr.Row():
                msg = gr.Textbox(label="提问", placeholder="请输入你的问题，按回车发送…", scale=4)
                send_btn = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("清空对话")
            sources_md = gr.Markdown("### 答案来源\n（暂无）")

            send_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg, sources_md])
            msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg, sources_md])
            clear_btn.click(
                lambda: ([], "", "### 答案来源\n（暂无）"),
                outputs=[chatbot, msg, sources_md],
            )

    return demo


def launch(config: AppConfig) -> None:
    demo = build_ui(config)
    demo.queue().launch(server_name=config.server.host, server_port=config.server.port)
