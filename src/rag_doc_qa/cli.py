"""命令行入口：构建知识库 / 问答。"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console

from .config import get_config
from .logging_config import setup_logging
from .pipeline import KnowledgeBase, QAService

console = Console()


def cmd_build(args) -> None:
    config = get_config(args.config)
    setup_logging(config.paths.log_dir)
    kb = KnowledgeBase(config)
    console.print("[bold cyan]开始构建知识库…[/bold cyan]")
    n_docs, n_chunks = kb.build(args.paths)
    console.print(f"[green]✅ 完成：{n_docs} 篇文档 → {n_chunks} 个片段[/green]")


def _print_sources(sources) -> None:
    for doc in sources:
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "-")
        console.print(f"  [dim]· {src}（第 {page} 页）[/dim]")


def cmd_ask(args) -> None:
    config = get_config(args.config)
    setup_logging(config.paths.log_dir)
    kb = KnowledgeBase(config).load()
    qa = QAService(config, kb)

    if args.question:
        result = qa.ask(args.question)
        console.print(f"[bold]回答：[/bold] {result['answer']}")
        _print_sources(result["sources"])
        return

    console.print("[bold cyan]进入交互问答模式（输入 exit 退出）[/bold cyan]")
    history = []
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        result = qa.ask(q, history)
        console.print(f"[bold green]助手：[/bold green] {result['answer']}")
        _print_sources(result["sources"])
        history.append((q, result["answer"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rag-doc-qa", description="RAG 文档智能问答系统")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="构建知识库")
    b.add_argument("paths", nargs="+", help="文档路径（文件或目录）")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="问答（不传问题则进入交互模式）")
    a.add_argument("question", nargs="?", default=None, help="问题")
    a.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
