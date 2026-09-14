"""Gradio 界面入口：python app.py"""
import sys
from pathlib import Path

# 支持不安装直接运行（src 布局）
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_doc_qa.config import get_config  # noqa: E402
from rag_doc_qa.logging_config import setup_logging  # noqa: E402
from rag_doc_qa.ui import launch  # noqa: E402

if __name__ == "__main__":
    config = get_config()
    setup_logging(config.paths.log_dir)
    launch(config)
