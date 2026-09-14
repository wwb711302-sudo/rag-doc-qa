# 📚 RAG 文档智能问答系统（rag-doc-qa）

基于 **检索增强生成（RAG）** 的本地知识库智能问答 Agent，针对传统文档检索「效率低、答案碎片化、无法结合上下文多轮问答」的痛点，支持 **PDF / Word / Markdown** 文档的上传解析、本地知识库构建，以及**多轮精准问答**与**答案来源溯源**。

## ✨ 特性

- **全自动文档流水线**：加载 → 递归字符切分（重叠窗口 200）→ BGE 向量化 → 本地向量库
- **混合检索**：向量相似度 + BM25 关键词召回，RRF 融合，互补提升召回准确率
- **重排序**：BGE CrossEncoder 对候选片段精排，优化 Top-K 质量
- **多轮对话**：对话记忆 + 历史感知检索，支持上下文追问（指代消解）
- **可视化界面**：Gradio 提供文档上传、实时问答、答案来源引用
- **提供商可换**：默认 DeepSeek，可切换到任意 OpenAI 兼容 API

## 🧱 技术栈

Python · LangChain · Chroma · BGE（嵌入 + 重排序）· DeepSeek · Gradio · PyMuPDF · python-docx · rank-bm25 · jieba

## 📁 目录结构

```
rag-doc-qa/
├── app.py                      # Gradio 界面入口（python app.py）
├── pyproject.toml              # 依赖与打包配置
├── configs/config.yaml         # 可调参数配置
├── .env.example                # 环境变量示例（复制为 .env 填写密钥）
├── src/rag_doc_qa/
│   ├── config.py               # 配置加载（YAML + .env）
│   ├── embeddings.py           # BGE 嵌入封装
│   ├── document/
│   │   ├── loader.py           # PDF/Word/Markdown 加载
│   │   └── splitter.py         # 递归字符切分（overlap=200）
│   ├── store/
│   │   ├── vectorstore.py      # Chroma 向量库
│   │   └── bm25.py             # BM25 关键词索引（jieba 分词）
│   ├── retrieval/
│   │   ├── hybrid.py           # 混合检索 + RRF 融合
│   │   └── reranker.py         # BGE CrossEncoder 重排序
│   ├── chain.py                # 历史感知检索 + 生成链
│   ├── pipeline.py             # 知识库构建 / 问答编排
│   ├── ui.py                   # Gradio 界面
│   └── cli.py                  # 命令行入口
├── scripts/evaluate.py         # 检索效果离线评估
├── docs/architecture.md        # 架构说明
└── tests/                      # 单元测试
```

## 🚀 快速开始

### 1. 环境准备

需要 Python 3.10+。推荐使用虚拟环境：

```bash
git clone https://github.com/<你的账号>/rag-doc-qa.git
cd rag-doc-qa
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -e .
```

> 提示：BGE 嵌入与重排序模型依赖 PyTorch，首次安装体积较大；如已有 GPU，可安装对应 CUDA 版本的 torch 以加速。

### 3. 配置密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 中国大陆访问 HuggingFace 较慢时，可在 `.env` 中加入 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载。

### 4. 运行

**方式 A：可视化界面**

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，在「构建知识库」页上传文档并构建，随后在「智能问答」页提问。

**方式 B：命令行**

```bash
# 构建知识库（支持文件或目录）
python -m rag_doc_qa.cli build ./docs
# 或安装后可执行
rag-doc-qa build ./docs

# 单次问答
rag-doc-qa ask "这份合同的违约金比例是多少？"

# 多轮交互问答
rag-doc-qa ask
```

## ⚙️ 配置说明

所有可调参数位于 `configs/config.yaml`，主要项：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `llm.model` | `deepseek-chat` | DeepSeek 对话模型 |
| `embedding.model` | `BAAI/bge-small-zh-v1.5` | BGE 中文嵌入模型 |
| `reranker.model` | `BAAI/bge-reranker-base` | 重排序模型 |
| `chunk.chunk_size` | `500` | 切分块大小 |
| `chunk.chunk_overlap` | `200` | 字符重叠窗口 |
| `retrieval.vector_top_k` | `20` | 向量召回候选数 |
| `retrieval.bm25_top_k` | `20` | 关键词召回候选数 |
| `retrieval.final_top_k` | `5` | 最终喂给 LLM 的片段数 |
| `memory.max_turns` | `5` | 保留最近 N 轮对话历史 |

换用 BGE v2 系列（如 `BAAI/bge-m3`）时，将 `embedding.query_instruction` 留空即可。

## 🧪 测试与评估

```bash
# 单元测试
pip install -e ".[dev]"
pytest

# 检索效果评估（对比纯向量 vs 混合检索命中率）
python scripts/evaluate.py --eval-file data/eval.jsonl --top-k 5
```

评测集 `data/eval.jsonl` 每行一条：

```json
{"query": "合同违约金比例是多少？", "expected_source": "合同模板.docx"}
```

> 注意：README 中提到的「相关度提升 42%」等性能指标，需在你自己的数据集上通过上述脚本复测，不可直接采信宣传数值。

## 🔍 工作原理

1. **文档处理**：加载 → 递归字符切分（`overlap=200`）→ BGE 向量化 → Chroma + BM25 双索引
2. **检索**：向量相似度 + BM25 关键词双路召回 → RRF 融合 → BGE CrossEncoder 重排序
3. **生成**：结合对话历史改写问题 → 检索 → 带上下文约束的 DeepSeek 生成 → 附来源引用

详见 [`docs/architecture.md`](docs/architecture.md)。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [BGE (BAAI)](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [DeepSeek](https://www.deepseek.com/)
- [Gradio](https://github.com/gradio-app/gradio)
