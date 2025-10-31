# Docuest

## Overview / 项目概述
Docuest is a document indexing and semantic search toolkit that pairs a desktop GUI with an embedding-based backend. It scans configured folders, extracts text from many common office formats, stores chunk-level embeddings in SQLite, and lets you query them with reranked cosine similarity and optional markdown answers rendered in the UI.

Docuest 是一个文档索引与语义搜索工具套件，包含基于嵌入向量的后台和桌面图形界面。它可以遍历配置的文件夹，从常见的办公文档中提取文本内容，将切分后的向量保存到 SQLite 中，并在界面里基于余弦相似度进行检索和重排，还可输出 Markdown 形式的回答。

## Features / 功能特点
- **Broad extractor coverage / 丰富的文本抽取支持**：内置对 TXT、PDF、DOCX、PPTX、XLS(X) 等格式的解析，同时在可用时优先调用可选依赖（如 `textract`）以兼容更多旧格式。
- **Chunked embedding index / 分块嵌入索引**：按照配置的字符长度切分文档片段，为每个片段生成嵌入向量并存储在 SQLite 数据库中，兼顾检索精度与效率。
- **Configurable similarity search / 可配置的相似度搜索**：支持自定义检索数量、余弦阈值、路径权重等参数，实现内容、文件名与路径多重信号的综合评分。
- **Tkinter GUI with Markdown rendering / 基于 Tkinter 的图形界面并支持 Markdown 渲染**：提供索引与检索按钮、日志输出及结果显示区域；当安装 `tkhtmlview` 或 `tkinterweb` 与 `markdown` 时，可直接在界面中渲染 Markdown。
- **SiliconFlow embedding client / SiliconFlow 嵌入模型接入**：内置对 SiliconFlow API 的调用逻辑，包含自动重试、超时设置和批量请求配置。

## Repository Layout / 仓库结构
- `core_finder.py`: Embedding pipeline, document parsing, and vector search logic.
  `core_finder.py`：嵌入生成流程、文档解析与向量检索核心逻辑。
- `doc_finder_gui.py`: Tkinter-based desktop application for indexing and querying.
  `doc_finder_gui.py`：用于执行索引与检索的 Tkinter 图形界面程序。
- `config template.yml`: Example configuration file describing search scope, embedding settings, and thresholds.
  `config template.yml`：示例配置文件，定义索引范围、嵌入参数与搜索阈值。

## Requirements / 环境依赖
1. **Python 3.10+ recommended / 推荐使用 Python 3.10+**。
2. **Install core dependencies / 安装核心依赖**：
   ```bash
   pip install -r requirements.txt
   ```
   > If a `requirements.txt` file is not available, install the packages observed in the code such as `numpy`, `requests`, `pyyaml`, `pdfminer.six`, `python-docx`, `python-pptx`, `pandas`, `tkinterweb`/`tkhtmlview`, and `markdown` based on your needs.
   > 若仓库未提供 `requirements.txt`，可根据源码中的引入自行安装所需库，例如 `numpy`、`requests`、`pyyaml`、`pdfminer.six`、`python-docx`、`python-pptx`、`pandas`、`tkinterweb`/`tkhtmlview` 以及 `markdown` 等。
3. **SiliconFlow API key / SiliconFlow API 密钥**：将密钥写入 `API_KEY.txt`（或在配置中指定的其他路径）。

## Configuration / 配置说明
1. 复制示例配置：
   ```bash
   cp "config template.yml" config.yml
   ```
2. 根据需求修改 `config.yml`：
   - `include_dirs`: 需要索引的根目录列表。
   - `exclude_globs`: 使用通配符排除不需要的文件或目录。
   - `allowed_exts`: 控制可解析的文件扩展名。
   - `embedding`: 配置 SiliconFlow API 的地址、模型、密钥文件路径以及超时时间等参数。
   - `chunking` 与 `search`: 调整分块策略与搜索阈值。

## Usage / 使用方式
### 1. Build or refresh the index / 构建或更新索引
Call the indexing routine from Python after preparing the configuration, for example:

在准备好配置文件后，可在 Python 中调用索引方法，例如：
```bash
python -c "from core_finder import index_all; index_all('config.yml')"
```
The command walks through the configured directories, extracts text, and writes chunk metadata plus embeddings into SQLite.

上述命令会遍历配置中的目录，提取文本并将片段元数据与嵌入写入 SQLite 数据库。

### 2. Launch the desktop GUI / 启动桌面图形界面
```bash
python doc_finder_gui.py
```
- 点击 **Index Documents** 按钮以在 GUI 中触发索引构建（可在后台线程执行）。
- 在 **Search Query** 输入框中输入问题或关键词，点击 **Search + Reason** 按钮即可检索并展示匹配片段与推理结果。

Press **Index Documents** to trigger indexing from the GUI (runs on a background thread). Enter a question or keyword in **Search Query** and click **Search + Reason** to retrieve relevant chunks and the reasoning output.

### 3. Command-line search helper / 命令行检索（若需要）
`core_finder.py` 暴露的 `search_docs` 函数可被其他脚本调用，快速获取相似度最高的片段及生成式答案。

The `search_docs` function in `core_finder.py` can be reused by other scripts to obtain the top-matching chunks and an optional generated answer.

## Tips / 使用建议
- 首次索引大型目录时建议分批进行，并合理设置 `exclude_globs` 以减少无关文件。
- 当没有安装 Markdown 渲染相关依赖时，界面会自动回退到纯文本显示。
- 通过调整 `weights` 中的 content/filename/path 权重，可以平衡语义匹配与文件路径信息的重要程度。

For large directory trees, consider indexing in batches and refining `exclude_globs` to avoid irrelevant files. The GUI falls back to plain text rendering when Markdown dependencies are missing. Tune the `weights` section to balance semantic match against filename and path signals.

## License / 许可证
This project is distributed under the terms of the MIT License. See [`LICENSE`](LICENSE) for details.

本项目基于 MIT 许可证发布，详情请参阅 [`LICENSE`](LICENSE)。
