"""PDF 教材解析、文本分块、ChromaDB 向量化入库。

将 PDF 教材按语义块切分，追踪页码，生成 embedding 存入 ChromaDB
集合，供下游语义检索使用。

分块策略:
    1. 按页提取文本（带页码追踪）。
    2. 按自然段落边界切分（双换行 / 章节标题）。
    3. 大段落按 token 预算截断并加重叠窗口。
    4. 每块附加页码元数据（page_start, page_end）。

典型用法:
    from learning_agent.rag.ingest import ingest_pdf
    from pathlib import Path

    collection = ingest_pdf(
        pdf_path=Path("textbook/ch02.pdf"),
        collection_name="econometrics",
        source_name="计量经济学 第2章",
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 默认参数 ─────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 500       # 每块最大字符数（~125 tokens）
DEFAULT_CHUNK_OVERLAP = 100    # 块间重叠字符数
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Ollama 默认 embedding 模型
MIN_CHUNK_LENGTH = 30          # 最短块（短于此长度合并到上一块）


# ── 数据类 ───────────────────────────────────────────────────────────


@dataclass
class TextChunk:
    """单个文本块及其页码元数据。

    Attributes:
        text: 文本内容。
        page_start: 起始页码（1-based）。
        page_end: 结束页码（含）。
        chunk_index: 块在文档中的序号。
        metadata: 附加元数据（source, 页码区域等）。
    """

    text: str
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── 章节检测正则 ─────────────────────────────────────────────────────

# 匹配中文章节标题: "第X章", "第X节", "X.Y"，英文 "Chapter X", "Section X.Y"
_CHAPTER_RE = re.compile(
    r"^(第[一二三四五六七八九十\d]+[章节部]|"
    r"[Cc]hapter\s*\d+|"
    r"[Ss]ection\s*\d+|"
    r"\d+(?:\.\d+)*\s)",
    re.MULTILINE,
)


# ── PDF 解析 ──────────────────────────────────────────────────────────


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """从 PDF 文件逐页提取文本。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        (page_number, text) 列表，page_number 为 1-based。

    Raises:
        FileNotFoundError: PDF 文件不存在。
        ValueError: PDF 无法解析（加密/损坏）。
    """
    import pdfplumber

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    pages: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages.append((i, text.strip()))
                else:
                    # 空页保留占位
                    logger.debug("第 %d 页无文本内容", i)
    except Exception as exc:
        raise ValueError(f"PDF 解析失败: {pdf_path} ({exc})") from exc

    logger.info("从 %s 提取了 %d 页文本", pdf_path.name, len(pages))
    return pages


# ── 分块 ──────────────────────────────────────────────────────────────


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source_name: str = "",
) -> list[TextChunk]:
    """将多页文本按语义边界切分为重叠的 TextChunk。

    分块流程:
        1. 在自然段落边界（双换行）处切分每页文本。
        2. 段落按 chunk_size 累积，超出则输出一个 chunk。
        3. 保留 overlap 长度的前一块末尾文本作为下一块的上下文。
        4. 超过 chunk_size 的单段落在句子边界处切分。

    Args:
        pages: (page_number, text) 列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块间重叠字符数。
        source_name: 来源名称（如书名），附加到 chunk metadata。

    Returns:
        TextChunk 列表，按文档顺序排列。
    """
    # Step 1: 解析段落，附带页码
    paragraphs: list[tuple[str, int, int]] = []
    for page_num, page_text in pages:
        for para in _split_paragraphs(page_text):
            if para:
                paragraphs.append((para, page_num, page_num))

    if not paragraphs:
        return []

    # Step 2: 累积段落生成重叠 chunk
    chunks: list[TextChunk] = []
    buf = ""
    buf_start = paragraphs[0][1]
    buf_end = paragraphs[0][2]
    overlap_text = ""

    for text, p_start, p_end in paragraphs:
        # 单段落超大 → 句子级切分后逐个输出
        if len(text) > chunk_size:
            if buf.strip():
                chunks.append(_make_chunk(buf.strip(), buf_start, buf_end, len(chunks), source_name, overlap_text))
                overlap_text = _overlap_tail(buf.strip(), chunk_overlap)
                buf, buf_start, buf_end = "", p_start, p_end

            for sc_text, sc_start, sc_end in _split_long_text(text, chunk_size, chunk_overlap, p_start, p_end):
                chunks.append(_make_chunk(sc_text, sc_start, sc_end, len(chunks), source_name, overlap_text))
                overlap_text = _overlap_tail(sc_text, chunk_overlap)
            continue

        # 累积到 buffer
        new_buf = f"{buf}\n\n{text}" if buf else text
        if len(new_buf) <= chunk_size:
            if not buf:
                buf_start = p_start
            buf = new_buf
            buf_end = max(buf_end, p_end)
        else:
            # buffer 满了，输出
            chunks.append(_make_chunk(buf.strip(), buf_start, buf_end, len(chunks), source_name, overlap_text))
            overlap_text = _overlap_tail(buf.strip(), chunk_overlap)
            buf, buf_start, buf_end = text, p_start, p_end

    # 尾部剩余
    if buf.strip():
        chunks.append(_make_chunk(buf.strip(), buf_start, buf_end, len(chunks), source_name, overlap_text))

    # 过滤过短块（合并到前一块）
    chunks = _merge_short_chunks(chunks, min_length=MIN_CHUNK_LENGTH)

    logger.info("生成了 %d 个文本块（source=%s）", len(chunks), source_name)
    return chunks


def _make_chunk(
    text: str,
    page_start: int,
    page_end: int,
    index: int,
    source_name: str,
    overlap_text: str = "",
) -> TextChunk:
    """构造一个 TextChunk，可选前缀重叠文本。"""
    display = f"{overlap_text} {text}".strip() if overlap_text else text
    return TextChunk(
        text=display,
        page_start=page_start,
        page_end=page_end,
        chunk_index=index,
        metadata={"source": source_name},
    )


def _overlap_tail(text: str, overlap: int) -> str:
    """提取文本末尾的 overlap 长度作为下一块的前缀。"""
    return text[-overlap:] if len(text) > overlap else ""


def _merge_short_chunks(chunks: list[TextChunk], min_length: int) -> list[TextChunk]:
    """将过短的 chunk 合并到前一个 chunk。"""
    if len(chunks) <= 1:
        return chunks

    merged: list[TextChunk] = []
    for ch in chunks:
        if merged and len(ch.text) < min_length:
            prev = merged[-1]
            merged[-1] = TextChunk(
                text=f"{prev.text}\n\n{ch.text}",
                page_start=min(prev.page_start, ch.page_start),
                page_end=max(prev.page_end, ch.page_end),
                chunk_index=prev.chunk_index,
                metadata=prev.metadata,
            )
        else:
            merged.append(ch)
    return merged


def _split_paragraphs(text: str) -> list[str]:
    """在段落边界（双换行/章节标题）处切分文本。"""
    raw = re.split(r"\n\s*\n", text)
    result: list[str] = []
    for para in raw:
        para = para.strip()
        if not para:
            continue
        parts = _CHAPTER_RE.split(para)
        if len(parts) > 1:
            current = ""
            for part in parts:
                if _CHAPTER_RE.match(part):
                    if current.strip():
                        result.append(current.strip())
                    current = part
                else:
                    current += part
            if current.strip():
                result.append(current.strip())
        else:
            result.append(para)
    return result


def _split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page_start: int,
    page_end: int,
) -> list[tuple[str, int, int]]:
    """将长文本按句子边界切分为重叠块；无句子边界时按固定大小切。"""
    sentences = re.split(r"(?<=[。.！!？?\n])\s*", text)
    sentences = [s for s in sentences if s.strip()]

    # 如果句子切分无效（如全是连续字符无标点），退化为固定大小切分
    if not sentences or max(len(s) for s in sentences) > chunk_size * 2:
        return _split_by_fixed_size(text, chunk_size, chunk_overlap, page_start, page_end)

    chunks: list[tuple[str, int, int]] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append((current.strip(), page_start, page_end))
            current = _overlap_tail(current, chunk_overlap) + sent
        else:
            current += sent

    if current.strip():
        chunks.append((current.strip(), page_start, page_end))

    return chunks


def _split_by_fixed_size(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page_start: int,
    page_end: int,
) -> list[tuple[str, int, int]]:
    """按固定大小切分文本（无自然边界的降级方案）。"""
    chunks: list[tuple[str, int, int]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append((text[pos:end].strip(), page_start, page_end))
        pos = end - chunk_overlap if end < len(text) else end
    return chunks


# ── ChromaDB 入库 ────────────────────────────────────────────────────


def create_collection(
    collection_name: str,
    persist_dir: str = "output/chroma",
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Any:
    """创建或获取 ChromaDB 集合并配置 embedding 函数。

    Args:
        collection_name: 集合名称（如 'econometrics'）。
        persist_dir: ChromaDB 持久化目录。
        embedding_model: Ollama embedding 模型名。

    Returns:
        ChromaDB collection 对象。
    """
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),  # type: ignore[attr-defined]
    )

    # 使用 Ollama embedding
    ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=embedding_model,
        url="http://localhost:11434/api/embeddings",
    )

    # 删除旧集合（如果存在）以支持重新入库；不存在时 ChromaDB 抛错，忽略即可
    try:
        client.delete_collection(collection_name)
        logger.info("删除旧集合 '%s'", collection_name)
    except Exception as exc:  # noqa: BLE001 - 集合不存在属预期，无需处理
        logger.debug("旧集合 '%s' 不存在或删除失败: %s", collection_name, exc)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("创建 ChromaDB 集合 '%s' (model=%s)", collection_name, embedding_model)
    return collection


def ingest_chunks(
    collection: Any,
    chunks: list[TextChunk],
) -> int:
    """将 TextChunk 列表批量写入 ChromaDB 集合。

    Args:
        collection: ChromaDB collection 对象。
        chunks: TextChunk 列表。

    Returns:
        写入的块数。
    """
    if not chunks:
        return 0

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for ch in chunks:
        chunk_id = f"chunk-{ch.chunk_index:04d}"
        ids.append(chunk_id)
        documents.append(ch.text)
        metadatas.append({
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "chunk_index": ch.chunk_index,
            **ch.metadata,
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("写入 %d 个文本块到集合", len(ids))
    return len(ids)


# ── 便捷入口 ──────────────────────────────────────────────────────────


def ingest_pdf(
    pdf_path: Path,
    collection_name: str,
    source_name: str = "",
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_dir: str = "output/chroma",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Any:
    """一站式 PDF 入库：解析 → 分块 → 向量化 → 写 ChromaDB。

    Args:
        pdf_path: PDF 文件路径。
        collection_name: ChromaDB 集合名称。
        source_name: 来源名称（辅助元数据）。
        chunk_size: 分块大小（字符数）。
        chunk_overlap: 块间重叠量（字符数）。
        persist_dir: ChromaDB 持久化目录。
        embedding_model: Ollama embedding 模型名。

    Returns:
        ChromaDB collection 对象（可传给 retrieve.query()）。
    """
    if not source_name:
        source_name = pdf_path.stem

    logger.info("开始入库: %s → collection='%s'", pdf_path, collection_name)

    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"PDF 无文本内容: {pdf_path}")

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_name=source_name,
    )

    collection = create_collection(
        collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
    )

    ingest_chunks(collection, chunks)
    logger.info(
        "入库完成: %d 页 → %d chunks → collection '%s'",
        len(pages), len(chunks), collection_name,
    )
    return collection
