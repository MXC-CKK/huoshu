"""语义检索：从 ChromaDB 集合中检索与查询最相关的教材段落。

使用 Ollama embedding 将查询向量化，在 ChromaDB 中做余弦相似度
检索，返回原文段落 + 页码引用。

典型用法:
    from learning_agent.rag.retrieve import query, open_collection

    collection = open_collection("econometrics")
    results = query(collection, "OLS 估计量的一致性证明", top_k=5)
    for r in results:
        print(f"p{r.page_start}: {r.text[:100]}...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 默认参数 ─────────────────────────────────────────────────────────

DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


# ── 数据类 ───────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        text: 段落文本。
        page_start: 起始页码（1-based）。
        page_end: 结束页码（含）。
        score: 相似度分数（0-1，越高越相关）。
        chunk_index: 块序号。
        metadata: 附加元数据。
    """

    text: str
    page_start: int
    page_end: int
    score: float
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_ref(self) -> str:
        """人类可读的页码范围。"""
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"pp.{self.page_start}-{self.page_end}"


# ── 集合管理 ──────────────────────────────────────────────────────────


def open_collection(
    collection_name: str,
    persist_dir: str = "output/chroma",
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Any:
    """打开已有的 ChromaDB 集合。

    Args:
        collection_name: 集合名称。
        persist_dir: ChromaDB 持久化目录。
        embedding_model: Ollama embedding 模型名。

    Returns:
        ChromaDB collection 对象。

    Raises:
        RuntimeError: 集合不存在或 ChromaDB 不可用。
    """
    import chromadb
    from chromadb.utils import embedding_functions

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise RuntimeError(f"ChromaDB 持久化目录不存在: {persist_dir}")

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),  # type: ignore[attr-defined]
    )

    ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=embedding_model,
        url="http://localhost:11434/api/embeddings",
    )

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=ef,  # type: ignore[arg-type]
        )
        logger.info(
            "打开集合 '%s' (%d 个文档)", collection_name, collection.count()
        )
        return collection
    except Exception as exc:
        raise RuntimeError(
            f"集合 '{collection_name}' 不存在或无法打开: {exc}"
        ) from exc


def list_collections(persist_dir: str = "output/chroma") -> list[str]:
    """列出 ChromaDB 中所有集合名称。

    Args:
        persist_dir: ChromaDB 持久化目录。

    Returns:
        集合名称列表。
    """
    import chromadb

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return []

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),  # type: ignore[attr-defined]
    )
    return [c.name for c in client.list_collections()]


def collection_stats(collection: Any) -> dict[str, Any]:
    """获取集合的统计信息。

    Args:
        collection: ChromaDB collection 对象。

    Returns:
        包含 count, name, metadata 的字典。
    """
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


# ── 检索 ──────────────────────────────────────────────────────────────


def query(
    collection: Any,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    page_filter: tuple[int, int] | None = None,
) -> list[SearchResult]:
    """在集合中检索与查询最相关的文本块。

    Args:
        collection: ChromaDB collection 对象。
        query_text: 查询文本（自然语言）。
        top_k: 返回结果数。
        page_filter: 可选的页码范围过滤 (start, end)，仅返回该范围内的结果。

    Returns:
        SearchResult 列表，按相似度降序排列。
    """
    where_filter: dict[str, Any] | None = None
    if page_filter is not None:
        where_filter = {
            "$and": [
                {"page_start": {"$gte": page_filter[0]}},
                {"page_end": {"$lte": page_filter[1]}},
            ]
        }

    chroma_results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where_filter,
    )

    results: list[SearchResult] = []

    # ChromaDB 返回格式: {ids: [[...]], documents: [[...]], metadatas: [[...]], distances: [[...]]}
    ids = chroma_results.get("ids", [[]])[0]
    documents = chroma_results.get("documents", [[]])[0]
    metadatas = chroma_results.get("metadatas", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 1.0

        # ChromaDB 用余弦距离，转为相似度分数
        if distance is not None:
            score = 1.0 - float(distance)
        else:
            score = 0.0

        results.append(
            SearchResult(
                text=documents[i] if i < len(documents) else "",
                page_start=int(meta.get("page_start", 0)),
                page_end=int(meta.get("page_end", 0)),
                score=max(0.0, min(1.0, score)),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            )
        )

    return results


def query_all(
    collection: Any,
    queries: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[list[SearchResult]]:
    """批量查询多个 query。

    Args:
        collection: ChromaDB collection 对象。
        queries: 查询文本列表。
        top_k: 每个查询返回结果数。

    Returns:
        每个查询一个 SearchResult 列表。
    """
    return [query(collection, q, top_k=top_k) for q in queries]


# ── 格式化输出 ───────────────────────────────────────────────────────


def format_results(results: list[SearchResult], *, max_chars: int = 300) -> str:
    """将检索结果格式化为人类可读的文本。

    Args:
        results: SearchResult 列表。
        max_chars: 每个结果最多展示的字符数。

    Returns:
        格式化的多行字符串。
    """
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        truncated = r.text[:max_chars]
        if len(r.text) > max_chars:
            truncated += "…"

        lines.append(
            f"[{i}] {r.page_ref} (score={r.score:.3f})\n"
            f"    {truncated}"
        )

    return "\n\n".join(lines)
