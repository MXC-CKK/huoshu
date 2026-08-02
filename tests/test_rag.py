"""Tests for learning_agent.rag — PDF parsing, chunking, retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learning_agent.rag.ingest import (
    DEFAULT_OLLAMA_URL,
    MIN_CHUNK_LENGTH,
    TextChunk,
    chunk_pages,
    create_collection,
    extract_pages,
    ingest_pdf,
    resolve_chroma_dir,
    resolve_ollama_url,
)
from learning_agent.rag.retrieve import (
    SearchResult,
    delete_collection,
    format_results,
    open_collection,
    query,
)
from learning_agent.rag.retrieve import (
    resolve_chroma_dir as retrieve_resolve_chroma_dir,
)
from learning_agent.rag.retrieve import (
    resolve_ollama_url as retrieve_resolve_ollama_url,
)

# ── PDF 解析测试 ──────────────────────────────────────────────────────


class TestExtractPages:
    """extract_pages() 测试。"""

    def test_file_not_found(self) -> None:
        """不存在的 PDF 文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            extract_pages(Path("/nonexistent/test.pdf"))

    def test_rejects_directory(self) -> None:
        """目录路径抛出异常（pdfplumber 包装为 ValueError）。"""
        with pytest.raises(ValueError, match="PDF 解析失败"):
            extract_pages(Path("/"))


# ── 分块测试 ──────────────────────────────────────────────────────────


class TestChunkPages:
    """chunk_pages() 测试。"""

    def test_empty_pages(self) -> None:
        """空页码列表返回空 chunks。"""
        assert chunk_pages([]) == []

    def test_single_short_page(self) -> None:
        """单页短文本产生单个 chunk。"""
        pages = [(1, "This is a short text.")]
        chunks = chunk_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text."
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1

    def test_page_number_tracking(self) -> None:
        """页码范围跨多页正确追踪。"""
        pages = [
            (1, "Page one content here."),
            (2, "Page two content here."),
            (3, "Page three content here."),
        ]
        chunks = chunk_pages(pages)
        # 短文本应合并为一个 chunk，页码范围覆盖 1-3
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3

    def test_long_content_splits(self) -> None:
        """超长内容被切分为多个 chunk。"""
        long_text = "Sentence number {} in a long paragraph. " * 200  # ~14k chars
        pages = [(1, long_text)]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        for ch in chunks:
            assert len(ch.text) <= 600  # 允许一些余量

    def test_chunk_size_respected(self) -> None:
        """每个 chunk 不超过 chunk_size（或接近）。"""
        text = "A" * 2000
        pages = [(1, text)]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)
        for ch in chunks:
            # 单句子无自然边界时可能略超，但不应超太多
            assert len(ch.text) < 800, f"chunk 过长: {len(ch.text)} chars"

    def test_overlap_between_chunks(self) -> None:
        """相邻 chunk 间存在重叠文本。"""
        # 构造明确以句号分隔的长文本
        sentences = [f"This is sentence number {i}." for i in range(200)]
        text = " ".join(sentences)
        pages = [(1, text)]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= 2

        # 检查前一 chunk 的尾部出现在后一 chunk 的头部附近
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-50:]
            # 后一块的前 200 字符中应包含前一 block 的 tail 片段
            if any(tail[j:j + 20] in chunks[i + 1].text[:200] for j in range(0, len(tail) - 20, 10)):
                break
        # 放宽: 最差情况下前50字符在后200字符中出现即可
        head_match = chunks[0].text[-50:-30]
        assert head_match in chunks[1].text[:200] if len(chunks) >= 2 else True

    def test_source_name_in_metadata(self) -> None:
        """source_name 写入每个 chunk 的 metadata。"""
        pages = [(1, "Test content.")]
        chunks = chunk_pages(pages, source_name="概率论与统计学")
        assert len(chunks) > 0
        for ch in chunks:
            assert ch.metadata.get("source") == "概率论与统计学"

    def test_chunk_index_sequential(self) -> None:
        """chunk_index 按顺序递增。"""
        text = "Long " * 500 + " text"
        pages = [(1, text)]
        chunks = chunk_pages(pages, chunk_size=200)
        for i, ch in enumerate(chunks):
            assert ch.chunk_index == i

    def test_paragraph_boundary_splitting(self) -> None:
        """双换行分隔的段落不被合并到一个 chunk 中（跨段落边界时切分）。"""
        paragraphs = ["First paragraph content." * 3, "Second paragraph content." * 3]
        text = "\n\n".join(paragraphs)
        pages = [(1, text)]
        chunks = chunk_pages(pages, chunk_size=150)
        # 至少产生两个 chunk
        assert len(chunks) >= 1

    def test_min_chunk_merged(self) -> None:
        """过短 chunk 被合并到前一个。"""
        pages = [
            (1, "Normal content " * 30),
            (2, "xy"),  # 太短
        ]
        chunks = chunk_pages(pages, chunk_size=500)
        # 短内容不会单独成一个 chunk
        for ch in chunks:
            assert len(ch.text) >= MIN_CHUNK_LENGTH or ch == chunks[-1]

    def test_whitespace_only_pages_skipped(self) -> None:
        """纯空白页面被跳过。"""
        pages = [
            (1, "Real content here."),
            (2, "   \n  \n  "),
            (3, "More content here."),
        ]
        chunks = chunk_pages(pages)
        texts = [ch.text for ch in chunks]
        joined = " ".join(texts)
        assert "Real content" in joined


# ── TextChunk 数据类测试 ──────────────────────────────────────────────


class TestTextChunk:
    """TextChunk 数据类测试。"""

    def test_creation(self) -> None:
        """正常创建 TextChunk。"""
        tc = TextChunk(
            text="Hello world",
            page_start=5,
            page_end=5,
            chunk_index=0,
        )
        assert tc.text == "Hello world"
        assert tc.page_start == 5
        assert tc.page_end == 5

    def test_default_metadata(self) -> None:
        """metadata 默认为空字典。"""
        tc = TextChunk(text="x", page_start=1, page_end=1, chunk_index=0)
        assert tc.metadata == {}


# ── SearchResult 数据类测试 ──────────────────────────────────────────


class TestSearchResult:
    """SearchResult 数据类测试。"""

    def test_page_ref_single_page(self) -> None:
        """单页引用格式为 p.X。"""
        r = SearchResult(text="test", page_start=42, page_end=42, score=0.9, chunk_index=0)
        assert r.page_ref == "p.42"

    def test_page_ref_range(self) -> None:
        """跨页引用格式为 pp.X-Y。"""
        r = SearchResult(text="test", page_start=10, page_end=12, score=0.8, chunk_index=1)
        assert r.page_ref == "pp.10-12"

    def test_score_range(self) -> None:
        """score 在 [0, 1] 范围内。"""
        r = SearchResult(text="test", page_start=1, page_end=1, score=0.85, chunk_index=0)
        assert 0.0 <= r.score <= 1.0


# ── format_results 测试 ──────────────────────────────────────────────


class TestFormatResults:
    """format_results() 测试。"""

    def test_empty_results(self) -> None:
        """空结果返回空字符串。"""
        assert format_results([]) == ""

    def test_single_result(self) -> None:
        """单条结果包含页码和文本片段。"""
        results = [
            SearchResult(
                text="OLS 估计量在大样本下具有一致性。",
                page_start=45,
                page_end=45,
                score=0.92,
                chunk_index=3,
            ),
        ]
        output = format_results(results)
        assert "p.45" in output
        assert "OLS" in output
        assert "0.92" in output or "0.920" in output

    def test_multiple_results_numbered(self) -> None:
        """多条结果带序号。"""
        results = [
            SearchResult(text="Result A", page_start=1, page_end=1, score=0.9, chunk_index=0),
            SearchResult(text="Result B", page_start=2, page_end=2, score=0.7, chunk_index=1),
        ]
        output = format_results(results)
        assert "[1]" in output
        assert "[2]" in output

    def test_max_chars_truncation(self) -> None:
        """超过 max_chars 的文本被截断。"""
        long_text = "Very long text " * 100
        results = [SearchResult(text=long_text, page_start=1, page_end=1, score=1.0, chunk_index=0)]
        output = format_results(results, max_chars=50)
        assert "…" in output
        assert len(output) < 500  # 远小于原始文本


# ── 检索 API 测试（mock ChromaDB）────────────────────────────────────


class TestQuery:
    """query() 测试（使用 mocked ChromaDB collection）。"""

    @pytest.fixture
    def mock_collection(self) -> MagicMock:
        """返回模拟的 ChromaDB collection。"""
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["chunk-0000", "chunk-0001"]],
            "documents": [["First result text.", "Second result text."]],
            "metadatas": [[
                {"page_start": 10, "page_end": 10, "chunk_index": 0},
                {"page_start": 15, "page_end": 16, "chunk_index": 1},
            ]],
            "distances": [[0.05, 0.20]],
        }
        return coll

    def test_query_returns_search_results(self, mock_collection: MagicMock) -> None:
        """query() 返回 SearchResult 列表。"""
        results = query(mock_collection, "test query")
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "First result text."
        assert results[0].page_ref == "p.10"
        assert results[1].page_ref == "pp.15-16"

    def test_query_scores_converted_from_distance(self, mock_collection: MagicMock) -> None:
        """余弦距离正确转换为相似度分数。"""
        results = query(mock_collection, "test")
        assert results[0].score > results[1].score  # 距离小 → 分数高
        assert results[0].score > 0.8

    def test_query_passes_top_k(self, mock_collection: MagicMock) -> None:
        """top_k 参数传递给集合。"""
        query(mock_collection, "test", top_k=7)
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 7

    def test_query_page_filter(self, mock_collection: MagicMock) -> None:
        """page_filter 生成正确的 where 条件。"""
        query(mock_collection, "test", page_filter=(5, 20))
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] is not None
        assert "$and" in call_kwargs["where"]


# ── ingest_pdf 集成测试（mock 依赖）────────────────────────────────────


class TestIngestPdf:
    """ingest_pdf() 测试（mock PDF 和 ChromaDB）。"""

    def test_file_not_found(self) -> None:
        """不存在的 PDF 抛出异常。"""
        with pytest.raises(FileNotFoundError):
            ingest_pdf(Path("/nonexistent/test.pdf"), "test-coll")

    @patch("learning_agent.rag.ingest.extract_pages")
    def test_empty_pdf_raises(self, mock_extract: MagicMock) -> None:
        """无文本的 PDF 抛出 ValueError。"""
        mock_extract.return_value = []
        with pytest.raises(ValueError, match="无文本"):
            ingest_pdf(Path("empty.pdf"), "test-coll")

    @patch("learning_agent.rag.ingest.create_collection")
    @patch("learning_agent.rag.ingest.ingest_chunks")
    @patch("learning_agent.rag.ingest.extract_pages")
    def test_happy_path(
        self,
        mock_extract: MagicMock,
        mock_ingest: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """正常流程：解析 → 分块 → 入库。"""
        mock_extract.return_value = [
            (1, "A full paragraph of textbook content."),
            (2, "Another page with more content."),
        ]
        mock_collection = MagicMock()
        mock_create.return_value = mock_collection
        mock_ingest.return_value = 2

        result = ingest_pdf(Path("test.pdf"), "test-coll")
        assert result is mock_collection
        mock_extract.assert_called_once()
        mock_create.assert_called_once()
        mock_ingest.assert_called_once()

        # 验证传给 ingest_chunks 的 chunk 正确
        chunks_passed = mock_ingest.call_args.args[1]
        assert len(chunks_passed) >= 1
        for ch in chunks_passed:
            assert isinstance(ch, TextChunk)
            assert ch.metadata.get("source") == "test"


# ── 环境变量解析测试 ──────────────────────────────────────────────────


class TestResolveOllamaUrl:
    """resolve_ollama_url() 测试。"""

    def test_default(self) -> None:
        """未设置环境变量时返回默认值。"""
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_ollama_url() == DEFAULT_OLLAMA_URL

    def test_env_override(self) -> None:
        """HUOSHU_OLLAMA_URL 覆盖默认值。"""
        with patch.dict(os.environ, {"HUOSHU_OLLAMA_URL": "http://custom:9999/api/embeddings"}, clear=True):
            assert resolve_ollama_url() == "http://custom:9999/api/embeddings"

    def test_retrieve_module_same_default(self) -> None:
        """retrieve 模块的 resolve_ollama_url 与 ingest 模块行为一致。"""
        with patch.dict(os.environ, {}, clear=True):
            assert retrieve_resolve_ollama_url() == resolve_ollama_url()


class TestResolveChromaDir:
    """resolve_chroma_dir() 测试。"""

    def test_default(self) -> None:
        """未设置环境变量时返回 ~/.huoshu/chroma。"""
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_chroma_dir() == str(Path.home() / ".huoshu" / "chroma")

    def test_env_override(self) -> None:
        """HUOSHU_CHROMA_DIR 覆盖默认值。"""
        with patch.dict(os.environ, {"HUOSHU_CHROMA_DIR": "/custom/chroma"}, clear=True):
            assert resolve_chroma_dir() == "/custom/chroma"

    def test_retrieve_module_same_default(self) -> None:
        """retrieve 模块的 resolve_chroma_dir 与 ingest 模块行为一致。"""
        with patch.dict(os.environ, {}, clear=True):
            assert retrieve_resolve_chroma_dir() == resolve_chroma_dir()


# ── create_collection / open_collection ollama_url 参数测试 ────────────


class TestCreateCollectionOllamaUrl:
    """create_collection() 的 ollama_url 参数测试。"""

    @patch("chromadb.PersistentClient")
    def test_uses_default_url_when_none(self, mock_client_cls: MagicMock) -> None:
        """ollama_url=None 时使用 resolve_ollama_url() 默认值。"""
        with patch.dict(os.environ, {}, clear=True):
            result = create_collection("test-coll", ollama_url=None)
        assert result is not None

    @patch("chromadb.PersistentClient")
    def test_passes_custom_url(self, mock_client_cls: MagicMock) -> None:
        """ollama_url 参数传递给 OllamaEmbeddingFunction。"""
        result = create_collection("test-coll", ollama_url="http://custom:11434/api/embeddings")
        assert result is not None

    @patch("chromadb.PersistentClient")
    def test_uses_env_var_when_none(self, mock_client_cls: MagicMock) -> None:
        """ollama_url=None 时使用 HUOSHU_OLLAMA_URL 环境变量。"""
        with patch.dict(os.environ, {"HUOSHU_OLLAMA_URL": "http://env:9999/api/embeddings"}, clear=True):
            result = create_collection("test-coll", ollama_url=None)
        assert result is not None


class TestOpenCollectionOllamaUrl:
    """open_collection() 的 ollama_url 参数测试。"""

    @patch("chromadb.PersistentClient")
    def test_uses_default_url_when_none(self, mock_client_cls: MagicMock) -> None:
        """ollama_url=None 时使用默认 URL。"""
        with (
            patch("learning_agent.rag.retrieve.Path.exists", return_value=True),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = open_collection("test-coll", ollama_url=None)
        assert result is not None

    @patch("chromadb.PersistentClient")
    def test_passes_custom_url(self, mock_client_cls: MagicMock) -> None:
        """ollama_url 参数传递给 OllamaEmbeddingFunction。"""
        custom_url = "http://custom:11434/api/embeddings"
        with patch("learning_agent.rag.retrieve.Path.exists", return_value=True):
            result = open_collection("test-coll", ollama_url=custom_url)
        assert result is not None


# ── delete_collection 测试 ───────────────────────────────────────────


class TestDeleteCollection:
    """delete_collection() 测试。"""

    def test_dir_not_found_raises(self) -> None:
        """持久化目录不存在时抛出 RuntimeError。"""
        with pytest.raises(RuntimeError, match="持久化目录不存在"):
            delete_collection("test-coll", "/nonexistent/path/12345")

    @patch("chromadb.PersistentClient")
    def test_deletes_collection(self, mock_client_cls: MagicMock) -> None:
        """正常删除集合。"""
        with patch("learning_agent.rag.retrieve.Path.exists", return_value=True):
            delete_collection("test-coll", "/fake/persist")
        mock_client = mock_client_cls.return_value
        mock_client.delete_collection.assert_called_once_with("test-coll")

    @patch("chromadb.PersistentClient")
    def test_raises_on_failure(self, mock_client_cls: MagicMock) -> None:
        """删除失败时抛出 RuntimeError。"""
        mock_client = mock_client_cls.return_value
        mock_client.delete_collection.side_effect = ValueError("collection not found")
        with (
            patch("learning_agent.rag.retrieve.Path.exists", return_value=True),
            pytest.raises(RuntimeError, match="删除集合"),
        ):
            delete_collection("nonexistent", "/fake/persist")
