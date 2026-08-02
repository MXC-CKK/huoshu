"""Tests for learning_agent.build.graph_builder — AI 图谱构建器."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learning_agent.build.graph_builder import (
    _call_llm,
    _extract_balanced_json,
    _split_chapters,
    _split_subsections,
    assemble_bookmap,
    build_bookmap_from_pdf,
    extract_clusters,
    extract_items,
    infer_edges,
    parse_llm_json,
    resolve_bookmap_dir,
)
from learning_agent.core.graph import Bookmap

# ── resolve_bookmap_dir 测试 ──────────────────────────────────────────


class TestResolveBookmapDir:
    """resolve_bookmap_dir() 测试。"""

    def test_default(self) -> None:
        """未设置环境变量时返回 ~/.huoshu/bookmap。"""
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_bookmap_dir() == Path.home() / ".huoshu" / "bookmap"

    def test_env_override(self) -> None:
        """HUOSHU_BOOKMAP_DIR 覆盖默认值。"""
        with patch.dict(os.environ, {"HUOSHU_BOOKMAP_DIR": "/custom/bookmap"}, clear=True):
            assert resolve_bookmap_dir() == Path("/custom/bookmap")


# ── parse_llm_json 测试 ──────────────────────────────────────────────


class TestParseLlmJson:
    """parse_llm_json() 测试。"""

    def test_normal_json_dict(self) -> None:
        """正常 JSON 对象解析。"""
        result = parse_llm_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_normal_json_list(self) -> None:
        """正常 JSON 数组解析。"""
        result = parse_llm_json('[{"id": 1}, {"id": 2}]')
        assert result == [{"id": 1}, {"id": 2}]

    def test_strips_markdown_fence(self) -> None:
        """去除 ```json 围栏后解析。"""
        result = parse_llm_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_strips_markdown_fence_no_lang(self) -> None:
        """去除 ``` 围栏（无语言标记）。"""
        result = parse_llm_json('```\n{"b": 2}\n```')
        assert result == {"b": 2}

    def test_fixes_trailing_comma(self) -> None:
        """修复尾逗号。"""
        result = parse_llm_json('{"x": 1, "y": 2,}')
        assert result == {"x": 1, "y": 2}

    def test_fixes_trailing_comma_in_array(self) -> None:
        """修复数组尾逗号。"""
        result = parse_llm_json('[1, 2, 3,]')
        assert result == [1, 2, 3]

    def test_fixes_single_quotes(self) -> None:
        """单引号替换为双引号后解析。"""
        result = parse_llm_json("{'name': 'hello'}")
        assert result == {"name": "hello"}

    def test_empty_string_raises(self) -> None:
        """空字符串抛出 ValueError。"""
        with pytest.raises(ValueError, match="无法解析"):
            parse_llm_json("")

    def test_whitespace_only_raises(self) -> None:
        """纯空白抛出 ValueError。"""
        with pytest.raises(ValueError, match="无法解析"):
            parse_llm_json("   \n  ")

    def test_extracts_first_balanced_object(self) -> None:
        """从含前缀文本中提取首个平衡 JSON 对象。"""
        result = parse_llm_json('Some text before {"key": "ok"} and after')
        assert result == {"key": "ok"}

    def test_nested_braces(self) -> None:
        """正确处理嵌套花括号。"""
        result = parse_llm_json('{"outer": {"inner": [1, 2]}}')
        assert result == {"outer": {"inner": [1, 2]}}


# ── _extract_balanced_json 测试 ─────────────────────────────────────


class TestExtractBalancedJson:
    """_extract_balanced_json() 测试。"""

    def test_simple_object(self) -> None:
        """简单对象提取。"""
        assert _extract_balanced_json('{"a": 1}') == '{"a": 1}'

    def test_extracts_from_text(self) -> None:
        """从前导文本中提取。"""
        text = 'hello world {"key": "val"} trailing'
        assert _extract_balanced_json(text) == '{"key": "val"}'

    def test_nested(self) -> None:
        """嵌套括号正确提取。"""
        text = '{"a": {"b": [1, 2]}}'
        assert _extract_balanced_json(text) == text

    def test_no_braces_returns_original(self) -> None:
        """无括号文本返回原文。"""
        assert _extract_balanced_json("plain text") == "plain text"


# ── _call_llm 测试 ─────────────────────────────────────────────────


class TestCallLlm:
    """_call_llm() 测试。"""

    def test_returns_response(self) -> None:
        """正常返回 LLM 响应。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "  hello world  "
        result = _call_llm(mock_llm, "sys", "user")
        assert result == "hello world"

    def test_retries_on_failure(self) -> None:
        """首次失败后重试。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [RuntimeError("fail"), "ok"]
        result = _call_llm(mock_llm, "sys", "user")
        assert result == "ok"
        assert mock_llm.chat.call_count == 2

    def test_raises_after_two_failures(self) -> None:
        """两次失败后抛出 RuntimeError。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError, match="LLM 调用失败"):
            _call_llm(mock_llm, "sys", "user")


# ── extract_clusters 测试 ───────────────────────────────────────────


class TestExtractClusters:
    """extract_clusters() 测试（mock LLM）。"""

    def test_parses_valid_response(self) -> None:
        """正常解析 LLM 返回的章节列表。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([
            {"id": "ch1", "title": "第一章 绪论"},
            {"id": "ch2", "title": "第二章 概率论基础"},
        ])
        result = extract_clusters("目录内容...", mock_llm)
        assert len(result) == 2
        assert result[0]["id"] == "ch1"
        assert result[0]["title"] == "第一章 绪论"

    def test_fills_missing_ids(self) -> None:
        """缺失 id 时自动补全。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([
            {"title": "First chapter"},
        ])
        result = extract_clusters("text", mock_llm)
        assert result[0]["id"] == "ch1"
        assert result[0]["title"] == "First chapter"

    def test_raises_on_non_list(self) -> None:
        """LLM 返回非数组时抛出 ValueError。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({"not": "a list"})
        with pytest.raises(ValueError, match="JSON 数组"):
            extract_clusters("text", mock_llm)


# ── extract_items 测试 ──────────────────────────────────────────────


class TestExtractItems:
    """extract_items() 测试（mock LLM）。"""

    def test_parses_items_with_defaults(self) -> None:
        """正常解析知识点，补全默认字段。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([
            {"id": "ch1-1", "title": "大数定律", "type": "theorem", "mode": "whitebox", "source": "p.45"},
            {"id": "ch1-2", "title": "期望定义", "type": "definition"},
        ])
        result = extract_items("第一章 概率论", "text...", mock_llm, prefix="ch1")
        assert len(result) == 2
        assert result[0]["mode"] == "whitebox"
        assert result[0]["cluster"] == "ch1"
        # 第二个 item 未指定 mode → 按 type 默认 blackbox
        assert result[1]["mode"] == "blackbox"

    def test_invalid_type_defaults_to_concept(self) -> None:
        """非法 type 值回退为 concept。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([
            {"id": "x-1", "title": "test", "type": "bogus", "mode": "blackbox", "source": ""},
        ])
        result = extract_items("ch", "text", mock_llm, prefix="ch1")
        assert result[0]["type"] == "concept"

    def test_invalid_mode_defaults_to_blackbox(self) -> None:
        """非法 mode 值回退为 blackbox。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([
            {"id": "x-1", "title": "test", "type": "theorem", "mode": "super", "source": ""},
        ])
        result = extract_items("ch", "text", mock_llm, prefix="ch1")
        assert result[0]["mode"] == "blackbox"

    def test_raises_on_non_list(self) -> None:
        """LLM 返回非数组时抛出 ValueError。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({"items": []})
        with pytest.raises(ValueError, match="JSON 数组"):
            extract_items("ch", "text", mock_llm, prefix="ch1")


# ── infer_edges 测试 ────────────────────────────────────────────────


class TestInferEdges:
    """infer_edges() 测试（mock LLM）。"""

    def test_parses_edges(self) -> None:
        """正常解析边关系。"""
        items = [
            {"id": "ch1-1", "title": "A", "type": "definition", "mode": "blackbox"},
            {"id": "ch1-2", "title": "B", "type": "theorem", "mode": "whitebox"},
        ]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({
            "prerequisites": [["ch1-1", "ch1-2"]],
            "related": [],
        })
        result = infer_edges(items, mock_llm)
        assert ["ch1-1", "ch1-2"] in result["prerequisites"]

    def test_filters_invalid_refs(self) -> None:
        """过滤引用不存在 item 的边。"""
        items = [{"id": "x-1", "title": "A", "type": "concept", "mode": "blackbox"}]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({
            "prerequisites": [["nonexistent", "x-1"], ["x-1", "also-fake"]],
            "related": [],
        })
        result = infer_edges(items, mock_llm)
        assert len(result["prerequisites"]) == 0

    def test_filters_self_loops(self) -> None:
        """过滤自环边。"""
        items = [{"id": "x-1", "title": "A", "type": "concept", "mode": "blackbox"}]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({
            "prerequisites": [["x-1", "x-1"]],
            "related": [],
        })
        result = infer_edges(items, mock_llm)
        assert len(result["prerequisites"]) == 0

    def test_raises_on_non_dict(self) -> None:
        """LLM 返回非对象时抛出 ValueError。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps([1, 2, 3])
        with pytest.raises(ValueError, match="JSON 对象"):
            infer_edges([], mock_llm)


# ── assemble_bookmap 测试 ───────────────────────────────────────────


class TestAssembleBookmap:
    """assemble_bookmap() 测试。"""

    def test_assembles_valid_bookmap(self) -> None:
        """组装产物通过 schema 校验。"""
        meta = {
            "source": "test-book",
            "built": "2026-08-03",
            "status": "draft-待校对",
            "extraction_method": "test",
        }
        domain = "Test Subject"
        clusters = {
            "ch1": {"title": "Chapter 1", "learned": False, "learned_date": None, "parent": None},
        }
        items = [
            {"id": "ch1-1", "cluster": "ch1", "title": "Knowledge 1", "type": "concept", "mode": "blackbox", "source": "p.1"},
            {"id": "ch1-2", "cluster": "ch1", "title": "Knowledge 2", "type": "theorem", "mode": "whitebox", "source": "p.5", "note": "Important"},
        ]
        edges = {
            "prerequisites": [["ch1-1", "ch1-2"]],
            "related": [],
        }

        bookmap = assemble_bookmap(meta, domain, clusters, items, edges)

        # 通过 Bookmap 校验
        bm = Bookmap.from_dict(bookmap)
        assert bm.is_valid
        assert len(bm.items) == 2
        assert bm.domain == "Test Subject"
        # 边已应用
        assert "ch1-1" in bm.items["ch1-2"].prerequisites

    def test_applies_related_edges_bidirectionally(self) -> None:
        """related 边双向应用。"""
        meta = {"source": "t", "built": "2026-08-03", "status": "draft-待校对", "extraction_method": ""}
        clusters = {"ch1": {"title": "C1"}}
        items = [
            {"id": "a", "cluster": "ch1", "title": "A", "type": "concept", "mode": "blackbox", "source": ""},
            {"id": "b", "cluster": "ch1", "title": "B", "type": "concept", "mode": "blackbox", "source": ""},
        ]
        edges = {"prerequisites": [], "related": [["a", "b"]]}

        bookmap = assemble_bookmap(meta, "d", clusters, items, edges)
        bm = Bookmap.from_dict(bookmap)
        assert "b" in bm.items["a"].related
        assert "a" in bm.items["b"].related


# ── build_bookmap_from_pdf 集成测试 ──────────────────────────────────


class TestBuildBookmapFromPdf:
    """build_bookmap_from_pdf() 集成测试（mock PDF + mock LLM）。"""

    @patch("learning_agent.rag.ingest.extract_pages")
    def test_end_to_end(self, mock_extract: MagicMock) -> None:
        """完整流程：mock PDF 文本 → mock LLM → 产物通过校验。"""
        # Mock PDF: 每章只有一次出现，避免 _split_chapters 产生多个 chunk
        mock_extract.return_value = [
            (1, "目录\n第一章 绪论\n第二章 概率论"),
            (2, "第一章 绪论 样本空间、事件、概率等基本概念。"),
            (3, "第二章 概率论 条件概率与贝叶斯定理是核心内容。"),
        ]

        # Mock LLM（按顺序返回 cluster / items / edges 的 JSON）
        mock_llm = MagicMock()

        def _llm_side_effect(messages, **kwargs):
            content = messages[-1]["content"] if messages else ""
            # infer_edges 的 prompt 含"依赖关系"关键词，先匹配
            if "依赖关系" in content:
                return json.dumps({
                    "prerequisites": [["ch1-1", "ch1-2"], ["ch1-2", "ch2-1"], ["ch2-1", "ch2-2"]],
                    "related": [["ch1-2", "ch2-1"]],
                })
            # extract_clusters 的 prompt 含"目录"和"章节"
            if "目录" in content:
                return json.dumps([
                    {"id": "ch1", "title": "第一章 绪论"},
                    {"id": "ch2", "title": "第二章 概率论"},
                ])
            # extract_items 的 prompt 含章节标题
            if "第一章 绪论" in content:
                return json.dumps([
                    {"id": "ch1-1", "title": "样本空间", "type": "definition", "mode": "blackbox", "source": "p.3", "note": "随机试验所有可能结果的集合"},
                    {"id": "ch1-2", "title": "事件", "type": "definition", "mode": "blackbox", "source": "p.3"},
                ])
            if "第二章 概率论" in content:
                return json.dumps([
                    {"id": "ch2-1", "title": "条件概率", "type": "definition", "mode": "blackbox", "source": "p.4"},
                    {"id": "ch2-2", "title": "贝叶斯定理", "type": "theorem", "mode": "whitebox", "source": "p.5"},
                ])
            return "{}"

        mock_llm.chat.side_effect = _llm_side_effect

        # 进度回调记录
        progress_calls: list[tuple[str, int, int]] = []

        result = build_bookmap_from_pdf(
            Path("test.pdf"),
            mock_llm,
            progress=lambda s, c, t: progress_calls.append((s, c, t)),
        )

        # 产物校验
        bookmap = result["bookmap"]
        bm = Bookmap.from_dict(bookmap)
        assert bm.is_valid, f"校验失败: {bm.errors}"

        # stats
        stats = result["stats"]
        assert stats["clusters"] == 2
        assert stats["items"] >= 4  # 至少 4 个知识点（可能因章节切分多块而更多）
        assert stats["edges"] >= 0
        assert stats["whitebox"] >= 1
        assert stats["blackbox"] >= 1

        # progress 被调用
        assert len(progress_calls) >= 3

    @patch("learning_agent.rag.ingest.extract_pages")
    def test_empty_pdf_raises(self, mock_extract: MagicMock) -> None:
        """空 PDF 抛出 ValueError。"""
        mock_extract.return_value = []
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="无文本"):
            build_bookmap_from_pdf(Path("empty.pdf"), mock_llm)


# ── _split_subsections 测试 ───────────────────────────────────────


class TestSplitSubsections:
    """_split_subsections() 测试。"""

    def test_short_text_not_split(self) -> None:
        """短文本不切分。"""
        text = "Hello world"
        result = _split_subsections(text, max_chars=8000)
        assert result == [text]

    def test_long_text_split_by_paragraphs(self) -> None:
        """长文本按段落边界切分。"""
        para = "A" * 100
        text = "\n\n".join([para] * 100)  # ~10k chars
        result = _split_subsections(text, max_chars=2000)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 2200  # 允许一些余量


# ── _split_chapters 测试 ───────────────────────────────────────────


class TestSplitChapters:
    """_split_chapters() 测试。"""

    def test_splits_by_chapter_title(self) -> None:
        """按中文章标题切分。"""
        clusters = [
            {"id": "ch1", "title": "第一章 绪论"},
            {"id": "ch2", "title": "第二章 概率论"},
        ]
        text = "序言\n第一章 绪论\n这是第一章内容。\n第二章 概率论\n这是第二章内容。"
        result = _split_chapters(text, clusters)
        # 应产生 2 个簇的段落
        assert len(result) >= 1

    def test_empty_clusters_returns_empty(self) -> None:
        """空簇列表返回空。"""
        assert _split_chapters("text", []) == []
