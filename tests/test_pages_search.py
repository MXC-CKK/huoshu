"""Tests for learning_agent.ui.pages_search — helper functions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from learning_agent.ui.pages_search import list_pdf_files, resolve_pdf_dir


class TestResolvePdfDir:
    """resolve_pdf_dir() 测试。"""

    def test_default(self) -> None:
        """未设置环境变量时返回默认 ~/.huoshu/pdf。"""
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_pdf_dir()
            assert result == Path.home() / ".huoshu" / "pdf"

    def test_env_override(self) -> None:
        """HUOSHU_PDF_DIR 环境变量覆盖默认值。"""
        with patch.dict(os.environ, {"HUOSHU_PDF_DIR": "/custom/pdf/dir"}, clear=True):
            result = resolve_pdf_dir()
            assert result == Path("/custom/pdf/dir")


class TestListPdfFiles:
    """list_pdf_files() 测试。"""

    def test_dir_not_found_returns_empty(self) -> None:
        """目录不存在返回空列表。"""
        result = list_pdf_files("/nonexistent/path/12345")
        assert result == []

    def test_empty_dir_returns_empty(self) -> None:
        """空目录返回空列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_pdf_files(tmpdir)
            assert result == []

    def test_only_pdf_files_returned(self) -> None:
        """仅返回 .pdf 文件，忽略其他类型和子目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "a.pdf").touch()
            (base / "b.PDF").touch()  # 大小写不敏感
            (base / "c.txt").touch()
            (base / "d.md").touch()
            (base / "subdir").mkdir()
            (base / "subdir" / "e.pdf").touch()  # 子目录中不应被列出

            result = list_pdf_files(tmpdir)
            names = [p.name for p in result]
            assert len(result) == 2
            assert "a.pdf" in names
            assert "b.PDF" in names
            assert "c.txt" not in names

    def test_sorted_by_name(self) -> None:
        """结果按文件名排序。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "z.pdf").touch()
            (base / "a.pdf").touch()
            (base / "m.pdf").touch()

            result = list_pdf_files(tmpdir)
            names = [p.name for p in result]
            assert names == ["a.pdf", "m.pdf", "z.pdf"]
