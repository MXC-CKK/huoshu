"""Tests for learning_agent.ui.pages_search — helper functions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from learning_agent.ui.pages_search import (
    ensure_pdf_dir,
    list_pdf_files,
    resolve_pdf_dir,
    sanitize_filename,
    save_uploaded_pdf,
)


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


# ── ensure_pdf_dir 测试 ───────────────────────────────────────────────


class TestEnsurePdfDir:
    """ensure_pdf_dir() 测试。"""

    def test_creates_dir(self, tmp_path: Path) -> None:
        """目录不存在时创建。"""
        target = tmp_path / "new_pdf_dir"
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=target):
            result = ensure_pdf_dir()
        assert result == target
        assert target.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        """目录已存在时幂等，不报错。"""
        target = tmp_path / "existing_dir"
        target.mkdir()
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=target):
            result = ensure_pdf_dir()
        assert result == target
        assert target.is_dir()

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """返回 resolve_pdf_dir 的路径。"""
        target = tmp_path / "check_path"
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=target):
            result = ensure_pdf_dir()
        assert result == target


# ── sanitize_filename 测试 ─────────────────────────────────────────────


class TestSanitizeFilename:
    """sanitize_filename() 测试。"""

    def test_normal_name(self) -> None:
        """正常文件名不变。"""
        assert sanitize_filename("textbook.pdf") == "textbook.pdf"

    def test_path_traversal_basename_only(self) -> None:
        """路径穿越只保留 basename。"""
        result = sanitize_filename("../../evil.pdf")
        assert result == "evil.pdf"
        assert "/" not in result
        assert ".." not in result

    def test_windows_illegal_chars_replaced(self) -> None:
        """Windows 非法字符替换为 _。"""
        result = sanitize_filename('test:file<name>.pdf')
        assert ":" not in result
        assert "<" not in result
        assert result == "test_file_name_.pdf"

    def test_adds_pdf_extension(self) -> None:
        """无 .pdf 后缀自动补充。"""
        result = sanitize_filename("my_notes")
        assert result == "my_notes.pdf"

    def test_adds_extension_case_insensitive(self) -> None:
        """大写 .PDF 也识别，不重复加后缀。"""
        result = sanitize_filename("book.PDF")
        assert result == "book.PDF"

    def test_empty_fallback(self) -> None:
        """空结果回退为 uploaded.pdf。"""
        result = sanitize_filename("")
        assert result == "uploaded.pdf"

    def test_only_illegal_chars_fallback(self) -> None:
        """全非法字符替换为 _，不触发回退。"""
        result = sanitize_filename('<>:"')
        # 每个非法字符替换为 _，结果 ____.pdf
        assert result == "____.pdf"
        assert ":" not in result
        assert "<" not in result

    def test_strips_leading_dots(self) -> None:
        """去除首尾空白和点。"""
        result = sanitize_filename("...hidden.pdf")
        assert result == "hidden.pdf"


# ── save_uploaded_pdf 测试 ────────────────────────────────────────────


class TestSaveUploadedPdf:
    """save_uploaded_pdf() 测试。"""

    def test_writes_content_and_returns_path(self, tmp_path: Path) -> None:
        """写入内容正确，返回路径含文件名。"""
        content = b"%PDF-1.4 mock content"
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=tmp_path):
            saved = save_uploaded_pdf("lecture.pdf", content)
        assert saved.exists()
        assert saved.read_bytes() == content
        assert saved.name == "lecture.pdf"
        assert saved.parent == tmp_path

    def test_creates_dir_automatically(self, tmp_path: Path) -> None:
        """自动创建目标目录。"""
        target = tmp_path / "auto_created_subdir"
        assert not target.exists()
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=target):
            saved = save_uploaded_pdf("notes.pdf", b"data")
        assert target.is_dir()
        assert saved.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """同名文件直接覆盖。"""
        existing = tmp_path / "report.pdf"
        existing.write_bytes(b"old")
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=tmp_path):
            saved = save_uploaded_pdf("report.pdf", b"new content")
        assert saved.read_bytes() == b"new content"

    def test_sanitizes_dangerous_filename(self, tmp_path: Path) -> None:
        """危险文件名被清洗后写入。"""
        content = b"safe content"
        with patch("learning_agent.ui.pages_search.resolve_pdf_dir", return_value=tmp_path):
            saved = save_uploaded_pdf("../etc/passwd:evil.pdf", content)
        assert saved.parent == tmp_path
        assert ".." not in saved.name
        assert ":" not in saved.name
        assert saved.exists()
