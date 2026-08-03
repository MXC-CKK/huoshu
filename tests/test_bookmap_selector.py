"""Tests for learning_agent.ui.bookmap_selector — multi-directory bookmap discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from learning_agent.ui.bookmap_selector import (
    LEGACY_BOOKMAP_DIR,
    format_bookmap_label,
    list_bookmap_files,
    resolve_bookmap_dirs,
)


class TestResolveBookmapDirs:
    """目录解析优先级与去重。"""

    def test_env_override_first(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """HUOSHU_BOOKMAP_DIR 环境变量优先且出现在第一位。"""
        env_dir = tmp_path / "env-bookmaps"
        monkeypatch.setenv("HUOSHU_BOOKMAP_DIR", str(env_dir))
        dirs = resolve_bookmap_dirs()
        assert str(dirs[0]) == str(env_dir.resolve())

    def test_contains_home_and_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认包含 ~/.huoshu/bookmap 与历史目录。"""
        monkeypatch.delenv("HUOSHU_BOOKMAP_DIR", raising=False)
        dirs = resolve_bookmap_dirs()
        paths = [str(d) for d in dirs]
        assert str(Path.home() / ".huoshu" / "bookmap") in paths
        assert str(LEGACY_BOOKMAP_DIR.resolve()) in paths

    def test_no_duplicates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """同一路径（如 cwd 恰好等于 home）不会重复出现。"""
        monkeypatch.setenv("HUOSHU_BOOKMAP_DIR", str(Path.home() / ".huoshu" / "bookmap"))
        dirs = resolve_bookmap_dirs()
        paths = [str(d) for d in dirs]
        assert len(paths) == len(set(paths))


class TestListBookmapFiles:
    """多目录合并扫描。"""

    def _isolate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """屏蔽真实 home/legacy/cwd 目录，避免测试机环境干扰。"""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "learning_agent.ui.bookmap_selector.LEGACY_BOOKMAP_DIR",
            tmp_path / "legacy-missing",
        )
        monkeypatch.setattr(
            "learning_agent.ui.bookmap_selector.Path.cwd",
            lambda: tmp_path / "cwd-missing",
        )

    def test_scans_multiple_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """env 目录与 home 目录的图谱都能被发现。"""
        env_dir = tmp_path / "a"
        home_book = tmp_path / "home" / ".huoshu" / "bookmap"
        env_dir.mkdir(parents=True)
        home_book.mkdir(parents=True)
        (env_dir / "book-a.json").write_text("{}", encoding="utf-8")
        (home_book / "book-b.json").write_text("{}", encoding="utf-8")

        monkeypatch.setenv("HUOSHU_BOOKMAP_DIR", str(env_dir))
        self._isolate(monkeypatch, tmp_path)

        files = list_bookmap_files()
        names = [p.name for p, _ in files]
        assert "book-a.json" in names
        assert "book-b.json" in names

    def test_filters_bak_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """.bak / .bak2 备份文件被过滤。"""
        env_dir = tmp_path / "a"
        env_dir.mkdir()
        (env_dir / "book.json").write_text("{}", encoding="utf-8")
        (env_dir / "book.json.bak").write_text("{}", encoding="utf-8")
        (env_dir / "book.json.bak2").write_text("{}", encoding="utf-8")

        monkeypatch.setenv("HUOSHU_BOOKMAP_DIR", str(env_dir))
        self._isolate(monkeypatch, tmp_path)

        files = list_bookmap_files()
        assert len(files) == 1
        assert files[0][0].name == "book.json"

    def test_missing_dirs_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """不存在的目录静默跳过。"""
        monkeypatch.setenv("HUOSHU_BOOKMAP_DIR", str(tmp_path / "missing"))
        self._isolate(monkeypatch, tmp_path)
        assert list_bookmap_files() == []


class TestFormatBookmapLabel:
    """选择器显示名。"""

    def test_multi_dir_shows_source(self) -> None:
        label = format_bookmap_label(
            Path("/x/bookmap/ch6.json"),
            Path("/x/bookmap"),
            multi_dir=True,
        )
        assert label == "ch6（bookmap）"

    def test_single_dir_plain_stem(self) -> None:
        label = format_bookmap_label(
            Path("/x/bookmap/ch6.json"),
            Path("/x/bookmap"),
            multi_dir=False,
        )
        assert label == "ch6"
