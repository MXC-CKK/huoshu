"""bookmap 文件发现 — 多目录合并扫描（知识图谱页与学习会话页共用）。

历史背景：早期图谱构建在框架仓库（/root/projects/learning-agent/bookmap），
独立版 App 保存到 ~/.huoshu/bookmap。单一目录扫描会漏掉旧图谱，
本模块统一按优先级扫描多个目录并去重合并。

用法:
    from learning_agent.ui.bookmap_selector import list_bookmap_files

    for path, source_dir in list_bookmap_files():
        print(path.stem, "←", source_dir)
"""

from __future__ import annotations

import os
from pathlib import Path

# 开发机历史目录（早期在框架仓库构建的图谱，向后兼容）
LEGACY_BOOKMAP_DIR = Path("/root/projects/learning-agent/bookmap")


def resolve_bookmap_dirs() -> list[Path]:
    """返回 bookmap 搜索目录列表（按优先级去重）。

    优先级:
        1. HUOSHU_BOOKMAP_DIR 环境变量（显式覆盖）
        2. ~/.huoshu/bookmap（独立版 App 保存目录）
        3. /root/projects/learning-agent/bookmap（历史目录）
        4. cwd/bookmap、cwd/data/bookmap（开发/便携回退）

    Returns:
        去重后的目录 Path 列表（仅保留存在或不存在的候选，调用方自行过滤）。
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        p = Path(p).expanduser()
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            dirs.append(p)

    env_dir = os.environ.get("HUOSHU_BOOKMAP_DIR")
    if env_dir:
        _add(Path(env_dir))

    _add(Path.home() / ".huoshu" / "bookmap")
    _add(LEGACY_BOOKMAP_DIR)
    _add(Path.cwd() / "bookmap")
    _add(Path.cwd() / "data" / "bookmap")

    return dirs


def list_bookmap_files() -> list[tuple[Path, Path]]:
    """扫描所有 bookmap 目录，返回 (json_path, source_dir) 列表。

    过滤规则: 仅 .json；排除 .bak / .bak2 备份文件；
    按目录优先级排序，同目录内按文件名排序。

    Returns:
        [(json 文件路径, 所在目录), ...]；无图谱时返回空列表。
    """
    results: list[tuple[Path, Path]] = []
    for directory in resolve_bookmap_dirs():
        if not directory.is_dir():
            continue
        for p in sorted(directory.glob("*.json")):
            if p.name.endswith(".bak") or p.name.endswith(".bak2"):
                continue
            results.append((p, directory))
    return results


def format_bookmap_label(path: Path, source_dir: Path, *, multi_dir: bool) -> str:
    """生成图谱选择器显示名（多目录时附来源目录名区分）。

    Args:
        path: bookmap JSON 路径。
        source_dir: 所在目录。
        multi_dir: 是否有多于一个来源目录（决定是否显示目录前缀）。

    Returns:
        显示标签。
    """
    if multi_dir:
        return f"{path.stem}（{source_dir.name}）"
    return path.stem
