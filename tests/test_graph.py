"""Tests for learning_agent.core.graph — Bookmap loading, validation, navigation."""

# ruff: noqa: DTZ011

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from learning_agent.core.graph import Bookmap, Cluster, Item

# ── 测试夹具 ─────────────────────────────────────────────────────────


@pytest.fixture
def minimal_bookmap_dict() -> dict:
    """最小合法 bookmap 字典。"""
    return {
        "meta": {"source": "test", "built": "2026-08-01", "status": "active"},
        "domain": "Test Domain",
        "clusters": {
            "c01": {"title": "Chapter 1", "learned": False},
            "c02": {"title": "Chapter 2", "learned": True, "learned_date": "2026-07-15"},
        },
        "items": [
            {
                "id": "c01-1",
                "cluster": "c01",
                "title": "Definition of X",
                "type": "definition",
                "mode": "whitebox",
                "source": "textbook §1.1",
                "note": "Core concept",
                "mastery": 0.5,
                "status": "learned",
                "next_review": "2026-08-05",
            },
            {
                "id": "c01-2",
                "cluster": "c01",
                "title": "Theorem Y",
                "type": "theorem",
                "mode": "whitebox",
                "prerequisites": ["c01-1"],
                "source": "textbook §1.2",
                "note": "Depends on definition",
                "mastery": 0.2,
                "status": "pending",
            },
            {
                "id": "c02-1",
                "cluster": "c02",
                "title": "Application Z",
                "type": "application",
                "mode": "blackbox",
                "prerequisites": ["c01-2"],
                "related": ["c01-1"],
                "source": "textbook §2.1",
                "mastery": 0.0,
                "status": "pending",
            },
        ],
    }


@pytest.fixture
def bookmap(minimal_bookmap_dict: dict) -> Bookmap:
    """从最小合法字典构建的 Bookmap。"""
    return Bookmap.from_dict(minimal_bookmap_dict)


# ── 加载测试 ─────────────────────────────────────────────────────────


class TestLoad:
    """Bookmap.load() 和 from_dict() 测试。"""

    def test_load_from_file(self, minimal_bookmap_dict: dict) -> None:
        """从临时 JSON 文件加载。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(minimal_bookmap_dict, f)
            tmp_path = Path(f.name)

        try:
            bm = Bookmap.load(tmp_path)
            assert bm.domain == "Test Domain"
            assert bm.node_count == 3
            assert bm.cluster_count == 2
            assert bm.is_valid
        finally:
            tmp_path.unlink()

    def test_load_from_dict(self, minimal_bookmap_dict: dict) -> None:
        """从字典构建。"""
        bm = Bookmap.from_dict(minimal_bookmap_dict)
        assert bm.domain == "Test Domain"
        assert bm.node_count == 3

    def test_load_skips_validation(self, minimal_bookmap_dict: dict) -> None:
        """validate_on_load=False 跳过校验。"""
        bad = dict(minimal_bookmap_dict)
        bad["items"][0]["type"] = "invalid_type"
        bm = Bookmap.from_dict(bad, validate_on_load=False)
        assert bm.errors == []  # 未校验
        errors = bm.validate()
        assert len(errors) > 0

    def test_load_missing_file(self) -> None:
        """加载不存在的文件应抛出异常。"""
        with pytest.raises(FileNotFoundError):
            Bookmap.load(Path("/nonexistent/bookmap.json"))

    def test_load_invalid_json(self) -> None:
        """加载无效 JSON 应抛出异常。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json")
            tmp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                Bookmap.load(tmp_path)
        finally:
            tmp_path.unlink()


# ── 校验测试 ─────────────────────────────────────────────────────────


class TestValidation:
    """Bookmap.validate() 测试。"""

    def test_valid_bookmap_passes(self, bookmap: Bookmap) -> None:
        """合法 bookmap 校验通过。"""
        assert bookmap.is_valid
        assert bookmap.errors == []

    def test_empty_items_fails(self) -> None:
        """空 items 列表应报错。"""
        data = {
            "meta": {"source": "test", "built": "2026-08-01", "status": "active"},
            "domain": "Test",
            "clusters": {"c01": {"title": "Ch1"}},
            "items": [],
        }
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("为空" in e for e in errors)

    def test_invalid_item_type(self, minimal_bookmap_dict: dict) -> None:
        """无效 item type 应报错。"""
        data = dict(minimal_bookmap_dict)
        data["items"] = [
            {
                "id": "bad-1",
                "cluster": "c01",
                "title": "Bad",
                "type": "nonexistent",
                "mode": "whitebox",
                "source": "§1",
            }
        ]
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("type 无效" in e for e in errors)

    def test_invalid_item_mode(self, minimal_bookmap_dict: dict) -> None:
        """无效 item mode 应报错。"""
        data = dict(minimal_bookmap_dict)
        data["items"] = [
            {
                "id": "bad-1",
                "cluster": "c01",
                "title": "Bad",
                "type": "concept",
                "mode": "graybox",
                "source": "§1",
            }
        ]
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("mode 无效" in e for e in errors)

    def test_invalid_meta_status(self, minimal_bookmap_dict: dict) -> None:
        """无效 meta.status 应报错。"""
        data = dict(minimal_bookmap_dict)
        data["meta"]["status"] = "unknown-status"
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("meta.status" in e for e in errors)

    def test_mastery_out_of_range(self, minimal_bookmap_dict: dict) -> None:
        """mastery 超出 [0, 1] 范围应报错。"""
        data = dict(minimal_bookmap_dict)
        data["items"][0]["mastery"] = 1.5
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("mastery 超出" in e for e in errors)

    def test_bad_cluster_ref(self, minimal_bookmap_dict: dict) -> None:
        """引用不存在的 cluster 应报错。"""
        data = dict(minimal_bookmap_dict)
        data["items"].append({
            "id": "orphan-1",
            "cluster": "nonexistent-cluster",
            "title": "Orphan",
            "type": "concept",
            "mode": "whitebox",
            "source": "§99",
        })
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("不存在的 cluster" in e for e in errors)

    def test_cross_bookmap_prerequisite_allowed(self, minimal_bookmap_dict: dict) -> None:
        """跨图谱 prerequisite 引用应被允许（仅 debug 日志，不报错）。"""
        data = dict(minimal_bookmap_dict)
        data["items"][0]["prerequisites"] = ["prob-05-4"]  # 跨图谱引用
        bm = Bookmap.from_dict(data)
        assert bm.is_valid  # 不报错，因为可能是跨 bookmap 的有效引用

    def test_cross_bookmap_related_allowed(self, minimal_bookmap_dict: dict) -> None:
        """跨图谱 related 引用应被允许（仅 debug 日志，不报错）。"""
        data = dict(minimal_bookmap_dict)
        data["items"][0]["related"] = ["ch4-2-1"]  # 跨章节引用
        bm = Bookmap.from_dict(data)
        assert bm.is_valid

    def test_self_prerequisite(self, minimal_bookmap_dict: dict) -> None:
        """不能以自身为前置依赖。"""
        data = dict(minimal_bookmap_dict)
        data["items"][0]["prerequisites"] = ["c01-1"]  # self-ref
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("自身为前置依赖" in e for e in errors)

    def test_cycle_detection(self, minimal_bookmap_dict: dict) -> None:
        """检测 prerequisites 环路。"""
        data = dict(minimal_bookmap_dict)
        data["items"] = [
            {
                "id": "a", "cluster": "c01", "title": "A",
                "type": "concept", "mode": "whitebox", "source": "§1",
                "prerequisites": ["b"],
            },
            {
                "id": "b", "cluster": "c01", "title": "B",
                "type": "concept", "mode": "whitebox", "source": "§2",
                "prerequisites": ["a"],
            },
        ]
        bm = Bookmap.from_dict(data, validate_on_load=False)
        errors = bm.validate()
        assert any("环路" in e for e in errors)

    def test_no_cycle_for_dag(self, minimal_bookmap_dict: dict) -> None:
        """DAG 无环路，校验通过。"""
        data = dict(minimal_bookmap_dict)
        data["items"] = [
            {
                "id": "a", "cluster": "c01", "title": "A",
                "type": "concept", "mode": "whitebox", "source": "§1",
            },
            {
                "id": "b", "cluster": "c01", "title": "B",
                "type": "concept", "mode": "whitebox", "source": "§2",
                "prerequisites": ["a"],
            },
            {
                "id": "c", "cluster": "c01", "title": "C",
                "type": "concept", "mode": "whitebox", "source": "§3",
                "prerequisites": ["a", "b"],
            },
        ]
        bm = Bookmap.from_dict(data)
        assert bm.is_valid


# ── 导航测试 ─────────────────────────────────────────────────────────


class TestNavigation:
    """图谱导航方法测试。"""

    def test_prerequisites_of(self, bookmap: Bookmap) -> None:
        """获取直接前置依赖。"""
        prereqs = bookmap.prerequisites_of("c01-2")
        assert len(prereqs) == 1
        assert prereqs[0].id == "c01-1"

    def test_prerequisites_of_nonexistent(self, bookmap: Bookmap) -> None:
        """不存在 item 的前置依赖返回空列表。"""
        assert bookmap.prerequisites_of("nonexistent") == []

    def test_prerequisites_of_no_prereqs(self, bookmap: Bookmap) -> None:
        """无前置依赖返回空列表。"""
        assert bookmap.prerequisites_of("c01-1") == []

    def test_prerequisite_chain(self, bookmap: Bookmap) -> None:
        """获取完整前置链。"""
        chain = bookmap.prerequisite_chain("c02-1")
        ids = [it.id for it in chain]
        # c02-1 → c01-2 → c01-1
        assert "c01-2" in ids
        assert "c01-1" in ids
        assert "c02-1" not in ids  # 不包含自身

    def test_prerequisite_chain_nonexistent(self, bookmap: Bookmap) -> None:
        """不存在的 item 返回空链。"""
        assert bookmap.prerequisite_chain("nonexistent") == []

    def test_related_of(self, bookmap: Bookmap) -> None:
        """获取相关概念。"""
        related = bookmap.related_of("c02-1")
        assert len(related) == 1
        assert related[0].id == "c01-1"

    def test_related_of_none(self, bookmap: Bookmap) -> None:
        """无 related 边返回空列表。"""
        related = bookmap.related_of("c01-1")
        assert related == []

    def test_items_in_cluster(self, bookmap: Bookmap) -> None:
        """按簇筛选知识点。"""
        items = bookmap.items_in_cluster("c01")
        assert len(items) == 2
        ids = {it.id for it in items}
        assert ids == {"c01-1", "c01-2"}

    def test_items_in_cluster_empty(self, bookmap: Bookmap) -> None:
        """不存在的簇返回空列表。"""
        assert bookmap.items_in_cluster("nonexistent") == []

    def test_items_by_mode(self, bookmap: Bookmap) -> None:
        """按学习模式筛选。"""
        whitebox = bookmap.items_by_mode("whitebox")
        blackbox = bookmap.items_by_mode("blackbox")
        assert len(whitebox) == 2
        assert len(blackbox) == 1

    def test_items_by_status(self, bookmap: Bookmap) -> None:
        """按学习状态筛选。"""
        learned = bookmap.items_by_status("learned")
        pending = bookmap.items_by_status("pending")
        assert len(learned) == 1
        assert len(pending) == 2

    def test_items_due_for_review(self, bookmap: Bookmap) -> None:
        """到期复习项筛选。"""
        # c01-1: next_review=2026-08-05, not due today (2026-08-01)
        due_today = bookmap.items_due_for_review(date.today())
        assert len(due_today) == 0

        # 用未来日期查
        due_future = bookmap.items_due_for_review(date.fromisoformat("2026-08-10"))
        assert len(due_future) == 1
        assert due_future[0].id == "c01-1"

    def test_topological_order(self, bookmap: Bookmap) -> None:
        """拓扑排序正确。"""
        order = bookmap.topological_order()
        ids = [it.id for it in order]
        # c01-1 应排在 c01-2 之前（无前置依赖）
        assert ids.index("c01-1") < ids.index("c01-2")
        # c01-2 应排在 c02-1 之前
        assert ids.index("c01-2") < ids.index("c02-1")

    def test_orphans(self, bookmap: Bookmap) -> None:
        """无前置依赖的知识点（入口节点）。"""
        orphans = bookmap.orphans()
        ids = {it.id for it in orphans}
        assert "c01-1" in ids
        assert "c01-2" not in ids  # depends on c01-1

    def test_leaves(self, bookmap: Bookmap) -> None:
        """叶子节点（不被任何其他节点作为前置依赖）。"""
        leaves = bookmap.leaves()
        ids = {it.id for it in leaves}
        assert "c02-1" in ids
        assert "c01-1" not in ids  # is a prerequisite for c01-2

    def test_get_item(self, bookmap: Bookmap) -> None:
        """按 id 获取知识点。"""
        item = bookmap.get_item("c01-1")
        assert item is not None
        assert item.title == "Definition of X"

    def test_get_item_nonexistent(self, bookmap: Bookmap) -> None:
        """不存在返回 None。"""
        assert bookmap.get_item("nonexistent") is None

    def test_get_cluster(self, bookmap: Bookmap) -> None:
        """按 id 获取簇。"""
        cluster = bookmap.get_cluster("c01")
        assert cluster is not None
        assert cluster.title == "Chapter 1"

    def test_get_cluster_nonexistent(self, bookmap: Bookmap) -> None:
        """不存在返回 None。"""
        assert bookmap.get_cluster("nonexistent") is None


# ── 统计测试 ─────────────────────────────────────────────────────────


class TestSummary:
    """图谱统计信息测试。"""

    def test_summary(self, bookmap: Bookmap) -> None:
        """summary() 返回完整摘要。"""
        s = bookmap.summary()
        assert s["domain"] == "Test Domain"
        assert s["node_count"] == 3
        assert s["cluster_count"] == 2
        assert s["edge_count"] == 3  # c01-2→c01-1, c02-1→c01-2, c02-1↔c01-1
        assert s["is_valid"] is True
        assert "mode_distribution" in s
        assert "status_distribution" in s
        assert "type_distribution" in s

    def test_repr(self, bookmap: Bookmap) -> None:
        """__repr__ 包含关键信息。"""
        r = repr(bookmap)
        assert "Test Domain" in r
        assert "nodes=3" in r

    def test_properties(self, bookmap: Bookmap) -> None:
        """便捷属性正确。"""
        assert bookmap.node_count == 3
        assert bookmap.cluster_count == 2
        assert bookmap.edge_count == 3

    def test_all_items(self, bookmap: Bookmap) -> None:
        """all_items() 返回所有项。"""
        assert len(bookmap.all_items()) == 3

    def test_all_clusters(self, bookmap: Bookmap) -> None:
        """all_clusters() 返回所有簇。"""
        assert len(bookmap.all_clusters()) == 2


# ── 数据类测试 ──────────────────────────────────────────────────────


class TestDataClasses:
    """Item 和 Cluster 数据类测试。"""

    def test_item_defaults(self) -> None:
        """Item 默认值正确。"""
        item = Item(id="x", cluster="c", title="T", type="concept", mode="whitebox", source="§1")
        assert item.prerequisites == []
        assert item.related == []
        assert item.mastery == 0.0
        assert item.status == "pending"
        assert item.note is None
        assert item.next_review is None

    def test_cluster_defaults(self) -> None:
        """Cluster 默认值正确。"""
        c = Cluster(id="c01", title="Ch1")
        assert c.learned is False
        assert c.learned_date is None
        assert c.parent is None


# ── 真实图谱加载测试 ─────────────────────────────────────────────────


class TestRealBookmaps:
    """用 learning-agent 中的真实 bookmap 测试。"""

    BOOKMAP_DIR = Path(__file__).parent / "fixtures" / "bookmap"

    @pytest.mark.parametrize("filename", [
        "probability.json",
        "econometrics-ch2.json",
        "econometrics-ch4.json",
        "econometrics-ch5.json",
        "econometrics-ch6.json",
    ])
    def test_load_real_bookmap(self, filename: str) -> None:
        """加载真实 bookmap 并通过校验。"""
        path = self.BOOKMAP_DIR / filename
        if not path.exists():
            pytest.skip(f"{path} 不存在")

        bm = Bookmap.load(path)

        # 基本存在性
        assert bm.domain
        assert bm.node_count > 0
        assert bm.cluster_count > 0

        # 校验通过
        if not bm.is_valid:
            error_summary = "\n".join(bm.errors)
            pytest.fail(f"{filename} 校验失败:\n{error_summary}")

    def test_no_cycles_in_real_bookmaps(self) -> None:
        """所有真实 bookmap 无环路。"""
        for path in self.BOOKMAP_DIR.glob("*.json"):
            if path.name.endswith(".bak") or path.name.endswith(".bak2"):
                continue
            bm = Bookmap.load(path)
            assert not bm._has_cycle(), f"{path.name} 存在环路"

    def test_topological_order_real_bookmaps(self) -> None:
        """所有真实 bookmap 可以拓扑排序。"""
        for path in self.BOOKMAP_DIR.glob("*.json"):
            if path.name.endswith(".bak") or path.name.endswith(".bak2"):
                continue
            bm = Bookmap.load(path)
            try:
                order = bm.topological_order()
                assert len(order) == bm.node_count
            except ValueError as e:
                pytest.fail(f"{path.name} 拓扑排序失败: {e}")
