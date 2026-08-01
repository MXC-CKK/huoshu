"""Tests for learning_agent.ui.graph_renderer — Bookmap → vis.js conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from learning_agent.core.graph import Bookmap
from learning_agent.ui.graph_renderer import (
    GraphEdge,
    GraphLayout,
    GraphNode,
    _blend,
    build_graph,
    build_node_detail,
)


# ── 测试夹具 ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_bm() -> Bookmap:
    """包含 3 个节点、2 个簇的示例 Bookmap。"""
    data: dict[str, Any] = {
        "meta": {"source": "test", "built": "2026-08-01", "status": "active"},
        "domain": "Test",
        "clusters": {
            "c01": {"title": "Chapter 1"},
            "c02": {"title": "Chapter 2"},
        },
        "items": [
            {
                "id": "a",
                "cluster": "c01",
                "title": "Definition A",
                "type": "definition",
                "mode": "whitebox",
                "source": "§1.1",
                "mastery": 0.9,
                "status": "learned",
            },
            {
                "id": "b",
                "cluster": "c01",
                "title": "Theorem B",
                "type": "theorem",
                "mode": "whitebox",
                "prerequisites": ["a"],
                "source": "§1.2",
                "mastery": 0.3,
                "status": "learned",
            },
            {
                "id": "c",
                "cluster": "c02",
                "title": "Method C",
                "type": "method",
                "mode": "blackbox",
                "prerequisites": ["b"],
                "related": ["a"],
                "source": "§2.1",
                "mastery": 0.0,
                "status": "pending",
            },
        ],
    }
    return Bookmap.from_dict(data)


# ── build_graph 测试 ──────────────────────────────────────────────────


class TestBuildGraph:
    """build_graph() 测试。"""

    def test_produces_correct_node_count(self, sample_bm: Bookmap) -> None:
        """节点数 = items + clusters（默认）。"""
        layout = build_graph(sample_bm)
        # 3 items + 2 clusters = 5 nodes
        assert len(layout.nodes) == 5

    def test_hide_clusters(self, sample_bm: Bookmap) -> None:
        """show_clusters=False 隐藏簇节点。"""
        layout = build_graph(sample_bm, show_clusters=False)
        assert len(layout.nodes) == 3

    def test_mode_filter(self, sample_bm: Bookmap) -> None:
        """mode_filter 正确筛选。"""
        layout = build_graph(sample_bm, mode_filter={"whitebox"}, show_clusters=False)
        node_ids = {n.id for n in layout.nodes}
        assert node_ids == {"a", "b"}
        assert "c" not in node_ids  # blackbox

    def test_status_filter(self, sample_bm: Bookmap) -> None:
        """status_filter 正确筛选。"""
        layout = build_graph(sample_bm, status_filter={"learned"}, show_clusters=False)
        node_ids = {n.id for n in layout.nodes}
        assert node_ids == {"a", "b"}
        assert "c" not in node_ids  # pending

    def test_cluster_filter(self, sample_bm: Bookmap) -> None:
        """cluster_filter 正确筛选。"""
        layout = build_graph(sample_bm, cluster_filter={"c02"}, show_clusters=False)
        node_ids = {n.id for n in layout.nodes}
        assert node_ids == {"c"}

    def test_whitebox_nodes_are_blue(self, sample_bm: Bookmap) -> None:
        """白箱节点为蓝色系。"""
        layout = build_graph(sample_bm)
        a_node = next(n for n in layout.nodes if n.id == "a")
        assert "4A90D9" in a_node.color or "D9" in a_node.color
        # 蓝色通道应大于红色
        r, g, b = _parse_hex(a_node.color)
        assert b > r or a_node.color == "#4A90D9"

    def test_blackbox_nodes_are_amber(self, sample_bm: Bookmap) -> None:
        """黑箱节点为琥珀色系。"""
        layout = build_graph(sample_bm)
        c_node = next(n for n in layout.nodes if n.id == "c")
        r, g, b = _parse_hex(c_node.color)
        # 红色和绿色通道应较高（琥珀色特征）
        assert r > 100 and g > 50

    def test_high_mastery_thicker_border(self, sample_bm: Bookmap) -> None:
        """高掌握度 → 更粗的边框。"""
        layout = build_graph(sample_bm)
        a_node = next(n for n in layout.nodes if n.id == "a")  # mastery=0.9
        b_node = next(n for n in layout.nodes if n.id == "b")  # mastery=0.3
        assert a_node.border_width > b_node.border_width

    def test_prerequisite_edges(self, sample_bm: Bookmap) -> None:
        """前置依赖边包含箭头和正确颜色。"""
        layout = build_graph(sample_bm)
        prereq_edges = [e for e in layout.edges if not e.dashes]
        assert len(prereq_edges) == 2  # b→a, c→b
        for e in prereq_edges:
            assert e.arrows == "to"
            assert not e.dashes

    def test_related_edges_are_dashed(self, sample_bm: Bookmap) -> None:
        """相关边为虚线，无箭头。"""
        layout = build_graph(sample_bm)
        related_edges = [e for e in layout.edges if e.dashes]
        assert len(related_edges) == 1  # c↔a
        assert related_edges[0].arrows == ""

    def test_edge_only_between_filtered_nodes(self, sample_bm: Bookmap) -> None:
        """过滤后的边只连接可见节点。"""
        # 只显示 whitebox → a 和 b，边只剩 a→b
        layout = build_graph(sample_bm, mode_filter={"whitebox"}, show_clusters=False)
        assert len(layout.edges) == 1
        assert layout.edges[0].source == "a"
        assert layout.edges[0].target == "b"

    def test_theorem_gets_diamond_shape(self, sample_bm: Bookmap) -> None:
        """定理节点使用菱形。"""
        layout = build_graph(sample_bm)
        b_node = next(n for n in layout.nodes if n.id == "b")
        assert b_node.shape == "diamond"

    def test_definition_gets_diamond_shape(self, sample_bm: Bookmap) -> None:
        """定义节点使用菱形。"""
        layout = build_graph(sample_bm)
        a_node = next(n for n in layout.nodes if n.id == "a")
        assert a_node.shape == "diamond"

    def test_method_gets_ellipse_shape(self, sample_bm: Bookmap) -> None:
        """方法节点使用椭圆。"""
        layout = build_graph(sample_bm)
        c_node = next(n for n in layout.nodes if n.id == "c")
        assert c_node.shape == "ellipse"

    def test_pending_node_desaturated(self, sample_bm: Bookmap) -> None:
        """pending 节点颜色被灰色混合（去饱和）。"""
        layout = build_graph(sample_bm)
        c_node = next(n for n in layout.nodes if n.id == "c")
        # pending 节点经灰色混合后饱和度降低，三通道差值应小于纯色
        r, g, b = _parse_hex(c_node.color)
        max_diff = max(r, g, b) - min(r, g, b)
        # 原始琥珀 #D9A44A 的 max_diff ≈ 139，混合后应 < 90
        assert max_diff < 90

    def test_returns_graph_layout_type(self, sample_bm: Bookmap) -> None:
        """返回 GraphLayout 类型。"""
        layout = build_graph(sample_bm)
        assert isinstance(layout, GraphLayout)
        assert isinstance(layout.nodes[0], GraphNode)
        assert isinstance(layout.edges[0], GraphEdge)

    def test_options_include_physics(self, sample_bm: Bookmap) -> None:
        """options 包含 physics 配置。"""
        layout = build_graph(sample_bm)
        assert "physics" in layout.options
        assert layout.options["physics"]["enabled"] is True

    def test_mastery_border_width_range(self, sample_bm: Bookmap) -> None:
        """边框宽度在 [1, 6] 范围内。"""
        layout = build_graph(sample_bm)
        for node in layout.nodes:
            if node.group == "cluster":
                continue  # cluster nodes have fixed border
            assert 1 <= node.border_width <= 6


# ── build_node_detail 测试 ───────────────────────────────────────────


class TestBuildNodeDetail:
    """build_node_detail() 测试。"""

    def test_existing_node(self, sample_bm: Bookmap) -> None:
        """已存在节点返回完整详情。"""
        detail = build_node_detail(sample_bm, "a")
        assert "error" not in detail
        assert detail["item"].id == "a"
        assert detail["item"].title == "Definition A"
        assert detail["prereq_count"] == 0  # 无前置
        assert detail["mode_label"] == "白箱 · 深入理解"
        assert detail["status_label"] == "已学"
        assert detail["type_label"] == "定义"

    def test_node_with_prerequisites(self, sample_bm: Bookmap) -> None:
        """有前置的节点返回前置链。"""
        detail = build_node_detail(sample_bm, "c")
        assert detail["prereq_count"] == 2
        chain_ids = [it.id for it in detail["prereq_chain"]]
        assert "b" in chain_ids
        assert "a" in chain_ids

    def test_node_with_related(self, sample_bm: Bookmap) -> None:
        """有相关概念的节点返回 related。"""
        detail = build_node_detail(sample_bm, "c")
        assert detail["related_count"] == 1
        assert detail["related_items"][0].id == "a"

    def test_nonexistent_node(self, sample_bm: Bookmap) -> None:
        """不存在节点返回错误信息。"""
        detail = build_node_detail(sample_bm, "nonexistent")
        assert "error" in detail

    def test_blackbox_label(self, sample_bm: Bookmap) -> None:
        """黑箱节点的模式标签正确。"""
        detail = build_node_detail(sample_bm, "c")
        assert detail["mode_label"] == "黑箱 · 会用即可"

    def test_cluster_included(self, sample_bm: Bookmap) -> None:
        """详情包含所属簇信息。"""
        detail = build_node_detail(sample_bm, "a")
        cluster = detail["cluster"]
        assert cluster is not None
        assert cluster.title == "Chapter 1"


# ── _blend 测试 ───────────────────────────────────────────────────────


class TestBlend:
    """_blend() 颜色混合测试。"""

    def test_ratio_zero_returns_first(self) -> None:
        assert _blend("#FF0000", "#0000FF", 0.0) == "#ff0000"

    def test_ratio_one_returns_second(self) -> None:
        assert _blend("#FF0000", "#0000FF", 1.0) == "#0000ff"

    def test_ratio_half(self) -> None:
        result = _blend("#000000", "#FFFFFF", 0.5)
        r, g, b = _parse_hex(result)
        assert r == 127
        assert g == 127
        assert b == 127

    def test_output_is_valid_hex(self) -> None:
        for r in [0.0, 0.3, 0.7, 1.0]:
            result = _blend("#123456", "#ABCDEF", r)
            assert result.startswith("#")
            assert len(result) == 7
            int(result[1:], 16)  # should not raise


# ── 数据类测试 ──────────────────────────────────────────────────────


class TestGraphNodeDefaults:
    """GraphNode 默认值测试。"""

    def test_minimal_creation(self) -> None:
        n = GraphNode(
            id="x", label="X", title="T",
            color="#fff", border_color="#000", border_width=2,
        )
        assert n.shape == "box"
        assert n.size == 20
        assert n.font_color == "#333333"


class TestGraphEdgeDefaults:
    """GraphEdge 默认值测试。"""

    def test_minimal_creation(self) -> None:
        e = GraphEdge(source="a", target="b")
        assert e.label == ""
        assert e.dashes is False
        assert e.color == "#666666"
        assert e.arrows == "to"


# ── 真实图谱渲染测试 ────────────────────────────────────────────────


class TestRealBookmapRendering:
    """用真实 bookmap 验证渲染逻辑。"""

    BOOKMAP_DIR = Path("/root/projects/learning-agent/bookmap")

    @pytest.mark.parametrize("filename", [
        "probability.json",
        "econometrics-ch2.json",
        "econometrics-ch4.json",
        "econometrics-ch5.json",
        "econometrics-ch6.json",
    ])
    def test_build_graph_does_not_raise(self, filename: str) -> None:
        """所有真实 bookmap 都能生成图谱而不抛异常。"""

        path = self.__class__.BOOKMAP_DIR / filename  # type: ignore[attr-defined]
        if not path.exists():
            pytest.skip(f"{path} 不存在")

        bm = Bookmap.load(path)
        layout = build_graph(bm)

        # 节点数和边数合理
        assert len(layout.nodes) >= bm.node_count  # items + possible clusters
        assert len(layout.edges) >= 0
        assert len(layout.options) > 0

    @pytest.mark.parametrize("filename", [
        "probability.json",
        "econometrics-ch5.json",
    ])
    def test_node_detail_all_items(self, filename: str) -> None:
        """每个 item 都能生成详情。"""
        path = Path(f"/root/projects/learning-agent/bookmap/{filename}")
        if not path.exists():
            pytest.skip(f"{path} 不存在")

        bm = Bookmap.load(path)
        for item in bm.all_items():
            detail = build_node_detail(bm, item.id)
            assert "error" not in detail, f"Failed for {item.id}"
            assert detail["item"].id == item.id


# ── 辅助 ─────────────────────────────────────────────────────────────


def _parse_hex(hex_color: str) -> tuple[int, int, int]:
    """解析 #RRGGBB 为 (R, G, B)。"""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
