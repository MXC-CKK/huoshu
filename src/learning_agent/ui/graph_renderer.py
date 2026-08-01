"""将 Bookmap 知识图谱转换为 vis.js 渲染所需的节点/边列表。

纯逻辑层，不含 Streamlit UI 代码，可直接在测试中使用。

典型用法:
    from learning_agent.ui.graph_renderer import build_graph
    from learning_agent.core.graph import Bookmap

    bm = Bookmap.load(Path("bookmap/probability.json"))
    nodes, edges, config = build_graph(bm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from learning_agent.core.graph import Bookmap

# ── 配色方案 ─────────────────────────────────────────────────────────


@dataclass
class ColorScheme:
    """vis.js 节点/边配色。

    Attributes:
        whitebox: 白箱节点颜色（深入理解）。
        whitebox_border: 白箱节点边框。
        blackbox: 黑箱节点颜色（会用即可）。
        blackbox_border: 黑箱节点边框。
        pending: 待学节点颜色。
        optional: 选学节点颜色。
        cluster: 簇（章节）节点颜色。
        prerequisite_edge: 前置依赖边颜色。
        related_edge: 相关概念边颜色。
        text: 文字颜色。
        highlight_border: 选中/悬停高亮边框。
    """

    whitebox: str = "#4A90D9"
    whitebox_border: str = "#2B5D8E"
    blackbox: str = "#D9A44A"
    blackbox_border: str = "#8E6B2B"
    pending: str = "#B0B0B0"
    optional: str = "#D0D0D0"
    cluster: str = "#E8E0D0"
    prerequisite_edge: str = "#666666"
    related_edge: str = "#999999"
    text: str = "#333333"
    highlight_border: str = "#FF6B35"


DEFAULT_COLORS = ColorScheme()


# ── 输出类型 ──────────────────────────────────────────────────────────


@dataclass
class GraphNode:
    """vis.js 节点规格。

    Attributes:
        id: 节点唯一 ID（item.id 或 cluster.id）。
        label: 显示标签。
        title: 悬停 tooltip（HTML）。
        color: 填充色。
        border_color: 边框色。
        border_width: 边框宽度（掌握度热力：1-6）。
        shape: 形状（box/ellipse/diamond）。
        size: 节点大小。
        group: 所属分组（用于 vis.js 层级）。
        font_color: 文字颜色。
    """

    id: str
    label: str
    title: str
    color: str
    border_color: str
    border_width: int
    shape: str = "box"
    size: int = 20
    group: str = ""
    font_color: str = "#333333"


@dataclass
class GraphEdge:
    """vis.js 边规格。

    Attributes:
        source: 源节点 ID。
        target: 目标节点 ID。
        label: 边标签（可选）。
        dashes: 是否虚线（related 边为 True）。
        color: 边颜色。
        arrows: 箭头方向（'to' / ''）。
    """

    source: str
    target: str
    label: str = ""
    dashes: bool = False
    color: str = "#666666"
    arrows: str = "to"


@dataclass
class GraphLayout:
    """完整的 vis.js 图谱布局数据。

    Attributes:
        nodes: 节点列表。
        edges: 边列表。
        options: vis.js options 字典。
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


# ── 构建函数 ─────────────────────────────────────────────────────────


def build_graph(
    bm: Bookmap,
    *,
    colors: ColorScheme | None = None,
    show_clusters: bool = True,
    cluster_filter: set[str] | None = None,
    mode_filter: set[str] | None = None,
    status_filter: set[str] | None = None,
) -> GraphLayout:
    """从 Bookmap 构建完整的 vis.js 图谱布局。

    配色规则:
        - whitebox → 蓝色系
        - blackbox → 琥珀色系
        - pending/optional → 灰色系
        - 掌握度 → 边框宽度（1=不熟, 6=稳固）
        - cluster 节点 → 米色背景分组框

    边规则:
        - prerequisites → 实线箭头
        - related → 虚线无箭头

    Args:
        bm: 已加载的 Bookmap。
        colors: 配色方案（None 使用默认）。
        show_clusters: 是否显示 cluster 分组节点。
        cluster_filter: 限定簇集合（None=全部）。
        mode_filter: 限定学习模式（None=全部）。
        status_filter: 限定学习状态（None=全部）。

    Returns:
        GraphLayout 包含节点、边和 vis.js options。
    """
    colors = colors or DEFAULT_COLORS
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    item_ids: set[str] = set()

    # ── Cluster 节点 ──
    if show_clusters:
        for cid, cluster in bm.clusters.items():
            if cluster_filter and cid not in cluster_filter:
                continue
            nodes.append(
                GraphNode(
                    id=f"cluster:{cid}",
                    label=cluster.title,
                    title=f"<b>{cluster.title}</b>",
                    color=colors.cluster,
                    border_color="#C0B8A0",
                    border_width=2,
                    shape="box",
                    size=30,
                    group="cluster",
                    font_color=colors.text,
                )
            )

    # ── Item 节点 ──
    for item in bm.all_items():
        if cluster_filter and item.cluster not in cluster_filter:
            continue
        if mode_filter and item.mode not in mode_filter:
            continue
        if status_filter and item.status not in status_filter:
            continue

        item_ids.add(item.id)

        # 配色
        if item.mode == "whitebox":
            fill = colors.whitebox
            border = colors.whitebox_border
        else:
            fill = colors.blackbox
            border = colors.blackbox_border

        # 状态彩调
        if item.status == "pending":
            fill = _blend(fill, colors.pending, 0.4)
            border = _blend(border, colors.pending, 0.4)
        elif item.status == "optional":
            fill = _blend(fill, colors.optional, 0.5)
            border = _blend(border, colors.optional, 0.5)

        # 掌握度 → 边框宽度（1=弱, 6=强）
        mastery_bw = int(item.mastery * 5) + 1
        mastery_bw = max(1, min(6, mastery_bw))

        # 形状：定理/定义为菱形，方法/应用为椭圆，其余为 box
        if item.type in ("theorem", "definition"):
            shape = "diamond"
        elif item.type in ("method", "application"):
            shape = "ellipse"
        else:
            shape = "box"

        # tooltip
        mode_label = "白箱-深入理解" if item.mode == "whitebox" else "黑箱-会用即可"
        status_label = {"learned": "已学", "pending": "待学", "optional": "选学"}.get(
            item.status, item.status
        )
        title_html = (
            f"<b>{item.title}</b><br>"
            f"类型: {item.type} | {mode_label}<br>"
            f"状态: {status_label} | 掌握度: {item.mastery:.0%}<br>"
            f"锚点: {item.source}"
        )
        if item.note:
            title_html += f"<br>要点: {item.note}"

        nodes.append(
            GraphNode(
                id=item.id,
                label=item.title,
                title=title_html,
                color=fill,
                border_color=border,
                border_width=mastery_bw,
                shape=shape,
                size=25,
                group=item.cluster,
                font_color=colors.text,
            )
        )

    # ── 边 ──
    for item in bm.all_items():
        if item.id not in item_ids:
            continue

        # prerequisite edges
        for pid in item.prerequisites:
            if pid in item_ids:
                edges.append(
                    GraphEdge(
                        source=pid,
                        target=item.id,
                        dashes=False,
                        color=colors.prerequisite_edge,
                        arrows="to",
                    )
                )

        # related edges
        for rid in item.related:
            if rid in item_ids:
                edges.append(
                    GraphEdge(
                        source=item.id,
                        target=rid,
                        dashes=True,
                        color=colors.related_edge,
                        arrows="",
                    )
                )

    # ── vis.js options ──
    options = _build_vis_options()

    return GraphLayout(nodes=nodes, edges=edges, options=options)


def _build_vis_options() -> dict[str, Any]:
    """构建 vis.js 渲染选项，适配知识图谱展示。"""
    return {
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08,
                "damping": 0.4,
            },
            "stabilization": {"enabled": True, "iterations": 200},
        },
        "edges": {
            "smooth": {"type": "cubicBezier", "forceDirection": "vertical"},
            "font": {"size": 10},
        },
        "nodes": {
            "font": {"size": 14, "face": "sans-serif"},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "navigationButtons": True,
            "keyboard": True,
        },
        "layout": {
            "hierarchical": {
                "enabled": False,  # 默认不启用层级布局，用户可切换
                "direction": "LR",
                "sortMethod": "directed",
            },
        },
    }


def _blend(hex1: str, hex2: str, ratio: float) -> str:
    """混合两个 HEX 颜色。

    Args:
        hex1: 第一个颜色 (#RRGGBB)。
        hex2: 第二个颜色 (#RRGGBB)。
        ratio: hex2 权重（0=纯 hex1, 1=纯 hex2）。

    Returns:
        混合后的 #RRGGBB 字符串。
    """
    r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
    r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_node_detail(
    bm: Bookmap,
    item_id: str,
) -> dict[str, Any]:
    """构建单个节点的详情数据，供 UI 面板展示。

    Args:
        bm: 已加载的 Bookmap。
        item_id: 知识点 ID。

    Returns:
        包含 item 字段、prereq_chain、related_items 的字典。
        item 不存在时返回 {"error": "..."}。
    """
    item = bm.get_item(item_id)
    if item is None:
        return {"error": f"节点 '{item_id}' 不存在"}

    prereq_chain = bm.prerequisite_chain(item_id)
    related = bm.related_of(item_id)
    cluster = bm.get_cluster(item.cluster)

    return {
        "item": item,
        "cluster": cluster,
        "prereq_chain": prereq_chain,
        "related_items": related,
        "prereq_count": len(prereq_chain),
        "related_count": len(related),
        "mode_label": "白箱 · 深入理解" if item.mode == "whitebox" else "黑箱 · 会用即可",
        "status_label": {"learned": "已学", "pending": "待学", "optional": "选学"}.get(
            item.status, item.status
        ),
        "type_label": {
            "definition": "定义",
            "concept": "概念",
            "theorem": "定理",
            "method": "方法",
            "example": "示例",
            "application": "应用",
            "section": "章节",
            "exercise": "习题",
        }.get(item.type, item.type),
    }
