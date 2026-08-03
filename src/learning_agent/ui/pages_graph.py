"""Streamlit 知识图谱可视化页。

交互式知识图谱：
- 白箱节点 = 蓝色系，黑箱节点 = 琥珀色系
- 待学/选学 = 灰色调
- 掌握度 = 边框宽度热力（细=弱, 粗=稳固）
- 前置依赖边 = 实线箭头，相关边 = 虚线
- 点击节点 → 右侧详情面板（标题/类型/掌握度/前置链/锚点）

用法:
    streamlit run src/learning_agent/ui/pages_graph.py
"""

from __future__ import annotations

# ---- Streamlit imports (optional — page usable as module without UI) ----
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from learning_agent.core.graph import Bookmap
from learning_agent.ui.graph_renderer import (
    GraphLayout,
    build_graph,
    build_node_detail,
)

st: Any = importlib.import_module("streamlit") if importlib.util.find_spec("streamlit") else None
agraph: Any = None
Node: Any = None
Edge: Any = None
Config: Any = None
if st is not None:
    try:
        _agraph_mod = importlib.import_module("streamlit_agraph")
        agraph = _agraph_mod.agraph
        Node = _agraph_mod.Node
        Edge = _agraph_mod.Edge
        Config = _agraph_mod.Config
    except Exception:  # noqa: BLE001 - 降级：无 agraph 仅影响图谱渲染
        agraph = None

# ── 常量 ─────────────────────────────────────────────────────────────

DEFAULT_GRAPH_HEIGHT = 600


# ── 页面入口 ─────────────────────────────────────────────────────────


def main() -> None:
    """Streamlit 图谱可视化页入口。"""
    if st is None:
        print("Streamlit 未安装。请运行: pip install streamlit streamlit-agraph")
        raise SystemExit(1)

    st.set_page_config(
        page_title="活书 · 知识图谱",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 知识图谱")
    st.caption("白箱=深入理解 · 黑箱=会用即可 · 边框粗细=掌握度")

    # ── 侧边栏 ──
    with st.sidebar:
        st.header("📂 图谱选择")
        bm, _bm_path = _render_file_selector()
        if bm is None or st is None:
            st.info("请选择一个 bookmap JSON 文件")
            st.stop()

        assert bm is not None  # mypy narrowing after st.stop()

        st.divider()

        st.header("🎨 显示选项")
        show_clusters = st.checkbox("显示章节分组", value=True)

        st.subheader("学习模式")
        show_whitebox = st.checkbox("白箱 (深入理解)", value=True)
        show_blackbox = st.checkbox("黑箱 (会用即可)", value=True)

        st.subheader("学习状态")
        show_learned = st.checkbox("已学", value=True)
        show_pending = st.checkbox("待学", value=True)
        show_optional = st.checkbox("选学", value=False)

        st.divider()

        st.header("🗂️ 章节筛选")
        cluster_ids = list(bm.clusters.keys())
        selected_clusters = st.multiselect(
            "限定章节（留空=全部）",
            options=cluster_ids,
            format_func=lambda cid: f"{cid}: {bm.clusters[cid].title}",
        )

    # ── 构建过滤参数 ──
    mode_filter: set[str] = set()
    if show_whitebox:
        mode_filter.add("whitebox")
    if show_blackbox:
        mode_filter.add("blackbox")

    status_filter: set[str] = set()
    if show_learned:
        status_filter.add("learned")
    if show_pending:
        status_filter.add("pending")
    if show_optional:
        status_filter.add("optional")

    cluster_filter = set(selected_clusters) if selected_clusters else None

    # ── 构建图谱布局 ──
    layout = build_graph(
        bm,
        show_clusters=show_clusters,
        mode_filter=mode_filter if mode_filter else None,
        status_filter=status_filter if status_filter else None,
        cluster_filter=cluster_filter,
    )

    # ── 主区域：顶部统计 + 图例 + 两栏（图谱 + 详情）──
    from streamlit_shadcn_ui import badge, metric_card

    # 顶部统计卡片
    summary = bm.summary()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(label="🧩 知识点", value=summary["node_count"], delta="", key="graph_mc_nodes")
    with c2:
        metric_card(label="📂 章节/簇", value=summary["cluster_count"], delta="", key="graph_mc_clusters")
    with c3:
        metric_card(label="🔗 边", value=summary["edge_count"], delta="", key="graph_mc_edges")

    # 图例区（badge）
    st.caption("")
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        badge("🔬 白箱", key="badge_whitebox", variant="default")
    with bc2:
        badge("🔧 黑箱", key="badge_blackbox", variant="outline")
    with bc3:
        badge("📖 待学", key="badge_pending", variant="secondary")
    with bc4:
        badge("✅ 已学", key="badge_learned", variant="outline")

    st.divider()

    col_graph, col_detail = st.columns([3, 1])

    with col_graph:
        _render_graph(layout)

    with col_detail:
        _render_detail_panel(bm)


# ── 文件选择器 ───────────────────────────────────────────────────────


def _render_file_selector() -> tuple[Bookmap | None, Path | None]:
    """渲染侧边栏的 bookmap 文件选择器（多目录合并扫描）。

    Returns:
        (Bookmap, path) 或 (None, None) 如果未选择/加载失败。
    """
    from learning_agent.ui.bookmap_selector import format_bookmap_label, list_bookmap_files

    # 发现可用 bookmap 文件（多目录合并：~/.huoshu/bookmap + 历史目录）
    found = list_bookmap_files()
    candidates = [p for p, _ in found]
    source_dirs = {str(p): d for p, d in found}
    multi_dir = len({str(d) for _, d in found}) > 1

    if not candidates:
        st.warning("未找到 bookmap JSON 文件")
        # 允许手动输入路径
        manual_path = st.text_input(
            "手动输入 bookmap 路径",
            placeholder="/path/to/bookmap.json",
        )
        if manual_path:
            p = Path(manual_path)
            if p.exists():
                candidates = [p]
            else:
                st.error("文件不存在")
                return None, None
        else:
            return None, None

    selected = st.selectbox(
        "选择图谱",
        options=[str(c) for c in candidates],
        format_func=lambda s: format_bookmap_label(
            Path(s), source_dirs.get(str(Path(s)), Path(s).parent), multi_dir=multi_dir
        ),
    )

    if not selected:
        return None, None

    bm_path = Path(selected)
    try:
        bm = Bookmap.load(bm_path)
        if not bm.is_valid:
            st.warning(f"⚠️ 图谱校验有 {len(bm.errors)} 个警告")
            with st.expander("查看校验详情"):
                for err in bm.errors:
                    st.text(f"· {err}")
        return bm, bm_path
    except Exception as exc:  # noqa: BLE001 - UI 层需捕获所有加载错误并提示
        st.error(f"加载失败: {exc}")
        return None, None


# ── 图谱渲染 ─────────────────────────────────────────────────────────


def _render_graph(layout: GraphLayout) -> str | None:
    """用 streamlit-agraph 渲染 vis.js 图谱。

    Args:
        layout: GraphLayout 含 nodes, edges, options。

    Returns:
        用户在图中点击的节点 ID（None 表示未点击）。
    """
    if agraph is None:
        st.error("streamlit-agraph 未安装")
        return None

    # 转换 nodes
    agraph_nodes: list[Node] = []
    for gn in layout.nodes:
        node = Node(
            id=gn.id,
            label=gn.label,
            title=gn.title,
            color={"background": gn.color, "border": gn.border_color},
            borderWidth=gn.border_width,
            shape=gn.shape,
            size=gn.size,
            group=gn.group,
            font={"color": gn.font_color},
        )
        agraph_nodes.append(node)

    # 转换 edges
    agraph_edges: list[Edge] = []
    for ge in layout.edges:
        edge = Edge(
            source=ge.source,
            target=ge.target,
            label=ge.label,
            arrows=ge.arrows,
            color={"color": ge.color},
            dashes=ge.dashes,
        )
        agraph_edges.append(edge)

    # 配置
    config = Config(
        width="100%",
        height=DEFAULT_GRAPH_HEIGHT,
        directed=True,
        physics=layout.options.get("physics", {}),
        hierarchical=layout.options.get("layout", {}).get("hierarchical", False),
        interaction=layout.options.get("interaction", {}),
    )

    # 渲染并返回选中节点
    selected = agraph(
        nodes=agraph_nodes,
        edges=agraph_edges,
        config=config,
    )

    return str(selected) if selected else None


# ── 详情面板 ─────────────────────────────────────────────────────────


def _render_detail_panel(bm: Bookmap) -> None:
    """渲染右侧详情面板：图谱摘要 + 节点详情。

    节点选择通过 session_state 中的 'selected_node' 键传递。
    """
    st.header("📋 节点详情")

    # 搜索节点
    search = st.text_input("🔍 搜索知识点", placeholder="输入标题或 ID...")
    if search:
        matches: list[Any] = []
        for item in bm.all_items():
            if search.lower() in item.title.lower() or search.lower() in item.id.lower():
                matches.append(item)
        if matches:
            selected_id = st.selectbox(
                f"找到 {len(matches)} 个结果",
                options=[m.id for m in matches],
                format_func=lambda mid: bm.get_item(mid).title if bm.get_item(mid) else mid,  # type: ignore[union-attr]
                key="search_select",
            )
            if selected_id:
                _show_item_detail(bm, selected_id)
            else:
                st.caption("未选择节点")
        else:
            st.caption("无匹配结果")
    else:
        # 图谱总览
        st.subheader("📊 图谱总览")
        summary = bm.summary()

        # 模式分布
        md = summary.get("mode_distribution", {})
        if md:
            st.caption(
                f"白箱: {md.get('whitebox', 0)} · 黑箱: {md.get('blackbox', 0)}"
            )

        # 状态分布
        sd = summary.get("status_distribution", {})
        if sd:
            st.caption(
                f"已学: {sd.get('learned', 0)} · "
                f"待学: {sd.get('pending', 0)} · "
                f"选学: {sd.get('optional', 0)}"
            )

        # 类型分布
        td = summary.get("type_distribution", {})
        if td:
            with st.expander("按类型分布"):
                for t, count in sorted(td.items(), key=lambda x: -x[1]):
                    st.caption(f"{t}: {count}")

        # 薄弱项 Top 5
        st.divider()
        st.subheader("⚠️ 薄弱项 (mastery < 0.5)")
        weak = [
            it for it in bm.all_items()
            if it.status == "learned" and it.mastery < 0.5
        ]
        weak.sort(key=lambda it: it.mastery)
        for w in weak[:5]:
            st.caption(f"· {w.title} ({w.mastery:.0%})")


def _show_item_detail(bm: Bookmap, item_id: str) -> None:
    """展示单个节点的完整详情。

    Args:
        bm: Bookmap 实例。
        item_id: 知识点 ID。
    """
    detail = build_node_detail(bm, item_id)
    if "error" in detail:
        st.warning(detail["error"])
        return

    item = detail["item"]
    st.subheader(item.title)
    st.caption(f"`{item.id}`")

    # 元信息卡片
    cols = st.columns(2)
    with cols[0]:
        st.metric("掌握度", f"{item.mastery:.0%}")
        st.caption(f"类型: {detail['type_label']}")
    with cols[1]:
        st.caption(f"模式: {detail['mode_label']}")
        st.caption(f"状态: {detail['status_label']}")

    # 锚点
    st.caption(f"📖 锚点: {item.source}")

    # 备注
    if item.note:
        st.info(f"💡 {item.note}")

    # 簇信息
    cluster = detail.get("cluster")
    if cluster:
        learned_icon = "✅" if cluster.learned else "📖"
        st.caption(f"{learned_icon} 所属章节: {cluster.title}")

    # 前置依赖链
    st.divider()
    prereqs = detail["prereq_chain"]
    if prereqs:
        st.subheader(f"📎 前置依赖链 ({len(prereqs)})")
        for p in prereqs:
            m_icon = _mastery_icon(p.mastery)
            m_color = _mastery_color(p.mastery)
            st.markdown(
                f"- {m_icon} `{p.id}` {p.title} "
                f"<span style='color:{m_color}'>({p.mastery:.0%})</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("无前置依赖（入口节点）")

    # 相关概念
    st.divider()
    related = detail["related_items"]
    if related:
        st.subheader(f"🔗 相关概念 ({len(related)})")
        for r in related:
            st.caption(f"- `{r.id}` {r.title}")
    else:
        st.caption("无相关概念")

    # 以本节点为前置的后置节点
    st.divider()
    dependents = [
        it for it in bm.all_items()
        if item_id in it.prerequisites
    ]
    if dependents:
        st.subheader(f"📤 后置依赖 ({len(dependents)})")
        for d in dependents:
            st.caption(f"- `{d.id}` {d.title}")
    else:
        st.caption("无后置依赖（叶子节点）")


def _mastery_icon(mastery: float) -> str:
    """掌握度 → emoji 图标。"""
    if mastery >= 0.8:
        return "🟢"
    elif mastery >= 0.5:
        return "🟡"
    elif mastery >= 0.2:
        return "🟠"
    else:
        return "🔴"


def _mastery_color(mastery: float) -> str:
    """掌握度 → 显示颜色。"""
    if mastery >= 0.8:
        return "#2E7D32"
    elif mastery >= 0.5:
        return "#F57F17"
    elif mastery >= 0.2:
        return "#E65100"
    else:
        return "#C62828"


# ── standalone 运行 ───────────────────────────────────────────────────

if __name__ == "__main__":
    main()
