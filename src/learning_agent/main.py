"""活书 huoshu — 统一启动入口。

把图谱可视化 / AI 建图谱 / 学习会话 / 教材检索 / 间隔复习 / 模型设置
六个页面聚合为带侧边栏导航的单应用。

用法:
    huoshu                        # 安装后直接运行（默认 8501 端口）
    huoshu --server.port 8600     # 指定端口
    python -m learning_agent.main # 等效
    streamlit run src/learning_agent/main.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]


# ── 全局 CSS 注入 ───────────────────────────────────────────────────────


def _inject_global_css() -> None:
    """一次性注入全局自定义 CSS（字体/按钮/卡片/聊天气泡/侧边栏等）。"""
    css = """
    <style>
    /* ===== 全局字体栈 ===== */
    html, body, [class*="css"] {
        font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
    }

    /* ===== 主容器最大宽度 + 内边距 ===== */
    .block-container {
        max-width: 1200px !important;
        padding: 1.5rem 2rem !important;
    }

    /* ===== 通用圆角卡片类 ===== */
    .huoshu-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
        margin-bottom: 1rem;
    }

    /* ===== 侧边栏样式 ===== */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1cypcdb {
        background-color: #FFFFFF;
    }

    /* 侧边栏导航项选中态 */
    [data-testid="stSidebarNavLink"][aria-current="page"] {
        background-color: #EEF2FF !important;
        border-left: 3px solid #6366F1 !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background-color: #F5F5F5 !important;
    }

    /* ===== 主按钮圆角 + hover 过渡 ===== */
    .stButton > button[kind="primary"] {
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,.3);
    }
    .stButton > button {
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    /* ===== Metric 卡片化 ===== */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    }

    /* ===== 聊天气泡样式 ===== */
    /* 用户消息：靛蓝底白字 */
    [data-testid="stChatMessage"][aria-label*="user"] {
        background: linear-gradient(135deg, #6366F1, #818CF8);
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        color: #FFFFFF;
    }
    /* AI 消息：浅灰底 */
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        background: #F3F4F6;
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        color: #1F2937;
    }

    /* ===== 容器 card 类（st.container border）===== */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
    }

    /* ===== expander 轻美化 ===== */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
    }

    /* ===== divider 柔和色 ===== */
    hr {
        border-color: #E5E7EB !important;
    }

    /* ===== tabs 选中态高亮 ===== */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #6366F1 !important;
    }

    /* ===== radio/checkbox 选中色 ===== */
    .stRadio [data-baseweb="radio"] [aria-checked="true"] {
        background-color: #6366F1 !important;
    }

    /* ===== 进度条主色 ===== */
    .stProgress > div > div {
        background-color: #6366F1 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── 首页跨图谱统计 ─────────────────────────────────────────────────────


def _aggregate_cross_bookmap_stats() -> dict[str, int]:
    """扫描所有 bookmap 文件，聚合跨图谱统计。

    Returns:
        {"bookmaps": n, "items": n, "learned": n} 字典。
    """
    from learning_agent.ui.bookmap_selector import list_bookmap_files

    found = list_bookmap_files()
    candidates = [p for p, _ in found]

    total_bookmaps = len(candidates)
    total_items = 0
    total_learned = 0

    for bm_path in candidates:
        try:
            data = json.loads(bm_path.read_text(encoding="utf-8"))
            items = data.get("items", []) if isinstance(data, dict) else []
            total_items += len(items)
            total_learned += sum(
                1 for it in items
                if isinstance(it, dict) and it.get("status") == "learned"
            )
        except (json.JSONDecodeError, OSError):
            continue

    return {
        "bookmaps": total_bookmaps,
        "items": total_items,
        "learned": total_learned,
    }


# ── 首页视图 ───────────────────────────────────────────────────────────


def home_main() -> None:
    """首页视图入口（用作 st.Page 的 callable）。"""
    from streamlit_shadcn_ui import card, metric_card

    # ── Hero 区 ──
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1.5rem 0;">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.5rem;">
                📚 活书 <span style="font-weight: 400; color: #6B7280; font-size: 1.25rem;">Huoshu</span>
            </h1>
            <p style="font-size: 1.1rem; color: #6B7280; max-width: 600px; margin: 0 auto;">
                把教材变成可交互知识图谱的自学工具
            </p>
            <p style="font-size: 0.9rem; color: #9CA3AF; margin-top: 0.25rem;">
                Knowledge Graph × Adaptive Learning × RAG × Spaced Repetition
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 统计三卡（跨图谱聚合）──
    stats = _aggregate_cross_bookmap_stats()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            label="📂 已建图谱",
            value=stats["bookmaps"],
            delta="",
            key="home_stat_bookmaps",
        )
    with c2:
        metric_card(
            label="🧩 知识点总数",
            value=stats["items"],
            delta="",
            key="home_stat_items",
        )
    with c3:
        metric_card(
            label="✅ 已学知识点",
            value=stats["learned"],
            delta="",
            key="home_stat_learned",
        )

    st.divider()

    # ── 六功能入口卡片 ──
    entries: list[dict[str, str]] = [
        {
            "icon": "🤖",
            "title": "AI 建图谱",
            "desc": "上传教材 PDF，AI 自动生成知识图谱初稿",
            "page": "builder",
        },
        {
            "icon": "📊",
            "title": "知识图谱",
            "desc": "交互式浏览知识图谱，查看节点详情与依赖关系",
            "page": "graph",
        },
        {
            "icon": "📖",
            "title": "学习会话",
            "desc": "Socratic 引导式学习 + 对话中补充完善图谱",
            "page": "study",
        },
        {
            "icon": "🔁",
            "title": "间隔复习",
            "desc": "SM-2 自适应出题，到期巩固，薄弱回炉",
            "page": "review",
        },
        {
            "icon": "📖",
            "title": "教材检索",
            "desc": "PDF 拖拽上传，RAG 语义检索原文段落",
            "page": "search",
        },
        {
            "icon": "⚙️",
            "title": "模型设置",
            "desc": "配置 LLM 提供商、API Key 与模型参数",
            "page": "settings",
        },
    ]

    row_size = 3
    for row_start in range(0, len(entries), row_size):
        cols = st.columns(row_size)
        for i, entry in enumerate(entries[row_start : row_start + row_size]):
            with cols[i]:
                card_content = (
                    f'<div style="text-align:center; padding: 0.5rem 0;">'
                    f'<div style="font-size:2rem;">{entry["icon"]}</div>'
                    f'<p style="color:#6B7280; font-size:0.85rem; margin:0.5rem 0 0 0;">'
                    f'{entry["desc"]}</p></div>'
                )
                card(
                    title=entry["title"],
                    content=card_content,
                    description="",
                    key=f"home_card_{entry['page']}",
                )
                if st.button(
                    f"进入 {entry['title']}",
                    key=f"home_enter_{entry['page']}",
                    use_container_width=True,
                ):
                    st.switch_page(f"pages/{entry['page']}.py")


def _build_app() -> None:
    """构建带侧边栏导航的多页面应用。

    仅在 Streamlit runtime 已就绪时调用（bootstrap 重执行场景）。
    """
    from learning_agent.ui.pages_builder import main as builder_main
    from learning_agent.ui.pages_graph import main as graph_main
    from learning_agent.ui.pages_review import main as review_main
    from learning_agent.ui.pages_search import main as search_main
    from learning_agent.ui.pages_settings import main as settings_main
    from learning_agent.ui.pages_study import main as study_main

    st.set_page_config(
        page_title="活书 huoshu",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 全局 CSS 注入（在所有页面生效）
    _inject_global_css()

    pages: list[Any] = [
        st.Page(home_main, title="首页", icon="🏠", url_path="home", default=True),
        st.Page(builder_main, title="AI 建图谱", icon="🤖", url_path="builder"),
        st.Page(graph_main, title="知识图谱", icon="📊", url_path="graph"),
        st.Page(study_main, title="学习会话", icon="📖", url_path="study"),
        st.Page(search_main, title="教材检索", icon="🔍", url_path="search"),
        st.Page(review_main, title="间隔复习", icon="🔁", url_path="review"),
        st.Page(settings_main, title="模型设置", icon="⚙️", url_path="settings"),
    ]

    st.navigation(pages).run()


def main() -> None:
    """Console script 入口：代理到 Streamlit CLI。

    说明：`huoshu` 安装后是 console script（learning_agent.main:main），
    直接构建页面会缺少 Streamlit Runtime。这里把启动代理给
    `streamlit run <main.py>`，由 bootstrap 重新执行本文件。
    """
    import streamlit.web.cli as st_cli

    script = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script, *sys.argv[1:]]
    st_cli.main()


if __name__ == "__main__":
    # 两种进入方式：
    # 1) python main.py / console script 代理：bootstrap 重执行本文件时，
    #    __name__ == "__main__" 且 Runtime 已存在 → 直接构建应用。
    # 2) streamlit run main.py：同样 Runtime 已存在 → 直接构建应用。
    # 反之（无 Runtime）→ 代理到 streamlit CLI。
    from streamlit.runtime import exists as _runtime_exists

    if _runtime_exists():
        _build_app()
    else:
        main()
