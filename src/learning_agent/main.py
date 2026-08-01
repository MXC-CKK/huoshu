"""活书 huoshu — 统一启动入口。

把图谱可视化 / 学习会话 / 间隔复习三个页面聚合为带侧边栏导航的单应用。

用法:
    huoshu                        # 安装后直接运行（默认 8501 端口）
    huoshu --server.port 8600     # 指定端口
    streamlit run src/learning_agent/main.py
"""

from __future__ import annotations

import sys
from typing import Any

# ---- Streamlit 导入（缺失时降级，保证 CLI --version 等可用）----
import importlib
import importlib.util

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]


def _build_app() -> Any:
    """构建带侧边栏导航的多页面应用。"""
    from learning_agent.ui.pages_graph import main as graph_main
    from learning_agent.ui.pages_review import main as review_main
    from learning_agent.ui.pages_study import main as study_main

    st.set_page_config(
        page_title="活书 huoshu",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages: list[Any] = [
        st.Page(graph_main, title="知识图谱", icon="📊", default=True),
        st.Page(study_main, title="学习会话", icon="📖"),
        st.Page(review_main, title="间隔复习", icon="🔁"),
    ]

    nav = st.navigation(pages)
    nav.run()


def main() -> None:
    """Console script 入口：代理到 Streamlit CLI。

    说明：`huoshu` 安装后是 console script，直接调用 main() 不会启动
    Web 服务器。这里通过 streamlit.web.cli 以 main.py 为脚本启动。
    """
    import streamlit.web.cli as st_cli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    st_cli.main()


if __name__ == "__main__":
    main()
