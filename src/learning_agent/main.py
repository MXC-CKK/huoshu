"""活书 huoshu — 统一启动入口。

把图谱可视化 / 学习会话 / 间隔复习三个页面聚合为带侧边栏导航的单应用。

用法:
    huoshu                        # 安装后直接运行（默认 8501 端口）
    huoshu --server.port 8600     # 指定端口
    python -m learning_agent.main # 等效
    streamlit run src/learning_agent/main.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]


def _build_app() -> None:
    """构建带侧边栏导航的多页面应用。

    仅在 Streamlit runtime 已就绪时调用（bootstrap 重执行场景）。
    """
    from learning_agent.ui.pages_graph import main as graph_main
    from learning_agent.ui.pages_review import main as review_main
    from learning_agent.ui.pages_settings import main as settings_main
    from learning_agent.ui.pages_study import main as study_main

    st.set_page_config(
        page_title="活书 huoshu",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages: list[Any] = [
        st.Page(graph_main, title="知识图谱", icon="📊", url_path="graph", default=True),
        st.Page(study_main, title="学习会话", icon="📖", url_path="study"),
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
