"""Streamlit 教材检索页。

提供 PDF 教材的自然语言语义检索：
- 入库：PDF 解析 + 分块 + ChromaDB 向量化
- 检索：自然语言提问 → 返回教材原文段落 + 页码引用
- 集合管理：查看统计信息 + 删除集合

环境变量:
    HUOSHU_OLLAMA_URL  — Ollama embedding API URL（默认 http://localhost:11434/api/embeddings）
    HUOSHU_CHROMA_DIR  — ChromaDB 持久化目录（默认 ~/.huoshu/chroma）
    HUOSHU_PDF_DIR     — PDF 教材目录（默认 ~/.huoshu/pdf）

用法:
    streamlit run src/learning_agent/ui/pages_search.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]


# ── 模块级纯函数（可直接测试）───────────────────────────────────────────


def resolve_pdf_dir() -> Path:
    """返回 PDF 教材目录。

    优先级: 环境变量 HUOSHU_PDF_DIR > 默认 ~/.huoshu/pdf。

    Returns:
        PDF 教材目录的 Path 对象。
    """
    return Path(
        os.environ.get(
            "HUOSHU_PDF_DIR",
            str(Path.home() / ".huoshu" / "pdf"),
        )
    )


def list_pdf_files(pdf_dir: Path | str) -> list[Path]:
    """列出目录中所有 PDF 文件（不含子目录）。

    Args:
        pdf_dir: 目标目录路径。

    Returns:
        PDF 文件 Path 列表，按文件名排序。目录不存在时返回空列表。
    """
    dir_path = Path(pdf_dir)
    if not dir_path.is_dir():
        return []
    pdf_files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    return pdf_files


# ── Streamlit 页面 ─────────────────────────────────────────────────────


def _build_sidebar() -> dict[str, Any]:
    """构建侧边栏配置区域，返回当前配置值字典。"""
    from learning_agent.rag.ingest import resolve_chroma_dir, resolve_ollama_url

    st.sidebar.header("🔧 检索配置")

    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value=st.session_state.get("search_ollama_url", resolve_ollama_url()),
        help="Ollama embedding API 地址（需运行 Ollama 服务）",
    )
    embedding_model = st.sidebar.text_input(
        "Embedding 模型",
        value=st.session_state.get("search_embedding_model", "nomic-embed-text"),
        help="Ollama 已 pull 的 embedding 模型名",
    )
    chroma_dir = resolve_chroma_dir()
    st.sidebar.caption(f"Chroma 目录: `{chroma_dir}`")

    if st.sidebar.button("🔄 刷新集合列表"):
        st.session_state.pop("search_collections", None)

    return {
        "ollama_url": ollama_url,
        "embedding_model": embedding_model,
        "chroma_dir": chroma_dir,
    }


def _get_collections(persist_dir: str) -> list[tuple[str, int]]:
    """获取集合列表及各自的文档数。

    Args:
        persist_dir: ChromaDB 持久化目录。

    Returns:
        (collection_name, count) 列表，目录不存在时返回空列表。
    """
    from learning_agent.rag.retrieve import list_collections, open_collection

    names = list_collections(persist_dir)
    result: list[tuple[str, int]] = []
    for name in names:
        try:
            coll = open_collection(name, persist_dir)
            result.append((name, coll.count()))
        except Exception:  # noqa: BLE001 - 个别集合损坏不阻塞其他集合
            result.append((name, -1))
    return result


def _render_search_tab(config: dict[str, Any]) -> None:
    """渲染检索 Tab。"""
    from learning_agent.rag.retrieve import open_collection, query

    st.subheader("🔍 语义检索")
    st.caption("输入自然语言问题，在已入库的教材中检索最相关的原文段落。")

    collections = _get_collections(config["chroma_dir"])

    if not collections:
        st.warning("⚠️ 尚未入库任何教材。请先切换到「📥 入库」Tab 导入 PDF。")
        return

    # 集合选择
    col_names = [c[0] for c in collections]
    selected_name = st.selectbox(
        "教材集合",
        options=col_names,
        format_func=lambda n: f"{n} ({dict(collections).get(n, '?')} 个段落)",
    )

    # 查询参数
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query_text = st.text_area(
            "查询问题",
            placeholder="例如：大数定律的证明过程是什么？OLS 估计量的一致性条件？",
            height=80,
        )
    with col_k:
        top_k = st.slider("返回条数", min_value=1, max_value=10, value=5)

    if st.button("🔍 检索", type="primary", disabled=not query_text.strip()):
        if not query_text.strip():
            st.error("请输入查询问题")
            return

        try:
            with st.spinner("正在检索..."):
                collection = open_collection(
                    selected_name,
                    persist_dir=config["chroma_dir"],
                    embedding_model=config.get("embedding_model", "nomic-embed-text"),
                    ollama_url=config.get("ollama_url"),
                )
                results = query(collection, query_text.strip(), top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - 检索失败需展示给用户
            st.error(f"❌ 检索失败：{exc}")
            st.caption("请确认 Ollama 正在运行，且已 pull embedding 模型。")
            return

        if not results:
            st.info("未找到相关结果，请尝试换个问法。")
            return

        st.success(f"找到 {len(results)} 条相关段落")
        for i, r in enumerate(results, start=1):
            with st.container(border=True):
                col_h, col_s = st.columns([4, 1])
                with col_h:
                    st.caption(f"📄 {r.page_ref}")
                with col_s:
                    st.caption(f"相似度: {r.score:.3f}")
                with st.expander(f"结果 #{i} — 查看原文", expanded=(i == 1)):
                    st.text(r.text)


def _render_ingest_tab(config: dict[str, Any]) -> None:
    """渲染入库 Tab。"""
    from learning_agent.rag.ingest import ingest_pdf

    st.subheader("📥 PDF 入库")
    st.caption(
        "将 PDF 教材解析、分块并向量化存入 ChromaDB，供检索使用。"
        "请确保 Ollama 正在运行且已 pull embedding 模型（如 `ollama pull nomic-embed-text`）。"
    )

    # PDF 目录
    pdf_dir = resolve_pdf_dir()
    st.info(f"📂 PDF 教材目录: `{pdf_dir}`")
    st.caption("将 PDF 文件放入此目录后刷新即可选择入库。可通过环境变量 `HUOSHU_PDF_DIR` 修改。")

    if not pdf_dir.is_dir():
        st.warning("⚠️ PDF 目录尚不存在，请先创建并放入教材 PDF。")
        return

    pdf_files = list_pdf_files(pdf_dir)
    if not pdf_files:
        st.warning("⚠️ PDF 目录中尚无 PDF 文件。")
        return

    # PDF 下拉选择
    pdf_options = {p.name: p for p in pdf_files}
    selected_name = st.selectbox(
        "选择 PDF 文件",
        options=list(pdf_options.keys()),
    )
    selected_pdf = pdf_options[selected_name]

    st.caption(f"文件大小: {selected_pdf.stat().st_size / 1024:.1f} KB")

    # 入库参数
    col1, col2 = st.columns(2)
    with col1:
        collection_name = st.text_input(
            "集合名称",
            value=st.session_state.get("ingest_collection_name", selected_pdf.stem),
            help="ChromaDB 集合名，默认取 PDF 文件名（不含扩展名）",
        )
    with col2:
        source_name = st.text_input(
            "来源名称（可选）",
            value=selected_pdf.stem,
            help="显示在检索结果元数据中",
        )

    col3, col4 = st.columns(2)
    with col3:
        chunk_size = st.slider(
            "分块大小（字符数）",
            min_value=200,
            max_value=2000,
            value=500,
            step=100,
            help="每块最大字符数，越小检索粒度越细",
        )
    with col4:
        chunk_overlap = st.slider(
            "块间重叠（字符数）",
            min_value=0,
            max_value=400,
            value=100,
            step=20,
            help="相邻块之间重叠字符数，避免切断语义",
        )

    # 入库状态防重复
    ingest_key = f"ingested_{collection_name}"
    if ingest_key not in st.session_state:
        st.session_state[ingest_key] = False

    if st.button("📥 开始入库", type="primary", disabled=st.session_state[ingest_key]):
        try:
            with st.spinner(f"正在解析 {selected_name} 并生成向量..."):
                collection = ingest_pdf(
                    pdf_path=selected_pdf,
                    collection_name=collection_name,
                    source_name=source_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    persist_dir=config["chroma_dir"],
                    embedding_model=config.get("embedding_model", "nomic-embed-text"),
                    ollama_url=config.get("ollama_url"),
                )
                count = collection.count()
                st.session_state[ingest_key] = True
                st.success(f"✅ 入库完成！共 {count} 个文本块 → 集合「{collection_name}」")
                st.session_state.pop("search_collections", None)  # 刷新集合列表
        except Exception as exc:  # noqa: BLE001 - 入库失败需展示给用户
            st.error(f"❌ 入库失败：{exc}")
            st.caption("请确认 Ollama 正在运行，PDF 文件完整且未被加密。")


def _render_manage_tab(config: dict[str, Any]) -> None:
    """渲染集合管理 Tab。"""
    from learning_agent.rag.retrieve import collection_stats, delete_collection, open_collection

    st.subheader("🗂 集合管理")
    st.caption("查看已入库集合的详细信息，或删除不需要的集合。")

    collections = _get_collections(config["chroma_dir"])

    if not collections:
        st.info("暂无集合。请先切换到「📥 入库」Tab 导入 PDF。")
        return

    col_names = [c[0] for c in collections]
    selected_name = st.selectbox(
        "选择集合",
        options=col_names,
        format_func=lambda n: f"{n} ({dict(collections).get(n, '?')} 个段落)",
        key="manage_collection_select",
    )

    if selected_name:
        # 集合统计
        try:
            collection = open_collection(
                selected_name,
                persist_dir=config["chroma_dir"],
                embedding_model=config.get("embedding_model", "nomic-embed-text"),
                ollama_url=config.get("ollama_url"),
            )
            stats = collection_stats(collection)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("名称", stats["name"])
            with col_b:
                st.metric("文档数", stats["count"])
            with col_c:
                hnsw = stats.get("metadata", {})
                st.metric("距离度量", hnsw.get("hnsw:space", "cosine") if hnsw else "cosine")
        except Exception as exc:  # noqa: BLE001 - 无法打开集合需展示
            st.warning(f"⚠️ 无法打开集合: {exc}")

        # 删除集合（二次确认）
        st.divider()
        st.subheader("🗑 删除集合")
        st.caption("删除后不可恢复，请谨慎操作。")

        confirm = st.checkbox(
            f"我确认要删除集合「{selected_name}」及其所有数据",
            key=f"confirm_delete_{selected_name}",
        )
        if st.button("🗑 删除集合", type="secondary", disabled=not confirm):
            try:
                delete_collection(selected_name, persist_dir=config["chroma_dir"])
                st.success(f"✅ 集合「{selected_name}」已删除")
                st.session_state.pop("search_collections", None)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - 删除失败需展示
                st.error(f"❌ 删除失败：{exc}")


def main() -> None:
    """教材检索页入口。"""
    if st is None:
        print("Streamlit 未安装。请运行: pip install streamlit")
        raise SystemExit(1)

    st.set_page_config(
        page_title="教材检索 · 活书",
        page_icon="📖",
        layout="wide",
    )
    st.title("📖 教材检索")
    st.caption(
        "基于 RAG 的语义检索：将 PDF 教材分块向量化，用自然语言提问，"
        "返回原文段落及精确页码引用。"
    )

    # 侧边栏配置
    config = _build_sidebar()

    # 主区三个 Tab
    tab_search, tab_ingest, tab_manage = st.tabs(["🔍 检索", "📥 入库", "🗂 集合管理"])

    with tab_search:
        _render_search_tab(config)
    with tab_ingest:
        _render_ingest_tab(config)
    with tab_manage:
        _render_manage_tab(config)


if __name__ == "__main__":
    main()
