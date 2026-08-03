"""Streamlit 教材检索页。

提供 PDF 教材的自然语言语义检索：
- 入库：上传 PDF（推荐）或从目录选择 → 解析 + 分块 + ChromaDB 向量化
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
import re
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


def ensure_pdf_dir() -> Path:
    """确保 PDF 教材目录存在，不存在则递归创建。

    Returns:
        PDF 教材目录的 Path 对象（已保证存在）。
    """
    pdf_dir = resolve_pdf_dir()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


def sanitize_filename(name: str) -> str:
    """清洗上传文件名，防路径穿越和非法字符。

    Args:
        name: 原始文件名（可能含路径）。

    Returns:
        安全的文件名：仅 basename、非法字符替换为 _、以 .pdf 结尾。
    """
    # 只取 basename，防路径穿越
    safe = Path(name).name

    # 去除 Windows 非法字符
    illegal_chars = r'\/:*?"<>|'
    for ch in illegal_chars:
        safe = safe.replace(ch, "_")

    # 去除首尾空白/点
    safe = safe.strip(". ")

    # 空结果回退
    if not safe:
        safe = "uploaded"

    # 保证以 .pdf 结尾
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"

    return safe


def sanitize_collection_name(name: str) -> str:
    """将任意文件名清洗为 ChromaDB 合法集合名。

    ChromaDB 集合名只允许 ``[a-zA-Z0-9._-]``（3-512 字符，首尾字母数字），
    中文教材文件名（如「高计_Ch5_2024」）必须转换。

    Args:
        name: 原始名称（通常是 PDF 文件名 stem）。

    Returns:
        合法集合名：非法字符替换为 ``_``，去首尾分隔符，最短 3 字符。
    """
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name or "")
    # 去首尾非字母数字字符（_ . -）
    cleaned = cleaned.strip("_.-")
    # 最短 3 字符（ChromaDB 下限），空串回退
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}" if cleaned else "col_book"
    # 最长 512 字符（ChromaDB 上限）
    return cleaned[:512]


def save_uploaded_pdf(filename: str, content: bytes) -> Path:
    """将上传的 PDF 内容保存到 PDF 教材目录。

    Args:
        filename: 原始文件名（会经过 sanitize_filename 清洗）。
        content: PDF 文件字节内容。

    Returns:
        保存后的文件 Path（含完整路径和文件名）。
    """
    safe_name = sanitize_filename(filename)
    pdf_dir = ensure_pdf_dir()
    dest = pdf_dir / safe_name
    dest.write_bytes(content)
    return dest


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
    st.subheader("📥 PDF 入库")
    st.caption(
        "将 PDF 教材解析、分块并向量化存入 ChromaDB，供检索使用。"
        "请确保 Ollama 正在运行且已 pull embedding 模型（如 `ollama pull nomic-embed-text`）。"
    )

    # 目录路径展示（可一键复制）
    pdf_dir = resolve_pdf_dir()
    st.caption("📂 PDF 存储目录（可一键复制）：")
    st.code(str(pdf_dir), language=None)
    st.caption("可在页面直接上传 PDF，无需手动操作文件系统。可通过环境变量 `HUOSHU_PDF_DIR` 修改路径。")

    # 两种来源选择
    source_mode = st.radio(
        "选择 PDF 来源",
        options=["upload", "directory"],
        format_func=lambda m: "📤 上传 PDF（推荐）" if m == "upload" else "📂 从目录选择",
        horizontal=True,
        key="ingest_source_mode",
    )

    if source_mode == "upload":
        _render_upload_ingest(config, pdf_dir)
    else:
        _render_directory_ingest(config, pdf_dir)


def _render_upload_ingest(config: dict[str, Any], pdf_dir: Path) -> None:
    """渲染上传模式的入库表单。"""
    from learning_agent.rag.ingest import ingest_pdf

    st.caption("拖拽或点击上传 PDF 文件，自动保存并入库。")

    uploaded_file = st.file_uploader(
        "上传教材 PDF",
        type=["pdf"],
        help="支持 PDF 格式，最大 200MB（Streamlit 默认限制）",
    )

    if uploaded_file is None:
        st.info("👆 请先上传一个 PDF 文件")
        return

    # 文件预览
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    safe_name = sanitize_filename(uploaded_file.name)
    dest = pdf_dir / safe_name

    col_preview, col_warn = st.columns([3, 1])
    with col_preview:
        st.success(f"✅ 已接收: **{uploaded_file.name}** ({file_size_kb:.1f} KB)")
        st.caption(f"将保存为: `{dest}`")
    with col_warn:
        if dest.exists():
            st.warning("⚠️ 同名文件已存在，入库将覆盖旧文件")

    # 入库参数
    col1, col2 = st.columns(2)
    with col1:
        default_coll = sanitize_collection_name(Path(uploaded_file.name).stem)
        collection_name = st.text_input(
            "集合名称",
            value=st.session_state.get("ingest_upload_coll", default_coll),
            help="ChromaDB 集合名（仅字母/数字/._-），中文文件名已自动转换，可修改",
            key="ingest_upload_coll",
        )
    with col2:
        source_name = st.text_input(
            "来源名称（可选）",
            value=Path(uploaded_file.name).stem,
            help="显示在检索结果元数据中",
            key="ingest_upload_source",
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
            key="ingest_upload_cs",
        )
    with col4:
        chunk_overlap = st.slider(
            "块间重叠（字符数）",
            min_value=0,
            max_value=400,
            value=100,
            step=20,
            help="相邻块之间重叠字符数，避免切断语义",
            key="ingest_upload_co",
        )

    # 入库状态防重复
    ingest_key = f"ingested_{collection_name}"
    if ingest_key not in st.session_state:
        st.session_state[ingest_key] = False

    if st.button("📥 开始入库", type="primary", disabled=st.session_state[ingest_key], key="ingest_upload_btn"):
        if not re.fullmatch(r"[a-zA-Z0-9._-]{3,512}", collection_name.strip()):
            st.error("❌ 集合名仅允许字母/数字/._-（3-512 字符），请修改后再入库")
            return
        try:
            content = uploaded_file.getvalue()
            with st.spinner(f"正在保存 {uploaded_file.name} 并生成向量..."):
                saved_path = save_uploaded_pdf(uploaded_file.name, content)
                collection = ingest_pdf(
                    pdf_path=saved_path,
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
                st.success(
                    f"✅ 入库完成！共 {count} 个文本块 → 集合「{collection_name}」\n\n"
                    f"文件已保存至: `{saved_path}`"
                )
                st.session_state.pop("search_collections", None)  # 刷新集合列表
        except Exception as exc:  # noqa: BLE001 - 入库失败需展示给用户
            st.error(f"❌ 入库失败：{exc}")
            st.caption("请确认 Ollama 正在运行，PDF 文件完整且未被加密。")


def _render_directory_ingest(config: dict[str, Any], pdf_dir: Path) -> None:
    """渲染目录选择模式的入库表单。"""
    from learning_agent.rag.ingest import ingest_pdf

    # 自动创建目录
    if not pdf_dir.is_dir():
        ensure_pdf_dir()
        st.info(f"📂 目录已自动创建: `{pdf_dir}`。请将 PDF 放入目录后点击刷新。")
        if st.button("🔄 刷新"):
            st.rerun()
        return

    pdf_files = list_pdf_files(pdf_dir)
    if not pdf_files:
        st.warning(f"⚠️ 目录中尚无 PDF 文件。可直接上传 PDF（推荐），或手动放入 `{pdf_dir}` 后点击刷新。")
        if st.button("🔄 刷新"):
            st.rerun()
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
            value=st.session_state.get(
                "ingest_dir_coll", sanitize_collection_name(selected_pdf.stem)
            ),
            help="ChromaDB 集合名（仅字母/数字/._-），中文文件名已自动转换，可修改",
            key="ingest_dir_coll",
        )
    with col2:
        source_name = st.text_input(
            "来源名称（可选）",
            value=selected_pdf.stem,
            help="显示在检索结果元数据中",
            key="ingest_dir_source",
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
            key="ingest_dir_cs",
        )
    with col4:
        chunk_overlap = st.slider(
            "块间重叠（字符数）",
            min_value=0,
            max_value=400,
            value=100,
            step=20,
            help="相邻块之间重叠字符数，避免切断语义",
            key="ingest_dir_co",
        )

    # 入库状态防重复
    ingest_key = f"ingested_{collection_name}"
    if ingest_key not in st.session_state:
        st.session_state[ingest_key] = False

    if st.button("📥 开始入库", type="primary", disabled=st.session_state[ingest_key], key="ingest_dir_btn"):
        if not re.fullmatch(r"[a-zA-Z0-9._-]{3,512}", collection_name.strip()):
            st.error("❌ 集合名仅允许字母/数字/._-（3-512 字符），请修改后再入库")
            return
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
            from streamlit_shadcn_ui import metric_card as smc

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                smc(label="名称", value=stats["name"], delta="", key="mgmt_name")
            with col_b:
                smc(label="文档数", value=stats["count"], delta="", key="mgmt_count")
            with col_c:
                hnsw = stats.get("metadata", {})
                smc(label="距离度量", value=hnsw.get("hnsw:space", "cosine") if hnsw else "cosine", delta="", key="mgmt_metric")
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
