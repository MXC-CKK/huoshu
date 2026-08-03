"""Streamlit AI 建图谱页。

上传教材 PDF，LLM 自动抽取知识图谱（章节结构 → 原子知识点 → 依赖边），
校对后保存为 bookmap JSON。

用法:
    streamlit run src/learning_agent/ui/pages_builder.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]


# ── Streamlit 页面 ─────────────────────────────────────────────────────


def main() -> None:
    """AI 建图谱页入口。"""
    if st is None:
        print("Streamlit 未安装。请运行: pip install streamlit")
        raise SystemExit(1)

    from learning_agent.llm import LLMConfig
    from learning_agent.ui.pages_search import list_pdf_files, resolve_pdf_dir

    st.set_page_config(
        page_title="AI 建图谱 · 活书",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 AI 建图谱")
    st.caption(
        "上传教材 PDF，AI 自动抽取知识图谱（章节 → 知识点 → 依赖关系）。"
        "生成后可在校对区编辑调整，确认无误后保存。"
    )

    # ── 检查 LLM 可用性 ──
    llm_config = LLMConfig.resolve()
    llm_available = (
        llm_config.provider == "ollama"
        or bool(llm_config.api_key)
    )
    if not llm_available:
        st.warning(
            "⚠️ LLM 未配置。AI 建图谱需要大模型支持。"
            "请先到「⚙️ 模型设置」页配置 API Key 或 Ollama。"
        )
        return

    # ── 推理模型警告（建图谱极慢，建议切换非推理模型）──
    _render_model_hint(llm_config)

    # ── PDF 选择 ──
    from streamlit_extras.stylable_container import stylable_container

    with stylable_container(
        key="builder_pdf_card",
        css_styles="""
            .builder-pdf-card {
                background: #FFFFFF;
                border-radius: 12px;
                padding: 1.25rem;
                box-shadow: 0 1px 3px rgba(0,0,0,.06);
                margin-bottom: 1rem;
            }
        """,
    ):
        st.subheader("📂 选择教材")

        pdf_dir = resolve_pdf_dir()
        pdf_files = list_pdf_files(pdf_dir)

        col_pdf, col_goal = st.columns([2, 1])
        with col_pdf:
            if pdf_files:
                pdf_options = {p.name: p for p in pdf_files}
                selected_name = st.selectbox(
                    "选择 PDF 文件",
                    options=list(pdf_options.keys()),
                    help="从 PDF 目录中选择教材",
                )
                selected_pdf: Path | None = pdf_options[selected_name]
            else:
                st.info(
                    f"PDF 目录 `{pdf_dir}` 中尚无文件。"
                    "请先到「📖 教材检索」页上传 PDF。"
                )
                selected_pdf = None

        # ── 生成前预估（调用次数 + 预计时间）──
        if selected_pdf is not None:
            _render_estimate(selected_pdf, llm_config)

        with col_goal:
            goal = st.radio(
                "学习目标（可选）",
                options=["", "考试复习", "系统读懂", "快速应用"],
                format_func=lambda g: {
                    "": "📚 不指定",
                    "考试复习": "📝 考试复习",
                    "系统读懂": "🔬 系统读懂",
                    "快速应用": "⚡ 快速应用",
                }.get(g, g),
                help="影响 AI 抽取知识点的粒度偏好",
            )

        # ── 生成按钮 ──
        gen_disabled = selected_pdf is None

        if st.button("🤖 AI 生成知识图谱", type="primary", disabled=gen_disabled, use_container_width=True):
            if selected_pdf is None:
                st.error("请先选择 PDF 文件")
                return

            _run_generation(selected_pdf, goal)


def _run_generation(pdf_path: Path, goal: str) -> None:
    """执行图谱生成流程（st.status 分阶段展示进度）。"""
    from learning_agent.build.graph_builder import build_bookmap_from_pdf
    from learning_agent.core.graph import Bookmap
    from learning_agent.llm import LLMClient

    # 进度回调：实时写入 status 容器（用户可见中间进度，不再最后一次性输出）
    progress_messages: list[str] = []

    def _progress(stage: str, current: int, total: int) -> None:
        if total > 0:
            msg = f"{stage} ({current}/{total})"
        else:
            msg = stage
        progress_messages.append(msg)
        st.write(f"⏳ {msg}")  # 实时显示（st.status 容器内逐条追加）

    # 分阶段展示
    with st.status("🤖 AI 正在分析教材...", expanded=True) as status:
        try:
            client = LLMClient.from_env()

            st.write(f"📖 解析 PDF 文本（模型: `{client.config.model}`）...")
            result = build_bookmap_from_pdf(
                pdf_path=pdf_path,
                llm=client,
                goal=goal,
                progress=_progress,
            )

            status.update(label="✅ 图谱生成完成！", state="complete")
        except Exception as exc:  # noqa: BLE001 - 生成失败需展示给用户
            status.update(label=f"❌ 生成失败: {exc}", state="error")
            st.error(f"图谱生成失败: {exc}")
            st.caption(
                "💡 建议：① 到「⚙️ 模型设置」把模型切换为 deepseek-chat（非推理模型，"
                "结构化输出更稳定）后重试；② 换一本更小的 PDF 或按章节拆分；③ 稍后再试（"
                "LLM 服务偶发不稳定）"
            )
            st.caption(
                "请确认 LLM 配置正确（设置页可测试连接），"
                "PDF 文件完整且含章节标题。"
            )
            return

    for msg in progress_messages:
        st.caption(f"  ✓ {msg}")

    bookmap = result["bookmap"]
    stats = result["stats"]
    failed = result.get("failed_chapters", [])

    if failed:
        st.warning(f"⚠️ {len(failed)} 个章节抽取失败: {'; '.join(failed)}")

    # ── 自动保存草稿（生成即落盘，刷新/断连不丢；图谱页立即可见）──
    auto_path = _auto_save_draft(bookmap, pdf_path)
    st.success(f"💾 已自动保存草稿到 `{auto_path}`（无需手动保存，可直接去「📊 知识图谱」页查看）")
    st.session_state["generated_bookmap"] = bookmap
    st.session_state["generated_stats"] = stats

    # ── 校验 ──
    st.subheader("🔍 Schema 校验")
    try:
        bm = Bookmap.from_dict(bookmap, validate_on_load=True)
        if bm.is_valid:
            st.success(f"✅ 校验通过 — {stats['items']} 个知识点，{stats['edges']} 条依赖边")
        else:
            st.warning(f"⚠️ 校验发现 {len(bm.errors)} 个问题（仍可手动校对后保存）")
            for err in bm.errors[:10]:
                st.caption(f"  • {err}")
    except Exception as exc:  # noqa: BLE001 - 校验失败需展示给用户
        st.error(f"❌ 图谱结构无效: {exc}")

    # ── 统计卡片 ──
    from streamlit_shadcn_ui import metric_card as mc

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mc(label="📚 章节", value=stats["clusters"], delta="", key="stat_chapters")
    with col2:
        mc(label="🧩 知识点", value=stats["items"], delta="", key="stat_items")
    with col3:
        mc(label="🔗 依赖边", value=stats["edges"], delta="", key="stat_edges")
    with col4:
        mc(label="🔬 白箱", value=stats["whitebox"], delta="", key="stat_whitebox")
    with col5:
        mc(label="🔧 黑箱", value=stats["blackbox"], delta="", key="stat_blackbox")

    # ── 校对区 ──
    st.divider()
    st.subheader("✏️ 校对编辑")

    # 保存到 session_state
    st.session_state["generated_bookmap"] = bookmap
    st.session_state["generated_stats"] = stats

    # data_editor 编辑 items
    items = bookmap.get("items", [])
    if items:
        st.caption("编辑知识点（可修改 title/type/mode/note，点右键删行）：")
        editor_rows: list[dict[str, Any]] = []
        for it in items:
            editor_rows.append({
                "id": it.get("id", ""),
                "title": it.get("title", ""),
                "type": it.get("type", "concept"),
                "mode": it.get("mode", "blackbox"),
                "cluster": it.get("cluster", ""),
                "source": it.get("source", ""),
                "note": it.get("note", ""),
            })

        edited = st.data_editor(
            editor_rows,
            column_config={
                "id": st.column_config.TextColumn("ID", disabled=True, width="small"),
                "title": st.column_config.TextColumn("标题", width="medium"),
                "type": st.column_config.SelectboxColumn(
                    "类型",
                    options=["definition", "concept", "theorem", "method", "example", "application", "section", "exercise"],
                    width="small",
                ),
                "mode": st.column_config.SelectboxColumn(
                    "模式",
                    options=["blackbox", "whitebox"],
                    width="small",
                ),
                "cluster": st.column_config.TextColumn("簇", disabled=True, width="small"),
                "source": st.column_config.TextColumn("教材锚点", width="medium"),
                "note": st.column_config.TextColumn("要点", width="medium"),
            },
            num_rows="dynamic",
            use_container_width=True,
            height=400,
            key="bookmap_editor",
        )
        st.session_state["edited_items"] = edited

    # ── 边预览 ──
    with st.expander("🔗 依赖边预览（只读）"):
        prereq_pairs: list[str] = []
        for it in items:
            for p in it.get("prerequisites", []):
                prereq_pairs.append(f"{p} → {it['id']}")
        if prereq_pairs:
            st.caption(f"共 {len(prereq_pairs)} 条前置依赖边")
            st.text("\n".join(prereq_pairs[:50]))
            if len(prereq_pairs) > 50:
                st.caption(f"... 还有 {len(prereq_pairs) - 50} 条")
        else:
            st.caption("无边")

    # ── JSON 预览 ──
    with st.expander("📄 完整 JSON 预览"):
        st.json(bookmap, expanded=False)

    # ── 保存 ──
    st.divider()
    st.subheader("💾 保存图谱")

    default_name = pdf_path.stem
    save_name = st.text_input(
        "文件名（不含扩展名）",
        value=st.session_state.get("builder_save_name", default_name),
        help="保存到 ~/.huoshu/bookmap/<name>.json",
    )

    if st.button("💾 保存为图谱", type="primary", use_container_width=True):
        _save_bookmap(save_name)


def _render_model_hint(llm_config: Any) -> None:
    """生成前提示：推理模型建图谱很慢，建议切换非推理模型（可一键切换）。"""
    from streamlit_shadcn_ui import alert

    from learning_agent.llm import LLMConfig

    model = llm_config.model
    is_reasoner = (
        "v4-pro" in model.lower()
        or "reasoner" in model.lower()
        or "thinking" in model.lower()
        or "r1" in model.lower()
    )
    if not is_reasoner:
        return

    alert(
        title="⚠️ 推理模型警告",
        description=(
            f"当前模型 `{model}` 是推理模型：每次调用思考时间长，"
            "建图谱会非常慢。建议切换为 **deepseek-chat**（非推理，快且稳）。"
        ),
        variant="destructive",
        key="reasoner_alert",
    )
    if st.button("⚡ 一键切换为 deepseek-chat", key="switch_model_btn"):
        cfg = LLMConfig.from_file()
        if cfg is not None:
            cfg.model = "deepseek-chat"
            cfg.save()
            st.success("✅ 已切换为 deepseek-chat，请重新点击生成。")
            st.rerun()


def _render_estimate(pdf_path: Path, llm_config: Any) -> None:
    """根据 PDF 规模估算 LLM 调用次数与预计耗时，透明化等待时间。"""
    from learning_agent.build.graph_builder import MAX_EXTRACT_CHARS
    from learning_agent.rag.ingest import extract_pages

    try:
        pages = extract_pages(pdf_path)
        total_chars = sum(len(text) for _, text in pages)
    except Exception:  # noqa: BLE001 - 预估失败不阻塞生成
        return

    # 目录抽取 1 次 + 子块抽取 N 次 + 边推断 1 次（降级时更少）
    sub_chunks = max(1, (total_chars + MAX_EXTRACT_CHARS - 1) // MAX_EXTRACT_CHARS)
    calls = 1 + sub_chunks + 1
    is_reasoner = (
        "v4-pro" in llm_config.model.lower()
        or "reasoner" in llm_config.model.lower()
    )
    per_call = "30~90 秒" if is_reasoner else "5~15 秒"
    total = calls * ("40~120 秒" if is_reasoner else "10~25 秒")

    st.info(
        f"📊 教材约 {total_chars:,} 字符 / {len(pages)} 页 → 预计 **{calls} 次** LLM 调用"
        f"（目录 1 + 知识点 {sub_chunks} + 关系 1）。"
        f"当前模型每条约 {per_call}，全程约 {total}。"
        f"生成过程会实时显示进度，无需担心卡住。"
    )


def _auto_save_draft(bookmap: dict[str, Any], pdf_path: Path) -> Path:
    """生成后自动保存草稿到 bookmap 目录（幂等，同名追加序号）。

    防止用户忘记手动保存 / 页面刷新导致 session_state 丢失。

    Args:
        bookmap: 生成的图谱字典。
        pdf_path: 来源 PDF 路径（用于默认文件名）。

    Returns:
        实际写入的文件路径。
    """
    import json

    from learning_agent.build.graph_builder import resolve_bookmap_dir

    save_dir = resolve_bookmap_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_save_name(pdf_path.stem)
    dest = save_dir / f"{safe_name}.json"
    counter = 1
    while dest.exists():
        dest = save_dir / f"{safe_name}-{counter}.json"
        counter += 1

    dest.write_text(
        json.dumps(bookmap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[pages_builder] 自动保存草稿: {dest}")
    return dest


def _sanitize_save_name(name: str) -> str:
    """清洗保存文件名，防路径穿越。

    Args:
        name: 用户输入的文件名（不含扩展名）。

    Returns:
        安全的文件名（仅保留字母/数字/中文/连字符/下划线/空格）。

    Raises:
        ValueError: 清洗后为空。
    """
    # 只取 basename（防路径穿越：Path(name).name）
    safe = Path(name).name

    # 移除扩展名（用户可能误输入 .json）
    if safe.lower().endswith(".json"):
        safe = safe[:-5]

    # 去除所有路径分隔符和非法字符，只保留安全字符集
    safe = re.sub(r"[^\w一-鿿\- ]", "_", safe)

    # 去除首尾空白和点
    safe = safe.strip(". ")

    if not safe:
        raise ValueError("文件名无效（清洗后为空）")
    return safe


def _save_bookmap(name: str) -> None:
    """保存生成的图谱到文件。"""
    from learning_agent.build.graph_builder import resolve_bookmap_dir

    bookmap = st.session_state.get("generated_bookmap")
    edited_items = st.session_state.get("edited_items")

    if bookmap is None:
        st.error("没有可保存的图谱，请先生成。")
        return

    # 清洗文件名（防路径穿越）
    try:
        safe_name = _sanitize_save_name(name)
    except ValueError:
        st.error("文件名无效，请使用字母/数字/中文/连字符/下划线。")
        return

    # 应用编辑
    if edited_items is not None:
        # 从编辑数据重建 items（保留 id/prerequisites/related 等非编辑字段）
        id_map: dict[str, dict[str, Any]] = {
            it["id"]: it for it in bookmap.get("items", [])
        }
        new_items: list[dict[str, Any]] = []
        for row in edited_items:
            row_id = row.get("id", "")
            original = id_map.pop(row_id, {})
            new_items.append({
                "id": row_id,
                "cluster": row.get("cluster", original.get("cluster", "")),
                "title": row.get("title", original.get("title", "")),
                "type": row.get("type", original.get("type", "concept")),
                "mode": row.get("mode", original.get("mode", "blackbox")),
                "source": row.get("source", original.get("source", "")),
                "prerequisites": original.get("prerequisites", []),
                "related": original.get("related", []),
                "note": row.get("note") if row.get("note") else None,
                "mastery": original.get("mastery", 0.0),
                "next_review": original.get("next_review"),
                "status": original.get("status", "pending"),
                "cross_refs": original.get("cross_refs", []),
            })
        # 保留未被编辑删除的原始 item（实际上 data_editor 已删行则不见）
        bookmap["items"] = new_items

    # 保存
    save_dir = resolve_bookmap_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 去重：同名文件追加序号
    dest = save_dir / f"{safe_name}.json"
    counter = 1
    while dest.exists():
        dest = save_dir / f"{safe_name}-{counter}.json"
        counter += 1

    # 二次防护：确保解析后路径仍在 save_dir 内
    resolved_dest = dest.resolve()
    resolved_dir = save_dir.resolve()
    if not str(resolved_dest).startswith(str(resolved_dir)):
        st.error("文件名包含非法路径字符，拒绝保存。")
        return

    resolved_dest.write_text(
        json.dumps(bookmap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.success(f"✅ 图谱已保存到 `{resolved_dest}`")
    st.caption("前往「📊 知识图谱」页加载该文件即可浏览。")


if __name__ == "__main__":
    main()
