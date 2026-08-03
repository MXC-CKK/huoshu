"""Streamlit 学习会话页。

交互式学习会话：
- 项目选择（bookmap JSON）
- 图谱导航（当前位置 + breadcrumb + 前置链）
- Socratic 引导式问答
- 下钻（drill-down）/ 返回（step-back）
- 迷航三栏展示（已完成 / 剩余 / 推荐）

用法:
    streamlit run src/learning_agent/ui/pages_study.py
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

st: Any = importlib.import_module("streamlit") if importlib.util.find_spec("streamlit") else None

from learning_agent.core.graph import Bookmap
from learning_agent.llm import is_llm_available
from learning_agent.ui.bookmap_selector import format_bookmap_label, list_bookmap_files
from learning_agent.ui.study_engine import (
    StudySession,
    add_item_to_bookmap,
    ask_llm,
    compute_three_column,
    extract_new_item,
    generate_socratic_prompt,
    get_navigation_context,
    locate_item,
    translate_mastery,
    translate_mode,
)

# ── 常量 ─────────────────────────────────────────────────────────────

if "study_session" not in locals():
    study_session: StudySession | None = None


def main() -> None:
    """Streamlit 学习会话页入口。"""
    if st is None:
        print("Streamlit 未安装。")
        raise SystemExit(1)

    st.set_page_config(
        page_title="活书 · 学习会话",
        page_icon="📖",
        layout="wide",
    )
    st.title("📖 学习会话")

    # ── 初始化 session state ──
    _init_state()

    # ── 侧边栏: 项目选择 + 三栏状态 ──
    with st.sidebar:
        _render_sidebar()

    # ── 主区域 ──
    if st.session_state.bookmap is None:
        st.info("👈 请先在侧边栏选择一个图谱文件")
        return

    bm: Bookmap = st.session_state.bookmap
    session: StudySession | None = st.session_state.study_session

    if session is None or not session.current_item_id:
        _render_welcome(bm, session)
        return

    # ── 活跃会话: 双栏布局 ──
    col_main, col_ctx = st.columns([3, 1])

    with col_main:
        _render_main_area(bm, session)

    with col_ctx:
        _render_context_panel(bm, session)


# ── 初始化 ───────────────────────────────────────────────────────────


def _init_state() -> None:
    """初始化 session_state。"""
    defaults: dict[str, Any] = {
        "bookmap": None,
        "bookmap_path": None,
        "study_session": None,
        "chat_history": [],
        "selected_item_id": "",
        "show_question_input": False,
        "show_progress": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 侧边栏 ───────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    """渲染侧边栏：项目选择 + LLM 状态 + 三栏状态。"""
    st.header("📂 项目")

    bm = _bookmap_selector()
    st.session_state.bookmap = bm

    if bm is None:
        return

    # LLM 状态
    llm_ok = is_llm_available()
    if llm_ok:
        st.success("🤖 LLM 已就绪")
    else:
        st.warning("⚠️ LLM 未配置 (设置 LLM_API_KEY)")

    session = st.session_state.study_session

    # 当前会话状态
    if session and session.current_item_id:
        item = bm.get_item(session.current_item_id)
        if item:
            st.success(f"📍 **{item.title}**")
            st.caption(f"`{item.id}` · {translate_mode(item.mode)}")
            if session.main_goal:
                st.caption(f"🎯 主线: {session.main_goal}")
            if session.breakdown_stack:
                st.caption(f"📚 下钻栈: {len(session.breakdown_stack)} 层")

    st.divider()

    # 三栏状态
    if bm:
        _render_three_column(bm, session)


def _bookmap_selector() -> Bookmap | None:
    """渲染 bookmap 文件选择器（多目录合并扫描）。"""

    found = list_bookmap_files()
    candidates = [p for p, _ in found]
    source_dirs = {str(p): d for p, d in found}
    multi_dir = len({str(d) for _, d in found}) > 1

    if not candidates:
        st.warning("未找到图谱文件")
        manual = st.text_input("手动输入路径", placeholder="/path/to/bookmap.json")
        if manual and Path(manual).exists():
            candidates = [Path(manual)]
        else:
            return None

    selected = st.selectbox(
        "选择图谱",
        options=[str(c) for c in candidates],
        format_func=lambda s: format_bookmap_label(
            Path(s), source_dirs.get(str(Path(s)), Path(s).parent), multi_dir=multi_dir
        ),
        key="bookmap_selector",
    )

    if not selected:
        return None

    try:
        bm = Bookmap.load(Path(selected))
        st.session_state.bookmap_path = Path(selected)
        if not bm.is_valid and bm.errors:
            with st.expander(f"⚠️ {len(bm.errors)} 个校验警告"):
                for e in bm.errors:
                    st.text(f"· {e}")
        return bm
    except Exception as exc:  # noqa: BLE001 - UI 层兜底
        st.error(f"加载失败: {exc}")
        return None


def _render_three_column(bm: Bookmap, session: StudySession | None) -> None:
    """渲染三栏：已完成 / 剩余 / 推荐。"""
    st.header("📊 学习状态")

    current_id = session.current_item_id if session else ""
    tc = compute_three_column(bm, current_id)

    tabs = st.tabs(["✅ 已完成", "📋 剩余", "⭐ 推荐"])

    with tabs[0]:
        if tc.completed:
            for item, note in tc.completed:
                st.caption(f"· **{item.title}** ({note})")
        else:
            st.caption("暂无")

    with tabs[1]:
        if tc.remaining:
            for item, reason in tc.remaining:
                st.caption(f"· **{item.title}** — {reason}")
        else:
            st.caption("全部完成！🎉")

    with tabs[2]:
        if tc.recommended:
            for item, reason in tc.recommended:
                if st.button(
                    f"▶ {item.title}",
                    help=reason,
                    key=f"rec_{item.id}",
                ):
                    _start_or_resume_session(bm, item.id)
                    st.rerun()
        else:
            st.caption("暂无推荐")


# ── 欢迎页 ───────────────────────────────────────────────────────────


def _render_welcome(bm: Bookmap, session: StudySession | None) -> None:
    """渲染新会话欢迎页：设置目标 + 快速开始。"""
    st.header("开始学习")

    # 三栏预览
    tc = compute_three_column(bm)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已完成", len(tc.completed))
    with col2:
        st.metric("剩余", len(tc.remaining))
    with col3:
        st.metric("推荐", len(tc.recommended))

    st.divider()

    # 设置目标
    goal = st.text_input(
        "🎯 本次学习目标（可选）",
        placeholder="例如：理解大数定律的证明、学完第3章",
    )

    # 推荐入口
    st.subheader("⭐ 推荐从这里开始")
    for item, reason in tc.recommended[:5]:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{item.title}** — {reason}")
        with cols[1]:
            if st.button("开始", key=f"start_{item.id}"):
                _start_or_resume_session(bm, item.id, goal)
                st.rerun()

    # 搜索
    st.divider()
    st.subheader("🔍 或者直接搜索")
    query = st.text_input("搜索知识点", placeholder="输入标题或关键词...")
    if query:
        matches = locate_item(bm, query)
        for m in matches:
            label = f"{m.title} ({m.id})"
            if st.button(f"▶ {label}", key=f"search_{m.id}"):
                _start_or_resume_session(bm, m.id, goal)
                st.rerun()


# ── 主区域 ───────────────────────────────────────────────────────────


def _render_main_area(bm: Bookmap, session: StudySession) -> None:
    """渲染主学习区域：当前位置 + 问答。"""
    item = bm.get_item(session.current_item_id)
    if item is None:
        st.error("当前位置不存在")
        return

    # breadcrumb
    _render_breadcrumb(bm, session)

    # 知识点卡片
    with st.container(border=True):
        st.subheader(f"📌 {item.title}")
        cols = st.columns(3)
        with cols[0]:
            st.caption(f"类型: {item.type}")
        with cols[1]:
            st.caption(f"模式: {translate_mode(item.mode)}")
        with cols[2]:
            st.caption(f"掌握度: {item.mastery:.0%} ({translate_mastery(item.mastery)})")

        st.caption(f"📖 锚点: {item.source}")
        if item.note:
            st.info(f"💡 {item.note}")

    # 操作按钮
    cols = st.columns(4)
    with cols[0]:
        if st.button("⬆ 返回上一层", disabled=not session.breakdown_stack):
            popped = session.step_back(bm)
            if popped:
                st.rerun()
    with cols[1]:
        if st.button("❓ 提问"):
            st.session_state.show_question_input = True
    with cols[2]:
        if st.button("📊 看进度"):
            st.session_state.show_progress = True
    with cols[3]:
        if st.button("🏠 回首页"):
            st.session_state.study_session = None
            st.rerun()

    # ── 标记已学（学习进度推进：学完自评 → 图谱更新 → 推荐下一个）──
    _render_mark_learned_area(bm, session, item)

    # ── 补充知识点（学习时图谱增量完善）──
    with st.expander("➕ 补充新知识点（加入图谱）", expanded=False):
        _render_add_item_area(bm, session, item)

    # ── 提问区域 ──
    if st.session_state.get("show_question_input"):
        _render_question_area(bm, session, item)

    # ── 聊天历史 ──
    if st.session_state.get("chat_history"):
        st.divider()
        st.subheader("💬 会话记录")
        for msg in st.session_state.chat_history[-10:]:
            role = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(role):
                st.markdown(msg["content"])


def _render_mark_learned_area(bm: Bookmap, session: StudySession, item: Any) -> None:
    """渲染「标记已学」区域：自评掌握度 → 更新图谱 → 保存 → 推荐推进。

    学习会话默认不自动改掌握度；学完一个知识点由用户自评标记，
    标记后 status/mastery/next_review 更新并保存回图谱文件，
    侧边栏三栏与推荐列表随即推进到下一个可学知识点。
    """
    from learning_agent.ui.study_engine import MASTERY_LEVELS, mark_item_learned

    st.divider()
    with st.container(border=True):
        st.caption(
            f"学完了「{item.title}」？标记学习状态即可推进到下一个知识点"
            "（会更新图谱并保存）。"
        )
        level = st.radio(
            "自评掌握度",
            options=list(MASTERY_LEVELS.keys()),
            format_func=lambda m: {
                "mastered": "✅ 掌握了（0.8）",
                "basics": "🟡 基本掌握（0.6）",
                "unsure": "🔴 还不熟（0.3，不标记已学）",
            }.get(m, m),
            horizontal=True,
            key="mastery_self_eval",
        )

        if st.button("💾 提交学习状态", type="primary", key="mark_learned_btn"):
            try:
                updated = mark_item_learned(bm, session.current_item_id, level)
                bm_path = st.session_state.get("bookmap_path")
                if bm_path is not None:
                    bm.save(bm_path)
                    saved_note = f"已保存到 `{bm_path}`"
                else:
                    saved_note = "⚠️ 未找到图谱文件路径，本次修改未落盘"

                if updated.status == "learned":
                    st.success(
                        f"✅ 「{updated.title}」已标记为已学（掌握度 {updated.mastery:.0%}，"
                        f"{updated.next_review} 复习）。{saved_note} 推荐列表已推进！"
                    )
                else:
                    st.info(
                        f"🔴 「{updated.title}」保持待学（掌握度 {updated.mastery:.0%}），"
                        "建议再看一遍教材或继续提问。"
                    )
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - UI 层兜底
                st.error(f"标记失败: {exc}")


def _render_add_item_area(bm: Bookmap, session: StudySession, item: Any) -> None:
    """渲染「补充新知识点」区域：学习会话中增量完善知识图谱。

    流程: 输入标题+描述 → LLM 抽取结构化字段 → 加入图谱 → 保存回文件。
    这样开始只需建基础图谱，学习过程中随时补充完善。
    """
    from learning_agent.llm import LLMClient

    st.caption(
        f"学习到新概念？补充进图谱，它会成为可导航的知识点（与「{item.title}」关联）。"
    )

    col_title, col_rel = st.columns([3, 2])
    with col_title:
        new_title = st.text_input(
            "新知识点名称",
            placeholder="如：GLS 估计量 / 平稳性定义 / 中心极限定理",
            key="new_item_title",
        )
    with col_rel:
        relation = st.radio(
            "与当前知识点的关系",
            options=["prerequisite", "extension", "independent"],
            format_func=lambda r: {
                "prerequisite": "🔙 它是前置知识",
                "extension": "🔗 延伸/相关",
                "independent": "🧩 独立补充",
            }.get(r, r),
            key="new_item_relation",
        )

    new_desc = st.text_area(
        "一句话描述（可选，帮助 AI 归类）",
        placeholder="如：当扰动项存在异方差时，GLS 比 OLS 更有效",
        key="new_item_desc",
        height=60,
    )

    if st.button("✨ 生成并加入图谱", type="primary", disabled=not new_title.strip(), key="add_item_btn"):
        try:
            client = LLMClient.from_env()
            if not client.available:
                st.error("LLM 未配置，无法生成新知识点。请先到「⚙️ 模型设置」配置。")
                return

            with st.spinner("🤖 AI 正在抽取知识点结构..."):
                item_data = extract_new_item(
                    client,
                    new_title.strip(),
                    new_desc,
                    current_item=item,
                )
                # 用户选择的关系优先（LLM 推断仅作参考）
                item_data["relation"] = relation
                new_item = add_item_to_bookmap(
                    bm, item_data, current_item_id=session.current_item_id
                )

            # 保存回图谱文件
            bm_path = st.session_state.get("bookmap_path")
            if bm_path is not None:
                bm.save(bm_path)
                saved_note = f"已保存到 `{bm_path}`"
            else:
                saved_note = "⚠️ 未找到图谱文件路径，本次修改未落盘（刷新后丢失）"

            st.success(
                f"✅ 新知识点「{new_item.title}」已加入图谱（`{new_item.id}`，"
                f"类型: {new_item.type}，模式: {new_item.mode}）。{saved_note}"
            )
            st.caption("刷新后可在「📊 知识图谱」页看到新节点。")
        except Exception as exc:  # noqa: BLE001 - UI 层兜底
            st.error(f"补充失败: {exc}")
            st.caption(
                "💡 建议：① 到「⚙️ 模型设置」确认模型配置；② 简化描述后重试。"
            )


def _render_question_area(bm: Bookmap, session: StudySession, item: Any) -> None:
    """渲染问答区域。"""
    st.divider()
    st.subheader("💬 提问")

    qtype = st.selectbox(
        "问题类型",
        options=["definition", "proof", "relationship", "application", "self_test"],
        format_func=lambda t: {
            "definition": "这个定义/定理怎么理解？",
            "proof": "这个证明是怎么来的？",
            "relationship": "和其他概念有什么关系？",
            "application": "有什么用？",
            "self_test": "考考我",
        }.get(t, t),
    )

    # Socratic 提示（真实判断 LLM 可用性，可用时展示 AI 引导）
    prompt = generate_socratic_prompt(item, qtype, llm_available=is_llm_available())
    st.info(prompt)

    # 回答输入
    user_answer = st.text_area("你的回答/追问", placeholder="写下你的理解或进一步的问题...")

    cols = st.columns(2)
    with cols[0]:
        if st.button("提交") and user_answer:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_answer}
            )

            # 调用 LLM（或降级模板）
            reply = ask_llm(
                bm=bm,
                item_id=session.current_item_id,
                qtype=qtype,
                user_message=user_answer,
                chat_history=st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else None,
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()

    with cols[1]:
        if st.button("取消"):
            st.session_state.show_question_input = False
            st.rerun()


# ── breadcrumb ────────────────────────────────────────────────────────


def _render_breadcrumb(bm: Bookmap, session: StudySession) -> None:
    """渲染导航面包屑。"""
    parts: list[str] = []

    # 主线目标
    if session.main_goal:
        parts.append(f"🎯 {session.main_goal}")

    # 下钻栈
    for iid, label in session.breakdown_stack:
        parts.append(f"… → {label}")

    # 当前位置
    item = bm.get_item(session.current_item_id)
    if item:
        parts.append(f"**{item.title}**")

    if parts:
        st.caption(" > ".join(parts))


# ── 上下文面板 ───────────────────────────────────────────────────────


def _render_context_panel(bm: Bookmap, session: StudySession) -> None:
    """渲染右侧上下文面板：前置链 / 相关概念 / 后置依赖。"""
    if not session.current_item_id:
        return

    nav = get_navigation_context(bm, session.current_item_id)

    # 前置链
    st.subheader("📎 前置依赖链")
    if nav.prereq_chain:
        for p in nav.prereq_chain:
            st.caption(f"· {p.title} ({p.mastery:.0%})")
            if st.button("🔍", key=f"ctx_{p.id}", help=f"跳到 {p.title}") and session.drill_down(p.id, f"补前置: {p.title}", bm):
                    st.rerun()
    else:
        st.caption("无前置依赖（入口节点）")

    # 相关概念
    st.divider()
    st.subheader("🔗 相关概念")
    if nav.related:
        for r in nav.related:
            st.caption(f"· {r.title}")
    else:
        st.caption("无相关概念")

    # 后置依赖
    st.divider()
    st.subheader("📤 后置依赖")
    if nav.dependents:
        for d in nav.dependents:
            emoji = "✅" if d.status == "learned" else "📖"
            st.caption(f"{emoji} {d.title}")
            if st.button("▶", key=f"dep_{d.id}", help=f"学 {d.title}") and session.drill_down(d.id, f"前进: {d.title}", bm):
                    st.rerun()
    else:
        st.caption("叶子节点")


# ── 会话管理 ─────────────────────────────────────────────────────────


def _start_or_resume_session(
    bm: Bookmap,
    item_id: str,
    goal: str = "",
) -> None:
    """开始或恢复学习会话。

    Args:
        bm: Bookmap 实例。
        item_id: 起始知识点 ID。
        goal: 主线目标。
    """
    session = st.session_state.study_session
    if session is None or session.project_name != bm.domain:
        session = StudySession(project_name=bm.domain, main_goal=goal)
    elif goal:
        session.set_goal(goal)

    session.move_to(item_id, bm)
    st.session_state.study_session = session
    st.session_state.chat_history = []


# ── standalone ────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
