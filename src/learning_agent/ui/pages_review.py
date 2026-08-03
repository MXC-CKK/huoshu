"""Streamlit 间隔复习页。

交互式复习会话：
- 自动筛选到期 items（按项目分组）
- 自适应出题（按掌握度分级：基础/理解/应用）
- 作答 → 判分 → 掌握度更新 → 间隔重排
- 薄弱项展示 + 教材锚点回炉

用法:
    streamlit run src/learning_agent/ui/pages_review.py
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

st: Any = importlib.import_module("streamlit") if importlib.util.find_spec("streamlit") else None

from learning_agent.core.graph import Bookmap
from learning_agent.ui.review_engine import (
    MAX_ITEMS_PER_SESSION,
    ReviewEngine,
    ReviewQuestion,
    format_review_summary,
)

# ── 常量 ─────────────────────────────────────────────────────────────


def main() -> None:
    """Streamlit 复习页入口。"""
    if st is None:
        print("Streamlit 未安装。")
        raise SystemExit(1)

    st.set_page_config(
        page_title="活书 · 间隔复习",
        page_icon="🔁",
        layout="wide",
    )
    st.title("🔁 间隔复习")

    _init_state()

    # ── 侧边栏 ──
    with st.sidebar:
        _render_sidebar()

    # ── 主区域 ──
    bm = st.session_state.get("review_bookmap")
    if bm is None:
        st.info("👈 请先选择图谱")
        return

    engine = st.session_state.get("review_engine")
    if engine is None:
        engine = ReviewEngine(bm)
        st.session_state.review_engine = engine

    # 双栏: 题目 + 结果
    col_q, col_r = st.columns([2, 1])

    with col_q:
        _render_question_area(bm, engine)

    with col_r:
        _render_results_panel(engine)


# ── 初始化 ───────────────────────────────────────────────────────────


def _init_state() -> None:
    """初始化 session_state。"""
    defaults: dict[str, Any] = {
        "review_bookmap": None,
        "review_engine": None,
        "review_questions": [],
        "current_question_idx": 0,
        "review_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 侧边栏 ───────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    """渲染侧边栏：图谱选择 + 到期项列表 + 控制。"""
    st.header("📂 图谱")
    bm = _bookmap_selector()
    st.session_state.review_bookmap = bm

    if bm is None:
        return

    st.divider()

    # 加载到期 items
    engine = st.session_state.review_engine
    if engine is None:
        engine = ReviewEngine(bm)
        st.session_state.review_engine = engine

    # 生成题目按钮
    if not st.session_state.review_questions:
        st.header("📅 到期项目")
        due = engine.get_due_items()

        if not due:
            st.success("🎉 没有到期项目！一切都在掌控中。")
            return

        st.caption(f"共 {len(due)} 个到期项")
        from streamlit_shadcn_ui import progress as shadcn_progress

        # 显示到期项概览进度条（以最大会话数衡量）
        session_cap = min(len(due), MAX_ITEMS_PER_SESSION)
        shadcn_progress(value=session_cap / max(len(due), 1), key="due_progress", show_value=False)
        st.caption(f"本场可复习 {session_cap}/{len(due)} 个到期项")

        for item in due[:10]:
            mastery_emoji = "🟢" if item.mastery >= 0.8 else "🟡" if item.mastery >= 0.5 else "🟠"
            st.caption(
                f"{mastery_emoji} **{item.title}** "
                f"(掌握度 {item.mastery:.0%}, 复习日 {item.next_review})"
            )
        if len(due) > 10:
            st.caption(f"... 还有 {len(due) - 10} 项")

        if st.button(f"🚀 开始复习 ({session_cap} 题)", type="primary"):
            questions = engine.generate_all_questions()
            st.session_state.review_questions = questions
            st.session_state.current_question_idx = 0
            st.session_state.review_done = False
            st.rerun()
    else:
        # 复习进行中
        idx = st.session_state.current_question_idx
        total = len(st.session_state.review_questions)

        from streamlit_shadcn_ui import progress as shadcn_progress

        if total > 0:
            shadcn_progress(value=idx / total, key="review_progress", show_value=True)
            st.caption(f"进度: {idx}/{total}")

        if idx >= total and not st.session_state.review_done:
            st.session_state.review_done = True

        if st.button("🔄 重新开始"):
            _reset_review()
            st.rerun()


def _bookmap_selector() -> Bookmap | None:
    """渲染图谱选择器（多目录合并扫描）。"""
    from learning_agent.ui.bookmap_selector import format_bookmap_label, list_bookmap_files

    found = list_bookmap_files()
    candidates = [p for p, _ in found]
    source_dirs = {str(p): d for p, d in found}
    multi_dir = len({str(d) for _, d in found}) > 1

    if not candidates:
        st.warning("未找到图谱")
        return None

    selected = st.selectbox(
        "选择图谱",
        options=[str(c) for c in candidates],
        format_func=lambda s: format_bookmap_label(
            Path(s), source_dirs.get(str(Path(s)), Path(s).parent), multi_dir=multi_dir
        ),
        key="review_bookmap_sel",
    )
    if not selected:
        return None

    try:
        return Bookmap.load(Path(selected))
    except Exception as exc:  # noqa: BLE001 - UI 层兜底
        st.error(f"加载失败: {exc}")
        return None


# ── 题目区域 ─────────────────────────────────────────────────────────


def _render_question_area(bm: Bookmap, engine: ReviewEngine) -> None:
    """渲染题目和作答区域。"""
    questions: list[ReviewQuestion] = st.session_state.review_questions
    if not questions:
        st.header("📝 准备复习")
        st.markdown("在侧边栏点击 **开始复习** 来生成题目。")
        return

    idx = st.session_state.current_question_idx

    if idx >= len(questions):
        _render_summary(engine)
        return

    question = questions[idx]

    # 题目卡片
    with st.container(border=True):
        st.subheader(f"第 {idx + 1}/{len(questions)} 题")
        st.caption(
            f"知识点: **{question.item_title}** "
            f"| 难度: {_difficulty_label(question.difficulty)} "
            f"| 📖 {question.source_anchor}"
        )

        # 题型标签
        qtype_label = {
            "choice": "📋 选择题",
            "explain": "✏️ 解释题",
            "apply": "🔧 应用题",
        }.get(question.question_type, "")
        st.markdown(f"**{qtype_label}** {question.question_text}")

    # 作答区域
    st.divider()
    st.subheader("✍️ 你的答案")
    answer = st.text_area(
        "输入你的回答",
        placeholder="写下你的理解...",
        key=f"answer_{idx}",
        height=120,
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ 提交答案", type="primary", key=f"submit_{idx}") and answer:
            result = engine.evaluate_answer(question.item_id, answer)
            if result:
                st.session_state.last_result = result
                st.session_state.current_question_idx = idx + 1
                st.rerun()

    with cols[1]:
        if st.button("⏭ 跳过", key=f"skip_{idx}"):
            st.session_state.current_question_idx = idx + 1
            st.rerun()


# ── 结果面板 ─────────────────────────────────────────────────────────


def _render_results_panel(engine: ReviewEngine) -> None:
    """渲染右侧结果面板。"""
    st.header("📊 结果")

    # 最近结果
    last = st.session_state.get("last_result")
    if last:
        with st.container(border=True):
            emoji = "✅" if last.score >= 0.8 else "⚠️" if last.score >= 0.4 else "❌"
            st.subheader(f"{emoji} {_score_label(last.score)}")
            st.caption(
                f"掌握度: {last.mastery_before:.0%} → {last.mastery_after:.0%}"
            )
            st.caption(f"下次复习: {last.next_review}")
            st.info(last.feedback)

    # 历史汇总
    if engine.results:
        from streamlit_shadcn_ui import metric_card as rmc

        st.divider()
        st.subheader("📋 本次汇总")
        summary = engine.summarize()
        rmc(label="已复习", value=f"{summary.reviewed}/{summary.total_due}", delta="", key="result_reviewed")
        c1, c2, c3 = st.columns(3)
        with c1:
            rmc(label="✅ 正确", value=summary.correct, delta="", key="result_correct")
        with c2:
            rmc(label="⚠️ 部分", value=summary.partial, delta="", key="result_partial")
        with c3:
            rmc(label="❌ 错误", value=summary.incorrect, delta="", key="result_incorrect")


def _render_summary(engine: ReviewEngine) -> None:
    """渲染复习完成汇总页。"""
    st.balloons()
    st.header("🎉 复习完成！")

    summary = engine.summarize()
    st.markdown(format_review_summary(summary))

    # 薄弱项回炉建议
    weak = [r for r in engine.results if r.score < 0.4]
    if weak:
        st.divider()
        from streamlit_extras.stylable_container import stylable_container

        with stylable_container(
            key="review_weak_card",
            css_styles="""
                .review-weak-card {
                    background: #FFFFFF;
                    border-radius: 12px;
                    padding: 1rem 1.25rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,.06);
                    border-left: 4px solid #EF4444;
                }
            """,
        ):
            st.subheader("⚠️ 需要回炉的知识点")
            for w in weak:
                st.markdown(
                    f"- **{w.question.item_title}** "
                    f"(掌握度 {w.mastery_before:.0%} → {w.mastery_after:.0%})\n"
                    f"  📖 请复习教材: {w.question.source_anchor}"
                )

    if st.button("🔄 开始新一轮复习"):
        _reset_review()
        st.rerun()


# ── 辅助 ─────────────────────────────────────────────────────────────


def _difficulty_label(d: str) -> str:
    """难度 → 中文标签。"""
    return {"basic": "基础", "understanding": "理解", "application": "应用"}.get(d, d)


def _score_label(s: float) -> str:
    """得分 → 中文标签。"""
    if s >= 0.8:
        return "全对"
    elif s >= 0.4:
        return "部分对"
    else:
        return "需回炉"


def _reset_review() -> None:
    """重置复习状态。"""
    st.session_state.review_questions = []
    st.session_state.current_question_idx = 0
    st.session_state.review_done = False
    st.session_state.last_result = None
    # 保留 engine 不变（结果历史保留），下次点开始会重新生成


if __name__ == "__main__":
    main()
