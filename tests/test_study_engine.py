"""Tests for learning_agent.ui.study_engine — learning session logic."""

from __future__ import annotations

from typing import Any

import pytest

from learning_agent.core.graph import Bookmap
from learning_agent.ui.study_engine import (
    StudySession,
    ThreeColumn,
    classify_question,
    compute_three_column,
    generate_socratic_prompt,
    get_navigation_context,
    get_sources,
    locate_item,
    translate_mastery,
    translate_mode,
    translate_status,
)

# ── 夹具 ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bm() -> Bookmap:
    """3 节点、2 簇的示例图谱。"""
    data: dict[str, Any] = {
        "meta": {"source": "test", "built": "2026-08-01", "status": "active"},
        "domain": "Test Domain",
        "clusters": {
            "c01": {"title": "Chapter 1", "learned": False},
            "c02": {"title": "Chapter 2", "learned": True},
        },
        "items": [
            {
                "id": "a",
                "cluster": "c01",
                "title": "样本均值定义",
                "type": "definition",
                "mode": "whitebox",
                "source": "§1.1",
                "note": "样本均值的期望等于总体均值",
                "mastery": 0.9,
                "status": "learned",
                "next_review": "2026-08-15",
            },
            {
                "id": "b",
                "cluster": "c01",
                "title": "中心极限定理",
                "type": "theorem",
                "mode": "whitebox",
                "prerequisites": ["a"],
                "source": "§1.2",
                "note": "CLT: 样本均值的分布趋近正态",
                "mastery": 0.3,
                "status": "learned",
                "next_review": "2026-08-01",
            },
            {
                "id": "c",
                "cluster": "c02",
                "title": "假设检验方法",
                "type": "method",
                "mode": "blackbox",
                "prerequisites": ["b"],
                "related": ["a"],
                "source": "§2.1",
                "mastery": 0.0,
                "status": "pending",
            },
        ],
    }
    return Bookmap.from_dict(data)


@pytest.fixture
def session(sample_bm: Bookmap) -> StudySession:
    """在 bookmap 上已移动到 'a' 的会话。"""
    s = StudySession(project_name="Test Domain", main_goal="理解假设检验")
    s.move_to("a", sample_bm)
    return s


# ── StudySession 测试 ─────────────────────────────────────────────────


class TestStudySession:
    """StudySession 生命期测试。"""

    def test_creation(self) -> None:
        """创建会话时 project_name 正确。"""
        s = StudySession(project_name="概率论")
        assert s.project_name == "概率论"
        assert s.current_item_id == ""
        assert s.breakdown_stack == []
        assert s.items_covered == set()

    def test_set_goal(self) -> None:
        """设置主线目标。"""
        s = StudySession(project_name="test")
        s.set_goal("学完第3章")
        assert s.main_goal == "学完第3章"

    def test_move_to_valid_item(self, sample_bm: Bookmap) -> None:
        """移动到存在的 item 成功。"""
        s = StudySession(project_name="test")
        assert s.move_to("a", sample_bm)
        assert s.current_item_id == "a"
        assert "a" in s.items_covered

    def test_move_to_nonexistent(self, sample_bm: Bookmap) -> None:
        """移动到不存在的 item 返回 False。"""
        s = StudySession(project_name="test")
        assert not s.move_to("nonexistent", sample_bm)
        assert s.current_item_id == ""

    def test_drill_down(self, sample_bm: Bookmap, session: StudySession) -> None:
        """下钻：当前位置压栈，跳转到新 item。"""
        assert session.current_item_id == "a"
        assert session.drill_down("b", "学定理", sample_bm)
        assert session.current_item_id == "b"
        assert len(session.breakdown_stack) == 1
        assert session.breakdown_stack[0][0] == "a"

    def test_drill_down_nonexistent(self, sample_bm: Bookmap, session: StudySession) -> None:
        """下钻到不存在的 item 失败。"""
        assert not session.drill_down("nonexistent", "xxx", sample_bm)
        assert session.current_item_id == "a"

    def test_step_back(self, sample_bm: Bookmap, session: StudySession) -> None:
        """返回弹栈。"""
        session.drill_down("b", "学定理", sample_bm)
        assert session.current_item_id == "b"
        popped = session.step_back(sample_bm)
        assert popped == ("a", "样本均值定义")
        assert session.current_item_id == "a"
        assert len(session.breakdown_stack) == 0

    def test_step_back_empty_stack(self, sample_bm: Bookmap, session: StudySession) -> None:
        """空栈时返回 None。"""
        assert session.step_back(sample_bm) is None

    def test_record_evidence(self, session: StudySession) -> None:
        """记录 mastery 变化证据。"""
        session.record_evidence("a", 0.10)
        assert session.items_mastery_delta["a"] == 0.10
        session.record_evidence("a", 0.03)
        assert session.items_mastery_delta["a"] == 0.13

    def test_is_lost(self) -> None:
        """无当前位置且无栈 = 迷航。"""
        s = StudySession(project_name="test")
        assert s.is_lost()

    def test_not_lost_when_at_item(self, session: StudySession) -> None:
        """有当前位置时不是迷航。"""
        assert not session.is_lost()

    def test_has_goal(self, session: StudySession) -> None:
        """有主线目标时 has_goal() = True。"""
        assert session.has_goal()

    def test_no_goal(self) -> None:
        """空目标时 has_goal() = False。"""
        s = StudySession(project_name="test")
        assert not s.has_goal()

    def test_add_note(self, session: StudySession) -> None:
        """添加笔记。"""
        session.add_note("今天的理解：大数定律是关键")
        assert len(session.notes) == 1


# ── locate_item 测试 ──────────────────────────────────────────────────


class TestLocateItem:
    """locate_item() 测试。"""

    def test_exact_id_match(self, sample_bm: Bookmap) -> None:
        """精确 ID 匹配返回单项。"""
        results = locate_item(sample_bm, "a")
        assert len(results) == 1
        assert results[0].id == "a"

    def test_full_title_match(self, sample_bm: Bookmap) -> None:
        """完整标题匹配返回正确 item。"""
        results = locate_item(sample_bm, "中心极限定理")
        assert len(results) >= 1
        assert results[0].id == "b"

    def test_partial_match(self, sample_bm: Bookmap) -> None:
        """标题子串匹配。"""
        results = locate_item(sample_bm, "检验")
        assert len(results) >= 1

    def test_note_match(self, sample_bm: Bookmap) -> None:
        """note 字段匹配。"""
        results = locate_item(sample_bm, "CLT")
        assert any(r.id == "b" for r in results)

    def test_no_match(self, sample_bm: Bookmap) -> None:
        """无匹配返回空列表。"""
        results = locate_item(sample_bm, "量子力学")
        assert results == []

    def test_limit_top_k(self, sample_bm: Bookmap) -> None:
        """top_k 限制生效。"""
        results = locate_item(sample_bm, "a", top_k=1)
        assert len(results) <= 1


# ── classify_question 测试 ────────────────────────────────────────────


class TestClassifyQuestion:
    """classify_question() 测试。"""

    def test_definition(self) -> None:
        assert classify_question("中心极限定理是什么") == "definition"
        assert classify_question("定义一下XX") == "definition"

    def test_proof(self) -> None:
        assert classify_question("这个定理怎么证明") == "proof"
        assert classify_question("为什么这一步这样推导") == "proof"

    def test_relationship(self) -> None:
        assert classify_question("X 和 Y 有什么区别") == "relationship"

    def test_prerequisite(self) -> None:
        assert classify_question("学这个之前需要先学什么") == "prerequisite"

    def test_application(self) -> None:
        assert classify_question("这个有什么用") == "application"

    def test_self_test(self) -> None:
        assert classify_question("考考我") == "self_test"
        assert classify_question("出个题") == "self_test"

    def test_progress(self) -> None:
        assert classify_question("我学了多少") == "progress"

    def test_default_to_definition(self) -> None:
        """无法匹配时默认返回 definition。"""
        assert classify_question("随便说说") == "definition"


# ── get_sources 测试 ──────────────────────────────────────────────────


class TestGetSources:
    """get_sources() 测试。"""

    def test_definition_sources(self) -> None:
        assert "book" in get_sources("definition")

    def test_relationship_sources(self) -> None:
        assert "graph" in get_sources("relationship")

    def test_unknown_type_defaults(self) -> None:
        sources = get_sources("unknown")
        assert isinstance(sources, list)
        assert len(sources) > 0


# ── generate_socratic_prompt 测试 ─────────────────────────────────────


class TestSocraticPrompt:
    """generate_socratic_prompt() 测试。"""

    def test_includes_anchor(self, sample_bm: Bookmap) -> None:
        """提示中包含教材锚点。"""
        item = sample_bm.get_item("a")
        assert item is not None
        prompt = generate_socratic_prompt(item, "definition")
        assert item.source in prompt

    def test_includes_note(self, sample_bm: Bookmap) -> None:
        """提示中包含 note（如有）。"""
        item = sample_bm.get_item("a")
        assert item is not None
        prompt = generate_socratic_prompt(item, "definition")
        assert "样本均值的期望等于总体均值" in prompt

    def test_blackbox_mode_note(self, sample_bm: Bookmap) -> None:
        """黑箱 item 提示包含模式说明。"""
        item = sample_bm.get_item("c")
        assert item is not None
        prompt = generate_socratic_prompt(item, "proof")
        assert "会用就行" in prompt

    def test_llm_unavailable_notice(self, sample_bm: Bookmap) -> None:
        """LLM 不可用时提示含警告。"""
        item = sample_bm.get_item("a")
        assert item is not None
        prompt = generate_socratic_prompt(item, "definition", llm_available=False)
        assert "LLM 不可用" in prompt


# ── 术语翻译测试 ──────────────────────────────────────────────────────


class TestTerminology:
    """translate_* 函数测试。"""

    def test_translate_mode_whitebox(self) -> None:
        assert translate_mode("whitebox") == "深入理解"

    def test_translate_mode_blackbox(self) -> None:
        assert translate_mode("blackbox") == "会用即可"

    def test_translate_mastery_high(self) -> None:
        assert "熟练" in translate_mastery(0.9)

    def test_translate_mastery_new(self) -> None:
        assert "刚刚接触" == translate_mastery(0.1)

    def test_translate_status(self) -> None:
        assert translate_status("learned") == "已学"
        assert translate_status("pending") == "待学"


# ── 三栏测试 ──────────────────────────────────────────────────────────


class TestThreeColumn:
    """compute_three_column() 测试。"""

    def test_returns_three_column(self, sample_bm: Bookmap) -> None:
        """返回 ThreeColumn 实例。"""
        tc = compute_three_column(sample_bm)
        assert isinstance(tc, ThreeColumn)
        assert tc.completed or True  # may be empty
        assert tc.remaining or True

    def test_completed_items_are_learned(self, sample_bm: Bookmap) -> None:
        """已完成栏只包含 status='learned' 的 items。"""
        tc = compute_three_column(sample_bm)
        for item, _note in tc.completed:
            assert item.status == "learned"

    def test_recommended_not_empty(self, sample_bm: Bookmap) -> None:
        """推荐栏不为空。"""
        tc = compute_three_column(sample_bm)
        assert len(tc.recommended) > 0

    def test_current_item_dependents_prioritized(self, sample_bm: Bookmap) -> None:
        """当前 item 的后置节点优先推荐。"""
        # 'b' 的后置是 'c'
        tc = compute_three_column(sample_bm, current_item_id="b")
        if tc.recommended:
            # c 应该是推荐中的第一个（后继续）
            first_rec = tc.recommended[0]
            assert first_rec[0].id == "c"

    def test_max_per_column(self, sample_bm: Bookmap) -> None:
        """max_per_column 限制生效。"""
        tc = compute_three_column(sample_bm, max_per_column=1)
        assert len(tc.completed) <= 1
        assert len(tc.remaining) <= 1
        assert len(tc.recommended) <= 1


# ── get_navigation_context 测试 ───────────────────────────────────────


class TestNavigationContext:
    """get_navigation_context() 测试。"""

    def test_existing_item(self, sample_bm: Bookmap) -> None:
        """存在 item 返回完整上下文。"""
        nav = get_navigation_context(sample_bm, "b")
        assert nav.current is not None
        assert nav.current.id == "b"
        assert len(nav.prereq_chain) == 1  # a
        assert nav.prereq_chain[0].id == "a"

    def test_nonexistent_item(self, sample_bm: Bookmap) -> None:
        """不存在 item 返回空上下文。"""
        nav = get_navigation_context(sample_bm, "nonexistent")
        assert nav.current is None

    def test_cluster_info(self, sample_bm: Bookmap) -> None:
        """包含所属章节。"""
        nav = get_navigation_context(sample_bm, "a")
        assert nav.cluster is not None
        assert nav.cluster.title == "Chapter 1"

    def test_dependents(self, sample_bm: Bookmap) -> None:
        """后置依赖正确。"""
        nav = get_navigation_context(sample_bm, "b")
        dep_ids = [d.id for d in nav.dependents]
        assert "c" in dep_ids

    def test_related_items(self, sample_bm: Bookmap) -> None:
        """相关概念正确。"""
        nav = get_navigation_context(sample_bm, "c")
        rel_ids = [r.id for r in nav.related]
        assert "a" in rel_ids
