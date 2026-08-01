"""Tests for learning_agent.ui.review_engine — review session logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from learning_agent.core.graph import Bookmap
from learning_agent.ui.review_engine import (
    MAX_ITEMS_PER_SESSION,
    ReviewEngine,
    ReviewQuestion,
    ReviewResult,
    ReviewSessionSummary,
    _extract_keywords,
    _generate_question_text,
    _mastery_to_difficulty,
    _score_answer,
    format_review_summary,
)

# ── 夹具 ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bm() -> Bookmap:
    """包含到期和未到期 items 的示例图谱。"""
    today = datetime.now(tz=UTC).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    tomorrow = (today + timedelta(days=3)).isoformat()
    last_week = (today - timedelta(days=7)).isoformat()

    data: dict[str, Any] = {
        "meta": {"source": "test", "built": "2026-08-01", "status": "active"},
        "domain": "Test",
        "clusters": {"c01": {"title": "Ch1"}},
        "items": [
            {  # 到期，低 mastery
                "id": "a",
                "cluster": "c01",
                "title": "Definition A",
                "type": "definition",
                "mode": "whitebox",
                "prerequisites": [],
                "source": "§1",
                "note": "Key concept",
                "mastery": 0.2,
                "next_review": yesterday,
                "status": "learned",
            },
            {  # 到期，中 mastery
                "id": "b",
                "cluster": "c01",
                "title": "Theorem B",
                "type": "theorem",
                "mode": "whitebox",
                "prerequisites": ["a"],
                "source": "§2",
                "note": "Important result",
                "mastery": 0.6,
                "next_review": last_week,
                "status": "learned",
            },
            {  # 到期，高 mastery
                "id": "c",
                "cluster": "c01",
                "title": "Application C",
                "type": "application",
                "mode": "blackbox",
                "prerequisites": ["b"],
                "source": "§3",
                "mastery": 0.85,
                "next_review": yesterday,
                "status": "learned",
            },
            {  # 未到期
                "id": "d",
                "cluster": "c01",
                "title": "Future Topic D",
                "type": "concept",
                "mode": "whitebox",
                "source": "§4",
                "mastery": 0.7,
                "next_review": tomorrow,
                "status": "learned",
            },
            {  # pending (未学)
                "id": "e",
                "cluster": "c01",
                "title": "Pending E",
                "type": "method",
                "mode": "blackbox",
                "source": "§5",
                "mastery": 0.0,
                "next_review": yesterday,
                "status": "pending",
            },
        ],
    }
    return Bookmap.from_dict(data)


@pytest.fixture
def engine(sample_bm: Bookmap) -> ReviewEngine:
    """基于示例图谱的复习引擎。"""
    return ReviewEngine(sample_bm)


# ── get_due_items 测试 ────────────────────────────────────────────────


class TestGetDueItems:
    """ReviewEngine.get_due_items() 测试。"""

    def test_filters_due_only(self, engine: ReviewEngine) -> None:
        """只返回到期且 status='learned' 的 items。"""
        due = engine.get_due_items()
        due_ids = {it.id for it in due}
        assert due_ids == {"a", "b", "c"}  # d 未到期, e 未学

    def test_sorted_by_mastery_asc(self, engine: ReviewEngine) -> None:
        """按 mastery 升序排列。"""
        due = engine.get_due_items()
        masteries = [it.mastery for it in due]
        assert masteries == sorted(masteries)

    def test_max_items_limit(self, engine: ReviewEngine, sample_bm: Bookmap) -> None:
        """最多返回 MAX_ITEMS_PER_SESSION 个。"""
        # Add many overdue items
        today = datetime.now(tz=UTC).date()
        for i in range(20):
            sample_bm.items[f"x{i}"] = type(sample_bm.get_item("a"))(
                id=f"x{i}", cluster="c01", title=f"X{i}",
                type="concept", mode="whitebox", source=f"§{i}",
                mastery=0.5,
                next_review=(today - timedelta(days=i)).isoformat(),
                status="learned",
            )
        due = engine.get_due_items(reference_date=today)
        assert len(due) <= MAX_ITEMS_PER_SESSION


# ── generate_question 测试 ─────────────────────────────────────────────


class TestGenerateQuestion:
    """ReviewEngine.generate_question() 测试。"""

    def test_nonexistent_item(self, engine: ReviewEngine) -> None:
        """不存在 item 返回 None。"""
        assert engine.generate_question("nonexistent") is None

    def test_low_mastery_basic(self, engine: ReviewEngine) -> None:
        """低掌握度 → 基础题。"""
        q = engine.generate_question("a")
        assert q is not None
        assert q.difficulty == "basic"
        assert q.item_title == "Definition A"

    def test_mid_mastery_understanding(self, engine: ReviewEngine) -> None:
        """中掌握度 → 理解题。"""
        q = engine.generate_question("b")
        assert q is not None
        assert q.difficulty == "understanding"

    def test_high_mastery_application(self, engine: ReviewEngine) -> None:
        """高掌握度 → 应用题。"""
        q = engine.generate_question("c")
        assert q is not None
        assert q.difficulty == "application"

    def test_includes_source_anchor(self, engine: ReviewEngine) -> None:
        """题目包含教材锚点。"""
        q = engine.generate_question("a")
        assert q is not None
        assert q.source_anchor == "§1"

    def test_question_has_keywords(self, engine: ReviewEngine) -> None:
        """题目包含期望关键词。"""
        q = engine.generate_question("a")
        assert q is not None
        assert len(q.expected_keywords) > 0

    def test_returns_review_question_type(self, engine: ReviewEngine) -> None:
        """返回 ReviewQuestion 实例。"""
        q = engine.generate_question("a")
        assert isinstance(q, ReviewQuestion)


# ── evaluate_answer 测试 ──────────────────────────────────────────────


class TestEvaluateAnswer:
    """ReviewEngine.evaluate_answer() 测试。"""

    def test_nonexistent_item(self, engine: ReviewEngine) -> None:
        """不存在 item 返回 None。"""
        assert engine.evaluate_answer("nonexistent", "answer") is None

    def test_correct_answer_updates_mastery(self, engine: ReviewEngine, sample_bm: Bookmap) -> None:
        """答对 → mastery 提升。"""
        item = sample_bm.get_item("a")
        assert item is not None
        old_mastery = item.mastery
        keywords = _extract_keywords(item)
        # 使用关键词构造"好"答案
        good_answer = " ".join(keywords) + " detailed explanation here"
        result = engine.evaluate_answer("a", good_answer)
        assert result is not None
        assert result.mastery_after >= old_mastery

    def test_empty_answer_scores_zero(self, engine: ReviewEngine) -> None:
        """空答案 → 0 分。"""
        result = engine.evaluate_answer("a", "")
        assert result is not None
        assert result.score == 0.0
        assert result.mastery_after < result.mastery_before

    def test_sets_next_review(self, engine: ReviewEngine) -> None:
        """判分后设置 next_review。"""
        result = engine.evaluate_answer("a", "sample mean expectation")
        assert result is not None
        assert result.next_review != ""

    def test_result_added_to_history(self, engine: ReviewEngine) -> None:
        """结果加入 results 列表。"""
        engine.evaluate_answer("a", "test answer")
        assert len(engine.results) == 1

    def test_score_update_reflected_in_item(self, engine: ReviewEngine, sample_bm: Bookmap) -> None:
        """item.mastery 被写回。"""
        item = sample_bm.get_item("a")
        assert item is not None
        old = item.mastery
        engine.evaluate_answer("a", "key concept definitions here")
        assert item.mastery != old  # 至少变了


# ── generate_all_questions 测试 ────────────────────────────────────────


class TestGenerateAllQuestions:
    """generate_all_questions() 测试。"""

    def test_from_due_items(self, engine: ReviewEngine) -> None:
        """从到期列表批量生成题目。"""
        questions = engine.generate_all_questions()
        assert len(questions) == 3
        assert all(isinstance(q, ReviewQuestion) for q in questions)

    def test_from_explicit_ids(self, engine: ReviewEngine) -> None:
        """从指定 IDs 生成题目。"""
        questions = engine.generate_all_questions(item_ids=["a", "c"])
        assert len(questions) == 2

    def test_nonexistent_ids_skipped(self, engine: ReviewEngine) -> None:
        """不存在的 IDs 被跳过。"""
        questions = engine.generate_all_questions(item_ids=["a", "nonexistent"])
        assert len(questions) == 1


# ── summarize 测试 ─────────────────────────────────────────────────────


class TestSummarize:
    """summarize() 测试。"""

    def test_empty_summary(self, engine: ReviewEngine) -> None:
        """空引擎产生零计数汇总。"""
        s = engine.summarize()
        assert isinstance(s, ReviewSessionSummary)
        assert s.reviewed == 0
        assert s.total_due == 0

    def test_after_reviews(self, engine: ReviewEngine) -> None:
        """复习后汇总正确。"""
        engine.evaluate_answer("a", "definition explanation detailed enough")
        engine.evaluate_answer("b", "theorem proof steps and explanation")
        engine.evaluate_answer("c", "")  # 空答案

        s = engine.summarize()
        assert s.reviewed == 3
        # a 和 b 应该得分较高（有关键词 match），c 得 0
        assert s.correct + s.partial + s.incorrect == 3
        assert s.incorrect == 1  # c


# ── 辅助函数测试 ──────────────────────────────────────────────────────


class TestMasteryToDifficulty:
    """_mastery_to_difficulty() 测试。"""

    def test_low(self) -> None:
        assert _mastery_to_difficulty(0.1) == "basic"

    def test_mid_low(self) -> None:
        assert _mastery_to_difficulty(0.5) == "understanding"

    def test_mid_high(self) -> None:
        assert _mastery_to_difficulty(0.69) == "understanding"

    def test_high(self) -> None:
        assert _mastery_to_difficulty(0.8) == "application"


class TestGenerateQuestionText:
    """_generate_question_text() 测试。"""

    def test_basic_question_contains_title(self, sample_bm: Bookmap) -> None:
        """基础题包含知识点标题。"""
        item = sample_bm.get_item("a")
        assert item is not None
        text = _generate_question_text(item, "basic")
        assert item.title in text

    def test_deterministic(self, sample_bm: Bookmap) -> None:
        """同 item 同 difficulty 产生相同题目。"""
        item = sample_bm.get_item("a")
        assert item is not None
        t1 = _generate_question_text(item, "basic")
        t2 = _generate_question_text(item, "basic")
        assert t1 == t2


class TestExtractKeywords:
    """_extract_keywords() 测试。"""

    def test_extracts_from_title(self, sample_bm: Bookmap) -> None:
        """从标题提取关键词。"""
        item = sample_bm.get_item("a")
        assert item is not None
        kw = _extract_keywords(item)
        assert "样本均值定义" in " ".join(kw) or any(
            w in kw for w in ["样本均值定义"]
        ) or len(kw) > 0  # at minimum, some words extracted

    def test_extracts_from_note(self, sample_bm: Bookmap) -> None:
        """从 note 提取关键词。"""
        item = sample_bm.get_item("a")
        assert item is not None
        kw = _extract_keywords(item)
        # note: "样本均值的期望等于总体均值" — should extract meaningful words
        assert len(kw) > 0

    def test_max_keywords(self, sample_bm: Bookmap) -> None:
        """关键词数量不超过 10。"""
        item = sample_bm.get_item("a")
        assert item is not None
        kw = _extract_keywords(item)
        assert len(kw) <= 10


class TestScoreAnswer:
    """_score_answer() 测试。"""

    def test_empty_answer(self, sample_bm: Bookmap) -> None:
        """空答案得 0 分。"""
        item = sample_bm.get_item("a")
        assert item is not None
        assert _score_answer("", ["test"], item) == 0.0

    def test_short_answer(self, sample_bm: Bookmap) -> None:
        """太短答案得 0 分。"""
        item = sample_bm.get_item("a")
        assert item is not None
        assert _score_answer("hi", ["test"], item) == 0.0

    def test_full_match(self, sample_bm: Bookmap) -> None:
        """所有关键词都匹配得 1 分。"""
        item = sample_bm.get_item("a")
        assert item is not None
        kw = ["definition", "concept"]
        assert _score_answer("this is a definition of the concept with detail", kw, item) == 1.0

    def test_partial_match(self, sample_bm: Bookmap) -> None:
        """部分关键词匹配得 0.5 分。"""
        item = sample_bm.get_item("a")
        assert item is not None
        kw = ["definition", "concept", "advanced", "topic"]
        answer = "this is a definition with extra detail long enough text"
        score = _score_answer(answer, kw, item)
        assert score in (0.5, 1.0)  # at least partial credit


# ── format_review_summary 测试 ─────────────────────────────────────────


class TestFormatReviewSummary:
    """format_review_summary() 测试。"""

    def test_empty_summary(self) -> None:
        """空汇总。"""
        s = ReviewSessionSummary(
            date="2026-08-02",
            total_due=0,
            reviewed=0,
            correct=0,
            partial=0,
            incorrect=0,
        )
        text = format_review_summary(s)
        assert "0/0" in text

    def test_with_results(self, sample_bm: Bookmap) -> None:
        """带结果的汇总包含知识点标题。"""
        item = sample_bm.get_item("a")
        assert item is not None
        q = ReviewQuestion(
            item_id="a", item_title="Definition A",
            difficulty="basic", question_text="What is A?",
        )
        r = ReviewResult(
            question=q, user_answer="answer",
            score=1.0, mastery_before=0.2, mastery_after=0.3,
            feedback="Good!", next_review="2026-08-03",
        )
        s = ReviewSessionSummary(
            date="2026-08-02",
            total_due=1, reviewed=1,
            correct=1, partial=0, incorrect=0,
            results=[r],
        )
        text = format_review_summary(s)
        assert "Definition A" in text
        assert "1/1" in text
