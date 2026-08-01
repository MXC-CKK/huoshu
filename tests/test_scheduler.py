"""Tests for learning_agent.core.scheduler — spaced repetition scheduling."""

# ruff: noqa: DTZ011

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from learning_agent.core.scheduler import (
    CORRECT_BOOST,
    INCORRECT_PENALTY,
    MAX_REVIEW_ITEMS_PER_SESSION,
    PARTIAL_BOOST,
    PENALTY_INTERVAL,
    ReviewSession,
    ScheduleResult,
    compute_mastery_delta,
    compute_next_review,
    get_due_items,
    initial_schedule,
)


class TestComputeNextReview:
    """compute_next_review() 测试。"""

    def test_correct_moves_forward(self) -> None:
        """答对 → 间隔增长，日期在未来。"""
        result = compute_next_review(0.5, 1.0, review_count=0)
        assert result.next_review > date.today()
        assert result.interval_days >= 1
        assert result.mastery_delta == CORRECT_BOOST
        assert result.new_mastery > 0.5

    def test_incorrect_resets_interval(self) -> None:
        """答错 → 间隔重置为 1 天。"""
        result = compute_next_review(0.5, 0.0, review_count=5)
        assert result.interval_days == PENALTY_INTERVAL
        assert result.mastery_delta == -INCORRECT_PENALTY
        assert result.new_mastery < 0.5

    def test_partial_moderate_change(self) -> None:
        """部分对 → 间隔适度调整。"""
        result = compute_next_review(0.5, 0.5, review_count=2)
        assert result.mastery_delta == PARTIAL_BOOST
        # 间隔可能保持或减半

    def test_review_count_increases(self) -> None:
        """复习次数正确递增。"""
        result = compute_next_review(0.5, 1.0, review_count=3)
        assert result.review_count == 4

    def test_reset_still_counts(self) -> None:
        """答错后 review_count 仍递增（但间隔重置）。"""
        result = compute_next_review(0.5, 0.0, review_count=3)
        assert result.review_count == 4
        assert result.interval_days == 1  # 间隔重置

    def test_high_mastery_longer_interval(self) -> None:
        """高掌握度 → 间隔更长（mastery bonus）。"""
        low = compute_next_review(0.4, 1.0, review_count=3)
        high = compute_next_review(0.9, 1.0, review_count=3)
        assert high.interval_days >= low.interval_days

    def test_mastery_bounded(self) -> None:
        """掌握度始终在合理范围内。"""
        # 极端低 → 不会低于下限
        result = compute_next_review(0.05, 0.0)
        assert result.new_mastery >= 0.05

        # 极端高 → 不会超过上限
        result = compute_next_review(0.95, 1.0)
        assert result.new_mastery <= 0.95

    def test_interval_progression(self) -> None:
        """连续答对 → 间隔指数增长。"""
        intervals: list[int] = []
        mastery = 0.4
        for i in range(5):
            result = compute_next_review(mastery, 1.0, review_count=i)
            intervals.append(result.interval_days)
            mastery = result.new_mastery

        # 间隔应递增（或至少不递减）
        for i in range(len(intervals) - 1):
            assert intervals[i] <= intervals[i + 1], (
                f"间隔从 {intervals[i]} 降到 {intervals[i+1]}"
            )

    def test_invalid_outcome_raises(self) -> None:
        """无效 outcome 抛出 ValueError。"""
        with pytest.raises(ValueError):
            compute_next_review(0.5, 1.5)
        with pytest.raises(ValueError):
            compute_next_review(0.5, -0.5)


class TestGetDueItems:
    """get_due_items() 测试。"""

    @pytest.fixture
    def sample_items(self) -> list[dict]:
        """示例 items 列表。"""
        today = date.today()
        yesterday = (today - timedelta(days=1)).isoformat()
        tomorrow = (today + timedelta(days=1)).isoformat()
        last_week = (today - timedelta(days=7)).isoformat()

        return [
            {
                "id": "a", "status": "learned", "mastery": 0.3,
                "next_review": yesterday,  # 到期
            },
            {
                "id": "b", "status": "learned", "mastery": 0.7,
                "next_review": last_week,  # 过期很久
            },
            {
                "id": "c", "status": "learned", "mastery": 0.5,
                "next_review": tomorrow,  # 未到期
            },
            {
                "id": "d", "status": "pending", "mastery": 0.1,
                "next_review": yesterday,  # 到期但未学完
            },
            {
                "id": "e", "status": "learned", "mastery": 0.9,
                "next_review": None,  # 未安排
            },
        ]

    def test_filters_due_only(self, sample_items: list[dict]) -> None:
        """只筛选到期且已学的项。"""
        due = get_due_items(sample_items)
        ids = {item["id"] for item in due}
        assert ids == {"b", "a"}  # 按 mastery 升序：a(0.3) 在 b(0.7) 之前

    def test_sorted_by_mastery_asc(self, sample_items: list[dict]) -> None:
        """按 mastery 升序排列（最弱优先）。"""
        due = get_due_items(sample_items)
        masteries = [item["mastery"] for item in due]
        assert masteries == sorted(masteries)

    def test_max_items_limit(self, sample_items: list[dict]) -> None:
        """max_items 限制生效。"""
        due = get_due_items(sample_items, max_items=1)
        assert len(due) == 1
        assert due[0]["id"] == "a"  # 最弱的

    def test_reference_date(self, sample_items: list[dict]) -> None:
        """自定义参考日期。"""
        # 用昨天作为参考 → c 未到期
        yesterday = date.today() - timedelta(days=1)
        due = get_due_items(sample_items, reference_date=yesterday)
        ids = {item["id"] for item in due}
        assert "c" not in ids  # c is on tomorrow

    def test_empty_list(self) -> None:
        """空列表返回空。"""
        assert get_due_items([]) == []

    def test_default_max_items(self, sample_items: list[dict]) -> None:
        """默认 max_items = 8。"""
        # 添加很多到期项
        many = []
        for i in range(20):
            many.append({
                "id": f"x{i}", "status": "learned", "mastery": 0.5,
                "next_review": (date.today() - timedelta(days=i)).isoformat(),
            })
        due = get_due_items(many)
        assert len(due) <= MAX_REVIEW_ITEMS_PER_SESSION


class TestInitialSchedule:
    """initial_schedule() 测试。"""

    def test_first_review_tomorrow(self) -> None:
        """新学知识点明天复习。"""
        result = initial_schedule()
        today = datetime.now(tz=UTC).date()
        assert result.next_review == today + timedelta(days=1)
        assert result.interval_days == 1
        assert result.review_count == 0
        assert result.mastery_delta == 0.0

    def test_custom_mastery(self) -> None:
        """自定义初始掌握度。"""
        result = initial_schedule(mastery=0.3)
        assert result.new_mastery == 0.3


class TestComputeMasteryDelta:
    """compute_mastery_delta() 测试。"""

    def test_correct(self) -> None:
        assert compute_mastery_delta(1.0) == CORRECT_BOOST
        assert compute_mastery_delta(0.8) == CORRECT_BOOST

    def test_partial(self) -> None:
        assert compute_mastery_delta(0.5) == PARTIAL_BOOST
        assert compute_mastery_delta(0.4) == PARTIAL_BOOST
        assert compute_mastery_delta(0.7) == PARTIAL_BOOST

    def test_incorrect(self) -> None:
        assert compute_mastery_delta(0.0) == -INCORRECT_PENALTY
        assert compute_mastery_delta(0.3) == -INCORRECT_PENALTY


class TestReviewSession:
    """ReviewSession 数据类测试。"""

    def test_empty_session(self) -> None:
        """空会话。"""
        session = ReviewSession(date=date.today())
        assert session.correct_count == 0
        assert session.incorrect_count == 0
        assert session.partial_count == 0

    def test_counts(self) -> None:
        """正确/部分对/错误计数。"""
        session = ReviewSession(
            date=date.today(),
            items_reviewed=[
                ScheduleResult(
                    item_id="a", next_review=date.today(),
                    interval_days=7, mastery_delta=0.10,
                    new_mastery=0.6, review_count=1,
                ),
                ScheduleResult(
                    item_id="b", next_review=date.today(),
                    interval_days=1, mastery_delta=-0.15,
                    new_mastery=0.3, review_count=1,
                ),
                ScheduleResult(
                    item_id="c", next_review=date.today(),
                    interval_days=3, mastery_delta=0.03,
                    new_mastery=0.5, review_count=1,
                ),
            ],
        )
        assert session.correct_count == 1   # a
        assert session.incorrect_count == 1  # b
        assert session.partial_count == 1    # c


class TestScheduleResult:
    """ScheduleResult 数据类测试。"""

    def test_creation(self) -> None:
        """正常创建。"""
        r = ScheduleResult(
            item_id="x-1",
            next_review=date.today(),
            interval_days=7,
            mastery_delta=0.1,
            new_mastery=0.6,
            review_count=2,
        )
        assert r.item_id == "x-1"
        assert r.interval_days == 7
