"""Tests for learning_agent.core.mastery — IRT mastery model."""

from __future__ import annotations

import pytest

from learning_agent.core.mastery import (
    MASTERY_MAX,
    MASTERY_MIN,
    MasteryUpdate,
    compute_difficulty,
    estimate_initial_mastery,
    probability_correct,
    update_mastery,
)


class TestProbabilityCorrect:
    """probability_correct() 测试。"""

    def test_low_mastery_low_probability(self) -> None:
        """低掌握度 → 低答对概率。"""
        p = probability_correct(0.1)
        assert p < 0.5

    def test_high_mastery_high_probability(self) -> None:
        """高掌握度 → 高答对概率。"""
        p = probability_correct(0.9)
        assert p > 0.7

    def test_mid_mastery_around_difficulty(self) -> None:
        """掌握度等于难度 → 概率约 0.5 + guessing 调整。"""
        p = probability_correct(0.6, difficulty=0.6)
        assert 0.5 < p < 0.7

    def test_blackbox_easier_than_whitebox(self) -> None:
        """同掌握度下 blackbox 答对概率更高。"""
        mastery = 0.5
        p_bb = probability_correct(mastery, mode="blackbox")
        p_wb = probability_correct(mastery, mode="whitebox")
        assert p_bb > p_wb  # blackbox 难度更低

    def test_guessing_floor(self) -> None:
        """猜测概率是最低下限。"""
        p = probability_correct(0.0, guessing=0.25)
        assert p >= 0.25

    def test_monotonic(self) -> None:
        """掌握度越高，答对概率越大。"""
        probs = [probability_correct(m) for m in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]

    def test_explicit_difficulty_overrides_mode(self) -> None:
        """明确指定 difficulty 覆盖 mode 默认值。"""
        p = probability_correct(0.5, difficulty=0.3, mode="whitebox")
        expected = probability_correct(0.5, difficulty=0.3)
        assert p == expected


class TestUpdateMastery:
    """update_mastery() 测试。"""

    def test_correct_increases_mastery(self) -> None:
        """答对 → 掌握度提升。"""
        result = update_mastery(0.5, 1.0)
        assert result.new_mastery > result.old_mastery
        assert result.delta > 0

    def test_incorrect_decreases_mastery(self) -> None:
        """答错 → 掌握度降低。"""
        result = update_mastery(0.5, 0.0)
        assert result.new_mastery < result.old_mastery
        assert result.delta < 0

    def test_partial_small_change(self) -> None:
        """部分对 → 微小变化。"""
        result = update_mastery(0.5, 0.5)
        # 变化应该很小
        assert abs(result.delta) < 0.1

    def test_high_mastery_wrong_big_penalty(self) -> None:
        """高掌握度答错 → 惩罚更大（hypercorrection）。"""
        low_result = update_mastery(0.3, 0.0)
        high_result = update_mastery(0.8, 0.0)
        # 高掌握度答错的绝对下降应该 ≥ 低掌握度答错
        assert abs(high_result.delta) >= abs(low_result.delta) * 0.8

    def test_bounded_output(self) -> None:
        """掌握度始终在 [0.05, 0.95] 范围内。"""
        # 极端低 → 不会低于下限
        result = update_mastery(0.05, 0.0)
        assert result.new_mastery >= MASTERY_MIN

        # 极端高 → 不会高于上限
        result = update_mastery(0.95, 1.0)
        assert result.new_mastery <= MASTERY_MAX

    def test_blackbox_smaller_fluctuation(self) -> None:
        """Blackbox 模式波动小于 whitebox。"""
        bb_result = update_mastery(0.5, 1.0, mode="blackbox")
        wb_result = update_mastery(0.5, 1.0, mode="whitebox")
        # blackbox 提升较小
        assert bb_result.delta < wb_result.delta

    def test_invalid_outcome_raises(self) -> None:
        """无效 outcome 抛出 ValueError。"""
        with pytest.raises(ValueError):
            update_mastery(0.5, 1.5)
        with pytest.raises(ValueError):
            update_mastery(0.5, -0.1)

    def test_invalid_mastery_raises(self) -> None:
        """无效 mastery 抛出 ValueError。"""
        with pytest.raises(ValueError):
            update_mastery(1.5, 0.0)
        with pytest.raises(ValueError):
            update_mastery(-0.5, 0.0)

    def test_return_type(self) -> None:
        """返回正确的 MasteryUpdate 实例。"""
        result = update_mastery(0.5, 1.0)
        assert isinstance(result, MasteryUpdate)
        assert result.outcome == 1.0
        assert result.old_mastery == 0.5
        assert 0 <= result.p_correct <= 1


class TestComputeDifficulty:
    """compute_difficulty() 测试。"""

    def test_theorem_harder_than_definition(self) -> None:
        """定理比定义更难。"""
        d_def = compute_difficulty("definition", "whitebox")
        d_thm = compute_difficulty("theorem", "whitebox")
        assert d_thm > d_def

    def test_whitebox_harder_than_blackbox(self) -> None:
        """同类型下 whitebox 比 blackbox 难。"""
        d_bb = compute_difficulty("concept", "blackbox")
        d_wb = compute_difficulty("concept", "whitebox")
        assert d_wb > d_bb

    def test_output_bounded(self) -> None:
        """难度始终在 [0.2, 0.8] 范围内。"""
        for t in ["definition", "concept", "theorem", "method", "example", "application", "exercise"]:
            for m in ["blackbox", "whitebox"]:
                d = compute_difficulty(t, m)
                assert 0.2 <= d <= 0.8, f"type={t}, mode={m}, difficulty={d}"

    def test_unknown_type_defaults(self) -> None:
        """未知类型使用默认值。"""
        d = compute_difficulty("unknown_type", "whitebox")
        assert 0.2 <= d <= 0.8


class TestEstimateInitialMastery:
    """estimate_initial_mastery() 测试。"""

    def test_no_prerequisites(self) -> None:
        """无前置知识 → 默认初始值。"""
        m = estimate_initial_mastery([])
        assert m == 0.15

    def test_with_prerequisites(self) -> None:
        """有前置知识 → 基于平均折扣估计。"""
        m = estimate_initial_mastery([0.8, 0.9])
        assert 0.15 <= m <= 0.5
        # avg=0.85, discount=0.4 → 0.34
        assert m > 0.3

    def test_blackbox_higher_initial(self) -> None:
        """Blackbox 模式初始估计更高。"""
        prereqs = [0.7, 0.8]
        m_bb = estimate_initial_mastery(prereqs, mode="blackbox")
        m_wb = estimate_initial_mastery(prereqs, mode="whitebox")
        assert m_bb >= m_wb

    def test_capped_at_0_5(self) -> None:
        """初始估计不超过 0.5。"""
        m = estimate_initial_mastery([1.0, 1.0, 1.0])
        assert m <= 0.5
