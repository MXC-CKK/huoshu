"""简化 IRT 掌握度模型。

使用 2PL 逻辑模型（简化版）建模知识点掌握度。掌握度 θ ∈ [0, 1]，
根据答题正误更新，区分 blackbox/whitebox 难度差异。

核心公式:
    P(correct | θ, b) = c + (1 - c) / (1 + exp(-a * (θ - b)))
    θ_new = θ_old + α * (outcome - P(correct)) * adjustment

其中:
    - θ: 掌握度
    - b: 难度参数（blackbox 0.4 / whitebox 0.6）
    - a: 区分度（默认 2.0）
    - c: 猜测概率（默认 0.2）
    - α: 学习率（默认 0.10）
    - adjustment: calibration adjustment factor

参考:
    - learn-session-wrap SKILL.md 的 mastery 更新规则
    - LearnLab 的简化 IRT 实践
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── 默认参数 ─────────────────────────────────────────────────────────

DEFAULT_DISCRIMINATION = 2.0   # a: 区分度
DEFAULT_GUESSING = 0.2         # c: 猜测概率（选择题蒙对）
DEFAULT_LEARNING_RATE = 0.10   # α: 基础学习率
DEFAULT_SLIP = 0.05            # s: 粗心犯错概率（上限）
MASTERY_MIN = 0.05             # 掌握度下限
MASTERY_MAX = 0.95             # 掌握度上限（避免绝对确定）
BLACKBOX_DIFFICULTY = 0.40     # 黑箱默认难度
WHITEBOX_DIFFICULTY = 0.60     # 白箱默认难度


# ── 结果类型 ─────────────────────────────────────────────────────────


@dataclass
class MasteryUpdate:
    """掌握度更新结果。

    Attributes:
        old_mastery: 更新前掌握度。
        new_mastery: 更新后掌握度。
        delta: 变化量（正=提升，负=下降）。
        p_correct: 答题前估计的正确概率。
        outcome: 实际结果（1.0=全对, 0.5=部分对, 0.0=全错）。
    """

    old_mastery: float
    new_mastery: float
    delta: float
    p_correct: float
    outcome: float


# ── 核心函数 ─────────────────────────────────────────────────────────


def probability_correct(
    mastery: float,
    difficulty: float | None = None,
    mode: str = "whitebox",
    *,
    discrimination: float = DEFAULT_DISCRIMINATION,
    guessing: float = DEFAULT_GUESSING,
) -> float:
    """计算给定掌握度下答对的概率。

    使用 3PL IRT 模型的简化形式:
        P(correct) = c + (1 - c) / (1 + exp(-a * (θ - b)))

    Args:
        mastery: 当前掌握度 θ ∈ [0, 1]。
        difficulty: 难度参数 b。None 时按 mode 自动选择默认值。
        mode: 学习模式（'blackbox' 或 'whitebox'），仅在 difficulty 为 None 时使用。
        discrimination: 区分度参数 a（越大越陡峭）。
        guessing: 猜测概率 c（纯随机猜对的概率）。

    Returns:
        答对概率 P ∈ [c, 1 - s]（含猜测下限，排除粗心上限）。
    """
    import math

    if difficulty is None:
        difficulty = BLACKBOX_DIFFICULTY if mode == "blackbox" else WHITEBOX_DIFFICULTY

    # 标准 3PL
    exponent = -discrimination * (mastery - difficulty)
    # 防止 exp 溢出
    if exponent > 50:
        exponent = 50.0
    elif exponent < -50:
        exponent = -50.0

    p = guessing + (1.0 - guessing) / (1.0 + math.exp(exponent))

    # 夹紧到合理区间
    return max(guessing, min(p, 1.0 - DEFAULT_SLIP))


def update_mastery(
    mastery: float,
    outcome: float,
    mode: str = "whitebox",
    *,
    difficulty: float | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    discrimination: float = DEFAULT_DISCRIMINATION,
    guessing: float = DEFAULT_GUESSING,
) -> MasteryUpdate:
    """根据答题结果更新掌握度。

    使用基于 IRT 的更新规则:
        θ_new = θ_old + α * (outcome - P(correct)) * adjustment

    其中 adjustment 考虑:
        - 答错高置信 → 惩罚更大（hypercorrection 效应的反面）
        - 答对低置信 → 奖励更谨慎
        - blackbox 模式波动更小（对"会用就行"要求低）

    Args:
        mastery: 当前掌握度 θ_old ∈ [0, 1]。
        outcome: 实际结果。1.0=全对，0.5=部分对，0.0=全错。
        mode: 学习模式，影响难度参数和调整因子。
        difficulty: 手动指定难度参数（覆盖 mode 默认值）。
        learning_rate: 基础学习率 α。
        discrimination: 区分度 a。
        guessing: 猜测概率 c。

    Returns:
        MasteryUpdate 包含旧值、新值、变化量、预测概率和结果。

    Raises:
        ValueError: outcome 不在 [0, 1] 范围内。
    """
    if not 0.0 <= outcome <= 1.0:
        raise ValueError(f"outcome 必须在 [0, 1] 范围内，实际值: {outcome}")
    if not 0.0 <= mastery <= 1.0:
        raise ValueError(f"mastery 必须在 [0, 1] 范围内，实际值: {mastery}")

    if difficulty is None:
        difficulty = BLACKBOX_DIFFICULTY if mode == "blackbox" else WHITEBOX_DIFFICULTY

    p_correct = probability_correct(
        mastery=mastery,
        difficulty=difficulty,
        discrimination=discrimination,
        guessing=guessing,
    )

    # 预测误差
    error = outcome - p_correct

    # 调整因子
    adjustment = _compute_adjustment(mastery=mastery, outcome=outcome, mode=mode)

    # 更新
    delta = learning_rate * error * adjustment
    new_mastery = mastery + delta

    # 夹紧
    new_mastery = max(MASTERY_MIN, min(MASTERY_MAX, new_mastery))

    logger.debug(
        "Mastery update: %.3f → %.3f (Δ=%+.3f, P(correct)=%.3f, outcome=%.1f, mode=%s)",
        mastery, new_mastery, new_mastery - mastery, p_correct, outcome, mode,
    )

    return MasteryUpdate(
        old_mastery=mastery,
        new_mastery=new_mastery,
        delta=new_mastery - mastery,
        p_correct=p_correct,
        outcome=outcome,
    )


def _compute_adjustment(mastery: float, outcome: float, mode: str) -> float:
    """计算 mastery 更新的调整因子。

    - 答错 + 高掌握度 → 大幅降低（hypercorrection 效应：高置信错误需要大修正）
    - 答对 + 低掌握度 → 适度提升（运气成分考虑）
    - blackbox 模式 → 整体波动减小（降低到 whitebox 的 70%）

    Args:
        mastery: 当前掌握度。
        outcome: 答题结果。
        mode: 学习模式。

    Returns:
        调整因子 multiplier。
    """
    # 基础因子
    factor = 1.0

    # 高掌握度答错 → 大修正（surprise effect）
    if outcome < 0.5 and mastery > 0.6:
        factor *= 1.5
    # 低掌握度答对 → 谨慎奖励（可能是猜的）
    elif outcome > 0.5 and mastery < 0.3:
        factor *= 0.7
    # 部分对 → 温和调整
    elif outcome == 0.5:
        factor *= 0.5

    # blackbox 模式波动降低
    if mode == "blackbox":
        factor *= 0.7

    return factor


def compute_difficulty(item_type: str, item_mode: str) -> float:
    """根据知识点类型和学习模式计算难度参数。

    Args:
        item_type: 知识点类型（definition/concept/theorem/method/...）。
        item_mode: 学习模式（blackbox/whitebox）。

    Returns:
        难度参数 b ∈ [0.2, 0.8]。
    """
    # 基础难度按类型
    type_difficulty: dict[str, float] = {
        "definition": 0.30,
        "concept": 0.45,
        "section": 0.35,
        "example": 0.40,
        "method": 0.55,
        "application": 0.60,
        "theorem": 0.65,
        "exercise": 0.50,
    }

    base = type_difficulty.get(item_type, 0.50)

    # whitebox 模式增加难度
    if item_mode == "whitebox":
        base += 0.15

    return max(0.2, min(0.8, base))


# ── 批量操作 ─────────────────────────────────────────────────────────


def estimate_initial_mastery(
    known_prerequisites: list[float],
    mode: str = "whitebox",
) -> float:
    """根据已知前置掌握度估计新知识点的初始掌握度。

    平均前置掌握度 → 以折扣因子映射到新知识点。

    Args:
        known_prerequisites: 已知前置知识点的 mastery 值列表（可为空）。
        mode: 学习模式。

    Returns:
        估计初始掌握度。
    """
    if not known_prerequisites:
        # 无前置信息 → 默认初始值
        return 0.15

    avg_prereq = sum(known_prerequisites) / len(known_prerequisites)

    # 折扣因子：新知识点从平均前置掌握度的 40% 开始
    discount = 0.40
    if mode == "blackbox":
        discount = 0.50  # 黑箱工具类更容易上手

    estimated = avg_prereq * discount
    return max(MASTERY_MIN, min(0.50, estimated))  # 上限 0.5（初始估计不能太高）
