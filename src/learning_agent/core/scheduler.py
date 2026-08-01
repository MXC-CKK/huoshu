"""间隔复习调度器。

基于 SM-2 算法的简化间隔复习调度。根据知识点掌握度、复习历史和
学习模式计算下次复习日期，生成到期复习列表。

核心规则（参考 learn-review SKILL.md）:
    - 答对 +0.10（上限 0.95）
    - 部分对 +0.03
    - 答错 -0.15（下限 0.10）
    - 间隔表: 1, 3, 7, 14, 30, 60, 120 天
    - 答错回炉：间隔重置为 1 天
    - 每次会话最多 8 题

参考:
    - SM-2 algorithm (Wozniak, 1990)
    - learn-review SKILL.md 的间隔表与判分规则
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ── 默认参数 ─────────────────────────────────────────────────────────

# SM-2 间隔表（天），指数增长
DEFAULT_INTERVALS: list[int] = [1, 3, 7, 14, 30, 60, 120]

# mastery 更新幅度
CORRECT_BOOST = 0.10       # 答对提升
PARTIAL_BOOST = 0.03       # 部分对提升
INCORRECT_PENALTY = 0.15   # 答错降低

# 每次复习会话上限
MAX_REVIEW_ITEMS_PER_SESSION = 8

# 首次学习后的默认间隔（天）
DEFAULT_FIRST_INTERVAL = 1

# 答错后回炉间隔（天）
PENALTY_INTERVAL = 1


# ── 结果类型 ─────────────────────────────────────────────────────────


@dataclass
class ScheduleResult:
    """单次复习调度结果。

    Attributes:
        item_id: 知识点 id。
        next_review: 下次复习日期。
        interval_days: 间隔天数。
        mastery_delta: 掌握度变化（Δ）。
        new_mastery: 更新后掌握度。
        review_count: 累计复习次数（本次后）。
    """

    item_id: str
    next_review: date
    interval_days: int
    mastery_delta: float
    new_mastery: float
    review_count: int


@dataclass
class ReviewSession:
    """一次复习会话的完整结果。

    Attributes:
        date: 会话日期。
        items_reviewed: 已复习项的结果列表。
        total_due: 到期总数（含未复习的）。
        remaining: 未复习的到期项 id 列表。
    """

    date: date
    items_reviewed: list[ScheduleResult] = field(default_factory=list)
    total_due: int = 0
    remaining: list[str] = field(default_factory=list)

    @property
    def correct_count(self) -> int:
        """答对数量（mastery 提升 ≥ CORRECT_BOOST）。"""
        return sum(1 for r in self.items_reviewed if r.mastery_delta >= CORRECT_BOOST)

    @property
    def incorrect_count(self) -> int:
        """答错数量（mastery 降低 ≥ INCORRECT_PENALTY * 0.5）。"""
        return sum(1 for r in self.items_reviewed if r.mastery_delta <= -INCORRECT_PENALTY * 0.5)

    @property
    def partial_count(self) -> int:
        """部分对数量。"""
        return len(self.items_reviewed) - self.correct_count - self.incorrect_count


# ── 核心函数 ─────────────────────────────────────────────────────────


def compute_next_review(
    current_mastery: float,
    outcome: float,
    *,
    review_count: int = 0,
    previous_interval: int = 0,
) -> ScheduleResult:
    """根据答题结果计算下次复习日期。

    Args:
        current_mastery: 当前掌握度 θ ∈ [0, 1]。
        outcome: 答题结果。1.0=全对，0.5=部分对，0.0=全错。
        review_count: 当前累计复习次数（本次之前）。
        previous_interval: 上次使用的间隔天数（首次为 0）。

    Returns:
        ScheduleResult 包含下次复习日期、新掌握度等。

    Raises:
        ValueError: outcome 不在 [0, 1] 范围内。
    """
    if not 0.0 <= outcome <= 1.0:
        raise ValueError(f"outcome 必须在 [0, 1] 范围内，实际值: {outcome}")

    # ── 更新掌握度 ──
    if outcome >= 0.8:
        mastery_delta = CORRECT_BOOST
    elif outcome >= 0.4:
        mastery_delta = PARTIAL_BOOST
    else:
        mastery_delta = -INCORRECT_PENALTY

    new_mastery = current_mastery + mastery_delta
    new_mastery = max(0.05, min(0.95, new_mastery))

    # ── 确定间隔 ──
    if outcome < 0.4:
        # 答错：回炉到最短间隔
        interval_days = PENALTY_INTERVAL
        new_review_count = review_count + 1  # 仍计数但重置间隔
    elif outcome >= 0.4 and outcome < 0.8:
        # 部分对：half step back
        half_idx = max(0, _interval_index(previous_interval) - 1)
        interval_days = DEFAULT_INTERVALS[max(0, half_idx)]
        new_review_count = review_count + 1
    else:
        # 答对：前进到下一档间隔
        new_review_count = review_count + 1
        idx = min(new_review_count, len(DEFAULT_INTERVALS) - 1)

        # 根据掌握度微调间隔
        mastery_bonus = 0
        if new_mastery > 0.8:
            mastery_bonus = 1  # 高掌握度可以跳一级

        idx = min(idx + mastery_bonus, len(DEFAULT_INTERVALS) - 1)
        interval_days = DEFAULT_INTERVALS[idx]

    # 计算日期
    next_date = datetime.now(tz=UTC).date() + timedelta(days=interval_days)

    logger.debug(
        "Schedule: mastery %.2f→%.2f, outcome=%.1f, interval=%dd, next=%s",
        current_mastery, new_mastery, outcome, interval_days, next_date.isoformat(),
    )

    return ScheduleResult(
        item_id="",  # 调用方填充
        next_review=next_date,
        interval_days=interval_days,
        mastery_delta=mastery_delta,
        new_mastery=new_mastery,
        review_count=new_review_count,
    )


def _interval_index(days: int) -> int:
    """找到 interval_days 在间隔表中的索引位置。

    Args:
        days: 间隔天数。

    Returns:
        索引（-1 表示未找到，按最近的返回）。
    """
    if days <= 0:
        return 0
    for i, d in enumerate(DEFAULT_INTERVALS):
        if days <= d:
            return i
    return len(DEFAULT_INTERVALS) - 1


def get_due_items(
    items: list[dict[str, Any]],
    reference_date: date | None = None,
    max_items: int = MAX_REVIEW_ITEMS_PER_SESSION,
) -> list[dict[str, Any]]:
    """从未处理的 item 字典列表中筛选到期复习项。

    筛选条件:
        - status == 'learned'
        - next_review 不为 None
        - next_review <= reference_date

    结果按 mastery 升序排列（最弱的优先）。

    Args:
        items: 知识点字典列表（需包含 id, status, mastery, next_review）。
        reference_date: 参考日期，默认为今天。
        max_items: 最多返回项数。

    Returns:
        到期复习项列表（按 mastery 升序，最多 max_items 项）。
    """
    if reference_date is None:
        reference_date = datetime.now(tz=UTC).date()

    ref_str = reference_date.isoformat()
    due: list[dict[str, Any]] = []

    for item in items:
        status = item.get("status", "")
        next_review = item.get("next_review")
        if status != "learned":
            continue
        if next_review is None:
            continue
        if str(next_review) <= ref_str:
            due.append(item)

    # 按 mastery 升序
    due.sort(key=lambda it: float(it.get("mastery", 0.0)))

    return due[:max_items]


def initial_schedule(mastery: float = 0.15) -> ScheduleResult:
    """为新学知识点生成初始复习计划。

    Args:
        mastery: 初始掌握度（默认为新知识点的 0.15）。

    Returns:
        ScheduleResult 下次复习在明天。
    """
    return ScheduleResult(
        item_id="",
        next_review=datetime.now(tz=UTC).date() + timedelta(days=DEFAULT_FIRST_INTERVAL),
        interval_days=DEFAULT_FIRST_INTERVAL,
        mastery_delta=0.0,
        new_mastery=mastery,
        review_count=0,
    )


def compute_mastery_delta(outcome: float) -> float:
    """仅计算 mastery 变化量，不涉及调度。

    Args:
        outcome: 答题结果（1.0 / 0.5 / 0.0）。

    Returns:
        掌握度变化量（正=提升，负=下降）。
    """
    if outcome >= 0.8:
        return CORRECT_BOOST
    elif outcome >= 0.4:
        return PARTIAL_BOOST
    else:
        return -INCORRECT_PENALTY
