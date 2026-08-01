"""复习引擎 — 纯逻辑层，可独立于 Streamlit 测试。

实现 learn-review 协议的调度逻辑：
- 到期 item 筛选（按项目分组）
- 自适应出题（按 mastery 分级）
- 判分与掌握度更新
- 复习间隔重排
- 会话结果汇总

典型用法:
    from learning_agent.ui.review_engine import ReviewEngine

    engine = ReviewEngine(bookmap=bm)
    due = engine.get_due_items()
    question = engine.generate_question(due[0].id)
    result = engine.evaluate_answer(due[0].id, "user's answer")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from learning_agent.core.graph import Bookmap, Item
from learning_agent.core.mastery import update_mastery

logger = logging.getLogger(__name__)


# ── 常量 ─────────────────────────────────────────────────────────────

MAX_ITEMS_PER_SESSION = 8

# mastery 更新幅度（复习场景，比学习场景更保守）
CORRECT_BOOST = 0.10
PARTIAL_BOOST = 0.03
INCORRECT_PENALTY = 0.15

# 间隔表（learn-session-wrap：按 mastery 档位）
INTERVAL_TABLE: list[tuple[float, int]] = [
    (0.0, 1),    # mastery < 0.5  → 次日
    (0.5, 3),    # 0.5 – 0.7  → 3 日
    (0.7, 7),    # 0.7 – 0.85 → 7 日
    (0.85, 14),  # > 0.85   → 14 日
]

# 题目难度分级
DIFFICULTY_BASIC = "basic"
DIFFICULTY_UNDERSTANDING = "understanding"
DIFFICULTY_APPLICATION = "application"


def _mastery_to_difficulty(mastery: float) -> str:
    """掌握度 → 题目难度。"""
    if mastery < 0.4:
        return DIFFICULTY_BASIC
    elif mastery < 0.7:
        return DIFFICULTY_UNDERSTANDING
    else:
        return DIFFICULTY_APPLICATION


def _compute_review_date(score: float, new_mastery: float) -> date:
    """根据答题结果和掌握度计算下次复习日期（SM-2 规则）。

    分离自 scheduler.compute_next_review：mastery 由 IRT 2PL 模型更新，
    间隔由这里的 SM-2 规则单独计算。

    Args:
        score: 答题得分（1.0/0.5/0.0）。
        new_mastery: 更新后的掌握度。

    Returns:
        下次复习日期。
    """
    today = datetime.now(tz=UTC).date()

    if score < 0.4:
        # 答错 → 明天回炉
        return today + timedelta(days=1)
    elif score < 0.8:
        # 部分对 → 3 天
        return today + timedelta(days=3)
    elif new_mastery > 0.85:
        # 稳固 → 14 天
        return today + timedelta(days=14)
    elif new_mastery > 0.7:
        # 中等 → 7 天
        return today + timedelta(days=7)
    else:
        # 低 mastery 但答对了 → 3 天
        return today + timedelta(days=3)


# ── 数据类 ───────────────────────────────────────────────────────────


@dataclass
class ReviewQuestion:
    """单道复习题。

    Attributes:
        item_id: 对应知识点 ID。
        item_title: 知识点标题。
        difficulty: 难度级别（basic/understanding/application）。
        question_text: 题目文本。
        question_type: 题型（choice/explain/apply/fill）。
        expected_keywords: 期望答案关键词（用于自动判分）。
        source_anchor: 教材锚点。
    """

    item_id: str
    item_title: str
    difficulty: str
    question_text: str
    question_type: str = "explain"
    expected_keywords: list[str] = field(default_factory=list)
    source_anchor: str = ""


@dataclass
class ReviewResult:
    """单题评判结果。

    Attributes:
        question: 原题。
        user_answer: 用户答案文本。
        score: 得分（1.0 全对 / 0.5 部分对 / 0.0 全错）。
        mastery_before: 答题前掌握度。
        mastery_after: 答题后掌握度。
        feedback: 反馈文本。
        next_review: 新复习日期。
    """

    question: ReviewQuestion
    user_answer: str
    score: float
    mastery_before: float
    mastery_after: float
    feedback: str
    next_review: str


@dataclass
class ReviewSessionSummary:
    """一次复习会话的汇总。

    Attributes:
        date: 复习日期。
        total_due: 到期总数。
        reviewed: 实际复习数。
        correct: 全对数。
        partial: 部分对数。
        incorrect: 全错数。
        results: 详细结果列表。
        remaining_ids: 剩余未复习的 item IDs。
    """

    date: str
    total_due: int
    reviewed: int
    correct: int
    partial: int
    incorrect: int
    results: list[ReviewResult] = field(default_factory=list)
    remaining_ids: list[str] = field(default_factory=list)


# ── 复习引擎 ─────────────────────────────────────────────────────────


class ReviewEngine:
    """复习会话管理。

    管理单次复习会话的完整生命周期：筛选到期 items →
    生成自适应题目 → 评判答案 → 更新掌握度/间隔。

    Attributes:
        bookmap: 已加载的 Bookmap。
        results: 累积的评判结果。
        remaining: 待复习的 item IDs。
    """

    def __init__(self, bookmap: Bookmap) -> None:
        """初始化复习引擎。

        Args:
            bookmap: 已加载的 Bookmap 实例。
        """
        self.bookmap = bookmap
        self.results: list[ReviewResult] = []
        self.remaining: list[str] = []

    # ── 到期筛选 ──────────────────────────────────────────────────

    def get_due_items(self, reference_date: date | None = None) -> list[Item]:
        """获取到期应复习的知识点，按 mastery 升序排列。

        Args:
            reference_date: 参考日期，默认为今天。

        Returns:
            到期的 Item 列表（最多 MAX_ITEMS_PER_SESSION 个）。
        """
        if reference_date is None:
            reference_date = datetime.now(tz=UTC).date()

        due = self.bookmap.items_due_for_review(reference_date)
        self.remaining = [it.id for it in due]
        result = due[:MAX_ITEMS_PER_SESSION]
        return result

    # ── 自适应出题 ──────────────────────────────────────────────────

    def generate_question(self, item_id: str) -> ReviewQuestion | None:
        """为指定知识点生成自适应复习题。

        按 mastery 选择难度: <0.4→基础题, 0.4-0.7→理解题, >0.7→应用题。
        whitebox 模式保证至少一道"用自己的话解释"。

        Args:
            item_id: 知识点 ID。

        Returns:
            ReviewQuestion 或 None（item 不存在）。
        """
        item = self.bookmap.get_item(item_id)
        if item is None:
            return None

        difficulty = _mastery_to_difficulty(item.mastery)

        return ReviewQuestion(
            item_id=item.id,
            item_title=item.title,
            difficulty=difficulty,
            question_text=_generate_question_text(item, difficulty),
            question_type=_question_type_for(item, difficulty),
            expected_keywords=_extract_keywords(item),
            source_anchor=item.source,
        )

    # ── 判分 ──────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        item_id: str,
        user_answer: str,
    ) -> ReviewResult | None:
        """评判用户答案，更新掌握度和复习间隔。

        Args:
            item_id: 知识点 ID。
            user_answer: 用户答案文本。

        Returns:
            ReviewResult 或 None（item 不存在）。
        """
        item = self.bookmap.get_item(item_id)
        if item is None:
            return None

        question = self.generate_question(item_id)
        if question is None:
            return None

        mastery_before = item.mastery
        score = _score_answer(user_answer, question.expected_keywords, item)
        feedback = _generate_feedback(score, item, user_answer)

        # 使用 core/mastery 的 2PL 模型更新掌握度（含 hypercorrection）
        mastery_result = update_mastery(
            mastery=mastery_before,
            outcome=score,
            mode=item.mode,
        )
        new_mastery = mastery_result.new_mastery

        # mastery 更新（IRT 2PL）和间隔调度（SM-2）分离
        next_review_date = _compute_review_date(score, new_mastery)

        # 写回 item
        item.mastery = new_mastery
        item.next_review = next_review_date.isoformat()

        result = ReviewResult(
            question=question,
            user_answer=user_answer,
            score=score,
            mastery_before=mastery_before,
            mastery_after=new_mastery,
            feedback=feedback,
            next_review=item.next_review,
        )
        self.results.append(result)

        # 从 remaining 中移除
        if item_id in self.remaining:
            self.remaining.remove(item_id)

        return result

    # ── 批量操作 ──────────────────────────────────────────────────

    def generate_all_questions(
        self,
        item_ids: list[str] | None = None,
    ) -> list[ReviewQuestion]:
        """为一批 items 生成题目。

        Args:
            item_ids: 知识点 ID 列表（None = 到期 items）。

        Returns:
            ReviewQuestion 列表。
        """
        if item_ids is None:
            due = self.get_due_items()
            item_ids = [it.id for it in due]

        questions: list[ReviewQuestion] = []
        for iid in item_ids:
            q = self.generate_question(iid)
            if q:
                questions.append(q)
        return questions

    # ── 汇总 ──────────────────────────────────────────────────────

    def summarize(self) -> ReviewSessionSummary:
        """生成复习会话汇总。

        Returns:
            ReviewSessionSummary。
        """
        correct = sum(1 for r in self.results if r.score >= 0.8)
        partial = sum(1 for r in self.results if 0.4 <= r.score < 0.8)
        incorrect = sum(1 for r in self.results if r.score < 0.4)

        total_due = len(self.results) + len(self.remaining)

        return ReviewSessionSummary(
            date=datetime.now(tz=UTC).date().isoformat(),
            total_due=total_due,
            reviewed=len(self.results),
            correct=correct,
            partial=partial,
            incorrect=incorrect,
            results=list(self.results),
            remaining_ids=list(self.remaining),
        )


# ── 题目生成辅助 ─────────────────────────────────────────────────────


def _generate_question_text(item: Item, difficulty: str) -> str:
    """按难度生成题目文本（模板化，LLM 可用时替换）。

    Args:
        item: 知识点。
        difficulty: 难度级别。

    Returns:
        题目文本。
    """
    templates = {
        DIFFICULTY_BASIC: [
            f"请用自己的话解释「{item.title}」的核心含义。",
            f"「{item.title}」的定义是什么？",
            f"判断：以下关于「{item.title}」的说法是否正确？为什么？",
        ],
        DIFFICULTY_UNDERSTANDING: [
            f"请解释「{item.title}」的关键推导步骤。",
            f"「{item.title}」和其他相关概念有什么联系和区别？",
            f"在什么条件下「{item.title}」适用 / 不适用？",
        ],
        DIFFICULTY_APPLICATION: [
            f"请给出一个「{item.title}」的实际应用场景，并说明如何运用。",
            f"下面的变式题需要用到「{item.title}」：...",
            f"请指出「{item.title}」的局限性或反例。",
        ],
    }

    # 按 item.type 微调
    type_templates: dict[str, dict[str, list[str]]] = {
        "theorem": {
            DIFFICULTY_BASIC: [
                f"请陈述「{item.title}」的条件和结论。",
                f"「{item.title}」的前提假设有哪些？",
            ],
            DIFFICULTY_UNDERSTANDING: [
                f"请补全「{item.title}」的证明中最关键的步骤。",
                f"如果不满足「{item.title}」的某个条件，结论还成立吗？",
            ],
        },
        "definition": {
            DIFFICULTY_BASIC: [
                f"请给出「{item.title}」的精确数学定义。",
                f"「{item.title}」这个定义中的每个符号代表什么？",
            ],
        },
        "method": {
            DIFFICULTY_APPLICATION: [
                f"给定以下场景，请说明如何用「{item.title}」解决。",
            ],
        },
    }

    # 选择模板
    choice = type_templates.get(item.type, {}).get(difficulty)
    if choice is None:
        choice = templates.get(difficulty, templates[DIFFICULTY_BASIC])

    # 确定性选择（按 item id hash）
    idx = hash(item.id) % len(choice)
    return choice[idx]


def _question_type_for(item: Item, difficulty: str) -> str:
    """按 item 类型 + 难度确定题型。"""
    if difficulty == DIFFICULTY_BASIC:
        return "choice" if item.type in ("definition", "concept") else "explain"
    elif difficulty == DIFFICULTY_UNDERSTANDING:
        return "explain"
    else:
        return "apply"


def _extract_keywords(item: Item) -> list[str]:
    """从 item 摘要中提取关键词用于判分。

    Args:
        item: 知识点。

    Returns:
        关键词列表。
    """
    keywords: list[str] = []

    # 从标题提取
    title_words = item.title.lower().split()
    # 过滤常见停用词
    stopwords = {"的", "of", "the", "a", "an", "in", "on", "to", "and", "or", "is"}
    keywords.extend(w for w in title_words if len(w) > 2 and w not in stopwords)

    # 从 note 提取
    if item.note:
        note_words = item.note.lower().split()
        keywords.extend(w for w in note_words if len(w) > 3 and w not in stopwords)

    # 去重，限制数量
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:10]


# ── 判分辅助 ─────────────────────────────────────────────────────────


def _score_answer(
    answer: str,
    keywords: list[str],
    item: Item,
) -> float:
    """基于关键词匹配 + 长度启发式自动判分。

    这是 LLM 不可用时的降级方案。LLM 可用时应替换为语义判分。

    Args:
        answer: 用户答案文本。
        keywords: 期望关键词列表。
        item: 知识点。

    Returns:
        得分（1.0 / 0.5 / 0.0）。
    """
    answer_lower = answer.lower().strip()

    # 空答案/太短
    if not answer_lower or len(answer_lower) < 5:
        return 0.0

    # 关键词覆盖率
    if keywords:
        hit = sum(1 for kw in keywords if kw in answer_lower)
        coverage = hit / len(keywords)
    else:
        coverage = 0.0

    # 长度启发式（太短可能不完整）
    if len(answer_lower) < 20:
        coverage *= 0.5

    if coverage >= 0.5:
        return 1.0
    elif coverage >= 0.2:
        return 0.5
    else:
        return 0.0


def _generate_feedback(
    score: float,
    item: Item,
    _user_answer: str,
) -> str:
    """生成反馈文本。

    Args:
        score: 得分。
        item: 知识点。
        _user_answer: 用户答案（预留 LLM 反馈接口）。

    Returns:
        反馈文本。
    """
    if score >= 0.8:
        return (
            f"✅ 很好！对「{item.title}」的掌握扎实。\n"
            f"掌握度已提升，下次复习间隔延长。"
        )
    elif score >= 0.4:
        return (
            f"⚠️ 部分正确。对「{item.title}」还需巩固。\n"
            f"📖 建议回顾教材：{item.source}\n"
            f"{'💡 要点：' + item.note if item.note else ''}"
        )
    else:
        return (
            f"❌ 需要加强。「{item.title}」是需要重点回炉的知识点。\n"
            f"📖 请仔细复习教材：{item.source}\n"
            f"下次复习安排在明天，掌握后间隔才会延长。\n"
            f"{'💡 要点：' + item.note if item.note else ''}"
        )


# ── 格式化输出 ───────────────────────────────────────────────────────


def format_review_summary(summary: ReviewSessionSummary) -> str:
    """将复习汇总格式化为人类可读的文本。

    Args:
        summary: ReviewSessionSummary 实例。

    Returns:
        多行格式化字符串。
    """
    lines = [
        "🔁 复习完成",
        f"📊 {summary.reviewed}/{summary.total_due} 到期项已复习",
        f"✅ 全对: {summary.correct} | ⚠️ 部分对: {summary.partial} | ❌ 需回炉: {summary.incorrect}",
        "",
    ]

    if summary.results:
        for r in summary.results:
            emoji = "✅" if r.score >= 0.8 else "⚠️" if r.score >= 0.4 else "❌"
            lines.append(
                f"{emoji} **{r.question.item_title}** "
                f"({r.mastery_before:.0%} → {r.mastery_after:.0%})"
            )

    if summary.remaining_ids:
        lines.append("")
        lines.append(f"📅 剩余 {len(summary.remaining_ids)} 项待复习")

    return "\n".join(lines)
