"""学习会话引擎 — 纯逻辑层，可独立于 Streamlit 测试。

实现 learn-session 协议的调度逻辑：
- 项目选择与绑定
- 知识点定位（模糊匹配）
- 问题分类与来源调度
- Socratic 引导生成
- 下钻/返回导航（breakdown stack）
- 迷航三栏展示（已完成/剩余/推荐）
- 黑箱/白箱术语翻译

典型用法:
    from learning_agent.ui.study_engine import StudySession, locate_item

    session = StudySession(bookmap=bm, project_name="概率论")
    session.set_goal("理解大数定律的证明")
    matches = locate_item(bm, "大数定律")
    session.drill_down(matches[0].id)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from learning_agent.core.graph import Bookmap, Item

logger = logging.getLogger(__name__)


# ── 常量 ─────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "definition": "定义/定理表述",
    "proof": "证明推导",
    "relationship": "概念关系",
    "prerequisite": "前置概念",
    "application": "应用场景",
    "self_test": "自测请求",
    "progress": "学习进度",
}

SOURCE_TABLE: dict[str, list[str]] = {
    "definition": ["book", "llm"],
    "proof": ["book", "llm"],
    "relationship": ["llm", "graph"],
    "prerequisite": ["graph", "cross_bookmap"],
    "application": ["llm"],
    "self_test": ["llm", "graph"],
    "progress": ["reader_model"],
}

STATUS_LABELS: dict[str, str] = {
    "completed": "已完成",
    "remaining": "剩余",
    "recommended": "推荐",
}

TYPE_LABELS_CN: dict[str, str] = {
    "definition": "定义",
    "concept": "概念",
    "theorem": "定理",
    "method": "方法",
    "example": "示例",
    "application": "应用",
    "section": "章节",
    "exercise": "习题",
}


# ── 数据类 ───────────────────────────────────────────────────────────


@dataclass
class StudySession:
    """单次学习会话的完整状态。

    Attributes:
        project_name: 活动项目名（bookmap 文件 stem）。
        main_goal: 本次会话主线目标。
        current_item_id: 当前所在知识点 ID。
        breakdown_stack: 下钻栈 [(item_id, label), ...]，每次 drill_down 压栈。
        items_covered: 本次会话涉及过的 item ID 集合。
        items_mastery_delta: item_id → 会话内 mastery 变化。
        started_at: 会话开始时间。
        notes: 用户笔记列表。
    """

    project_name: str
    main_goal: str = ""
    current_item_id: str = ""
    breakdown_stack: list[tuple[str, str]] = field(default_factory=list)
    items_covered: set[str] = field(default_factory=set)
    items_mastery_delta: dict[str, float] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    notes: list[str] = field(default_factory=list)

    def set_goal(self, goal: str) -> None:
        """设置本次会话的主线目标。"""
        self.main_goal = goal

    def move_to(self, item_id: str, bm: Bookmap) -> bool:
        """移动到指定知识点（不压栈，直接跳转）。

        Args:
            item_id: 目标知识点 ID。
            bm: Bookmap 实例。

        Returns:
            True 如果跳转成功。
        """
        item = bm.get_item(item_id)
        if item is None:
            return False
        self.current_item_id = item_id
        self.items_covered.add(item_id)
        return True

    def drill_down(self, item_id: str, label: str, bm: Bookmap) -> bool:
        """下钻：保存当前位置到栈，跳转到目标知识点。

        Args:
            item_id: 目标知识点 ID。
            label: 本次下钻的标签（如 "补前置 → 概率公理"）。
            bm: Bookmap 实例。

        Returns:
            True 如果下钻成功。
        """
        item = bm.get_item(item_id)
        if item is None:
            return False

        if self.current_item_id:
            current = bm.get_item(self.current_item_id)
            current_label = current.title if current else self.current_item_id
            self.breakdown_stack.append((self.current_item_id, current_label))

        self.current_item_id = item_id
        self.items_covered.add(item_id)
        return True

    def step_back(self, bm: Bookmap) -> tuple[str, str] | None:
        """返回上一层（弹栈）。

        Args:
            bm: Bookmap 实例。

        Returns:
            (item_id, label) 如果栈非空，否则 None。
        """
        if not self.breakdown_stack:
            return None
        prev_id, prev_label = self.breakdown_stack.pop()
        self.current_item_id = prev_id
        return prev_id, prev_label

    def record_evidence(self, item_id: str, delta: float) -> None:
        """记录会话内 mastery 变化证据。

        Args:
            item_id: 知识点 ID。
            delta: mastery 变化量（正=提升）。
        """
        self.items_covered.add(item_id)
        old = self.items_mastery_delta.get(item_id, 0.0)
        self.items_mastery_delta[item_id] = old + delta

    def is_lost(self) -> bool:
        """判断用户是否处于迷航状态（无当前位置 + 无栈）。"""
        return not self.current_item_id and not self.breakdown_stack

    def has_goal(self) -> bool:
        """是否有明确主线目标。"""
        return bool(self.main_goal)

    def add_note(self, text: str) -> None:
        """添加用户笔记。"""
        self.notes.append(text)


# ── 定位 ─────────────────────────────────────────────────────────────


def locate_item(
    bm: Bookmap,
    query: str,
    *,
    top_k: int = 5,
) -> list[Item]:
    """模糊搜索定位知识点。

    匹配优先级:
        1. 精确 ID 匹配
        2. 标题精确匹配
        3. 标题子串匹配
        4. note 字段子串匹配

    Args:
        bm: Bookmap 实例。
        query: 用户查询文本。
        top_k: 最多返回结果数。

    Returns:
        匹配的 Item 列表（按匹配质量排序）。
    """
    query_lower = query.strip().lower()

    # Level 1: 精确 ID
    exact_id = bm.get_item(query_lower)
    if exact_id:
        return [exact_id]

    scored: list[tuple[int, Item]] = []

    for item in bm.all_items():
        score = 0
        title_lower = item.title.lower()

        # 标题精确匹配
        if query_lower == title_lower:
            score = 100
        # 标题子串匹配
        elif query_lower in title_lower:
            score = 80
        # 标题 token 重叠
        else:
            query_tokens = set(query_lower.split())
            title_tokens = set(title_lower.split())
            overlap = query_tokens & title_tokens
            if overlap:
                score = 40 + len(overlap) * 10

        # note 匹配
        if item.note and query_lower in item.note.lower():
            score += 20

        # source 匹配（查章/节/页码）
        if query_lower in item.source.lower():
            score += 10

        # ID 部分匹配
        if query_lower in item.id.lower():
            score += 5

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    result = [item for _, item in scored[:top_k]]

    if not result:
        logger.info("locate_item: 未找到 '%s' 的匹配", query)

    return result


# ── 问题分类 ─────────────────────────────────────────────────────────


def classify_question(query: str) -> str:
    """将用户问题分类到问题类型。

    Args:
        query: 用户输入的问题文本。

    Returns:
        问题类型 key（见 QUESTION_TYPES）。
    """
    q = query.strip().lower()

    # 关键词匹配
    patterns: list[tuple[str, str]] = [
        (r"(什么是|定义|是什么|什么是…|的含义)", "definition"),
        (r"(证明|推导|怎么来的|为什么|原因|how.*prove|why)", "proof"),
        (r"(关系|区别|对比|联系|vs|和.*什么|difference)", "relationship"),
        (r"(前置|基础|预备|先修|需要先|prerequisite)", "prerequisite"),
        (r"(应用|用途|怎么用|有什么用|例子)", "application"),
        (r"(考考我|自测|测试|quiz|出.*题|我学得怎么样)", "self_test"),
        (r"(进度|我学了什么|学了多少|还剩多少|overview)", "progress"),
    ]

    for pattern, qtype in patterns:
        if re.search(pattern, q):
            return qtype

    # 默认: 定义类（用户可能在问概念含义）
    return "definition"


def get_sources(qtype: str) -> list[str]:
    """获取问题类型的推荐信息来源。

    Args:
        qtype: 问题类型 key。

    Returns:
        来源列表（优先级从高到低）。
    """
    return SOURCE_TABLE.get(qtype, ["book", "llm"])


# ── Socratic 引导 ────────────────────────────────────────────────────


def generate_socratic_prompt(
    item: Item,
    qtype: str,
    *,
    llm_available: bool = True,
) -> str:
    """为指定知识点生成 Socratic 引导提示。

    按 learn-session 协议：Socratic 优先，先引导后解释。

    Args:
        item: 目标知识点。
        qtype: 问题类型。
        llm_available: LLM 是否可用（False 时返回模板化引导）。

    Returns:
        Socratic 引导文本（可直接显示给用户）。
    """
    mode_note = ""
    if item.mode == "blackbox":
        mode_note = "（这个知识点记住公式会用就行，不需要深入证明）"

    prompts: dict[str, str] = {
        "definition": (
            f"关于 **{item.title}**，我们先来试着自己理解一下：\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"你能从字面上拆解一下这个定义吗？关键术语有哪些？\n"
            f"{mode_note}"
        ),
        "proof": (
            f"**{item.title}** 的推导过程，我们先梳理一下思路：\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"这个证明的前提条件是什么？结论需要从哪些已知事实出发？\n"
            f"试着画一下证明链的结构，然后我们逐步看每一步的动机。\n"
            f"{mode_note}"
        ),
        "relationship": (
            f"**{item.title}** 和其他概念的关系：\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"你觉得它可能和哪些概念有关？为什么？\n"
            f"试着用一句话概括它在这个知识体系中的「位置」。\n"
            f"{mode_note}"
        ),
        "prerequisite": (
            f"在深入 **{item.title}** 之前，我们先确认前置基础：\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"这个知识点依赖哪些前置概念？你之前学过吗？\n"
            f"如果有模糊的地方，我们先回头补一下。\n"
            f"{mode_note}"
        ),
        "application": (
            f"**{item.title}** 有什么用？\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"你能先试着想一个它能用到的场景吗？\n"
            f"然后我们看教材中的经典应用。\n"
            f"{mode_note}"
        ),
        "self_test": (
            f"来检验一下 **{item.title}** 掌握得如何：\n\n"
            f"📖 锚点：{item.source}\n\n"
            f"先用自己的话复述一遍核心内容。\n"
            f"然后我会根据你的掌握程度出一道变式题。"
        ),
    }

    prompt = prompts.get(qtype, prompts["definition"])

    if item.note:
        prompt += f"\n\n💡 提示：{item.note}"

    if not llm_available:
        prompt += "\n\n> ⚠️ LLM 不可用，以上为模板化引导。请基于教材原文自行理解。"

    return prompt


# ── 术语翻译 ─────────────────────────────────────────────────────────


def translate_mode(mode: str) -> str:
    """黑箱/白箱 → 用户友好表述（硬性规则）。"""
    return "深入理解" if mode == "whitebox" else "会用即可"


def translate_mastery(mastery: float) -> str:
    """掌握度数值 → 用户友好表述。"""
    if mastery >= 0.8:
        return "已经很熟练了"
    elif mastery >= 0.6:
        return "基本掌握"
    elif mastery >= 0.4:
        return "正在熟悉中"
    elif mastery >= 0.2:
        return "还不太熟"
    else:
        return "刚刚接触"


def translate_status(status: str) -> str:
    """状态枚举 → 用户友好表述。"""
    return {"learned": "已学", "pending": "待学", "optional": "选学"}.get(status, status)


# ── 三栏展示（迷航处理核心）──────────────────────────────────────────


@dataclass
class ThreeColumn:
    """已完成 / 剩余 / 推荐 三栏数据。

    迷航处理的核心数据结构：绝不空手问"你想做什么"，永远带选项。

    Attributes:
        completed: 已完成的 items（status='learned'）。
        remaining: 剩余的 items（status='pending'）。
        recommended: 推荐的下一步 items（带推荐理由）。
    """

    completed: list[tuple[Item, str]] = field(default_factory=list)
    remaining: list[tuple[Item, str]] = field(default_factory=list)
    recommended: list[tuple[Item, str]] = field(default_factory=list)


def compute_three_column(
    bm: Bookmap,
    current_item_id: str = "",
    max_per_column: int = 5,
) -> ThreeColumn:
    """计算三栏数据。

    推荐算法:
        1. 无前置依赖且 status='pending' 的 items（入口节点）
        2. 前置依赖已满足（全部 'learned'）的 pending items
        3. mastery 最低的 learned items（需要复习的薄弱项）

    Args:
        bm: Bookmap 实例。
        current_item_id: 当前所在知识点 ID（可选）。
        max_per_column: 每栏最多显示数。

    Returns:
        ThreeColumn 实例。
    """
    completed: list[tuple[Item, str]] = []
    remaining: list[tuple[Item, str]] = []
    recommended: list[tuple[Item, str]] = []

    for item in bm.all_items():
        if item.status == "learned":
            note = f"掌握度 {item.mastery:.0%}"
            if item.mastery < 0.5:
                note += " · ⚠️ 薄弱需复习"
            completed.append((item, note))
        elif item.status == "pending":
            prereqs = item.prerequisites
            all_prereqs_learned = all(
                bm.get_item(pid) and bm.get_item(pid).status == "learned"  # type: ignore[union-attr]
                for pid in prereqs
            )
            if all_prereqs_learned:
                reason = "前置已就绪"
            else:
                missing = [
                    pid for pid in prereqs
                    if bm.get_item(pid) and bm.get_item(pid).status != "learned"  # type: ignore[union-attr]
                ]
                reason = f"前置待补: {', '.join(missing[:2])}"
            remaining.append((item, reason))
        elif item.status == "optional":
            remaining.append((item, "选学内容"))

    # 已完成按掌握度升序（薄弱优先）
    completed.sort(key=lambda x: x[0].mastery)
    # 剩余按前置就绪优先
    remaining.sort(key=lambda x: 0 if "就绪" in x[1] else 1)

    # 推荐: 前置就绪的 pending items + 薄弱 learned items
    ready_pending = [(it, reason) for it, reason in remaining if "就绪" in reason]
    weak_learned = [(it, reason) for it, reason in completed if it.mastery < 0.5]

    recommended = ready_pending[:3] + weak_learned[:2]

    # 如果有 current_item_id，将其后置节点优先推荐
    if current_item_id:
        dependents = [it for it in bm.all_items() if current_item_id in it.prerequisites]
        for dep in dependents:
            if dep.status == "pending":
                # 去重：后置节点可能已在前置就绪推荐中（同一节点只出现一次）
                if any(it.id == dep.id for it, _ in recommended):
                    continue
                recommended.insert(0, (dep, f"← 接着 {current_item_id} 继续"))

    # 兜底去重（保序保留首个 reason），防止 UI 按钮 key 冲突
    seen_ids: set[str] = set()
    deduped: list[tuple[Item, str]] = []
    for it, reason in recommended:
        if it.id not in seen_ids:
            seen_ids.add(it.id)
            deduped.append((it, reason))
    recommended = deduped

    return ThreeColumn(
        completed=completed[:max_per_column],
        remaining=remaining[:max_per_column],
        recommended=recommended[:max_per_column],
    )


# ── 会话导航上下文 ───────────────────────────────────────────────────


@dataclass
class NavigationContext:
    """当前位置的完整导航上下文。

    Attributes:
        current: 当前知识点。
        cluster: 所属章节。
        prereq_chain: 前置依赖链。
        related: 相关概念。
        dependents: 后置依赖项。
        siblings: 同章节其他项。
    """

    current: Item | None
    cluster: Any | None
    prereq_chain: list[Item]
    related: list[Item]
    dependents: list[Item]
    siblings: list[Item]


def get_navigation_context(bm: Bookmap, item_id: str) -> NavigationContext:
    """获取当前位置的完整导航信息。

    Args:
        bm: Bookmap 实例。
        item_id: 当前知识点 ID。

    Returns:
        NavigationContext 包含所有导航数据。
    """
    item = bm.get_item(item_id)
    cluster = bm.get_cluster(item.cluster) if item else None
    prereq_chain = bm.prerequisite_chain(item_id) if item else []
    related = bm.related_of(item_id) if item else []
    dependents = [it for it in bm.all_items() if item_id in it.prerequisites] if item else []
    siblings = bm.items_in_cluster(item.cluster) if item else []

    return NavigationContext(
        current=item,
        cluster=cluster,
        prereq_chain=prereq_chain,
        related=related,
        dependents=dependents,
        siblings=siblings,
    )


# ── LLM 调用接口 ─────────────────────────────────────────────────────


def build_llm_context(
    bm: Bookmap,
    item_id: str,
    qtype: str,
) -> dict[str, Any]:
    """构建送给 LLM 的完整上下文。

    Args:
        bm: Bookmap 实例。
        item_id: 目标知识点 ID。
        qtype: 问题类型。

    Returns:
        包含 item、prereq、related、instruction 的上下文字典。
    """
    nav = get_navigation_context(bm, item_id)
    if nav.current is None:
        return {"error": f"item '{item_id}' 不存在"}

    return {
        "item": {
            "id": nav.current.id,
            "title": nav.current.title,
            "type": nav.current.type,
            "mode": nav.current.mode,
            "source": nav.current.source,
            "note": nav.current.note,
            "mastery": nav.current.mastery,
        },
        "prerequisites": [
            {"id": p.id, "title": p.title, "mastery": p.mastery}
            for p in nav.prereq_chain
        ],
        "related": [
            {"id": r.id, "title": r.title}
            for r in nav.related
        ],
        "instruction": _llm_instruction(qtype, nav.current),
        "socratic_priority": True,
    }


def _llm_instruction(qtype: str, item: Item) -> str:
    """按问题类型生成 LLM 指令。"""
    instructions: dict[str, str] = {
        "definition": (
            f"请用 Socratic 方法引导理解「{item.title}」。"
            f"先给一个引导性追问，不要直接给答案。"
            f"如果用户表示不理解，再逐步解释。"
            f"所有解释必须引用教材锚点：{item.source}。"
            f"解释后钉回图谱节点 {item.id}。"
        ),
        "proof": (
            f"请解释「{item.title}」的证明。"
            f"先摆出证明链的结构，再逐段解释动机（为什么要这步）。"
            f"如果 item.mode == 'blackbox'，只讲结论和使用条件，不讲证明细节。"
        ),
        "relationship": (
            f"请对比「{item.title}」和相关概念。"
            f"指出相同点/不同点/适用场景。"
            f"用图谱 related 边找到关联概念进行对比。"
        ),
        "prerequisite": (
            f"现在是补缺模式。"
            f"用户在学后置概念时发现缺了前置知识「{item.title}」。"
            f"请就地补讲这个前置概念。补完后回到主线。"
        ),
        "application": (
            f"请介绍「{item.title}」的应用场景。"
            f"优先使用教材中的例子（source: {item.source}）。"
            f"可以补充领域内的经典应用。"
        ),
        "self_test": (
            f"请根据「{item.title}」出题。"
            f"当前掌握度: {item.mastery:.0%}。"
            f"{'低掌握度，请出基础题（定义/判断/选择）。' if item.mastery < 0.4 else ''}"
            f"{'中等掌握度，请出理解题（解释/补全）。' if 0.4 <= item.mastery < 0.7 else ''}"
            f"{'高掌握度，请出应用题（变式/对比/反例）。' if item.mastery >= 0.7 else ''}"
            f"{'必须有一道用自己的话解释的题。' if item.mode == 'whitebox' else ''}"
        ),
    }
    return instructions.get(qtype, instructions["definition"])


# ── LLM 对话接口 ─────────────────────────────────────────────────────


def ask_llm(
    bm: Bookmap,
    item_id: str,
    qtype: str,
    user_message: str,
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """向 LLM 发起 Socratic 教学对话（含图谱上下文）。

    LLM 可用时：调用 llm.LLMClient.socratic_teach()，返回模型回复。
    LLM 不可用时：返回模板化降级回复。

    Args:
        bm: Bookmap 实例。
        item_id: 当前知识点 ID。
        qtype: 问题类型。
        user_message: 用户输入。
        chat_history: 之前的对话历史。

    Returns:
        模型回复或降级模板文本。
    """
    item = bm.get_item(item_id)
    if item is None:
        return f"知识点 '{item_id}' 不存在。"

    # 构建 LLM 上下文
    ctx = build_llm_context(bm, item_id, qtype)
    if "error" in ctx:
        return str(ctx["error"])

    # 尝试调用真实 LLM
    try:
        from learning_agent.llm import LLMClient

        client = LLMClient.from_env()
        if client.available:
            return client.socratic_teach(
                item_context=ctx,
                user_question=user_message,
                chat_history=chat_history,
            )
    except Exception as exc:  # noqa: BLE001 - LLM 失败降级为模板回复
        logger.warning("LLM 调用失败，降级为模板回复: %s", exc)

    # 降级: 模板化
    mode_note = "（记住公式会用即可）" if item.mode == "blackbox" else ""
    return (
        f"收到！关于 **{item.title}**：\n\n"
        f"📖 教材锚点: {item.source}\n\n"
        f"{'💡 要点: ' + item.note if item.note else '请参照 Socratic 引导自行思考。'}\n"
        f"{mode_note}\n\n"
        f"> ⚠️ LLM 不可用，当前为模板化回复。\n"
        f"> 设置 LLM_PROVIDER 和 LLM_API_KEY 环境变量以启用 AI 教学。"
    )


# ── 补充知识点（学习时图谱增量完善） ─────────────────────────────────


def extract_new_item(
    llm: Any,
    title: str,
    description: str,
    *,
    current_item: Item | None = None,
) -> dict[str, Any]:
    """LLM 抽取新知识点的结构化字段（学习会话中补充图谱）。

    Args:
        llm: LLMClient 实例。
        title: 用户输入的新知识点标题（必填）。
        description: 用户补充描述（可为空字符串）。
        current_item: 当前学习中的知识点（用于推断关系与锚点）。

    Returns:
        {"title", "type", "mode", "note", "source", "relation"}。
        relation ∈ {"prerequisite", "extension", "independent"}：
        prerequisite = 新知识点是当前知识点的前置；extension = 延伸/相关；independent = 独立。

    Raises:
        RuntimeError: LLM 不可用或输出解析失败。
    """
    current_title = current_item.title if current_item else ""
    current_source = current_item.source if current_item else ""
    desc_line = f"用户描述: {description}" if description.strip() else "用户未提供描述"

    prompt = (
        f"为知识点「{title}」补全结构化字段，输出 JSON 对象:\n"
        '{"type": "...", "mode": "...", "note": "...", "source": "...", "relation": "..."}\n\n'
        "规则:\n"
        "- type: definition/concept/theorem/method/example/application/exercise\n"
        "- mode: theorem/核心机制→whitebox；其余→blackbox\n"
        "- note: 一句话要点（≤40字）\n"
        "- source: 若与当前知识点同源则沿用其锚点，否则 \"学习补充\"\n"
        f"- relation: 与「{current_title}」的关系 → prerequisite(它是前置)/extension(延伸相关)/independent(独立)\n"
        f"- {desc_line}；只依据用户描述与教材锚点，不编造内容\n\n"
        f"当前知识点: {current_title}（锚点: {current_source}）"
    )

    from learning_agent.build.graph_builder import _SYSTEM_JSON_PROMPT, _call_llm_json

    raw = _call_llm_json(llm, _SYSTEM_JSON_PROMPT, prompt, max_tokens=1024)
    if not isinstance(raw, dict):
        raise TypeError(f"新知识点抽取期望 JSON 对象，实际: {type(raw).__name__}")

    item_type = str(raw.get("type", "concept"))
    if item_type not in {
        "definition", "concept", "theorem", "method",
        "example", "application", "section", "exercise",
    }:
        item_type = "concept"

    mode = str(raw.get("mode", "blackbox"))
    if mode not in {"whitebox", "blackbox"}:
        mode = "blackbox"

    relation = str(raw.get("relation", "independent"))
    if relation not in {"prerequisite", "extension", "independent"}:
        relation = "independent"

    return {
        "title": str(raw.get("title", title)).strip() or title,
        "type": item_type,
        "mode": mode,
        "note": str(raw.get("note", "")).strip() or None,
        "source": str(raw.get("source", "学习补充")).strip() or "学习补充",
        "relation": relation,
    }


def add_item_to_bookmap(
    bm: Bookmap,
    item_data: dict[str, Any],
    *,
    current_item_id: str = "",
) -> Item:
    """将新知识点加入 Bookmap 并连边（纯逻辑，原地修改，可测）。

    Args:
        bm: Bookmap 实例（原地修改）。
        item_data: extract_new_item 返回的字段字典。
        current_item_id: 当前学习知识点 id（用于 relation 连边）。

    Returns:
        新增的 Item。
    """
    # 生成唯一 id: ext-1, ext-2, ...
    n = 1
    while f"ext-{n}" in bm.items:
        n += 1
    new_id = f"ext-{n}"

    # cluster 跟随当前知识点（保持同章归属）
    cluster = "ch1"
    if current_item_id and current_item_id in bm.items:
        cluster = bm.items[current_item_id].cluster

    relation = item_data.get("relation", "independent")
    prereqs: list[str] = []
    related: list[str] = []
    if relation == "prerequisite" and current_item_id and current_item_id in bm.items:
        # 新知识点是当前知识点的前置 → 新节点无前置，当前节点增加前置边
        if new_id not in bm.items[current_item_id].prerequisites:
            bm.items[current_item_id].prerequisites.append(new_id)
    elif relation == "extension" and current_item_id and current_item_id in bm.items:
        related.append(current_item_id)
        if new_id not in bm.items[current_item_id].related:
            bm.items[current_item_id].related.append(new_id)

    item = Item(
        id=new_id,
        cluster=cluster,
        title=item_data.get("title", ""),
        type=item_data.get("type", "concept"),
        mode=item_data.get("mode", "blackbox"),
        source=item_data.get("source", "学习补充"),
        prerequisites=prereqs,
        related=related,
        note=item_data.get("note") or None,
        mastery=0.0,
        next_review=None,
        status="pending",
        cross_refs=[],
    )
    bm.items[new_id] = item
    logger.info("已补充知识点 %s（relation=%s）到图谱", new_id, relation)
    return item


# ── 标记已学（学习进度推进） ─────────────────────────────────────────


# 自评掌握度档位 → mastery 值
MASTERY_LEVELS: dict[str, float] = {
    "mastered": 0.8,  # 掌握了
    "basics": 0.6,    # 基本掌握
    "unsure": 0.3,    # 还不熟（不标记已学）
}


def mark_item_learned(
    bm: Bookmap,
    item_id: str,
    mastery_level: str,
) -> Item:
    """标记知识点学习状态（自评驱动，纯逻辑可测）。

    学完一个知识点后由用户自评掌握度：
    - mastered / basics → status='learned'，mastery 设为对应档位，
      并按档位安排 next_review（3/7 天后）
    - unsure → 保持 pending，仅更新 mastery（后续复习再提升）

    Args:
        bm: Bookmap 实例（原地修改）。
        item_id: 知识点 id。
        mastery_level: MASTERY_LEVELS 的键（mastered/basics/unsure）。

    Returns:
        更新后的 Item。

    Raises:
        KeyError: 知识点不存在。
    """
    item = bm.get_item(item_id)
    if item is None:
        raise KeyError(f"知识点不存在: {item_id}")

    level = mastery_level if mastery_level in MASTERY_LEVELS else "basics"
    new_mastery = MASTERY_LEVELS[level]

    item.mastery = new_mastery
    if level == "unsure":
        item.status = "pending"
        item.next_review = None
    else:
        item.status = "learned"
        days = 7 if level == "mastered" else 3
        item.next_review = (
            datetime.now(UTC).date() + timedelta(days=days)
        ).isoformat()

    logger.info("标记已学: %s (level=%s, mastery=%.2f)", item_id, level, new_mastery)
    return item
