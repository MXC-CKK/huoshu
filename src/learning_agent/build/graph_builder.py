"""AI 驱动知识图谱构建器 — PDF 教材 → bookmap JSON。

通过 LLM 从教材文本中自动抽取原子知识点、章节结构、依赖边，
生成符合 bookmap-schema 的知识图谱 JSON。

流程:
    1. parse_llm_json — 容错解析 LLM 输出的 JSON
    2. extract_clusters — 从目录/前言抽章节结构
    3. extract_items — 逐章抽原子知识点
    4. infer_edges — 推断前置依赖和相关边
    5. assemble_bookmap — 组装为 schema 结构
    6. build_bookmap_from_pdf — 编排入口

典型用法:
    from learning_agent.build.graph_builder import build_bookmap_from_pdf
    from learning_agent.llm import LLMClient
    from pathlib import Path

    client = LLMClient.from_env()
    result = build_bookmap_from_pdf(Path("textbook.pdf"), client)
    # result["bookmap"] → 符合 schema 的字典
    # result["stats"] → {clusters, items, edges, whitebox, blackbox}
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 常量 ─────────────────────────────────────────────────────────────

DEFAULT_CLUSTER_SCAN_CHARS = 12000  # 目录/前言扫描字符上限

# 单次 LLM 抽取的文本上限。实测 deepseek 对 >5000 字符输入+长 JSON 输出
# 组合不稳定（返回空/残缺响应），4000 字符内可靠。
MAX_EXTRACT_CHARS = 4000

# 章/节检测正则（中英文兼容）
_CHAPTER_RE = re.compile(
    r"(第[一二三四五六七八九十百\d]+章|"
    r"[Cc]hapter\s*\d+|"
    r"第[一二三四五六七八九十百\d]+节|"
    r"[Ss]ection\s*\d+[\.\d]*|"
    r"\d+[\.\s]+[A-Z一-鿿])",
)

# LLM 系统提示：强调 JSON 格式约束
_SYSTEM_JSON_PROMPT = (
    "你是一位严谨的知识图谱构建助手。请严格输出 JSON，不要任何 Markdown 围栏、"
    "不要注释、不要额外解释。只输出要求的 JSON 结构。"
)

# 知识点类型 → 默认 mode 建议
_TYPE_DEFAULT_MODE: dict[str, str] = {
    "definition": "blackbox",
    "concept": "blackbox",
    "theorem": "whitebox",
    "method": "blackbox",
    "example": "blackbox",
    "application": "blackbox",
    "section": "blackbox",
    "exercise": "blackbox",
}

# 单次抽取知识点数量上限（实测输出超过 ~10 个条目后模型容易截断）
MAX_ITEMS_PER_CALL = 8


# ── 环境变量解析 ─────────────────────────────────────────────────────


def resolve_bookmap_dir() -> Path:
    """返回 bookmap JSON 保存目录。

    优先级: 环境变量 HUOSHU_BOOKMAP_DIR > 默认 ~/.huoshu/bookmap。

    Returns:
        bookmap 目录的 Path 对象（调用方需自行 mkdir）。
    """
    return Path(
        os.environ.get(
            "HUOSHU_BOOKMAP_DIR",
            str(Path.home() / ".huoshu" / "bookmap"),
        )
    )


# ── JSON 容错解析 ────────────────────────────────────────────────────


def parse_llm_json(text: str) -> dict[str, Any] | list[Any]:
    """容错解析 LLM 输出的 JSON。

    处理常见 LLM 输出噪声：```json 围栏、尾逗号、单引号、控制字符。

    Args:
        text: LLM 原始输出文本。

    Returns:
        解析后的 dict 或 list。

    Raises:
        ValueError: 无法解析为有效 JSON。
    """
    if not text or not text.strip():
        raise ValueError("LLM 输出为空，无法解析 JSON")

    cleaned = text.strip()

    # 去除 ```json / ``` 围栏
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    # 截取首个平衡的 { 或 [
    cleaned = _extract_balanced_json(cleaned)

    # 尝试直接解析
    try:
        return json.loads(cleaned)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 修复尝试: 尾逗号
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(fixed)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 修复尝试: 单引号 → 双引号（简单场景）
    try:
        # 仅替换作为 key/value 边界的单引号，避免内容中的引号
        fixed = cleaned.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return json.loads(fixed)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    raise ValueError(f"无法解析 LLM 输出为 JSON（前200字符）: {text[:200]}")


def _extract_balanced_json(text: str) -> str:
    """从文本中截取首个平衡的 JSON 对象或数组。"""
    # 跳到第一个 { 或 [
    start_chars = {"{": "}", "[": "]"}
    start_pos = -1
    start_char = ""
    for i, ch in enumerate(text):
        if ch in start_chars:
            start_pos = i
            start_char = ch
            break

    if start_pos < 0:
        # 无括号，回退原文本
        return text

    end_char = start_chars[start_char]
    depth = 0
    in_string = False
    escape = False

    for i in range(start_pos, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start_pos : i + 1]

    return text


def _salvage_partial_json(text: str) -> list[Any] | None:
    """从被截断的 JSON 数组中抢救完整对象。

    实测 LLM（deepseek）生成长 JSON 数组时偶尔中途截断（无 finish_reason
    提前停止），json.loads 必然失败。本函数逐个提取数组中已完整的对象，
    至少抢救出 1 个则返回，否则返回 None。

    Args:
        text: LLM 原始输出（可能是截断的 JSON 数组）。

    Returns:
        完整对象列表；无法抢救时返回 None。
    """
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    if not cleaned.startswith("["):
        return None

    objects: list[Any] = []
    pos = 0
    while True:
        start = cleaned.find("{", pos)
        if start < 0:
            break
        # 找该对象的平衡右括号（忽略字符串内括号）
        depth = 0
        in_string = False
        escape = False
        end = start
        while end < len(cleaned):
            ch = cleaned[end]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            end += 1
        if depth != 0 or end >= len(cleaned):
            break  # 该对象不完整 → 后面的也都不完整
        try:
            objects.append(json.loads(cleaned[start : end + 1]))
        except json.JSONDecodeError:
            pass
        pos = end + 1

    return objects if objects else None


# ── LLM 调用封装 ─────────────────────────────────────────────────────


def _call_llm(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
) -> Any:
    """调用 LLM 并返回原始文本，带重试。

    Args:
        llm: LLMClient 实例。
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        max_tokens: 最大输出 token。

    Returns:
        LLM 回复文本。

    Raises:
        RuntimeError: 调用失败（含重试后）。
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            if response and response.strip():
                return response.strip()
            last_error = RuntimeError("LLM 返回空响应")
        except Exception as exc:  # noqa: BLE001 - 重试机制需要捕获所有异常
            last_error = exc
            logger.warning("LLM 调用失败 (attempt %d/2): %s", attempt + 1, exc)

    raise RuntimeError(f"LLM 调用失败（已重试）: {last_error}")


def _call_llm_json(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    retries: int = 3,
) -> dict[str, Any] | list[Any]:
    """调用 LLM 并容错解析 JSON，解析失败自动重试。

    应对 LLM 输出的两类不稳定：
    1. 空响应/调用失败 → _call_llm 内部已重试
    2. 截断/格式损坏 → 每次尝试先 parse_llm_json，失败后尝试
       _salvage_partial_json 抢救完整对象，仍失败则重试

    Args:
        llm: LLMClient 实例。
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        max_tokens: 最大输出 token。
        retries: 总尝试次数。

    Returns:
        解析后的 dict 或 list。

    Raises:
        RuntimeError: 所有尝试均失败。
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = _call_llm(llm, system_prompt, user_prompt, max_tokens=max_tokens)
            try:
                return parse_llm_json(raw)
            except ValueError:
                salvaged = _salvage_partial_json(raw)
                if salvaged is not None:
                    logger.warning(
                        "LLM 输出截断，抢救出 %d 个完整对象 (attempt %d/%d)",
                        len(salvaged), attempt + 1, retries,
                    )
                    return salvaged
                raise
        except (ValueError, RuntimeError) as exc:
            last_err = exc
            logger.warning(
                "LLM JSON 解析失败 (attempt %d/%d): %s", attempt + 1, retries, exc
            )

    raise RuntimeError(f"LLM 输出解析失败（已重试 {retries} 次）: {last_err}")


# ── 章节抽取 ─────────────────────────────────────────────────────────


def extract_clusters(text: str, llm: Any) -> list[dict[str, str]]:
    """从教材前若干页（目录/前言）抽取章节结构。

    Args:
        text: 教材前若干页的文本（建议前 15 页或 ~12000 字符）。
        llm: LLMClient 实例。

    Returns:
        章节列表 [{"id": "ch1", "title": "第一章 绪论"}, ...]。
    """
    prompt = (
        "请从以下教材目录/前言文本中提取所有章节（章级别）。\n\n"
        "要求:\n"
        "1. 只提取章级别（Chapter），不要节级别\n"
        "2. id 使用 ch1, ch2, ... 格式\n"
        "3. title 保留原文完整标题（含编号）\n"
        '4. 按章节顺序输出 JSON 数组: [{"id": "ch1", "title": "..."}, ...]\n\n'
        f"教材文本:\n{text[:DEFAULT_CLUSTER_SCAN_CHARS]}"
    )

    raw = _call_llm_json(llm, _SYSTEM_JSON_PROMPT, prompt, max_tokens=2048)
    result = raw

    if not isinstance(result, list):
        raise TypeError(f"extract_clusters 期望 JSON 数组，实际: {type(result).__name__}")

    # 校验并补全
    clusters: list[dict[str, str]] = []
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", f"ch{i + 1}"))
        title = str(item.get("title", f"第{i + 1}章"))
        clusters.append({"id": cid, "title": title})

    logger.info("抽取了 %d 个章节", len(clusters))
    return clusters


# ── 知识点抽取 ───────────────────────────────────────────────────────


def extract_items(
    cluster_title: str,
    chunk_text: str,
    llm: Any,
    prefix: str,
) -> list[dict[str, Any]]:
    """从单章文本中抽取原子知识点。

    遵循四层分解（Domain→Cluster→Item），每 item 满足原子性三标准：
    Atomic（不能再分）、Assessable（能出题）、Meaningful（真知识点）。

    Args:
        cluster_title: 章节标题（用于 LLM 上下文）。
        chunk_text: 该章节的文本。
        llm: LLMClient 实例。
        prefix: item id 前缀（如 "ch1" → id 为 ch1-1, ch1-2, ...）。

    Returns:
        知识点列表（dict 含 id/title/type/mode/note/source 字段）。
    """
    prompt = (
        f"你正在分析教材章节「{cluster_title}」。请从以下文本中提取所有原子知识点。\n\n"
        "## 原子性三标准（每个知识点必须满足）\n"
        "- **Atomic**：不能或不应再分为更小的子知识点（学生要么掌握、要么没有）\n"
        "- **Assessable**：能针对它出一道独立的测试题\n"
        "- **Meaningful**：是真正的知识点（定义、定理、方法、概念），不是琐碎碎片\n\n"
        "## 字段说明\n"
        f"- id: \"{prefix}-序号\"（如 {prefix}-1）\n"
        '- type: 从以下选择 → definition(定义) / concept(概念机制) / theorem(定理证明) / '
        "method(方法算法) / example(例子) / application(应用) / exercise(习题)\n"
        "- mode: 定理/核心机制 → whitebox（需深入理解）；定义/工具/应用例 → blackbox（会用即可）\n"
        "- source: 教材锚点（页码范围），格式如「第X章 pp.Y-Z」\n"
        "- note: 一句话要点（助记，可选）\n\n"
        "## 输出格式\n"
        '输出 JSON 数组（不要 Markdown 围栏）:\n'
        '[{"id": "...", "title": "...", "type": "...", "mode": "...", "source": "...", "note": "..."}, ...]\n\n'
        f"## 输出要求\n最多输出 {MAX_ITEMS_PER_CALL} 个最重要的知识点。若本段知识点更多，只输出最重要的 {MAX_ITEMS_PER_CALL} 个（其余会在后续分段继续抽取）。\n\n"
        f"## 教材文本\n{chunk_text[:MAX_EXTRACT_CHARS]}"
    )

    raw = _call_llm_json(llm, _SYSTEM_JSON_PROMPT, prompt, max_tokens=4096)
    result = raw

    if not isinstance(result, list):
        raise TypeError(f"extract_items 期望 JSON 数组，实际: {type(result).__name__}")

    items: list[dict[str, Any]] = []
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "concept"))
        if item_type not in {
            "definition", "concept", "theorem", "method",
            "example", "application", "section", "exercise",
        }:
            item_type = "concept"

        mode = str(item.get("mode", _TYPE_DEFAULT_MODE.get(item_type, "blackbox")))
        if mode not in {"whitebox", "blackbox"}:
            mode = "blackbox"

        items.append({
            "id": str(item.get("id", f"{prefix}-{i + 1}")),
            "cluster": prefix,  # 所属簇 id
            "title": str(item.get("title", f"知识点 {i + 1}")),
            "type": item_type,
            "mode": mode,
            "source": str(item.get("source", f"{cluster_title}")),
            "note": str(item.get("note", "")) if item.get("note") else None,
        })

    logger.info("从「%s」抽取了 %d 个知识点", cluster_title, len(items))
    return items


# ── 依赖边推断 ───────────────────────────────────────────────────────


def infer_edges(items: list[dict[str, Any]], llm: Any) -> dict[str, list[list[str]]]:
    """推断知识点间的依赖边。

    使用 QUERY 算法核心原则：「掌握 b 是否逻辑必然蕴含掌握 a？」
    只建必要前置边，置信度 ≥0.6 才收录。

    Args:
        items: 所有知识点列表。
        llm: LLMClient 实例。

    Returns:
        {"prerequisites": [[from_id, to_id], ...], "related": [[a_id, b_id], ...]}
    """
    # 构造简洁的 item 列表给 LLM
    item_summaries: list[str] = []
    for it in items:
        item_summaries.append(
            f"- {it['id']}: {it['title']} (type={it['type']}, mode={it['mode']})"
        )
    items_text = "\n".join(item_summaries)

    prompt = (
        "请推断下列知识点之间的依赖关系。\n\n"
        "## 前置依赖边 (prerequisites)\n"
        '核心问题：「掌握 b 是否逻辑必然蕴含掌握 a？」\n'
        "- 只建逻辑必要边，「有帮助但可跳过」不是边\n"
        "- 置信度 ≥0.6 才收录\n"
        "- 不要建传递依赖（a→b→c 只需 a→b 和 b→c，不要 a→c）\n\n"
        "## 相关边 (related)\n"
        "- 易混概念（confusable）：容易混淆的知识点对\n"
        "- 同现概念（cooccurring）：经常一起出现的知识点\n"
        "- 类比对比：相似或对立的知识点对\n\n"
        "## 输出格式\n"
        '输出一个 JSON 对象（不要 Markdown 围栏）:\n'
        '{\n'
        '  "prerequisites": [["前置id", "后置id"], ...],\n'
        '  "related": [["id_a", "id_b"], ...]\n'
        '}\n\n'
        f"## 知识点列表\n{items_text}"
    )

    raw = _call_llm_json(llm, _SYSTEM_JSON_PROMPT, prompt, max_tokens=4096)
    result = raw

    if not isinstance(result, dict):
        raise TypeError(f"infer_edges 期望 JSON 对象，实际: {type(result).__name__}")

    prerequisites = result.get("prerequisites", [])
    related = result.get("related", [])

    # 校验边：确保引用的 item id 存在
    valid_ids = {it["id"] for it in items}
    clean_pairs: list[list[str]] = []
    for pair in prerequisites:
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and pair[0] in valid_ids
            and pair[1] in valid_ids
            and pair[0] != pair[1]
        ):
            clean_pairs.append([str(pair[0]), str(pair[1])])

    clean_related: list[list[str]] = []
    for pair in related:
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and pair[0] in valid_ids
            and pair[1] in valid_ids
            and pair[0] != pair[1]
        ):
            clean_related.append([str(pair[0]), str(pair[1])])

    logger.info("推断 %d 条前置边, %d 条相关边", len(clean_pairs), len(clean_related))
    return {"prerequisites": clean_pairs, "related": clean_related}


# ── 组装 ─────────────────────────────────────────────────────────────


def assemble_bookmap(
    meta: dict[str, Any],
    domain: str,
    clusters: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
    edges: dict[str, list[list[str]]],
) -> dict[str, Any]:
    """将各部分组装为符合 bookmap-schema 的完整图谱字典。

    Args:
        meta: 元信息字典（source/built/status/extraction_method）。
        domain: 学科领域。
        clusters: {id: {title, learned, learned_date, parent}} 字典。
        items: 知识点列表（含 cluster/title/type/mode/source/note）。
        edges: {"prerequisites": [[a,b],...], "related": [[a,b],...]}。

    Returns:
        符合 bookmap-schema 的完整字典。
    """
    # 应用边到 items
    prereq_map: dict[str, list[str]] = {}
    related_map: dict[str, list[str]] = {}
    for a, b in edges.get("prerequisites", []):
        prereq_map.setdefault(b, []).append(a)
    for a, b in edges.get("related", []):
        related_map.setdefault(a, []).append(b)
        related_map.setdefault(b, []).append(a)

    assembled_items: list[dict[str, Any]] = []
    for it in items:
        item_id = str(it["id"])
        assembled = {
            "id": item_id,
            "cluster": str(it.get("cluster", "")),
            "title": str(it.get("title", "")),
            "type": str(it.get("type", "concept")),
            "mode": str(it.get("mode", "blackbox")),
            "source": str(it.get("source", "")),
            "prerequisites": prereq_map.get(item_id, []),
            "related": list(dict.fromkeys(related_map.get(item_id, []))),  # 去重保序
            "note": it.get("note") if it.get("note") else None,
            "mastery": 0.0,
            "next_review": None,
            "status": "pending",
            "cross_refs": [],
        }
        assembled_items.append(assembled)

    return {
        "meta": meta,
        "domain": domain,
        "clusters": clusters,
        "items": assembled_items,
    }


# ── 章节切分 ─────────────────────────────────────────────────────────


def _split_chapters(
    full_text: str,
    clusters: list[dict[str, str]],
) -> list[tuple[dict[str, str], str]]:
    """将教材全文按章节标题切分到各簇。

    使用正则匹配章标题边界，将文本分配到最近的簇。
    无法匹配的文本归入第一个簇或标记为序言。

    Args:
        full_text: 教材全文（所有页文本拼接）。
        clusters: extract_clusters 的输出。

    Returns:
        [(cluster_dict, chapter_text), ...] 列表。
    """
    if not clusters:
        return []

    # 构建簇标题 → 簇的映射
    # 为每个簇提取关键匹配词（取标题中可匹配的部分）
    cluster_patterns: list[tuple[dict[str, str], str]] = []
    for cl in clusters:
        title = cl["title"]
        # 尝试从标题中提取可正则匹配的部分
        # 如 "第一章 绪论" → "第一章"
        # 或 "Chapter 1 Introduction" → "Chapter 1"
        match = re.search(
            r"(第[一二三四五六七八九十百\d]+章|"
            r"[Cc]hapter\s*\d+)",
            title,
        )
        if match:
            cluster_patterns.append((cl, re.escape(match.group(1))))
        else:
            # 无法提取模式，用完整标题
            cluster_patterns.append((cl, re.escape(title)))

    # 构建正则：匹配所有章标题
    all_patterns = "|".join(p for _, p in cluster_patterns)
    if not all_patterns:
        return [(clusters[0], full_text)]

    boundary_re = re.compile(f"({all_patterns})")

    # 按章边界切分
    parts = boundary_re.split(full_text)

    # parts 格式: [text_before, match1, text_after_match1, match2, ...]
    result: list[tuple[dict[str, str], str]] = []

    # 第一个匹配前的文本（序言）
    preamble = parts[0] if parts else ""

    # 将每个匹配与紧随的文本配对
    current_cluster: dict[str, str] | None = None
    current_text_parts: list[str] = []

    for i in range(1, len(parts), 2):
        match_text = parts[i] if i < len(parts) else ""
        after_text = parts[i + 1] if i + 1 < len(parts) else ""

        # 找匹配的簇
        matched_cluster = None
        for cl, pat in cluster_patterns:
            if re.match(pat, match_text):
                matched_cluster = cl
                break

        if matched_cluster is None:
            # 未匹配的标题，作为当前簇的一部分
            if current_cluster is not None:
                current_text_parts.append(match_text + after_text)
            else:
                preamble += match_text + after_text
            continue

        # 保存上一个簇的结果
        if current_cluster is not None:
            result.append((current_cluster, "".join(current_text_parts)))

        current_cluster = matched_cluster
        current_text_parts = [match_text + " " + after_text]

    # 最后一个簇
    if current_cluster is not None:
        result.append((current_cluster, "".join(current_text_parts)))

    # 序言归入第一个簇
    if preamble.strip() and result:
        first_cluster, first_text = result[0]
        result[0] = (first_cluster, preamble + " " + first_text)

    logger.info("切分了 %d 个章节段落", len(result))
    return result


# ── 编排入口 ─────────────────────────────────────────────────────────


def build_bookmap_from_pdf(
    pdf_path: Path,
    llm: Any,
    *,
    goal: str = "",
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """从 PDF 教材自动构建知识图谱。

    编排流程:
        1. 解析 PDF 全文
        2. 从目录/前言抽取章节结构
        3. 按章节切分文本
        4. 逐章抽取原子知识点
        5. 全局推断依赖边
        6. 组装 bookmap

    Args:
        pdf_path: PDF 文件路径。
        llm: LLMClient 实例。
        goal: 可选学习目标（考试复习/系统读懂/快速应用），影响抽取提示。
        progress: 进度回调 (stage: str, current: int, total: int)。

    Returns:
        {
            "bookmap": dict,   # 符合 bookmap-schema 的完整图谱
            "stats": {
                "clusters": int,
                "items": int,
                "edges": int,
                "whitebox": int,
                "blackbox": int,
            }
        }
    """
    from learning_agent.rag.ingest import extract_pages

    # ── Step 1: 解析 PDF ──
    if progress:
        progress("解析PDF文本", 0, 0)
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"PDF 无文本内容: {pdf_path}")
    full_text = "\n\n".join(text for _, text in pages)
    first_pages_text = "\n\n".join(text for _, text in pages[:15])
    logger.info("PDF 解析完成: %d 页, 全文 %d 字符", len(pages), len(full_text))

    # ── Step 2: 抽取章节 ──
    if progress:
        progress("抽取目录结构", 0, 0)
    clusters = extract_clusters(first_pages_text, llm)
    if not clusters:
        raise ValueError("未能从教材中抽取到章节结构，请确认 PDF 包含目录或章节标题")

    # ── Step 3: 按章切分 ──
    if progress:
        progress("按章节切分文本", 0, len(clusters))
    chapter_chunks = _split_chapters(full_text, clusters)
    if not chapter_chunks:
        # 降级：整本书作为一个簇
        chapter_chunks = [
            ({"id": "ch1", "title": "全书"}, full_text)
        ]
        clusters = [{"id": "ch1", "title": "全书"}]

    # ── Step 4: 逐章抽取知识点 ──
    all_items: list[dict[str, Any]] = []
    failed_chapters: list[str] = []

    for idx, (cluster, chunk_text) in enumerate(chapter_chunks):
        cluster_id = cluster["id"]
        cluster_title = cluster["title"]
        if progress:
            progress(f"抽取知识点: {cluster_title}", idx + 1, len(chapter_chunks))

        # 单章 >4000 字符按节再切（保证每次 LLM 调用输入在可靠区间）
        sub_chunks = _split_subsections(chunk_text, max_chars=MAX_EXTRACT_CHARS)

        for sub_idx, sub_text in enumerate(sub_chunks):
            try:
                items = extract_items(
                    cluster_title=cluster_title,
                    chunk_text=sub_text,
                    llm=llm,
                    prefix=f"{cluster_id}-{sub_idx + 1}" if len(sub_chunks) > 1 else cluster_id,
                )
                # 确保每个 item 的 cluster 字段正确
                for it in items:
                    it["cluster"] = cluster_id
                all_items.extend(items)
            except Exception as exc:  # noqa: BLE001 - 单章失败不中断整体构建
                sub_label = (
                    f"第{sub_idx + 1}/{len(sub_chunks)}段" if len(sub_chunks) > 1 else ""
                )
                failed_chapters.append(f"{cluster_title} {sub_label}: {exc}")
                logger.warning("知识点抽取失败: %s %s", cluster_title, exc)

    if not all_items:
        raise ValueError(
            "所有章节的知识点抽取均失败。请检查 LLM 配置和 API 可用性。"
            + (f"\n失败详情: {'; '.join(failed_chapters)}" if failed_chapters else "")
        )

    # ── Step 5: 推断边 ──
    if progress:
        progress("推断依赖边", 0, 0)
    edges = infer_edges(all_items, llm)

    # ── Step 6: 组装 ──
    if progress:
        progress("组装图谱", 0, 0)

    # 构建 clusters 字典
    clusters_dict: dict[str, dict[str, Any]] = {}
    for cl in clusters:
        clusters_dict[cl["id"]] = {
            "title": cl["title"],
            "learned": False,
            "learned_date": None,
            "parent": None,
        }

    # 推断 domain
    domain_text = ""
    for page_text in [text for _, text in pages[:3]]:
        # 尝试从首页提取书名
        domain_text += page_text[:200]

    domain = pdf_path.stem
    goal_hint = {"考试复习": "exam", "系统读懂": "systematic", "快速应用": "quick"}.get(goal, "")

    meta = {
        "source": pdf_path.stem,
        "source_files": [str(pdf_path)],
        "built": datetime.now(UTC).date().isoformat(),
        "status": "draft-待校对",
        "extraction_method": "四层分解 (Domain→Cluster→Item) + 原子性三标准 + QUERY 边推断",
    }
    if goal_hint:
        meta["goal"] = goal

    bookmap = assemble_bookmap(meta, domain, clusters_dict, all_items, edges)

    # 统计
    prereq_count = sum(len(it.get("prerequisites", [])) for it in bookmap["items"])
    whitebox_count = sum(1 for it in bookmap["items"] if it.get("mode") == "whitebox")
    blackbox_count = sum(1 for it in bookmap["items"] if it.get("mode") == "blackbox")

    stats = {
        "clusters": len(clusters_dict),
        "items": len(bookmap["items"]),
        "edges": prereq_count,
        "whitebox": whitebox_count,
        "blackbox": blackbox_count,
    }

    logger.info(
        "图谱构建完成: %d 簇, %d items, %d 边, %d whitebox, %d blackbox",
        stats["clusters"], stats["items"], stats["edges"],
        stats["whitebox"], stats["blackbox"],
    )

    if failed_chapters:
        logger.warning("部分章节抽取失败: %s", "; ".join(failed_chapters))

    return {
        "bookmap": bookmap,
        "stats": stats,
        "failed_chapters": failed_chapters,
    }


def _split_subsections(text: str, max_chars: int = MAX_EXTRACT_CHARS) -> list[str]:
    """将长文本按节或自然段落边界切分为子块。

    Args:
        text: 长文本。
        max_chars: 每块最大字符数。

    Returns:
        子块列表。
    """
    if len(text) <= max_chars:
        return [text]

    # 尝试按节标题切分
    sub_boundaries = list(re.finditer(
        r"(第[一二三四五六七八九十百\d]+节|"
        r"[Ss]ection\s*\d+|"
        r"\d+\.\d+\s+[A-Z一-鿿])",
        text,
    ))

    if sub_boundaries:
        chunks: list[str] = []
        prev = 0
        for m in sub_boundaries:
            if m.start() > prev:
                chunks.append(text[prev : m.start()])
            prev = m.start()
        if prev < len(text):
            chunks.append(text[prev:])

        # 合并过短的块
        merged: list[str] = []
        buf = ""
        for ch in chunks:
            if len(buf) + len(ch) <= max_chars:
                buf += "\n" + ch if buf else ch
            else:
                if buf:
                    merged.append(buf)
                buf = ch
        if buf:
            merged.append(buf)
        return merged if merged else [text]

    # 无节标题，按段落边界切
    paragraphs = text.split("\n\n")
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) <= max_chars:
            buf += "\n\n" + para if buf else para
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]
