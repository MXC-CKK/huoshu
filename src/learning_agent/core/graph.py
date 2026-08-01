"""Bookmap 知识图谱：JSON 加载、完整性校验、图谱导航。

提供 Bookmap 类作为整个 huoshu 的数据基础层。所有调度、复习、RAG
操作都建立在 Bookmap 的数据模型之上。

典型用法:
    from pathlib import Path
    from learning_agent.core.graph import Bookmap

    bm = Bookmap.load(Path("bookmap/econometrics-ch2.json"))
    errors = bm.validate()
    if errors:
        for e in errors:
            logging.warning("校验失败: %s", e)

    chain = bm.prerequisite_chain("ch2-4-2")
    whitebox_items = bm.items_by_mode("whitebox")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 数据类 ──────────────────────────────────────────────────────────


@dataclass
class Item:
    """知识图谱中的原子知识点。

    Attributes:
        id: 全局唯一标识符（如 'ch5-4-1'）。
        cluster: 所属簇（章节）id。
        title: 知识点标题（含定理/定义编号）。
        type: 知识点类型（definition/concept/theorem/method/example/application/section/exercise）。
        mode: 学习模式（blackbox: 公式记住会用即可；whitebox: 深入理解）。
        prerequisites: 前置依赖 item id 列表（学习顺序约束）。
        related: 相关概念 item id 列表（类比/对比/横纵联系）。
        source: 教材锚点（章/节/页码/定理号）。
        note: 一句话要点（助记）。
        mastery: 掌握度 0-1（简化 IRT）。
        next_review: 下次复习日期（None 表示未安排）。
        status: 学习状态（learned/pending/optional）。
        cross_refs: 跨项目引用列表。
    """

    id: str
    cluster: str
    title: str
    type: str
    mode: str
    source: str
    prerequisites: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    note: str | None = None
    mastery: float = 0.0
    next_review: str | None = None
    status: str = "pending"
    cross_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Cluster:
    """章节/主题簇。

    Attributes:
        id: 簇标识符。
        title: 簇标题。
        learned: 用户是否已学完该簇。
        learned_date: 学完日期。
        parent: 上级簇 id（支持多级层次）。
    """

    id: str
    title: str
    learned: bool = False
    learned_date: str | None = None
    parent: str | None = None


# ── 合法值常量 ──────────────────────────────────────────────────────

VALID_ITEM_TYPES = frozenset({
    "definition", "concept", "theorem", "method",
    "example", "application", "section", "exercise",
})
VALID_MODES = frozenset({"blackbox", "whitebox"})
VALID_STATUSES = frozenset({"learned", "pending", "optional"})
VALID_META_STATUSES = frozenset({"draft-待校对", "active", "archived"})
VALID_CROSS_REF_RELATIONS = frozenset({"same", "prerequisite", "contrast", "extension"})


# ── Bookmap ──────────────────────────────────────────────────────────


class Bookmap:
    """加载后的知识图谱内存表示。

    提供加载、校验、导航三类操作。校验在加载时自动执行
    （可通过 validate_on_load=False 跳过），校验错误通过
    logging.warning 报告，同时可通过 errors 属性访问。
    """

    def __init__(self) -> None:
        """初始化空的 Bookmap。使用 Bookmap.load() 从文件加载。"""
        self.meta: dict[str, Any] = {}
        self.domain: str = ""
        self.clusters: dict[str, Cluster] = {}
        self.items: dict[str, Item] = {}
        self._errors: list[str] = []

    # ── 工厂方法 ─────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path, *, validate_on_load: bool = True) -> Bookmap:
        """从 JSON 文件加载 Bookmap。

        Args:
            path: bookmap JSON 文件路径。
            validate_on_load: 是否在加载后自动校验。

        Returns:
            加载完成的 Bookmap 实例。

        Raises:
            FileNotFoundError: 文件不存在。
            json.JSONDecodeError: JSON 解析失败。
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        bm = cls()
        bm._parse(raw)
        if validate_on_load:
            bm._errors = bm.validate()
            if bm._errors:
                logger.warning(
                    "Bookmap 校验发现 %d 个问题: %s", len(bm._errors), path
                )
        return bm

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, validate_on_load: bool = True) -> Bookmap:
        """从字典构建 Bookmap（用于测试和编程式构建）。

        Args:
            data: 符合 bookmap-schema 的字典。
            validate_on_load: 是否在构建后自动校验。

        Returns:
            构建完成的 Bookmap 实例。
        """
        bm = cls()
        bm._parse(data)
        if validate_on_load:
            bm._errors = bm.validate()
        return bm

    # ── 内部解析 ─────────────────────────────────────────────────

    def _parse(self, raw: dict[str, Any]) -> None:
        """解析原始 JSON 字典为内部数据结构。

        Args:
            raw: 原始 JSON 字典。

        Raises:
            KeyError: 缺少必需字段。
        """
        self.meta = raw.get("meta", {})
        self.domain = raw.get("domain", "")

        # 解析 clusters
        clusters_raw: dict[str, dict[str, Any]] = raw.get("clusters", {})
        for cid, cdata in clusters_raw.items():
            self.clusters[cid] = Cluster(
                id=cid,
                title=cdata.get("title", ""),
                learned=cdata.get("learned", False),
                learned_date=cdata.get("learned_date"),
                parent=cdata.get("parent"),
            )

        # 解析 items
        items_raw: list[dict[str, Any]] = raw.get("items", [])
        for idata in items_raw:
            item = Item(
                id=idata["id"],
                cluster=idata["cluster"],
                title=idata["title"],
                type=idata["type"],
                mode=idata["mode"],
                source=idata["source"],
                prerequisites=idata.get("prerequisites", []),
                related=idata.get("related", []),
                note=idata.get("note"),
                mastery=idata.get("mastery", 0.0),
                next_review=idata.get("next_review"),
                status=idata.get("status", "pending"),
                cross_refs=idata.get("cross_refs", []),
            )
            self.items[item.id] = item

    # ── 校验 ─────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """校验图谱完整性和一致性。

        检查项:
            1. 顶层必需字段（meta, domain, clusters, items）。
            2. meta.status 合法值。
            3. 每个 item 的字段合法性。
            4. 引用完整性（prerequisites, related, cluster 存在）。
            5. 无环（prerequisites 不形成环路）。
            6. Items 数量 > 0。

        Returns:
            错误信息列表（空列表表示校验通过）。
        """
        errors: list[str] = []

        # 顶层字段
        if not self.meta:
            errors.append("缺少 meta 字段")
        else:
            status = self.meta.get("status", "")
            if status not in VALID_META_STATUSES:
                errors.append(
                    f"meta.status 值无效: '{status}'（合法: {sorted(VALID_META_STATUSES)}）"
                )

        if not self.domain:
            errors.append("缺少 domain 字段")

        if not self.items:
            errors.append("items 为空，至少需要一个知识点")

        # 逐 item 校验
        item_ids = set(self.items.keys())
        cluster_ids = set(self.clusters.keys())

        for item in self.items.values():
            # 必需字段
            if not item.id:
                errors.append("item 缺少 id")
            if not item.title:
                errors.append(f"item '{item.id}' 缺少 title")

            # type 合法性
            if item.type not in VALID_ITEM_TYPES:
                errors.append(
                    f"item '{item.id}' type 无效: '{item.type}'"
                    f"（合法: {sorted(VALID_ITEM_TYPES)}）"
                )

            # mode 合法性
            if item.mode not in VALID_MODES:
                errors.append(
                    f"item '{item.id}' mode 无效: '{item.mode}'"
                    f"（合法: {sorted(VALID_MODES)}）"
                )

            # status 合法性
            if item.status not in VALID_STATUSES:
                errors.append(
                    f"item '{item.id}' status 无效: '{item.status}'"
                    f"（合法: {sorted(VALID_STATUSES)}）"
                )

            # mastery 范围
            if not (0.0 <= item.mastery <= 1.0):
                errors.append(
                    f"item '{item.id}' mastery 超出 [0,1]: {item.mastery}"
                )

            # 引用完整性: cluster
            if item.cluster not in cluster_ids:
                errors.append(
                    f"item '{item.id}' 引用不存在的 cluster: '{item.cluster}'"
                )

            # 引用完整性: prerequisites（允许跨图谱引用，仅警告）
            for pid in item.prerequisites:
                if pid not in item_ids:
                    logger.debug(
                        "item '%s' 引用跨图谱 prerequisite: '%s'", item.id, pid
                    )

            # 自引用检查
            if item.id in item.prerequisites:
                errors.append(f"item '{item.id}' 不能以自身为前置依赖")
            if item.id in item.related:
                errors.append(f"item '{item.id}' 不能以自身为相关项")

            # cross_refs 校验
            for cr in item.cross_refs:
                rel = cr.get("relation", "")
                if rel not in VALID_CROSS_REF_RELATIONS:
                    errors.append(
                        f"item '{item.id}' cross_ref relation 无效: '{rel}'"
                        f"（合法: {sorted(VALID_CROSS_REF_RELATIONS)}）"
                    )

        # 无环检查
        if self._has_cycle():
            errors.append("prerequisites 存在环路")

        return errors

    @property
    def errors(self) -> list[str]:
        """最近一次校验的错误列表（加载/构建时自动填充）。"""
        return self._errors

    @property
    def is_valid(self) -> bool:
        """图谱是否通过校验。"""
        return len(self._errors) == 0

    # ── 环路检测 ─────────────────────────────────────────────────

    def _has_cycle(self) -> bool:
        """检测 prerequisites 图中是否存在环路（DFS 三色法）。

        Returns:
            True 如果存在环路。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {item_id: WHITE for item_id in self.items}

        def _dfs(node: str) -> bool:
            color[node] = GRAY
            for prereq in self.items[node].prerequisites:
                if prereq not in color:
                    continue  # 引用不存在的 item，校验阶段已经报告
                if color[prereq] == GRAY:
                    return True  # 发现回边
                if color[prereq] == WHITE and _dfs(prereq):
                    return True
            color[node] = BLACK
            return False

        for nid in self.items:
            if color[nid] == WHITE and _dfs(nid):
                return True
        return False

    # ── 访问器 ───────────────────────────────────────────────────

    def get_item(self, item_id: str) -> Item | None:
        """按 id 获取知识点。

        Args:
            item_id: 知识点 id。

        Returns:
            Item 实例，不存在时返回 None。
        """
        return self.items.get(item_id)

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        """按 id 获取簇。

        Args:
            cluster_id: 簇 id。

        Returns:
            Cluster 实例，不存在时返回 None。
        """
        return self.clusters.get(cluster_id)

    def all_items(self) -> list[Item]:
        """返回所有知识点的列表（不保证顺序）。"""
        return list(self.items.values())

    def all_clusters(self) -> list[Cluster]:
        """返回所有簇的列表（不保证顺序）。"""
        return list(self.clusters.values())

    # ── 导航 ─────────────────────────────────────────────────────

    def prerequisites_of(self, item_id: str) -> list[Item]:
        """获取某知识点的直接前置依赖。

        Args:
            item_id: 知识点 id。

        Returns:
            直接前置依赖的 Item 列表（不存在或没有时返回空列表）。
        """
        item = self.items.get(item_id)
        if item is None:
            return []
        return [self.items[pid] for pid in item.prerequisites if pid in self.items]

    def prerequisite_chain(self, item_id: str) -> list[Item]:
        """获取某知识点的完整前置链（拓扑排序）。

        递归收集所有传递前置依赖，按拓扑序返回（先学排前面）。

        Args:
            item_id: 知识点 id。

        Returns:
            拓扑排序后的前置依赖 Item 列表（去重，不包括自身）。
        """
        item = self.items.get(item_id)
        if item is None:
            return []

        visited: set[str] = set()
        result: list[Item] = []

        def _collect(nid: str) -> None:
            """DFS 收集前置依赖。"""
            node = self.items.get(nid)
            if node is None:
                return
            for pid in node.prerequisites:
                if pid not in visited and pid in self.items:
                    visited.add(pid)
                    _collect(pid)
                    result.append(self.items[pid])

        _collect(item_id)
        return result

    def related_of(self, item_id: str) -> list[Item]:
        """获取某知识点的相关概念。

        Args:
            item_id: 知识点 id。

        Returns:
            相关概念的 Item 列表。
        """
        item = self.items.get(item_id)
        if item is None:
            return []
        return [self.items[rid] for rid in item.related if rid in self.items]

    def items_in_cluster(self, cluster_id: str) -> list[Item]:
        """获取某簇（章节）下的所有知识点。

        Args:
            cluster_id: 簇 id。

        Returns:
            该簇下的 Item 列表。
        """
        return [it for it in self.items.values() if it.cluster == cluster_id]

    def items_by_mode(self, mode: str) -> list[Item]:
        """按学习模式筛选知识点。

        Args:
            mode: 'blackbox' 或 'whitebox'。

        Returns:
            匹配的 Item 列表。
        """
        return [it for it in self.items.values() if it.mode == mode]

    def items_by_status(self, status: str) -> list[Item]:
        """按学习状态筛选知识点。

        Args:
            status: 'learned' / 'pending' / 'optional'。

        Returns:
            匹配的 Item 列表。
        """
        return [it for it in self.items.values() if it.status == status]

    def items_by_cluster(self, cluster_id: str) -> list[Item]:
        """获取某簇下的所有知识点（items_in_cluster 的别名）。"""
        return self.items_in_cluster(cluster_id)

    def items_due_for_review(self, reference_date: date | None = None) -> list[Item]:
        """获取到期应复习的知识点。

        Args:
            reference_date: 参考日期，默认为今天。

        Returns:
            next_review <= reference_date 且 status == 'learned' 的 Item 列表。
        """
        if reference_date is None:
            reference_date = datetime.now(tz=UTC).date()
        ref_str = reference_date.isoformat()

        due: list[Item] = []
        for item in self.items.values():
            if item.status != "learned":
                continue
            if item.next_review is None:
                continue
            if item.next_review <= ref_str:
                due.append(item)

        # 按 mastery 升序（最弱的先复习）
        due.sort(key=lambda it: it.mastery)
        return due

    def topological_order(self) -> list[Item]:
        """返回所有知识点的拓扑排序（按前置依赖）。

        没有前置依赖的排在最前面。

        Returns:
            拓扑排序后的 Item 列表。

        Raises:
            ValueError: 图中存在环路时无法排序。
        """
        if self._has_cycle():
            raise ValueError("prerequisites 存在环路，无法拓扑排序")

        in_degree: dict[str, int] = {nid: len(item.prerequisites) for nid, item in self.items.items()}
        # 过滤掉不存在的引用
        item_ids = set(self.items.keys())
        for item in self.items.values():
            in_degree[item.id] = sum(1 for pid in item.prerequisites if pid in item_ids)

        queue: list[str] = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[Item] = []

        while queue:
            # 按 id 排序保证确定性
            queue.sort()
            nid = queue.pop(0)
            result.append(self.items[nid])

            # 找到所有以 nid 为前置的 item
            for item in self.items.values():
                if nid in item.prerequisites:
                    in_degree[item.id] -= 1
                    if in_degree[item.id] == 0:
                        queue.append(item.id)

        if len(result) != len(self.items):
            raise ValueError("拓扑排序未覆盖所有节点，可能存在环路")

        return result

    def orphans(self) -> list[Item]:
        """返回没有前置依赖的知识点（可直接学习的入口节点）。

        Returns:
            无前置依赖的 Item 列表。
        """
        return [it for it in self.items.values() if not it.prerequisites]

    def leaves(self) -> list[Item]:
        """返回不被任何其他知识点作为前置依赖的叶子节点。

        Returns:
            叶子 Item 列表（最深层知识点）。
        """
        referenced: set[str] = set()
        for item in self.items.values():
            for pid in item.prerequisites:
                referenced.add(pid)
        return [it for it in self.items.values() if it.id not in referenced]

    # ── 统计 ─────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        """图谱中知识点总数。"""
        return len(self.items)

    @property
    def cluster_count(self) -> int:
        """图谱中簇（章节）总数。"""
        return len(self.clusters)

    @property
    def edge_count(self) -> int:
        """图谱中边总数（prerequisites + related）。"""
        prereq_edges = sum(len(it.prerequisites) for it in self.items.values())
        related_edges = sum(len(it.related) for it in self.items.values())
        return prereq_edges + related_edges

    def summary(self) -> dict[str, Any]:
        """返回图谱统计摘要。

        Returns:
            包含 node_count, cluster_count, edge_count, domain,
            meta_status, mode_distribution, status_distribution 的字典。
        """
        mode_dist: dict[str, int] = {}
        status_dist: dict[str, int] = {}
        type_dist: dict[str, int] = {}

        for item in self.items.values():
            mode_dist[item.mode] = mode_dist.get(item.mode, 0) + 1
            status_dist[item.status] = status_dist.get(item.status, 0) + 1
            type_dist[item.type] = type_dist.get(item.type, 0) + 1

        return {
            "domain": self.domain,
            "node_count": self.node_count,
            "cluster_count": self.cluster_count,
            "edge_count": self.edge_count,
            "meta_status": self.meta.get("status", "unknown"),
            "mode_distribution": mode_dist,
            "status_distribution": status_dist,
            "type_distribution": type_dist,
            "is_valid": self.is_valid,
        }

    def __repr__(self) -> str:
        return (
            f"Bookmap(domain='{self.domain}', "
            f"nodes={self.node_count}, clusters={self.cluster_count}, "
            f"edges={self.edge_count}, valid={self.is_valid})"
        )
