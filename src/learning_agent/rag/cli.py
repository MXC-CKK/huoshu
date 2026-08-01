"""RAG 模块 CLI 验证入口。

支持两个子命令:
    python -m learning_agent.rag.cli ingest <pdf_path> [--name COLLECTION] [--source NAME]
    python -m learning_agent.rag.cli search <query> [--name COLLECTION] [--top-k K]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _cmd_ingest(args: argparse.Namespace) -> None:
    """PDF 入库子命令。"""
    from learning_agent.rag.ingest import ingest_pdf

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("PDF 文件不存在: %s", pdf_path)
        sys.exit(1)

    try:
        collection = ingest_pdf(
            pdf_path=pdf_path,
            collection_name=args.name,
            source_name=args.source or pdf_path.stem,
        )
        count = collection.count()
        print(f"✓ 入库完成: {count} 个 chunks → collection '{args.name}'")
    except Exception as exc:  # noqa: BLE001 - CLI 入口需兜底所有错误并退出
        logger.error("入库失败: %s", exc)
        sys.exit(1)


def _cmd_search(args: argparse.Namespace) -> None:
    """检索子命令。"""
    from learning_agent.rag.retrieve import format_results, open_collection, query

    try:
        collection = open_collection(args.name)
    except RuntimeError as exc:
        logger.error("打开集合失败: %s", exc)
        sys.exit(1)

    results = query(collection, args.query, top_k=args.top_k)
    if not results:
        print("(无结果)")
        return

    print(format_results(results))


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="活书 huoshu RAG — PDF 入库与语义检索",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest 子命令
    p_ingest = subparsers.add_parser("ingest", help="PDF 入库")
    p_ingest.add_argument("pdf_path", help="PDF 文件路径")
    p_ingest.add_argument("--name", default="default", help="ChromaDB 集合名称 (default: default)")
    p_ingest.add_argument("--source", default="", help="来源名称 (default: PDF 文件名)")

    # search 子命令
    p_search = subparsers.add_parser("search", help="语义检索")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("--name", default="default", help="ChromaDB 集合名称 (default: default)")
    p_search.add_argument("--top-k", type=int, default=5, help="返回结果数 (default: 5)")

    args = parser.parse_args()

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "search":
        _cmd_search(args)


if __name__ == "__main__":
    main()
