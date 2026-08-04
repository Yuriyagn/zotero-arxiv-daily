"""Read-only smoke test for Zotero access and arXiv retrieval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from pyzotero import zotero

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = "cs.AI,cs.CV,cs.LG,cs.CL"


def parse_categories(value: str) -> list[str]:
    categories = [category.strip() for category in value.split(",") if category.strip()]
    if not categories:
        raise ValueError("At least one arXiv category is required.")
    return categories


def check_zotero(limit: int) -> None:
    user_id = os.getenv("ZOTERO_ID")
    api_key = os.getenv("ZOTERO_KEY")
    if not user_id or not api_key:
        raise RuntimeError(
            "ZOTERO_ID and ZOTERO_KEY are required. Copy .env.example to .env "
            "and fill in a Zotero API key with read access."
        )

    client = zotero.Zotero(user_id, "user", api_key)
    items = client.items(
        itemType="conferencePaper || journalArticle || preprint",
        limit=limit,
    )
    papers = [item["data"] for item in items if item.get("data")]

    logger.info("Zotero connection succeeded; received {} paper items.", len(papers))
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title") or "<untitled>"
        has_abstract = bool(paper.get("abstractNote"))
        logger.info("Zotero {}: {} (abstract: {})", index, title, "yes" if has_abstract else "no")


def check_arxiv(categories: list[str], limit: int) -> None:
    config = OmegaConf.create(
        {
            "source": {
                "arxiv": {
                    "category": categories,
                    "include_cross_list": False,
                }
            },
            "executor": {"debug": True},
        }
    )
    papers = ArxivRetriever(config)._retrieve_raw_papers()[:limit]

    logger.info(
        "arXiv connection succeeded; received {} recent papers for {}.",
        len(papers),
        ", ".join(categories),
    )
    for index, paper in enumerate(papers, start=1):
        logger.info("arXiv {}: {} ({})", index, paper.title, paper.entry_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Zotero and arXiv access without downloading full text, "
            "calling an LLM, reranking, or sending email."
        )
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum items to display per source.")
    parser.add_argument(
        "--categories",
        default=os.getenv("ARXIV_CATEGORIES", DEFAULT_CATEGORIES),
        help="Comma-separated arXiv categories.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--zotero-only", action="store_true", help="Check only Zotero.")
    mode.add_argument("--arxiv-only", action="store_true", help="Check only arXiv.")
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        args = build_parser().parse_args()
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")

        categories = parse_categories(args.categories)
        if not args.arxiv_only:
            check_zotero(args.limit)
        if not args.zotero_only:
            check_arxiv(categories, args.limit)
        return 0
    except Exception as exc:
        logger.error("Source check failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
