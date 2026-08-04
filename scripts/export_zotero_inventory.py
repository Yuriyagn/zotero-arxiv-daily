"""Export the Zotero papers relevant to similarity ranking as Markdown."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pyzotero import zotero


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "zotero_inventory.md"
PAPER_ITEM_TYPES = "conferencePaper || journalArticle || preprint"


def markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("\n", " ").strip()


def build_collection_paths(collections: dict[str, dict]) -> dict[str, str]:
    paths: dict[str, str] = {}

    def resolve(key: str) -> str:
        if key in paths:
            return paths[key]
        data = collections[key]["data"]
        parent = data.get("parentCollection")
        paths[key] = f"{resolve(parent)}/{data['name']}" if parent else data["name"]
        return paths[key]

    for collection_key in collections:
        resolve(collection_key)
    return paths


def export_inventory(output_path: Path) -> tuple[int, int]:
    user_id = os.getenv("ZOTERO_ID")
    api_key = os.getenv("ZOTERO_KEY")
    if not user_id or not api_key:
        raise RuntimeError("ZOTERO_ID and ZOTERO_KEY must be set in .env.")

    client = zotero.Zotero(user_id, "user", api_key)
    collections = {item["key"]: item for item in client.everything(client.collections())}
    collection_paths = build_collection_paths(collections)
    items = client.everything(client.items(itemType=PAPER_ITEM_TYPES))

    eligible = [item for item in items if item["data"].get("abstractNote", "")]
    automatic_exclusions = [item for item in items if not item["data"].get("abstractNote", "")]
    eligible.sort(key=lambda item: (item["data"].get("title") or "").casefold())
    automatic_exclusions.sort(key=lambda item: (item["data"].get("title") or "").casefold())

    paper_ids = {item["key"]: f"P{index:03d}" for index, item in enumerate(eligible, start=1)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    paper_paths: dict[str, list[str]] = {}
    for item in eligible:
        paths = sorted(
            collection_paths[key]
            for key in item["data"].get("collections", [])
            if key in collection_paths
        )
        if not paths:
            paths = ["[未分类]"]
        paper_paths[item["key"]] = paths
        for path in paths:
            grouped[path].append(item)

    lines = [
        "# Zotero 相似度语料清单",
        "",
        f"- 会进入相似度排序：{len(eligible)} 篇（具有摘要）",
        f"- 自动排除：{len(automatic_exclusions)} 篇（没有摘要）",
        f"- Zotero 论文条目总数：{len(items)} 篇",
        "",
        "> 同一篇论文可能属于多个目录，因此目录计数之和可能大于论文总数。",
        "",
        "## 按目录查看",
        "",
    ]

    for path in sorted(grouped, key=str.casefold):
        lines.extend([f"### {markdown_text(path)}（{len(grouped[path])} 篇）", ""])
        for item in grouped[path]:
            title = markdown_text(item["data"].get("title") or "<无标题>")
            lines.append(f"- `{paper_ids[item['key']]}` {title}")
        lines.append("")

    lines.extend(["## 全部会参与排序的论文", ""])
    for item in eligible:
        title = markdown_text(item["data"].get("title") or "<无标题>")
        paths = "；".join(markdown_text(path) for path in paper_paths[item["key"]])
        lines.append(f"- `{paper_ids[item['key']]}` {title} — {paths}")

    lines.extend(["", "## 已因缺少摘要而自动排除", ""])
    for item in automatic_exclusions:
        title = markdown_text(item["data"].get("title") or "<无标题>")
        lines.append(f"- {title}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(eligible), len(automatic_exclusions)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    eligible, excluded = export_inventory(args.output)
    print(f"Exported {eligible} eligible papers and {excluded} automatic exclusions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
