#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from html_utils import load_styles, markdown_to_html

TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")


def resolve_google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError(
            "GOOGLE_DRIVE_BASE is not set. "
            "Set it to the root directory containing deal folders and status.md."
        )
    return Path(base_raw).expanduser().resolve()


def resolve_website_dir() -> Path:
    base_raw = os.getenv("WEBSITE_BASE")
    if not base_raw:
        raise ValueError(
            "WEBSITE_BASE is not set. "
            "Set it to the parent directory where the website/ output folder should be created."
        )
    return Path(base_raw).expanduser().resolve() / "website"


def list_deal_folders(base: Path) -> list[Path]:
    return sorted(
        entry
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def summary_path_for_deal(deal_folder: Path) -> Path:
    return deal_folder / "ai-generated" / "summary.md"


def delete_html_files(website_dir: Path) -> int:
    removed = 0
    for path in website_dir.glob("*.html"):
        path.unlink()
        removed += 1
    return removed


def split_table_row(line: str) -> list[str] | None:
    match = TABLE_ROW_RE.match(line)
    if not match:
        return None

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in match.group(1):
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def link_deal_names_in_status_table(
    markdown_text: str,
    linked_deal_names: set[str],
) -> str:
    lines = markdown_text.splitlines()
    if not lines:
        return markdown_text

    output_lines: list[str] = []
    in_table = False

    for line in lines:
        if not TABLE_ROW_RE.match(line):
            in_table = False
            output_lines.append(line)
            continue

        if TABLE_SEPARATOR_RE.match(line):
            in_table = True
            output_lines.append(line)
            continue

        cells = split_table_row(line)
        if cells is None:
            output_lines.append(line)
            continue

        if in_table and cells:
            deal_name = cells[0]
            if deal_name in linked_deal_names:
                cells[0] = f"[{deal_name}]({deal_name}.html)"

        output_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(output_lines) + ("\n" if markdown_text.endswith("\n") else "")


def build_website_page(title: str, body_html: str, *, is_index: bool) -> str:
    styles = load_styles()
    page_title = html.escape(title)
    nav_html = "" if is_index else '    <a class="back-link" href="index.html">← All deals</a>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
{styles}
  </style>
</head>
<body>
  <main>
{nav_html}    <div class="content">
{body_html}
    </div>
  </main>
</body>
</html>
"""


def generate_deal_pages(
    deal_folders: list[Path],
    website_dir: Path,
) -> tuple[set[str], int]:
    linked_deal_names: set[str] = set()
    written = 0

    for deal_folder in deal_folders:
        summary_path = summary_path_for_deal(deal_folder)
        if not summary_path.is_file():
            print(
                f"Warning: skipping {deal_folder.name} (no ai-generated/summary.md)",
                file=sys.stderr,
            )
            continue

        summary_text = summary_path.read_text(encoding="utf-8")
        body_html = markdown_to_html(summary_text, demote_h1=False)
        document_html = build_website_page(
            deal_folder.name,
            body_html,
            is_index=False,
        )

        output_path = website_dir / f"{deal_folder.name}.html"
        output_path.write_text(document_html, encoding="utf-8")
        linked_deal_names.add(deal_folder.name)
        written += 1
        print(f"Wrote {output_path.name}", file=sys.stderr)

    return linked_deal_names, written


def generate_index_page(
    base: Path,
    website_dir: Path,
    linked_deal_names: set[str],
) -> None:
    status_path = base / "status.md"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"status.md not found at {status_path}. Run summarizer.py first."
        )

    status_text = status_path.read_text(encoding="utf-8")
    linked_status = link_deal_names_in_status_table(status_text, linked_deal_names)
    body_html = markdown_to_html(linked_status, demote_h1=False)
    document_html = build_website_page("Deal Portfolio", body_html, is_index=True)

    output_path = website_dir / "index.html"
    output_path.write_text(document_html, encoding="utf-8")
    print(f"Wrote {output_path.name}", file=sys.stderr)


def main() -> int:
    load_dotenv()

    try:
        base = resolve_google_drive_base()
        website_dir = resolve_website_dir()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.is_dir():
        print(f"Error: GOOGLE_DRIVE_BASE is not a directory: {base}", file=sys.stderr)
        return 1

    website_dir.mkdir(parents=True, exist_ok=True)

    removed = delete_html_files(website_dir)
    if removed:
        print(f"Removed {removed} existing HTML file(s)", file=sys.stderr)

    deal_folders = list_deal_folders(base)
    linked_deal_names, deal_count = generate_deal_pages(deal_folders, website_dir)

    try:
        generate_index_page(base, website_dir, linked_deal_names)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done: wrote index.html and {deal_count} deal page(s) to {website_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
