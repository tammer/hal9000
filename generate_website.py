#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from html_utils import load_styles, markdown_to_html

TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")
DAILY_JSON_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def resolve_google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError(
            "GOOGLE_DRIVE_BASE is not set. "
            "Set it to the root directory containing deal folders and ai-generated/status.md."
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
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name != "ai-generated"
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


def build_website_page(
    title: str,
    body_html: str,
    *,
    back_href: str | None = None,
    back_label: str | None = None,
) -> str:
    styles = load_styles()
    page_title = html.escape(title)
    if back_href and back_label:
        nav_html = (
            f'    <a class="back-link" href="{html.escape(back_href, quote=True)}">'
            f"{html.escape(back_label)}</a>\n"
        )
    else:
        nav_html = ""

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
        body_html = (
            f"<h1>{html.escape(deal_folder.name)}</h1>\n"
            + markdown_to_html(summary_text, demote_h1=False)
        )
        document_html = build_website_page(
            deal_folder.name,
            body_html,
            back_href="deals.html",
            back_label="← All deals",
        )

        output_path = website_dir / f"{deal_folder.name}.html"
        output_path.write_text(document_html, encoding="utf-8")
        linked_deal_names.add(deal_folder.name)
        written += 1
        print(f"Wrote {output_path.name}", file=sys.stderr)

    return linked_deal_names, written


def generate_deals_page(
    base: Path,
    website_dir: Path,
    linked_deal_names: set[str],
) -> None:
    status_path = base / "ai-generated" / "status.md"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"status.md not found at {status_path}. Run summarizer.py first."
        )

    status_text = status_path.read_text(encoding="utf-8")
    linked_status = link_deal_names_in_status_table(status_text, linked_deal_names)
    body_html = markdown_to_html(linked_status, demote_h1=False)
    document_html = build_website_page(
        "Deal Portfolio",
        body_html,
        back_href="index.html",
        back_label="← Home",
    )

    output_path = website_dir / "deals.html"
    output_path.write_text(document_html, encoding="utf-8")
    print(f"Wrote {output_path.name}", file=sys.stderr)


def list_recent_daily_files(dailies_dir: Path, limit: int = 5) -> list[Path]:
    if not dailies_dir.is_dir():
        return []

    dated: list[tuple[str, Path]] = []
    for path in dailies_dir.glob("*.json"):
        match = DAILY_JSON_RE.match(path.name)
        if match:
            dated.append((match.group(1), path))

    dated.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in dated[:limit]]


def render_daily_summaries_html(
    daily_files: list[Path],
    linked_deal_names: set[str],
) -> str:
    if not daily_files:
        return "<p>No daily summaries yet.</p>"

    sections: list[str] = []
    for path in daily_files:
        match = DAILY_JSON_RE.match(path.name)
        if not match:
            continue
        day = date.fromisoformat(match.group(1))
        day_label = f"{day.strftime('%A, %B')} {day.day}"

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(raw, list):
            print(f"Warning: {path} is not a JSON list; skipping", file=sys.stderr)
            continue

        parts = [f"<h1>{html.escape(day_label)}</h1>"]
        for item in raw:
            if not isinstance(item, dict):
                continue
            deal = item.get("deal")
            summary = item.get("summary")
            if not isinstance(deal, str) or not isinstance(summary, str):
                continue
            if deal in linked_deal_names:
                deal_html = (
                    f'<a href="{html.escape(deal, quote=True)}.html">'
                    f"{html.escape(deal)}</a>"
                )
            else:
                deal_html = html.escape(deal)
            parts.append(f"<h2>{deal_html}</h2>")
            parts.append(f"<p>{html.escape(summary)}</p>")

        sections.append("\n".join(parts))

    if not sections:
        return "<p>No daily summaries yet.</p>"
    return "\n".join(sections)


def generate_dailys_page(
    base: Path,
    website_dir: Path,
    linked_deal_names: set[str],
) -> None:
    dailies_dir = base / "ai-generated" / "dailies" / "deals"
    daily_files = list_recent_daily_files(dailies_dir, limit=5)
    body_html = render_daily_summaries_html(daily_files, linked_deal_names)
    document_html = build_website_page(
        "Daily Summaries",
        body_html,
        back_href="index.html",
        back_label="← Home",
    )

    output_path = website_dir / "dailys.html"
    output_path.write_text(document_html, encoding="utf-8")
    print(f"Wrote {output_path.name}", file=sys.stderr)


def generate_index_page(website_dir: Path) -> None:
    body_html = """<h1>Deal Hub</h1>
<ul>
  <li><a href="deals.html">Deals</a></li>
  <li><a href="dailys.html">Activity</a></li>
</ul>"""
    document_html = build_website_page("Deal Hub", body_html)

    output_path = website_dir / "index.html"
    output_path.write_text(document_html, encoding="utf-8")
    print(f"Wrote {output_path.name}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the deal portfolio website from summaries."
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Publish the generated site to ICDSoft over SFTP after generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        generate_deals_page(base, website_dir, linked_deal_names)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    generate_dailys_page(base, website_dir, linked_deal_names)
    generate_index_page(website_dir)

    print(
        f"Done: wrote index.html, deals.html, dailys.html, and {deal_count} "
        f"deal page(s) to {website_dir}",
        file=sys.stderr,
    )

    if args.deploy:
        from website_deploy import deploy_website

        return deploy_website(website_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
