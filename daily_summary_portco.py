#!/usr/bin/env python3
"""Infer per-portco daily activity from ai-generated/portco.json via Groq.

Import and call ``generate_daily_summary_portco(day)`` to get a JSON-serializable
list of ``{"portco", "summary"}`` dicts. Run as a script to write that JSON under
``GOOGLE_DRIVE_BASE/ai-generated/dailies/portcos/YYYY-MM-DD.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from daily_summary import (
    DEFAULT_MODEL,
    default_date,
    filter_entries_for_day,
    load_deal_entries,
    parse_day,
    resolve_day,
    strip_claude_stats,
)
from paths import list_company_folders, portcos_base, shared_ai_dir
from process_portco import PORTCO_JSON_NAME

__all__ = ["generate_daily_summary_portco"]

SUMMARY_INSTRUCTIONS = """You are an expert VC analyst working for us, Antler Canada. You infer what happened with a portfolio company on a given day.

You are given metadata entries (emails, transcripts, notes, etc.) whose created_at
falls on that day. Write a single short paragraph summarizing what happened with
this portfolio company that day.

Rules:
- Stick to facts supported by the entries; do not invent details.
- For team members (Bernie, Tammer, Alex, Shambhavi, Daphne, Matt), use first names only
  ("Alex" NOT "Alex Wright").
- For everyone else, use full names when available.
- Prefer concrete events: meetings held, emails sent, notes written, next steps.
- If activity is thin, say so briefly based on what is present.
- Output plain text only — no headings, bullets, or markdown.
- There is no need to mention the date in the summary.

Example:
Tammer and Alex met the team and talked about PMF. They concluded that more work is needed to get to PMF.
"""


def portco_json_path(folder: Path) -> Path:
    return folder / "ai-generated" / PORTCO_JSON_NAME


def summarize_portco_day(
    *,
    portco_name: str,
    day: date,
    entries: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> str:
    payload = json.dumps(strip_claude_stats(entries), indent=2, ensure_ascii=False)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SUMMARY_INSTRUCTIONS.strip()},
            {
                "role": "user",
                "content": (
                    f"Portfolio company: {portco_name}\n"
                    f"Date: {day.isoformat()}\n\n"
                    "Entries for this day:\n"
                    f"{payload}\n"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def generate_daily_summary_portco(day: date | str) -> list[dict[str, str]]:
    """Return JSON-ready per-portco summaries for a calendar day.

    Args:
        day: ``YYYY-MM-DD`` string or a ``date``.

    Returns:
        A list of ``{"portco": "...", "summary": "..."}`` dicts for portcos with
        activity that day. Progress goes to stderr; failed portcos are omitted.
        If the portcos root is missing, prints a warning and returns an empty list.
    """
    load_dotenv()
    resolved = resolve_day(day)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    base = portcos_base()
    if not base.is_dir():
        print(
            f"Warning: portcos folder not found or not a directory: {base}",
            file=sys.stderr,
        )
        return []

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    day_label = resolved.isoformat()
    results: list[dict[str, str]] = []

    for folder in list_company_folders(base):
        portco_name = folder.name
        path = portco_json_path(folder)
        entries = load_deal_entries(path)
        if entries is None:
            continue

        day_entries = filter_entries_for_day(entries, resolved)
        if not day_entries:
            continue

        print(
            f"Summarizing {portco_name} ({len(day_entries)} entr"
            f"{'y' if len(day_entries) == 1 else 'ies'} on {day_label})...",
            file=sys.stderr,
        )
        try:
            summary_text = summarize_portco_day(
                portco_name=portco_name,
                day=resolved,
                entries=day_entries,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            print(f"Error summarizing {portco_name}: {exc}", file=sys.stderr)
            continue

        if not summary_text:
            print(f"Skipping {portco_name}: empty summary", file=sys.stderr)
            continue

        results.append({"portco": portco_name, "summary": summary_text})

    if not results:
        print(f"No portco activity on {day_label}.", file=sys.stderr)

    return results


def main() -> int:
    """CLI entry point: write daily summary JSON under ai-generated/dailies/portcos/."""
    parser = argparse.ArgumentParser(
        description=(
            "For each portco, filter ai-generated/portco.json to a calendar day "
            "and write {portco, summary} JSON to "
            "GOOGLE_DRIVE_BASE/ai-generated/dailies/portcos/YYYY-MM-DD.json."
        )
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help=(
            "Calendar day as YYYY-MM-DD "
            "(default: yesterday before 16:30 local, otherwise today)"
        ),
    )
    args = parser.parse_args()

    try:
        day = parse_day(args.date) if args.date is not None else default_date()
        results = generate_daily_summary_portco(day)
        output_dir = shared_ai_dir() / "dailies" / "portcos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{day.isoformat()}.json"
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
