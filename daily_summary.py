#!/usr/bin/env python3
"""Infer per-deal daily activity from ai-generated/deal.json via Groq.

Import and call ``generate_daily_summary(day)`` to get a JSON-serializable list of
``{"deal", "summary"}`` dicts. Run as a script to write that JSON under
``GOOGLE_DRIVE_BASE/ai-generated/dailies/deals/YYYY-MM-DD.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from consolidator import DATETIME_FMT

__all__ = ["generate_daily_summary"]

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEAL_JSON_NAME = "deal.json"


def default_date() -> date:
    """Yesterday before 16:30 local time; today at or after 16:30."""
    now = datetime.now()
    cutoff = now.replace(hour=16, minute=30, second=0, microsecond=0)
    if now >= cutoff:
        return now.date()
    return now.date() - timedelta(days=1)

SUMMARY_INSTRUCTIONS = """You infer what happened with a venture capital deal on a given day.

You are given metadata entries (emails, transcripts, notes, etc.) whose created_at
falls on that day. Write a single short paragraph summarizing what happened with
this deal that day.

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


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def resolve_day(day: date | str) -> date:
    if isinstance(day, date):
        return day
    return parse_day(day)


def resolve_google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    return Path(base_raw).expanduser().resolve()


def list_deal_folders(base: Path) -> list[Path]:
    return sorted(
        entry
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def deal_json_path(folder: Path) -> Path:
    return folder / "ai-generated" / DEAL_JSON_NAME


def parse_created_at_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.lower() == "unknown":
        return None
    try:
        return datetime.strptime(raw, DATETIME_FMT).date()
    except ValueError:
        return None


def strip_claude_stats(entries: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {key: value for key, value in item.items() if key != "claude_stats"}
        )
    return cleaned


def load_deal_entries(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(raw, list):
        print(f"Warning: {path} is not a JSON list; skipping", file=sys.stderr)
        return None
    return [item for item in raw if isinstance(item, dict)]


def filter_entries_for_day(
    entries: list[dict[str, Any]], day: date
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for entry in entries:
        created = parse_created_at_date(entry.get("created_at"))
        if created == day:
            matched.append(entry)
    return matched


def summarize_deal_day(
    *,
    deal_name: str,
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
                    f"Deal: {deal_name}\n"
                    f"Date: {day.isoformat()}\n\n"
                    "Entries for this day:\n"
                    f"{payload}\n"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def generate_daily_summary(day: date | str) -> list[dict[str, str]]:
    """Return JSON-ready per-deal summaries for a calendar day.

    Args:
        day: ``YYYY-MM-DD`` string or a ``date``.

    Returns:
        A list of ``{"deal": "...", "summary": "..."}`` dicts for deals with
        activity that day. Progress goes to stderr; failed deals are omitted.
    """
    load_dotenv()
    resolved = resolve_day(day)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    base = resolve_google_drive_base()
    if not base.is_dir():
        raise NotADirectoryError(f"GOOGLE_DRIVE_BASE is not a directory: {base}")

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    day_label = resolved.isoformat()
    results: list[dict[str, str]] = []

    for folder in list_deal_folders(base):
        deal_name = folder.name
        path = deal_json_path(folder)
        entries = load_deal_entries(path)
        if entries is None:
            continue

        day_entries = filter_entries_for_day(entries, resolved)
        if not day_entries:
            continue

        print(
            f"Summarizing {deal_name} ({len(day_entries)} entr"
            f"{'y' if len(day_entries) == 1 else 'ies'} on {day_label})...",
            file=sys.stderr,
        )
        try:
            summary_text = summarize_deal_day(
                deal_name=deal_name,
                day=resolved,
                entries=day_entries,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            print(f"Error summarizing {deal_name}: {exc}", file=sys.stderr)
            continue

        if not summary_text:
            print(f"Skipping {deal_name}: empty summary", file=sys.stderr)
            continue

        results.append({"deal": deal_name, "summary": summary_text})

    if not results:
        print(f"No deal activity on {day_label}.", file=sys.stderr)

    return results


def main() -> int:
    """CLI entry point: write daily summary JSON under ai-generated/dailies/deals/."""
    parser = argparse.ArgumentParser(
        description=(
            "For each deal, filter ai-generated/deal.json to a calendar day "
            "and write {deal, summary} JSON to "
            "GOOGLE_DRIVE_BASE/ai-generated/dailies/deals/YYYY-MM-DD.json."
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
        results = generate_daily_summary(day)
        output_dir = resolve_google_drive_base() / "ai-generated" / "dailies" / "deals"
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
