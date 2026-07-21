#!/usr/bin/env python3
"""Consolidate Antler team notes from a deal folder into dated JSON entries."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from document_utils import read_file_as_text
from fetch_transcripts import TRANSCRIPT_FILENAME_MARKER, groq_json_chat
from generate_contents import resolve_folder_path

DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMAIL_FILENAME_PREFIX = "email_"
ALLOWED_EXTENSIONS = frozenset({".md", ".txt", ".docx", ".gdoc"})
ALLOWED_AUTHORS = frozenset(
    {"Tammer", "Bernie", "Shambhavi", "Alex", "Matt", "Daphne"}
)
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

AUTHOR_ALIASES: dict[str, str] = {
    "tammer": "Tammer",
    "tammer kamel": "Tammer",
    "tk": "Tammer",
    "bernie": "Bernie",
    "bernie li": "Bernie",
    "bl": "Bernie",
    "shambhavi": "Shambhavi",
    "shambhavi mishra": "Shambhavi",
    "sm": "Shambhavi",
    "alex": "Alex",
    "alex wright": "Alex",
    "aw": "Alex",
    "matt": "Matt",
    "daphne": "Daphne",
    "daphne mclarty": "Daphne",
    "dm": "Daphne",
}

SYSTEM_PROMPT = """You extract dated note entries written by Antler team members from a single deal-folder document.

Return valid JSON only with this exact shape:
{
  "is_team_note": true,
  "entries": [
    {
      "datetime": "2026-01-01 12:00:00",
      "author": "Tammer",
      "content": "note text for this entry"
    }
  ]
}

## What counts as a team note

Set is_team_note=true ONLY when the document is an internal note written BY Antler / the investing team ABOUT the startup (meeting notes, diligence notes, personal scratch notes, open questions, assessments).

Set is_team_note=false and entries=[] for:
- Raw meeting/call transcripts (speaker turns, verbatim dialogue)
- Founder-generated materials (pitch decks, one-pagers, company memos in founder voice)
- Emails or email dumps
- Purely administrative or empty files
- Anything not clearly authored by the Antler team

## Entries

- If is_team_note is false, entries must be [].
- If the file contains multiple distinct dated notes, return one entry per note.
- If the file is one undated or single-dated team note, return one entry with the full note content.
- content must be the note text for that entry (not unrelated boilerplate). Preserve meaning; do not invent facts.
- author must be exactly one of: Tammer, Bernie, Shambhavi, Alex, Matt, Daphne, or unknown.
  Map initials/full names: TK/Tammer Kamel→Tammer, BL/Bernie Li→Bernie, SM/Shambhavi Mishra→Shambhavi, AW/Alex Wright→Alex, DM/Daphne McLarty→Daphne, Matt→Matt.
  Infer author from document text or filename when clear; otherwise "unknown".
- datetime must be "YYYY-MM-DD HH:MM:SS" when a clear date is available, otherwise the string "unknown".
  Sources for datetime, in order:
  1. An explicit date in the document text (e.g. "2026-07-14", "July 14, 2026", "July 14th", "Jul 20").
  2. A timestamp or date embedded in the filename (the Filename field is provided for this).
  If neither the document nor the filename gives a clear date, set datetime to "unknown".
  If a date has a month/day but no year, assume the current year (provided in the user message).
  If only a date is known and no time, use 00:00:00.
"""


class NoteEntry(TypedDict):
    datetime: str
    author: str
    content: str
    source: str


class ConsolidateResult(TypedDict):
    files: list[str]
    entries: list[NoteEntry]


def should_skip_filename(name: str) -> bool:
    if name.startswith(".") or name.startswith("~$"):
        return True
    if name == "contents.json":
        return True
    if name.startswith(EMAIL_FILENAME_PREFIX):
        return True
    if TRANSCRIPT_FILENAME_MARKER in name:
        return True
    return False


def list_candidate_note_files(folder: Path) -> list[Path]:
    candidates: list[Path] = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        if should_skip_filename(entry.name):
            continue
        candidates.append(entry)
    return candidates


def normalize_author(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    cleaned = value.strip()
    if not cleaned:
        return "unknown"
    if cleaned in ALLOWED_AUTHORS:
        return cleaned
    mapped = AUTHOR_ALIASES.get(cleaned.lower())
    if mapped:
        return mapped
    # First token only (e.g. "Tammer notes")
    first = cleaned.split()[0]
    mapped = AUTHOR_ALIASES.get(first.lower())
    if mapped:
        return mapped
    return "unknown"


def file_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def format_datetime(value: datetime) -> str:
    return value.strftime(DATETIME_FMT)


def parse_llm_datetime(value: object, *, default_year: int | None = None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None

    for fmt in (
        DATETIME_FMT,
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b-%d-%Y",
        "%b %d %Y",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in {
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%b %d, %Y",
                "%B %d, %Y",
                "%b-%d-%Y",
                "%b %d %Y",
            }:
                return parsed.replace(hour=0, minute=0, second=0)
            return parsed
        except ValueError:
            continue

    # Month/day without year — assume default_year (current year).
    year = default_year if default_year is not None else datetime.now().year
    for fmt in ("%b %d", "%B %d", "%b-%d", "%B-%d", "%m/%d", "%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(year=year, hour=0, minute=0, second=0)
        except ValueError:
            continue

    # "July 14th" / "Jul 20th"
    ordinal = re.sub(r"(?i)(\d+)(st|nd|rd|th)\b", r"\1", text)
    if ordinal != text:
        return parse_llm_datetime(ordinal, default_year=year)

    return None


def resolve_entry_datetime(
    llm_value: object,
    *,
    mtime: datetime,
) -> str:
    """Use LLM datetime when present; otherwise fall back to file mtime."""
    parsed = parse_llm_datetime(llm_value, default_year=datetime.now().year)
    if parsed is not None:
        return format_datetime(parsed)
    return format_datetime(mtime)


def build_user_prompt(path: Path, text: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    year = datetime.now().year
    return (
        f"Filename: {path.name}\n"
        "(The filename may itself contain a date or timestamp — use it if the "
        "document body has no clear date.)\n"
        f"Today's date: {today}. Current year: {year}. "
        f"If a date has no year, assume {year}.\n\n"
        f"Document content:\n{text}"
    )


def extract_entries_from_file(
    path: Path,
    text: str,
    *,
    api_key: str,
    model: str,
) -> list[NoteEntry]:
    payload = groq_json_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(path, text),
        api_key=api_key,
        model=model,
    )

    if not isinstance(payload, dict):
        return []

    if not payload.get("is_team_note", False):
        return []

    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        return []

    mtime = file_mtime(path)
    entries: list[NoteEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        entries.append(
            {
                "datetime": resolve_entry_datetime(
                    item.get("datetime"), mtime=mtime
                ),
                "author": normalize_author(item.get("author")),
                "content": content,
                "source": path.name,
            }
        )
    return entries


def consolidate(
    relative_path: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> ConsolidateResult:
    folder = resolve_folder_path(relative_path)
    if not folder.exists():
        raise FileNotFoundError(f"path does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"path is not a directory: {folder}")

    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY is not set")
    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    processed_files: list[str] = []
    all_entries: list[NoteEntry] = []

    for path in list_candidate_note_files(folder):
        text = read_file_as_text(path)
        if text is None:
            print(f"Warning: could not read {path.name}", file=sys.stderr)
            continue

        processed_files.append(path.name)
        try:
            entries = extract_entries_from_file(
                path,
                text,
                api_key=resolved_api_key,
                model=resolved_model,
            )
        except Exception as exc:
            print(
                f"Warning: Groq extraction failed for {path.name}: {exc}",
                file=sys.stderr,
            )
            continue
        all_entries.extend(entries)

    all_entries.sort(key=lambda entry: entry["datetime"])
    return {"files": processed_files, "entries": all_entries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dated Antler team notes from top-level deal-folder files "
            "and print consolidated JSON."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under GOOGLE_DRIVE_BASE to the deal folder",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        result = consolidate(args.relative_path)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: consolidation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
