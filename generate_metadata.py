#!/usr/bin/env python3
"""Generate Claude-backed metadata for a single file under GOOGLE_DRIVE_BASE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv

from claude_common import MODEL, estimate_cost, extract_response_text
from consolidator import (
    ALLOWED_AUTHORS,
    AUTHOR_ALIASES,
    DATETIME_FMT,
    file_mtime,
    resolve_entry_datetime,
)
from document_utils import read_file_as_text
from get_facts import parse_json_response

DEFAULT_MODEL = MODEL
MAX_OUTPUT_TOKENS = 8192

SYSTEM_PROMPT = """You analyze a single document and return metadata as JSON.

Return valid JSON only with this exact shape:
{
  "creator": "Tammer",
  "created_at": "2026-01-01 12:00:00",
  "file_type": "internal note",
  "short summary": "brief summary of the document",
  "full summary": "fuller summary capturing the meaning of the document",
  "url": "https://example.com/page"
}

## creator

Must be exactly one of: Tammer, Bernie, Shambhavi, Alex, Matt, Daphne, other, unknown.

- Map initials/full names: TK/Tammer Kamel→Tammer, BL/Bernie Li→Bernie,
  SM/Shambhavi Mishra→Shambhavi, AW/Alex Wright→Alex, DM/Daphne McLarty→Daphne, Matt→Matt.
- Infer creator from document text or filename when clear.
- If you can determine a creator but they are not in the named list, use "other".
- If you cannot determine the creator, use "unknown".

## created_at

Must be "YYYY-MM-DD HH:MM:SS" when a clear date is available, otherwise the string "unknown".
Sources for created_at, in order:
1. An explicit date in the document text (e.g. "2026-07-14", "July 14, 2026", "July 14th", "Jul 20").
2. A timestamp or date embedded in the filename (the Filename field is provided for this).
If neither the document nor the filename gives a clear date, set created_at to "unknown".
If a date has a month/day but no year, assume the current year (provided in the user message).
If only a date is known and no time, use 00:00:00.

## file_type

Must be exactly one of:
- "internal note": a note or comment made by a team member
- "email": the file appears to be an email
- "transcript": a transcript of a conversation
- "facts": nothing but facts without commentary
- "founder generated": founder materials such as a pitch deck, proposal, or data-room document

## summaries

- short summary: ~100 words or less; capture the gist.
- full summary: as long as necessary to capture the meaning; do not invent facts. always include all email addresses mentioned, all linkedin urls mentioned, any phone numbers mentioned. all critical facts and numbers must be included.

## url

Required string field. Pick one primary URL associated with this file:
1. Prefer a MeetGeek / meeting-recording link when present (common in transcripts).
2. Otherwise use the most relevant http(s) URL in the document body.
3. If no http(s) URL is present, set url to exactly "no URL found".
Do not invent URLs. Do not leave url empty or omit it.
"""




class ClaudeStats(TypedDict):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    model: str


ALLOWED_FILE_TYPES = frozenset(
    {
        "internal note",
        "email",
        "transcript",
        "facts",
        "founder generated",
    }
)

MetadataResult = TypedDict(
    "MetadataResult",
    {
        "filename": str,
        "creator": str,
        "created_at": str,
        "file_type": str,
        "timestamp": str,
        "short summary": str,
        "full summary": str,
        "url": str,
        "claude_stats": ClaudeStats,
    },
)


def resolve_file_path(relative_path: str) -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).resolve()
    path = (base / relative_path.lstrip("/")).resolve()

    if base not in path.parents and path != base:
        raise ValueError(f"path escapes Google Drive root: {relative_path}")

    return path


def normalize_creator(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    cleaned = value.strip()
    if not cleaned:
        return "unknown"
    lowered = cleaned.lower()
    if lowered in {"unknown", "none", "null", "n/a"}:
        return "unknown"
    if cleaned in ALLOWED_AUTHORS:
        return cleaned
    mapped = AUTHOR_ALIASES.get(lowered)
    if mapped:
        return mapped
    first = cleaned.split()[0]
    mapped = AUTHOR_ALIASES.get(first.lower())
    if mapped:
        return mapped
    if lowered == "other":
        return "other"
    return "other"


def normalize_file_type(value: object) -> str:
    if not isinstance(value, str):
        return "internal note"
    cleaned = value.strip().lower()
    for allowed in ALLOWED_FILE_TYPES:
        if cleaned == allowed:
            return allowed
    return "internal note"


def normalize_url(value: object) -> str:
    if not isinstance(value, str):
        return "no URL found"
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "null", "n/a", "no url found"}:
        return "no URL found"
    return cleaned


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


def call_claude(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
) -> tuple[dict[str, Any], object]:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    content = extract_response_text(response)
    payload = parse_json_response(content)
    return payload, response


def generate_metadata(
    relative_path: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> MetadataResult:
    path = resolve_file_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"path is not a file: {path}")

    text = read_file_as_text(path)
    if text is None:
        raise ValueError(f"could not read file as text: {relative_path}")

    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    resolved_model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    payload, response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(path, text),
        api_key=resolved_api_key,
        model=resolved_model,
    )

    usage = response.usage
    cost = estimate_cost(resolved_model, usage)
    created_at = resolve_entry_datetime(
        payload.get("created_at"),
        mtime=file_mtime(path),
    )

    return {
        "filename": relative_path,
        "creator": normalize_creator(payload.get("creator")),
        "created_at": created_at,
        "file_type": normalize_file_type(payload.get("file_type")),
        "timestamp": datetime.now().strftime(DATETIME_FMT),
        "short summary": str(payload.get("short summary", "")).strip(),
        "full summary": str(payload.get("full summary", "")).strip(),
        "url": normalize_url(payload.get("url")),
        "claude_stats": {
            "input_tokens": int(usage.input_tokens),
            "output_tokens": int(usage.output_tokens),
            "cost_usd": cost,
            "model": resolved_model,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Claude metadata (creator, dates, summaries) for a single "
            "file under GOOGLE_DRIVE_BASE and print JSON."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under GOOGLE_DRIVE_BASE to the file",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        result = generate_metadata(args.relative_path)
    except (ValueError, FileNotFoundError, IsADirectoryError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: metadata generation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
