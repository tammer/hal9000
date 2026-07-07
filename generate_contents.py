#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from groq import Groq

from document_utils import read_file_as_text
from get_facts import parse_json_response

ContentType = Literal[
    "transcript",
    "founder generated content",
    "note",
    "unknown",
]

CONTENT_TYPES: tuple[ContentType, ...] = (
    "transcript",
    "founder generated content",
    "note",
    "unknown",
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
CONTENT_SAMPLE_CHARS = 2_000

CLASSIFIER_SYSTEM_PROMPT = """You classify documents in an Antler deal folder.

Return valid JSON only with this exact shape:
{
  "classifications": [
    {"file_name": "example.pdf", "type": "transcript"}
  ]
}

Allowed type values (use exactly one of these strings):
- "transcript"
- "founder generated content"
- "note"
- "unknown"

## Critical distinction

These folders mix two very different kinds of writing:

1. **Materials FROM the founders/startup** (classify as "founder generated content")
2. **Materials written BY Antler, investors, or AI ABOUT the startup** (classify as "note")

When unsure between "founder generated content" and "note", ask:
- Who is the author/speaker? If Antler, an investor, or an analyst writing about the company, it is a "note".
- Is the document selling/presenting the company in the founder's voice? That is "founder generated content".
- Is the document evaluating, summarizing, diligencing, or taking notes on the company? That is a "note".

## Type definitions

### "transcript"
Raw record of a meeting or call. Signals:
- Speaker names or labels with back-and-forth dialogue
- Timestamps or turn-by-turn conversation
- Header with meeting date, attendees, and often a meeting/recording URL
- Reads like a verbatim conversation, not a polished summary

Examples: Zoom/Meet transcripts, Otter/Fireflies exports, call recordings transcribed to text.

### "founder generated content"
Documents created by the startup/founders and shared outward (usually with investors). Signals:
- Founder or company voice: "we built", "our product", "our traction", "we are raising"
- Pitch deck, investor deck, demo day deck, data room materials
- One-pager, executive summary, product memo, GTM plan, financial model from the company
- Company branding, product screenshots, market slides, fundraising ask
- Promotional or presentational tone aimed at convincing investors/customers
- Describes the business from inside the company, not from an external evaluator

Examples: pitch deck PDF, founder one-pager, product overview from the startup, cap table or model sent by founders.

NOT founder generated content:
- Antler/investor diligence writeups
- Deal memos, IC memos, investment notes
- Post-meeting summaries written by someone on the investing team
- AI-generated deal analysis or report sections
- Third-person evaluation: "the founders claim...", "our view is...", "key risks include..."

### "note"
Internal analysis or commentary written by Antler, investors, or AI — not by the founders. Signals:
- Investor/diligence perspective on the startup
- Meeting notes taken by an Antler team member (summary, not raw transcript)
- Deal screening notes, open questions, concerns, pros/cons
- References to "Antler", "we met with", "our assessment", "diligence", "IC", "pass/invest"
- AI-generated analysis, summaries, or structured report output about the deal
- Evaluative or analytical tone about the company from the outside

Examples: diligence memo, investor meeting notes, deal summary, AI analysis output, concern lists.

### "unknown"
Unreadable, empty, purely administrative, or truly ambiguous.

## Rules
- Classify every file listed by the user.
- Use filename and content excerpt together; content tone and perspective matter more than filename alone.
- Filenames like "deck" or "memo" are weak signals — decide from who is writing and why.
- A polished summary of a founder conversation written by an investor is a "note", not a "transcript".
- A polished company presentation sent by founders is "founder generated content", not a "note".
- If content is "[unreadable]", use filename only; prefer "unknown" unless the filename is strongly indicative.
"""


class ContentEntry(TypedDict):
    file_name: str
    date_created: str
    type: ContentType


def resolve_folder_path(relative_path: str) -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).resolve()
    folder = (base / relative_path.lstrip("/")).resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes Google Drive root: {relative_path}")

    return folder


def should_skip_file(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith("~$"):
        return True
    if name == "contents.json":
        return True
    return False


def file_creation_timestamp(path: Path) -> float:
    stat = path.stat()
    return getattr(stat, "st_birthtime", stat.st_ctime)


def contents_json_path(folder: Path) -> Path:
    return folder.expanduser().resolve() / "contents.json"


def load_existing_contents(folder: Path) -> list[ContentEntry]:
    path = contents_json_path(folder)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid contents.json: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("contents.json must be a JSON array")

    entries: list[ContentEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name", "")).strip()
        date_created = str(item.get("date_created", "")).strip()
        if not file_name or not date_created:
            continue
        entries.append(
            {
                "file_name": file_name,
                "date_created": date_created,
                "type": normalize_content_type(item.get("type")),
            }
        )
    return entries


def scan_folder(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"path is not a directory: {folder}")

    files = sorted(
        entry
        for entry in folder.iterdir()
        if entry.is_file() and not should_skip_file(entry)
    )
    return files


def get_file_metadata(path: Path) -> dict[str, str]:
    created_at = datetime.fromtimestamp(file_creation_timestamp(path)).astimezone()
    return {
        "file_name": path.name,
        "date_created": created_at.isoformat(timespec="seconds"),
    }


def normalize_content_type(value: object) -> ContentType:
    if not isinstance(value, str):
        return "unknown"

    normalized = value.strip().lower()
    for content_type in CONTENT_TYPES:
        if normalized == content_type.lower():
            return content_type
    return "unknown"


def build_classification_prompt(files: list[tuple[Path, str | None]]) -> str:
    blocks: list[str] = []
    for path, text in files:
        excerpt = text[:CONTENT_SAMPLE_CHARS] if text else "[unreadable]"
        blocks.append(f"File: {path.name}\nExcerpt:\n{excerpt}")
    return "Classify each file:\n\n" + "\n\n".join(blocks)


def classify_files(
    files: list[tuple[Path, str | None]],
    *,
    api_key: str,
    model: str,
) -> dict[str, str]:
    if not files:
        return {}

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": build_classification_prompt(files)},
        ],
    )

    payload = parse_json_response(response.choices[0].message.content or "")
    raw_classifications = payload.get("classifications", [])

    classifications: dict[str, str] = {}
    if isinstance(raw_classifications, list):
        for item in raw_classifications:
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file_name", "")).strip()
            if not file_name:
                continue
            classifications[file_name] = normalize_content_type(item.get("type"))

    return classifications


def build_entries(
    files: list[Path],
    classifications: dict[str, str],
) -> list[ContentEntry]:
    entries: list[ContentEntry] = []
    for path in files:
        metadata = get_file_metadata(path)
        content_type = normalize_content_type(
            classifications.get(path.name, "unknown")
        )
        entries.append(
            {
                "file_name": metadata["file_name"],
                "date_created": metadata["date_created"],
                "type": content_type,
            }
        )

    entries.sort(key=lambda entry: entry["file_name"].lower())
    return entries


def resolve_api_config(
    api_key: str | None,
    model: str | None,
) -> tuple[str, str]:
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY is not set")

    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return resolved_api_key, resolved_model


def generate_contents(
    folder: Path,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> list[ContentEntry]:
    folder = folder.expanduser().resolve()
    existing_entries = load_existing_contents(folder)
    existing_by_name = {entry["file_name"]: entry for entry in existing_entries}

    files = scan_folder(folder)
    new_files = [path for path in files if path.name not in existing_by_name]
    if not new_files:
        return sorted(
            existing_entries,
            key=lambda entry: entry["file_name"].lower(),
        )

    resolved_api_key, resolved_model = resolve_api_config(api_key, model)
    file_samples = [(path, read_file_as_text(path)) for path in new_files]
    classifications = classify_files(
        file_samples,
        api_key=resolved_api_key,
        model=resolved_model,
    )
    new_entries = build_entries(new_files, classifications)

    merged = list(existing_by_name.values()) + new_entries
    merged.sort(key=lambda entry: entry["file_name"].lower())
    return merged


def write_contents_json(folder: Path, entries: list[ContentEntry]) -> Path:
    folder = folder.expanduser().resolve()
    output_path = contents_json_path(folder)
    output_path.write_text(
        json.dumps(entries, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def generate_and_write_contents(
    folder: Path,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> Path:
    folder = folder.expanduser().resolve()
    existing_entries = load_existing_contents(folder)
    existing_names = {entry["file_name"] for entry in existing_entries}

    entries = generate_contents(folder, api_key=api_key, model=model)
    output_path = contents_json_path(folder)

    if {entry["file_name"] for entry in entries} == existing_names:
        return output_path

    return write_contents_json(folder, entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan top-level deal folder files, classify new files with an LLM, "
            "and write contents.json in the same folder."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the folder to scan",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        folder = resolve_folder_path(args.relative_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not folder.exists():
        print(f"Error: path does not exist: {folder}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: path is not a directory: {folder}", file=sys.stderr)
        return 1

    try:
        output_path = generate_and_write_contents(folder)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: classification failed: {exc}", file=sys.stderr)
        return 1

    print(f"Contents written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
