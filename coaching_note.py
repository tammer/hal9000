#!/usr/bin/env python3
"""Generate a coaching note from the latest portco transcript using Claude."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from claude_common import (
    MODEL,
    build_payload,
    load_system_prompt,
    parse_relative_path_args,
    print_usage_report,
    require_api_key,
    run_claude,
)
from document_utils import read_file_as_text
from fetch_transcripts import TRANSCRIPT_FILENAME_MARKER, transcripts_dir
from generate_portco_report import validate_portco_folder

COACHING_NOTE_NAME = "coaching_note.md"
COACHING_NOTE_PROMPT_PATH = Path(__file__).parent / "coaching_note_prompt.md"

# MeetGeek filenames embed UTC as YYYY-MM-DDTHH_MM_SSZ (colons → underscores).
_SENTENCES_TS_RE = re.compile(
    re.escape(TRANSCRIPT_FILENAME_MARKER)
    + r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)"
)


def parse_sentences_timestamp(name: str) -> datetime | None:
    """Parse the MeetGeek ``_sentences_<UTC>`` timestamp from a filename."""
    match = _SENTENCES_TS_RE.search(name)
    if match is None:
        return None
    raw = match.group(1)
    # Reverse filename_timestamp(): restore colons in the time portion.
    date_part, _, time_part = raw.partition("T")
    if not time_part:
        return None
    restored = f"{date_part}T{time_part.replace('_', ':')}"
    normalized = restored.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return None


def transcript_sort_key(path: Path) -> datetime:
    """Prefer embedded MeetGeek meeting time; otherwise use file mtime."""
    embedded = parse_sentences_timestamp(path.name)
    if embedded is not None:
        return embedded
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def find_latest_transcript(folder: Path) -> Path | None:
    """Return the newest readable file under ``transcripts/``, or None."""
    target = transcripts_dir(folder)
    if not target.is_dir():
        return None

    candidates: list[Path] = []
    try:
        entries = list(target.iterdir())
    except OSError as exc:
        print(f"Error: failed to list {target}: {exc}", file=sys.stderr)
        return None

    for entry in entries:
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if read_file_as_text(entry) is None:
            continue
        candidates.append(entry)

    if not candidates:
        return None

    return max(candidates, key=transcript_sort_key)


def write_coaching_note(folder: Path, content: str) -> Path:
    ai_generated_dir = folder / "ai-generated"
    ai_generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_generated_dir / COACHING_NOTE_NAME
    output_path.write_text(content, encoding="utf-8")
    return output_path


def run_coaching_note(relative_path: str, *, dry_run: bool) -> int:
    folder = validate_portco_folder(relative_path)
    if folder is None:
        return 1

    transcript_path = find_latest_transcript(folder)
    if transcript_path is None:
        print(
            f"Error: no readable transcripts found under {transcripts_dir(folder)}",
            file=sys.stderr,
        )
        return 1

    text = read_file_as_text(transcript_path)
    if text is None:
        print(
            f"Error: could not read transcript: {transcript_path}",
            file=sys.stderr,
        )
        return 1

    print(f"Latest transcript: {transcript_path}")
    if dry_run:
        return 0

    api_key = require_api_key()
    if api_key is None:
        return 1

    system_prompt = load_system_prompt(COACHING_NOTE_PROMPT_PATH)
    if system_prompt is None:
        return 1

    user_content = f"Transcript:\n{build_payload([(transcript_path, text)])}"

    try:
        note, response = run_claude(
            system_prompt,
            user_content,
            api_key,
            MODEL,
        )
    except Exception as exc:
        print(f"Error: Anthropic API call failed: {exc}", file=sys.stderr)
        return 1

    try:
        output_path = write_coaching_note(folder, note)
    except OSError as exc:
        print(f"Error: failed to write {COACHING_NOTE_NAME}: {exc}", file=sys.stderr)
        return 1

    print_usage_report(MODEL, response.usage)
    print(f"Coaching note written to {output_path}")
    return 0


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        "Generate a coaching note from the latest portco transcript using Claude.",
        "Folder name under portcos/ (e.g. Central-Agent)",
        with_dry_run=True,
    )
    return run_coaching_note(args.relative_path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
