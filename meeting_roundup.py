#!/usr/bin/env python3
"""Summarize MeetGeek team meetings for a UTC day via Groq.

Import and call ``meeting_roundup()`` to get a JSON-serializable list of
``{"meeting_id", "summary"}`` dicts. Run as a script to write that JSON under
``GOOGLE_DRIVE_BASE/ai-generated/dailies/meetgeeks/YYYY-MM-DD.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from fetch_transcripts import format_transcript_text
from meetgeek_client import (
    MeetGeekError,
    MeetingSummary,
    get_meeting,
    get_transcript,
    list_team_meetings,
)

__all__ = ["meeting_roundup"]

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SUMMARY_INSTRUCTIONS = """You summarize meeting transcripts for a venture capital team.

Write a single paragraph of about 50 words covering:
1. Who was in the meeting
2. What type of meeting was it? It could be a) a meeting with a prospective or current investor in out fund,
b) a meeting with a founder we could invest in, c) a meeting with a founder we have already invested in (a portco meeting)
d) an internal meeting with some or all members of the team and no one else.
e) some other type of meeting.
2. What was discussed
3. Any next steps

Rules:
- Stick to facts supported by the transcript; do not invent details.
- Prefer speaker names from the transcript when available.
- For team members (Bernie, Tammmer, Alex, Shambhavi, Daphne), just use our first names ("Alex" NOT "Alex Wright)
- For everyone else, just full names.
- If next steps are unclear, say so briefly.
- Output plain text only — no headings, bullets, or markdown.

Examples:

For June 22, 2026:

- Bernie met with Tom Henderson who seems to be a prospective investor. Bernie will follow up by sending him some more information.
- Tammer met with Ethan (Quis). They discussed his go-to-market strategy and potential partnerships.
- Tammer, Alex and Shambhavi met to discuss several potential investments. It was decided that Tammer would meet Peter from Roper.com
and Alex would work on the investment memo for Trails.com

"""


def resolve_google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    return Path(base_raw).expanduser().resolve()


def load_facts_md() -> str:
    facts_path = resolve_google_drive_base().parent / "facts.md"
    if not facts_path.is_file():
        raise FileNotFoundError(f"facts.md not found at {facts_path}")
    return facts_path.read_text(encoding="utf-8").strip()


def build_system_prompt(facts_text: str) -> str:
    return f"{facts_text}\n\n{SUMMARY_INSTRUCTIONS.strip()}\n"


def default_date() -> date:
    """Yesterday before 16:30 local time; today at or after 16:30."""
    now = datetime.now()
    cutoff = now.replace(hour=16, minute=30, second=0, microsecond=0)
    if now >= cutoff:
        return now.date()
    return now.date() - timedelta(days=1)


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def parse_meeting_start(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def meetings_for_day(
    team_id: str,
    day: date,
) -> list[MeetingSummary]:
    day_start, day_end = day_bounds(day)
    summaries = list_team_meetings(team_id, day_start)
    filtered: list[MeetingSummary] = []
    for summary in summaries:
        start = parse_meeting_start(summary.timestamp_start_utc)
        if start is None:
            continue
        if day_start <= start < day_end:
            filtered.append(summary)
    filtered.sort(key=lambda item: item.timestamp_start_utc)
    return filtered


def summarize_transcript(
    *,
    transcript_text: str,
    system_prompt: str,
    api_key: str,
    model: str,
) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Summarize this meeting transcript in about 50 words:\n\n"
                    f"{transcript_text.rstrip()}\n"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def meeting_roundup(day: date | str | None = None) -> list[dict[str, str]]:
    """Return JSON-ready meeting summaries for a UTC day.

    Args:
        day: ``YYYY-MM-DD`` string, a ``date``, or ``None`` for the
            default day (yesterday before 16:30 local, otherwise today).

    Returns:
        A list of ``{"meeting_id": "...", "summary": "..."}`` dicts.
        Progress goes to stderr; skipped/failed meetings are omitted.
    """
    load_dotenv()

    if day is None:
        resolved = default_date()
    elif isinstance(day, date):
        resolved = day
    else:
        resolved = parse_day(day)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    if not os.getenv("MEETGEEK_API_KEY"):
        raise ValueError("MEETGEEK_API_KEY is not set")
    team_id = os.getenv("MEETGEEK_TEAM_ID", "").strip()
    if not team_id:
        raise ValueError("MEETGEEK_TEAM_ID is not set")

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    system_prompt = build_system_prompt(load_facts_md())
    summaries = meetings_for_day(team_id, resolved)

    day_label = resolved.isoformat()
    if not summaries:
        print(f"No meetings on {day_label}.", file=sys.stderr)
        return []

    print(
        f"Summarizing {len(summaries)} meeting(s) on {day_label}...",
        file=sys.stderr,
    )

    results: list[dict[str, str]] = []
    for summary in summaries:
        try:
            meeting = get_meeting(summary.meeting_id)
            sentences = get_transcript(summary.meeting_id)
        except MeetGeekError as exc:
            print(
                f"Error fetching meeting {summary.meeting_id}: {exc}",
                file=sys.stderr,
            )
            continue

        if not sentences:
            print(
                f"Skipping {meeting.title} ({meeting.meeting_id}): no transcript",
                file=sys.stderr,
            )
            continue

        transcript_text = format_transcript_text(meeting, sentences)
        print(
            f"Summarizing: {meeting.title} ({meeting.timestamp_start_utc})",
            file=sys.stderr,
        )

        try:
            summary_text = summarize_transcript(
                transcript_text=transcript_text,
                system_prompt=system_prompt,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            print(
                f"Error summarizing {meeting.meeting_id}: {exc}",
                file=sys.stderr,
            )
            continue

        results.append(
            {
                "meeting_id": meeting.meeting_id,
                "summary": summary_text,
            }
        )

    return results


def main() -> int:
    """CLI entry point: write meeting roundup JSON under ai-generated/dailies/meetgeeks/."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all MeetGeek team meetings for a UTC day and write "
            "{meeting_id, summary} JSON to "
            "GOOGLE_DRIVE_BASE/ai-generated/dailies/meetgeeks/YYYY-MM-DD.json."
        )
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help=(
            "UTC calendar day as YYYY-MM-DD "
            "(default: yesterday before 16:30 local, otherwise today)"
        ),
    )
    args = parser.parse_args()

    try:
        day = parse_day(args.date) if args.date is not None else default_date()
        results = meeting_roundup(day)
        output_dir = (
            resolve_google_drive_base() / "ai-generated" / "dailies" / "meetgeeks"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{day.isoformat()}.json"
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ValueError, FileNotFoundError, OSError, MeetGeekError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path}", file=sys.stderr)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
