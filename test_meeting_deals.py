#!/usr/bin/env python3
"""Standalone smoke test: report which deals a MeetGeek meeting discussed."""
# this is a program that summarizes an internal meeting about deals we have in progress

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from fetch_transcripts import format_transcript_text, groq_json_chat
from meetgeek_client import MeetGeekError, get_meeting, get_transcript

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert venture capital analyst reviewing a meeting transcript.

You are given:
1. A deal catalog (status.md) listing active deals with Deal Name, Product, Founder(s), and Status.
2. A MeetGeek meeting transcript.

Identify which catalog deals were discussed in the meeting and extract what was said and decided.

Return valid JSON only with this exact shape:
{
  "deals": [
    {
      "deal_name": "exact Deal Name from the catalog",
      "what_was_said": "2-5 bullet points of key discussion topics, each on its own line starting with '- '",
      "decisions": "what was decided, or empty string if none",
      "next_steps": "owner + action when clear, or empty string if none"
    }
  ],
  "notes": "brief note if no deals matched, meeting was unrelated, or useful context about multi-deal coverage; otherwise empty string"
}

Hard rules:
- Only include deals whose Deal Name appears in the catalog. Use the exact Deal Name spelling from the catalog.
- Do NOT invent deals or company names that are not in the catalog.
- Omit deals that were not discussed.
- Only report facts supported by the transcript; do not guess or import outside knowledge.
- Match by company/product name, founder names, or clear deal-specific discussion.
- Antler team members (e.g. Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li) appear on many deals and alone do not identify a deal.
- If nothing in the catalog was discussed, return {"deals": [], "notes": "..."}.
"""


def resolve_status_path() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).expanduser().resolve()
    status_path = base / "status.md"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"status.md not found at {status_path}. Run summarizer.py first."
        )
    return status_path


def build_user_prompt(*, status_text: str, transcript_text: str) -> str:
    return (
        "Deal catalog (status.md):\n"
        f"{status_text.rstrip()}\n\n"
        "Meeting transcript:\n"
        f"{transcript_text.rstrip()}\n"
    )


def print_report(
    *,
    meeting_title: str,
    meeting_date: str,
    meeting_id: str,
    payload: dict,
) -> None:
    print(f"Meeting: {meeting_title}")
    print(f"Date: {meeting_date}")
    print(f"MeetGeek ID: {meeting_id}")
    print()

    deals = payload.get("deals") or []
    if not isinstance(deals, list):
        deals = []

    if not deals:
        print("No catalog deals discussed.")
        notes = str(payload.get("notes") or "").strip()
        if notes:
            print(f"Notes: {notes}")
        return

    print(f"Deals discussed ({len(deals)}):")
    print()
    for index, deal in enumerate(deals, start=1):
        if not isinstance(deal, dict):
            continue
        name = str(deal.get("deal_name") or "").strip() or "(unknown)"
        what_was_said = str(deal.get("what_was_said") or "").strip()
        decisions = str(deal.get("decisions") or "").strip()
        next_steps = str(deal.get("next_steps") or "").strip()

        print(f"{index}. {name}")
        if what_was_said:
            print("   Discussion:")
            for line in what_was_said.splitlines():
                cleaned = line.strip()
                if cleaned:
                    print(f"   {cleaned}")
        else:
            print("   Discussion: (none)")
        print(f"   Decisions: {decisions or '(none)'}")
        print(f"   Next steps: {next_steps or '(none)'}")
        print()

    notes = str(payload.get("notes") or "").strip()
    if notes:
        print(f"Notes: {notes}")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Fetch a MeetGeek transcript by meeting ID and report which "
            "status.md deals were discussed (via Groq)."
        )
    )
    parser.add_argument("meeting_id", help="MeetGeek meeting record ID")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1
    if not os.getenv("MEETGEEK_API_KEY"):
        print("Error: MEETGEEK_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    try:
        status_path = resolve_status_path()
        status_text = status_path.read_text(encoding="utf-8")
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        meeting = get_meeting(args.meeting_id)
        sentences = get_transcript(args.meeting_id)
    except MeetGeekError as exc:
        print(f"Error fetching MeetGeek meeting: {exc}", file=sys.stderr)
        return 1

    if not sentences:
        print(
            f"Error: no transcript sentences for meeting {args.meeting_id}",
            file=sys.stderr,
        )
        return 1

    transcript_text = format_transcript_text(meeting, sentences)
    user_prompt = build_user_prompt(
        status_text=status_text,
        transcript_text=transcript_text,
    )

    try:
        payload = groq_json_chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            api_key=api_key,
            model=model,
        )
    except Exception as exc:
        print(f"Error calling Groq: {exc}", file=sys.stderr)
        return 1

    print_report(
        meeting_title=meeting.title,
        meeting_date=meeting.timestamp_start_utc,
        meeting_id=meeting.meeting_id,
        payload=payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
