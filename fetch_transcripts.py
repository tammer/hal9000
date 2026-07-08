#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from document_utils import collect_documents, read_file_as_text
from generate_contents import resolve_folder_path
from meetgeek_client import (
    MeetGeekError,
    Meeting,
    Sentence,
    get_meeting,
    get_transcript,
    list_recent_meetings,
)

ANTLER_TEAM = [
    "Tammer Kamel",
    "Shambhavi Mishra",
    "Alex Wright",
    "Daphne McLarty",
    "Bernie Li",
]
LOOKBACK_DAYS = 4
MAX_DEAL_DOC_CHARS = 100_000
TRANSCRIPT_EXCERPT_CHARS = 3_000
MEETING_LINK_PREFIX = "https://app.meetgeek.ai/meeting/"

RELEVANCE_SYSTEM_PROMPT = """You decide whether a MeetGeek meeting transcript belongs to a specific startup deal folder.

Return valid JSON only with this exact shape:
{"relevant": true, "reason": "short explanation"}

Rules:
- The transcript is relevant if external participants on the call match people associated with this deal, or if the meeting title/content clearly references the deal's company.
- Use the deal documents to identify founders, company name, customer names, and other external people tied to this deal.
- These Antler team members appear on ALL deals and must NOT be used to determine relevance:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Participant email overlap with deal contacts is a strong signal.
- Company name mentioned in the meeting title or transcript is a strong signal.
- If evidence is weak or ambiguous, set relevant=false.
- reason must be one concise sentence explaining the decision.
"""

JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class RelevanceResult:
    relevant: bool
    reason: str


@dataclass(frozen=True)
class MeetingOutcome:
    status: str
    title: str
    date_label: str
    filename: str | None = None
    reason: str = ""


def parse_json_response(text: str) -> dict[str, object]:
    stripped = text.strip()
    match = JSON_FENCE_RE.match(stripped)
    if match:
        stripped = match.group(1).strip()
    return json.loads(stripped)


def build_deal_payload(documents: list[tuple[Path, str]]) -> str:
    sections = [f"### {path.name}\n{content}" for path, content in documents]
    payload = "\n\n".join(sections)
    if len(payload) > MAX_DEAL_DOC_CHARS:
        payload = (
            payload[:MAX_DEAL_DOC_CHARS]
            + "\n\n[Note: deal documents were truncated due to size limits.]"
        )
    return payload


def filename_timestamp(timestamp_start_utc: str) -> str:
    if not timestamp_start_utc:
        return "unknown"
    return timestamp_start_utc.replace(":", "_")


def transcript_basename(title: str, timestamp_start_utc: str) -> str:
    safe_title = title.strip() or "Untitled Meeting"
    encoded_title = safe_title.replace(" ", "+")
    return f"{encoded_title}_sentences_{filename_timestamp(timestamp_start_utc)}"


def meeting_link(meeting_id: str) -> str:
    return f"{MEETING_LINK_PREFIX}{meeting_id}"


def email_to_display_name(email: str) -> str:
    local = email.split("@", 1)[0]
    parts = re.split(r"[._+-]+", local)
    return " ".join(part.capitalize() for part in parts if part)


def attendee_names(meeting: Meeting) -> list[str]:
    emails = list(meeting.participant_emails)
    if meeting.host_email and meeting.host_email not in emails:
        emails.insert(0, meeting.host_email)

    names: list[str] = []
    seen: set[str] = set()
    for email in emails:
        display = email_to_display_name(email)
        key = display.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(display)
    return names


def parse_meeting_start(timestamp_start_utc: str) -> datetime | None:
    if not timestamp_start_utc:
        return None
    normalized = timestamp_start_utc.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return None


def sentence_offset(sentence: Sentence, meeting_start: datetime | None) -> str:
    if meeting_start is None or not sentence.timestamp:
        return "00:00"
    normalized = sentence.timestamp.replace("Z", "+00:00")
    try:
        sentence_time = datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return "00:00"

    total_seconds = max(0, int((sentence_time - meeting_start).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_transcript_text(meeting: Meeting, sentences: list[Sentence]) -> str:
    attendees = ", ".join(attendee_names(meeting))
    link = meeting.join_link or meeting_link(meeting.meeting_id)
    meeting_start = parse_meeting_start(meeting.timestamp_start_utc)

    lines = [
        meeting.title,
        "Metadata",
        f"Title: {meeting.title}",
        "Location: Meet",
        f"Date: {meeting.timestamp_start_utc}",
        f"Attendees: {attendees}",
        f"Link: {link}",
        "",
        "MeetGeek Transcript",
    ]

    for sentence in sentences:
        offset = sentence_offset(sentence, meeting_start)
        lines.append(f"{sentence.speaker} - {offset}")
        lines.append(sentence.transcript)

    return "\n".join(lines).rstrip() + "\n"


def transcript_excerpt(sentences: list[Sentence]) -> str:
    parts: list[str] = []
    total = 0
    for sentence in sentences:
        line = f"{sentence.speaker}: {sentence.transcript}"
        if total + len(line) > TRANSCRIPT_EXCERPT_CHARS:
            remaining = TRANSCRIPT_EXCERPT_CHARS - total
            if remaining > 0:
                parts.append(line[:remaining])
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts)


def find_existing_transcript(
    folder: Path,
    basename: str,
    meeting_id: str,
) -> Path | None:
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        if entry.stem == basename:
            return entry

    needle = meeting_id.lower()
    for entry in folder.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        text = read_file_as_text(entry)
        if text and needle in text.lower():
            return entry
    return None


def build_relevance_prompt(
    meeting: Meeting,
    sentences: list[Sentence],
    deal_payload: str,
) -> str:
    attendees = ", ".join(attendee_names(meeting)) or "unknown"
    participant_emails = ", ".join(meeting.participant_emails) or "unknown"
    excerpt = transcript_excerpt(sentences) or "[no transcript text]"

    return (
        "Deal documents:\n"
        f"{deal_payload}\n\n"
        "Meeting metadata:\n"
        f"- Title: {meeting.title}\n"
        f"- Date: {meeting.timestamp_start_utc}\n"
        f"- Attendees: {attendees}\n"
        f"- Participant emails: {participant_emails}\n"
        f"- Host email: {meeting.host_email or 'unknown'}\n\n"
        "Transcript excerpt:\n"
        f"{excerpt}"
    )


def classify_relevance(
    meeting: Meeting,
    sentences: list[Sentence],
    deal_payload: str,
    *,
    api_key: str,
    model: str,
) -> RelevanceResult:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_relevance_prompt(meeting, sentences, deal_payload),
            },
        ],
    )

    payload = parse_json_response(response.choices[0].message.content or "")
    relevant = bool(payload.get("relevant"))
    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    return RelevanceResult(relevant=relevant, reason=reason)


def meeting_date_label(timestamp_start_utc: str) -> str:
    meeting_start = parse_meeting_start(timestamp_start_utc)
    if meeting_start is None:
        return "unknown date"
    return meeting_start.date().isoformat()


def process_meeting(
    folder: Path,
    meeting_id: str,
    deal_payload: str,
    *,
    api_key: str,
    model: str,
) -> MeetingOutcome:
    meeting = get_meeting(meeting_id)
    sentences = get_transcript(meeting_id)
    basename = transcript_basename(meeting.title, meeting.timestamp_start_utc)
    date_label = meeting_date_label(meeting.timestamp_start_utc)

    existing = find_existing_transcript(folder, basename, meeting.meeting_id)
    if existing is not None:
        return MeetingOutcome(
            status="skipped",
            title=meeting.title,
            date_label=date_label,
            filename=existing.name,
            reason="Transcript already present in deal folder.",
        )

    relevance = classify_relevance(
        meeting,
        sentences,
        deal_payload,
        api_key=api_key,
        model=model,
    )
    if not relevance.relevant:
        return MeetingOutcome(
            status="not_relevant",
            title=meeting.title,
            date_label=date_label,
            reason=relevance.reason,
        )

    filename = f"{basename}.txt"
    output_path = folder / filename
    output_path.write_text(
        format_transcript_text(meeting, sentences),
        encoding="utf-8",
    )
    return MeetingOutcome(
        status="written",
        title=meeting.title,
        date_label=date_label,
        filename=filename,
        reason=relevance.reason,
    )


def print_outcome(outcome: MeetingOutcome) -> None:
    if outcome.status == "written":
        print(f"WRITTEN: {outcome.filename}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "skipped":
        print(f"SKIPPED (already present): {outcome.filename}")
        if outcome.reason:
            print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "not_relevant":
        print(f"NOT RELEVANT: {outcome.title} ({outcome.date_label})")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "error":
        print(f"ERROR: {outcome.title} ({outcome.date_label})")
        print(f"  Reason: {outcome.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent MeetGeek transcripts and write relevant ones "
            "into a deal folder."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the deal folder",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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

    documents = collect_documents(folder, recursive=False)
    if not documents:
        print(
            f"Warning: no readable top-level files found in {folder}",
            file=sys.stderr,
        )
    deal_payload = build_deal_payload(documents) if documents else "[no deal documents]"

    try:
        meeting_summaries = list_recent_meetings(days=LOOKBACK_DAYS)
    except MeetGeekError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(meeting_summaries) >= 20:
        print(
            "Warning: MeetGeek free-tier plans allow about 100 API requests per day.",
            file=sys.stderr,
        )

    outcomes: list[MeetingOutcome] = []
    for summary in meeting_summaries:
        try:
            outcome = process_meeting(
                folder,
                summary.meeting_id,
                deal_payload,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            outcome = MeetingOutcome(
                status="error",
                title=summary.meeting_id,
                date_label=meeting_date_label(summary.timestamp_start_utc),
                reason=str(exc),
            )
            print(f"Error processing meeting {summary.meeting_id}: {exc}", file=sys.stderr)
        outcomes.append(outcome)
        print_outcome(outcome)

    written = sum(1 for outcome in outcomes if outcome.status == "written")
    skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    not_relevant = sum(1 for outcome in outcomes if outcome.status == "not_relevant")
    errors = sum(1 for outcome in outcomes if outcome.status == "error")

    print()
    print(
        "FETCHED: "
        f"{len(meeting_summaries)} meetings in last {LOOKBACK_DAYS} days | "
        f"{written} written | {skipped} skipped | {not_relevant} not relevant"
        + (f" | {errors} errors" if errors else "")
    )
    return 1 if errors and written == 0 and skipped == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
