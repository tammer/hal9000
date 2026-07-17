#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from document_utils import read_file_as_text
from generate_contents import resolve_folder_path
from get_facts import parse_json_response
from meetgeek_client import (
    MeetGeekError,
    Meeting,
    Sentence,
    get_meeting,
    get_transcript,
    list_recent_meetings,
)

LOOKBACK_DAYS = 8
MAX_DEAL_DOC_CHARS = 100_000
TRANSCRIPT_EXCERPT_CHARS = 3_000
MEETING_LINK_PREFIX = "https://app.meetgeek.ai/meeting/"

TRANSCRIPT_FILENAME_MARKER = "_sentences_"
TRANSCRIPTS_DIR_NAME = "transcripts"

PROCESSED_MEETINGS_PATH = (
    Path(__file__).resolve().parent / "processed_meetgeek_meetings.txt"
)
RECORDABLE_PROCESSED_STATUSES = frozenset(
    {"written", "no_match", "skipped", "not_relevant"}
)

ANTLER_STAFF = {
    "tammer kamel",
    "shambhavi mishra",
    "alex wright",
    "daphne mclarty",
    "bernie li",
}

IDENTITY_EXTRACTION_SYSTEM_PROMPT = """You extract the canonical identity of a startup deal from its deal documents.

Return valid JSON only with this exact shape:
{"company_name": "Acme Inc" or null, "human_names": ["Full Name", ...]}

Include:
- company_name: the startup or company name if one is clearly named; otherwise null
- human_names: full names of founders (not Antler staff)

Rules:
- These Antler team members appear on many deals and must be excluded from human_names:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Use full names as written in the documents when possible
- The deal folder name is often a founder's first name and may be misspelled; use it only as a weak hint
- Do not invent names or companies not supported by the documents
- human_names must contain only person name strings; never include explanations or commentary in JSON values
"""

JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return only valid JSON with no commentary inside values or arrays."
)
MAX_JSON_RETRIES = 3

TRANSCRIPT_RELEVANCE_SYSTEM_PROMPT = """You decide whether a MeetGeek meeting belongs to a specific startup deal.

You are given the deal's company name (if any) and human names extracted from deal documents.

Return valid JSON only with this exact shape:
{"relevant": true, "reason": "explanation of why the meeting is relevant referring to the identity evidence and conversation nature"}

A meeting is relevant ONLY if BOTH of the following are true:

1. Identity evidence: the company name and/or one of the human names appears in the meeting title, attendee names, participant emails, host email, or transcript text.
2. Conversation nature: the transcript excerpt shows a deal-assessment discussion — Antler or investor-side people asking diligence-style questions (market, product, traction, team, fundraising, GTM, etc.) and founders or startup-side people answering about their business.

Hard rules:
- These Antler team members appear on ALL deals and must NEVER determine relevance:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Do NOT match similar-sounding or partially similar names (e.g. Chen is not Chan)
- Do NOT infer company matches from email domains or substrings
- Do NOT mark relevant based on shared generic topics alone
- A lone name or single-word match with no supporting deal-assessment conversation context is NOT enough; set relevant=false
- If the meeting discusses a different company or topic and a deal name appears only coincidentally, set relevant=false
- Casual mentions, social catch-ups, or unrelated work meetings are NOT relevant even if a name appears
- If no company name or human name from the deal identity appears in the meeting, set relevant=false
- If conversation context is missing or thin (empty excerpt, greetings only) and identity evidence is weak, set relevant=false
- When evidence is ambiguous, set relevant=false

reason must be one concise sentence. If relevant=true, name the matching company or person and where it appears, and briefly note why the conversation fits a deal assessment.
"""


@dataclass(frozen=True)
class DealIdentity:
    company_name: str | None
    human_names: list[str]


@dataclass(frozen=True)
class RelevanceResult:
    relevant: bool
    reason: str


@dataclass(frozen=True)
class DealMatchTarget:
    folder_name: str
    identity: DealIdentity


@dataclass(frozen=True)
class MatchResult:
    deal_folder: str | None
    reason: str


@dataclass(frozen=True)
class MeetingOutcome:
    status: str
    title: str
    date_label: str
    filename: str | None = None
    reason: str = ""


def load_processed_meeting_ids(
    path: Path = PROCESSED_MEETINGS_PATH,
) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.add(stripped.split()[0])
    return ids


def append_processed_meeting_id(
    meeting_id: str,
    path: Path = PROCESSED_MEETINGS_PATH,
    *,
    known_ids: set[str] | None = None,
) -> None:
    cleaned = meeting_id.strip()
    if not cleaned:
        return
    if known_ids is not None and cleaned in known_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(cleaned + "\n")
    if known_ids is not None:
        known_ids.add(cleaned)


def should_record_processed(status: str) -> bool:
    return status in RECORDABLE_PROCESSED_STATUSES


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


def sanitize_title_for_filename(title: str) -> str:
    safe_title = title.strip() or "Untitled Meeting"
    for char in ':/\\?*|"<>':
        safe_title = safe_title.replace(char, "_")
    return safe_title.replace(" ", "+")


def transcript_basename(title: str, timestamp_start_utc: str) -> str:
    return (
        f"{sanitize_title_for_filename(title)}"
        f"{TRANSCRIPT_FILENAME_MARKER}{filename_timestamp(timestamp_start_utc)}"
    )


def is_meetgeek_transcript(path: Path) -> bool:
    return path.suffix.lower() == ".txt" and TRANSCRIPT_FILENAME_MARKER in path.name


def transcripts_dir(folder: Path) -> Path:
    return folder / TRANSCRIPTS_DIR_NAME


def transcript_relative_path(filename: str) -> str:
    return f"{TRANSCRIPTS_DIR_NAME}/{filename}"


def collect_deal_context(
    folder: Path,
    *,
    summary_only: bool = False,
) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []

    summary_path = folder / "ai-generated" / "summary.md"
    if summary_path.is_file():
        summary_text = read_file_as_text(summary_path)
        if summary_text:
            documents.append((summary_path, summary_text))
        if summary_only:
            return documents

    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith(".") or entry.name.startswith("~$"):
            continue
        if is_meetgeek_transcript(entry):
            continue

        text = read_file_as_text(entry)
        if text is None:
            continue
        documents.append((entry, text))

    return documents


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
    meetgeek_link = meeting_link(meeting.meeting_id)
    meeting_start = parse_meeting_start(meeting.timestamp_start_utc)

    lines = [
        meeting.title,
        "Metadata",
        f"Title: {meeting.title}",
        "Location: Meet",
        f"Date: {meeting.timestamp_start_utc}",
        f"Attendees: {attendees}",
        f"Link: {meetgeek_link}",
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
    target = transcripts_dir(folder)
    if not target.is_dir():
        return None

    for entry in target.iterdir():
        if not entry.is_file():
            continue
        if entry.stem == basename:
            return entry

    needle = meeting_id.lower()
    for entry in target.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() != ".txt":
            continue
        text = read_file_as_text(entry)
        if text and needle in text.lower():
            return entry
    return None


def build_identity_extraction_prompt(
    deal_payload: str,
    *,
    deal_folder_name: str,
) -> str:
    return (
        f"Deal folder name: {deal_folder_name}\n\n"
        "Deal documents:\n"
        f"{deal_payload}"
    )


def groq_json_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
) -> dict:
    client = Groq(api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None

    for _ in range(MAX_JSON_RETRIES):
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        try:
            return parse_json_response(content)
        except Exception as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": JSON_RETRY_PROMPT})

    if last_error is not None:
        raise last_error
    raise RuntimeError("Groq JSON chat failed without a response")


def parse_identity_payload(payload: dict) -> DealIdentity:
    company_raw = payload.get("company_name")
    company_name = str(company_raw).strip() if company_raw else None
    if company_name and company_name.lower() in {"null", "none", ""}:
        company_name = None

    human_names: list[str] = []
    for name in payload.get("human_names", []):
        cleaned = str(name).strip()
        if cleaned:
            human_names.append(cleaned)

    return DealIdentity(company_name=company_name, human_names=human_names)


def extract_deal_identity(
    deal_payload: str,
    *,
    deal_folder_name: str,
    api_key: str,
    model: str,
) -> DealIdentity:
    payload = groq_json_chat(
        system_prompt=IDENTITY_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_identity_extraction_prompt(
            deal_payload,
            deal_folder_name=deal_folder_name,
        ),
        api_key=api_key,
        model=model,
    )
    return parse_identity_payload(payload)


def print_deal_identity(identity: DealIdentity) -> None:
    print("Deal identity (from documents):")
    if identity.company_name:
        print(f"  Company: {identity.company_name}")
    else:
        print("  Company: (none identified)")
    if identity.human_names:
        print(f"  People: {', '.join(identity.human_names)}")
    else:
        print("  People: (none identified)")
    print()


def build_meeting_metadata_prompt(
    meeting: Meeting,
    sentences: list[Sentence],
) -> str:
    attendees = ", ".join(attendee_names(meeting)) or "unknown"
    participant_emails = ", ".join(meeting.participant_emails) or "unknown"
    excerpt = transcript_excerpt(sentences) or "[no transcript text]"

    return (
        "Meeting metadata:\n"
        f"- Title: {meeting.title}\n"
        f"- Date: {meeting.timestamp_start_utc}\n"
        f"- Attendees: {attendees}\n"
        f"- Participant emails: {participant_emails}\n"
        f"- Host email: {meeting.host_email or 'unknown'}\n\n"
        "Transcript excerpt:\n"
        f"{excerpt}"
    )


def build_relevance_prompt(
    meeting: Meeting,
    sentences: list[Sentence],
    identity: DealIdentity,
) -> str:
    company = identity.company_name or "(none)"
    people = ", ".join(identity.human_names) or "(none)"

    return (
        "Deal identity:\n"
        f"- Company: {company}\n"
        f"- People: {people}\n\n"
        f"{build_meeting_metadata_prompt(meeting, sentences)}"
    )


def classify_relevance(
    meeting: Meeting,
    sentences: list[Sentence],
    identity: DealIdentity,
    *,
    api_key: str,
    model: str,
) -> RelevanceResult:
    payload = groq_json_chat(
        system_prompt=TRANSCRIPT_RELEVANCE_SYSTEM_PROMPT,
        user_prompt=build_relevance_prompt(meeting, sentences, identity),
        api_key=api_key,
        model=model,
    )
    relevant = bool(payload.get("relevant"))
    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    return RelevanceResult(relevant=relevant, reason=reason)


def word_in_text(word: str, text: str) -> bool:
    if not word or len(word) < 2:
        return False
    return bool(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def meeting_match_haystack(meeting: Meeting, sentences: list[Sentence]) -> str:
    parts = [
        meeting.title,
        " ".join(attendee_names(meeting)),
        " ".join(meeting.participant_emails),
        meeting.host_email or "",
        transcript_excerpt(sentences),
    ]
    return " ".join(part for part in parts if part)


def find_programmatic_deal_match_from_haystack(
    haystack: str,
    targets: list[DealMatchTarget],
    *,
    source_label: str = "content",
) -> MatchResult | None:
    matched_folders: list[str] = []

    for target in targets:
        if word_in_text(target.folder_name, haystack):
            matched_folders.append(target.folder_name)
            continue

        if target.identity.company_name and word_in_text(
            target.identity.company_name,
            haystack,
        ):
            matched_folders.append(target.folder_name)
            continue

        for name in target.identity.human_names:
            if name.lower() in ANTLER_STAFF:
                continue
            name_parts = name.split()
            first_name = name_parts[0] if name_parts else ""
            if word_in_text(name, haystack) or (
                len(first_name) >= 3 and word_in_text(first_name, haystack)
            ):
                matched_folders.append(target.folder_name)
                break

    unique = sorted(set(matched_folders))
    if len(unique) == 1:
        return MatchResult(
            deal_folder=unique[0],
            reason=f"Matched {unique[0]} by name or folder in {source_label}.",
        )
    return None


def find_programmatic_deal_match(
    meeting: Meeting,
    sentences: list[Sentence],
    targets: list[DealMatchTarget],
) -> MatchResult | None:
    haystack = meeting_match_haystack(meeting, sentences)
    return find_programmatic_deal_match_from_haystack(
        haystack,
        targets,
        source_label="meeting content",
    )


def find_matching_deal(
    meeting: Meeting,
    sentences: list[Sentence],
    targets: list[DealMatchTarget],
    *,
    api_key: str,
    model: str,
) -> MatchResult:
    matches: list[tuple[str, str]] = []
    for target in targets:
        relevance = classify_relevance(
            meeting,
            sentences,
            target.identity,
            api_key=api_key,
            model=model,
        )
        if relevance.relevant:
            matches.append((target.folder_name, relevance.reason))

    if len(matches) == 1:
        return MatchResult(deal_folder=matches[0][0], reason=matches[0][1])
    if len(matches) > 1:
        folder_names = ", ".join(name for name, _ in matches)
        return MatchResult(
            deal_folder=None,
            reason=f"Multiple deals matched: {folder_names}.",
        )
    return MatchResult(
        deal_folder=None,
        reason="No deal identity matched the meeting.",
    )


def meeting_date_label(timestamp_start_utc: str) -> str:
    meeting_start = parse_meeting_start(timestamp_start_utc)
    if meeting_start is None:
        return "unknown date"
    return meeting_start.date().isoformat()


def process_meeting(
    folder: Path,
    meeting_id: str,
    identity: DealIdentity,
    *,
    api_key: str,
    model: str,
    dry_run: bool = False,
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
            filename=transcript_relative_path(existing.name),
            reason="Transcript already present in transcripts folder.",
        )

    relevance = classify_relevance(
        meeting,
        sentences,
        identity,
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
    relative_filename = transcript_relative_path(filename)
    if dry_run:
        return MeetingOutcome(
            status="would_write",
            title=meeting.title,
            date_label=date_label,
            filename=relative_filename,
            reason=relevance.reason,
        )

    output_dir = transcripts_dir(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(
        format_transcript_text(meeting, sentences),
        encoding="utf-8",
    )
    return MeetingOutcome(
        status="written",
        title=meeting.title,
        date_label=date_label,
        filename=relative_filename,
        reason=relevance.reason,
    )


def print_outcome(outcome: MeetingOutcome) -> None:
    if outcome.status == "written":
        print(f"WRITTEN: {outcome.filename}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "would_write":
        print(f"WOULD WRITE: {outcome.filename}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "skipped":
        print(f"SKIPPED (already present): {outcome.filename}")
        if outcome.reason:
            print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "already_processed":
        print(f"ALREADY PROCESSED: {outcome.title} ({outcome.date_label})")
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
            "into a deal transcripts folder."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the deal folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without writing files.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Ignore the processed-meetings log and re-analyze all meetings.",
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

    documents = collect_deal_context(folder)
    if not documents:
        print(
            f"Warning: no deal context documents found in {folder}",
            file=sys.stderr,
        )
    deal_payload = build_deal_payload(documents) if documents else "[no deal documents]"

    try:
        identity = extract_deal_identity(
            deal_payload,
            deal_folder_name=folder.name,
            api_key=api_key,
            model=model,
        )
    except Exception as exc:
        print(f"Error: failed to extract deal identity: {exc}", file=sys.stderr)
        return 1

    print_deal_identity(identity)

    if args.dry_run:
        print("Dry run: no files will be written.")
        print()

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

    processed_ids = load_processed_meeting_ids()

    outcomes: list[MeetingOutcome] = []
    for summary in meeting_summaries:
        if not args.reprocess and summary.meeting_id in processed_ids:
            outcome = MeetingOutcome(
                status="already_processed",
                title=summary.meeting_id,
                date_label=meeting_date_label(summary.timestamp_start_utc),
                reason="Meeting ID already in processed_meetgeek_meetings.txt.",
            )
            outcomes.append(outcome)
            print_outcome(outcome)
            continue

        try:
            outcome = process_meeting(
                folder,
                summary.meeting_id,
                identity,
                api_key=api_key,
                model=model,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            outcome = MeetingOutcome(
                status="error",
                title=summary.meeting_id,
                date_label=meeting_date_label(summary.timestamp_start_utc),
                reason=str(exc),
            )
            print(f"Error processing meeting {summary.meeting_id}: {exc}", file=sys.stderr)

        if (
            not args.dry_run
            and should_record_processed(outcome.status)
        ):
            append_processed_meeting_id(
                summary.meeting_id,
                known_ids=processed_ids,
            )

        outcomes.append(outcome)
        print_outcome(outcome)

    written = sum(1 for outcome in outcomes if outcome.status == "written")
    would_write = sum(1 for outcome in outcomes if outcome.status == "would_write")
    skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    already_processed = sum(
        1 for outcome in outcomes if outcome.status == "already_processed"
    )
    not_relevant = sum(1 for outcome in outcomes if outcome.status == "not_relevant")
    errors = sum(1 for outcome in outcomes if outcome.status == "error")

    print()
    summary_parts = [
        f"{len(meeting_summaries)} meetings in last {LOOKBACK_DAYS} days",
    ]
    if args.dry_run:
        summary_parts.append(f"{would_write} would write")
    else:
        summary_parts.append(f"{written} written")
    summary_parts.extend(
        [
            f"{skipped} skipped",
            f"{already_processed} already processed",
            f"{not_relevant} not relevant",
        ]
    )
    if errors:
        summary_parts.append(f"{errors} errors")

    print("FETCHED: " + " | ".join(summary_parts))
    return 1 if errors and written == 0 and would_write == 0 and skipped == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
