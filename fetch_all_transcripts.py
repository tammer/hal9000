#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from document_utils import read_file_as_text
from fetch_transcripts import (
    DealIdentity,
    attendee_names,
    build_deal_payload,
    extract_deal_identity,
    find_existing_transcript,
    format_transcript_text,
    groq_json_chat,
    is_meetgeek_transcript,
    meeting_date_label,
    transcript_basename,
    transcript_excerpt,
)
from meetgeek_client import (
    MeetGeekError,
    Meeting,
    Sentence,
    get_meeting,
    get_transcript,
    list_team_meetings,
)

DEFAULT_LOOKBACK_DAYS = 2

ANTLER_STAFF = {
    "tammer kamel",
    "shambhavi mishra",
    "alex wright",
    "daphne mclarty",
    "bernie li",
}

TRANSCRIPT_IDENTITY_SYSTEM_PROMPT = """You extract the company and people associated with a MeetGeek meeting.

Return valid JSON only with this exact shape:
{"company_name": "Acme Inc" or null, "human_names": ["Full Name", ...]}

Include:
- company_name: the startup or company discussed in the meeting, if clearly named; otherwise null
- human_names: full names of founders and external contacts (not Antler staff)

Rules:
- These Antler team members appear on many meetings and must be excluded from human_names:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Use full names as they appear in the meeting title, attendee list, emails, or transcript
- Include first names from the meeting title when that is the only form present (e.g. "Alex <> Jad" -> include "Jad")
- Do not invent names or companies not supported by the meeting content
- human_names must contain only person name strings; never include explanations or commentary in JSON values
"""

DEAL_MATCH_SYSTEM_PROMPT = """You decide which single startup deal folder a MeetGeek meeting belongs to.

You are given meeting metadata, the meeting's extracted company/people identity, and a catalog of deal folders with their identities.

Return valid JSON only with this exact shape:
{"deal_folder": "FolderName" or null, "reason": "short explanation"}

A meeting matches a deal when the company name and/or a deal person's name appears in the meeting title, attendees, emails, host email, transcript, or extracted meeting identity.

Matching rules:
- First-name matches count: "Jad" in a meeting title matches deal person "Jad Fadlallah" in folder "Jad"
- Deal folder names are often a founder's first name; if the folder name appears in the meeting title or content, that is strong evidence
- Company name matches count when the company name clearly appears in the meeting
- These Antler team members appear on ALL deals and must NEVER determine a match:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Do NOT match different similar-sounding names for different people (e.g. Chen is not Chan)
- Do NOT infer company matches from email domains alone
- Do NOT match based on shared generic topics alone
- Return at most one deal_folder; if no deal matches, return null
- If multiple deals could match, pick the one with the strongest name/company evidence; if still tied, return null

reason must be one concise sentence. If deal_folder is set, name the matching company or person and the deal folder.
- deal_folder must be a deal folder name string or null; never include explanations in JSON values
"""


@dataclass(frozen=True)
class DealCatalogEntry:
    folder: Path
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
    deal_folder: str | None = None
    filename: str | None = None
    reason: str = ""


def deals_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    return Path(base_raw).resolve()


def default_cutoff_date() -> date:
    return date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def parse_cutoff_date(value: str) -> datetime:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid cutoff date {value!r}; expected YYYY-MM-DD") from exc
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)


def collect_deal_context_for_matching(folder: Path) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []

    summary_path = folder / "ai-generated" / "summary.md"
    if summary_path.is_file():
        summary_text = read_file_as_text(summary_path)
        if summary_text:
            documents.append((summary_path, summary_text))
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


def folder_has_any_files(folder: Path) -> bool:
    for entry in folder.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            return True
        if entry.is_dir():
            for child in entry.iterdir():
                if not child.name.startswith(".") and child.is_file():
                    return True
    return False


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


def load_deal_catalog(
    base: Path,
    *,
    api_key: str,
    model: str,
) -> list[DealCatalogEntry]:
    catalog: list[DealCatalogEntry] = []

    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not folder_has_any_files(entry):
            continue

        documents = collect_deal_context_for_matching(entry)
        if not documents:
            continue

        deal_payload = build_deal_payload(documents)
        try:
            identity = extract_deal_identity(
                deal_payload,
                deal_folder_name=entry.name,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            print(
                f"Warning: skipping deal {entry.name}; failed to extract identity: {exc}",
                file=sys.stderr,
            )
            continue
        catalog.append(
            DealCatalogEntry(
                folder=entry,
                folder_name=entry.name,
                identity=identity,
            )
        )

    return catalog


def print_deal_catalog(catalog: list[DealCatalogEntry]) -> None:
    print(f"Loaded {len(catalog)} deal(s):")
    for deal in catalog:
        company = deal.identity.company_name or "(none)"
        people = ", ".join(deal.identity.human_names) or "(none)"
        print(f"  {deal.folder_name}: company={company}; people={people}")
    print()


def build_transcript_identity_prompt(
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


def extract_transcript_identity(
    meeting: Meeting,
    sentences: list[Sentence],
    *,
    api_key: str,
    model: str,
) -> DealIdentity:
    payload = groq_json_chat(
        system_prompt=TRANSCRIPT_IDENTITY_SYSTEM_PROMPT,
        user_prompt=build_transcript_identity_prompt(meeting, sentences),
        api_key=api_key,
        model=model,
    )
    return parse_identity_payload(payload)


def format_deal_catalog_for_prompt(catalog: list[DealCatalogEntry]) -> str:
    lines: list[str] = []
    for deal in catalog:
        company = deal.identity.company_name or "(none)"
        people = ", ".join(deal.identity.human_names) or "(none)"
        lines.append(
            f"- folder: {deal.folder_name}\n"
            f"  company: {company}\n"
            f"  people: {people}"
        )
    return "\n".join(lines)


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
    catalog: list[DealCatalogEntry],
    *,
    source_label: str = "content",
) -> MatchResult | None:
    matched_folders: list[str] = []

    for deal in catalog:
        if word_in_text(deal.folder_name, haystack):
            matched_folders.append(deal.folder_name)
            continue

        if deal.identity.company_name and word_in_text(
            deal.identity.company_name,
            haystack,
        ):
            matched_folders.append(deal.folder_name)
            continue

        for name in deal.identity.human_names:
            if name.lower() in ANTLER_STAFF:
                continue
            name_parts = name.split()
            first_name = name_parts[0] if name_parts else ""
            if word_in_text(name, haystack) or (
                len(first_name) >= 3 and word_in_text(first_name, haystack)
            ):
                matched_folders.append(deal.folder_name)
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
    catalog: list[DealCatalogEntry],
) -> MatchResult | None:
    haystack = meeting_match_haystack(meeting, sentences)
    return find_programmatic_deal_match_from_haystack(
        haystack,
        catalog,
        source_label="meeting content",
    )


def build_deal_match_prompt(
    meeting: Meeting,
    sentences: list[Sentence],
    transcript_identity: DealIdentity,
    catalog: list[DealCatalogEntry],
) -> str:
    attendees = ", ".join(attendee_names(meeting)) or "unknown"
    participant_emails = ", ".join(meeting.participant_emails) or "unknown"
    company = transcript_identity.company_name or "(none)"
    people = ", ".join(transcript_identity.human_names) or "(none)"

    return (
        "Meeting metadata:\n"
        f"- Title: {meeting.title}\n"
        f"- Date: {meeting.timestamp_start_utc}\n"
        f"- Attendees: {attendees}\n"
        f"- Participant emails: {participant_emails}\n"
        f"- Host email: {meeting.host_email or 'unknown'}\n\n"
        "Extracted meeting identity:\n"
        f"- Company: {company}\n"
        f"- People: {people}\n\n"
        "Deal catalog:\n"
        f"{format_deal_catalog_for_prompt(catalog)}"
    )


def match_transcript_to_deal(
    meeting: Meeting,
    sentences: list[Sentence],
    transcript_identity: DealIdentity,
    catalog: list[DealCatalogEntry],
    *,
    api_key: str,
    model: str,
) -> MatchResult:
    programmatic = find_programmatic_deal_match(meeting, sentences, catalog)
    if programmatic is not None:
        return programmatic

    payload = groq_json_chat(
        system_prompt=DEAL_MATCH_SYSTEM_PROMPT,
        user_prompt=build_deal_match_prompt(
            meeting,
            sentences,
            transcript_identity,
            catalog,
        ),
        api_key=api_key,
        model=model,
    )
    deal_folder_raw = payload.get("deal_folder")
    deal_folder = str(deal_folder_raw).strip() if deal_folder_raw else None
    if deal_folder and deal_folder.lower() in {"null", "none", ""}:
        deal_folder = None

    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    return MatchResult(deal_folder=deal_folder, reason=reason)


def resolve_catalog_entry(
    catalog: list[DealCatalogEntry],
    deal_folder: str,
) -> DealCatalogEntry | None:
    for deal in catalog:
        if deal.folder_name == deal_folder:
            return deal
    return None


def process_meeting(
    meeting_id: str,
    catalog: list[DealCatalogEntry],
    *,
    api_key: str,
    model: str,
    dry_run: bool,
) -> MeetingOutcome:
    meeting = get_meeting(meeting_id)
    sentences = get_transcript(meeting_id)
    basename = transcript_basename(meeting.title, meeting.timestamp_start_utc)
    date_label = meeting_date_label(meeting.timestamp_start_utc)

    transcript_identity = extract_transcript_identity(
        meeting,
        sentences,
        api_key=api_key,
        model=model,
    )
    match = match_transcript_to_deal(
        meeting,
        sentences,
        transcript_identity,
        catalog,
        api_key=api_key,
        model=model,
    )

    if not match.deal_folder:
        return MeetingOutcome(
            status="no_match",
            title=meeting.title,
            date_label=date_label,
            reason=match.reason,
        )

    deal = resolve_catalog_entry(catalog, match.deal_folder)
    if deal is None:
        return MeetingOutcome(
            status="no_match",
            title=meeting.title,
            date_label=date_label,
            reason=f"Model returned unknown deal folder: {match.deal_folder}",
        )

    existing = find_existing_transcript(deal.folder, basename, meeting.meeting_id)
    if existing is not None:
        return MeetingOutcome(
            status="skipped",
            title=meeting.title,
            date_label=date_label,
            deal_folder=deal.folder_name,
            filename=existing.name,
            reason="Transcript already present in deal folder.",
        )

    filename = f"{basename}.txt"
    if dry_run:
        return MeetingOutcome(
            status="would_write",
            title=meeting.title,
            date_label=date_label,
            deal_folder=deal.folder_name,
            filename=filename,
            reason=match.reason,
        )

    output_path = deal.folder / filename
    output_path.write_text(
        format_transcript_text(meeting, sentences),
        encoding="utf-8",
    )
    return MeetingOutcome(
        status="written",
        title=meeting.title,
        date_label=date_label,
        deal_folder=deal.folder_name,
        filename=filename,
        reason=match.reason,
    )


def print_outcome(outcome: MeetingOutcome) -> None:
    if outcome.status == "written":
        print(f"WRITTEN: {outcome.deal_folder}/{outcome.filename}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "would_write":
        print(f"WOULD WRITE: {outcome.deal_folder}/{outcome.filename}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "skipped":
        print(f"SKIPPED (already present): {outcome.deal_folder}/{outcome.filename}")
        if outcome.reason:
            print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "no_match":
        print(f"NO MATCH: {outcome.title} ({outcome.date_label})")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "error":
        print(f"ERROR: {outcome.title} ({outcome.date_label})")
        print(f"  Reason: {outcome.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch team MeetGeek transcripts since a cutoff date and write "
            "each relevant one to its deal folder."
        )
    )
    parser.add_argument(
        "--cutoff-date",
        default=default_cutoff_date().isoformat(),
        help=(
            "Include meetings on or after this date (YYYY-MM-DD). "
            f"Default: {DEFAULT_LOOKBACK_DAYS} days ago."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    team_id = os.getenv("MEETGEEK_TEAM_ID", "").strip()
    if not team_id:
        print("Error: MEETGEEK_TEAM_ID is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        cutoff = parse_cutoff_date(args.cutoff_date)
        base = deals_base()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.exists() or not base.is_dir():
        print(f"Error: deals base is not a directory: {base}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run: no files will be written.")
        print()

    try:
        catalog = load_deal_catalog(base, api_key=api_key, model=model)
    except Exception as exc:
        print(f"Error: failed to load deal catalog: {exc}", file=sys.stderr)
        return 1

    if not catalog:
        print("Error: no deal folders with usable documents found", file=sys.stderr)
        return 1

    print_deal_catalog(catalog)

    try:
        meeting_summaries = list_team_meetings(team_id, cutoff)
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
                summary.meeting_id,
                catalog,
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
            print(
                f"Error processing meeting {summary.meeting_id}: {exc}",
                file=sys.stderr,
            )
        outcomes.append(outcome)
        print_outcome(outcome)

    written = sum(1 for outcome in outcomes if outcome.status == "written")
    would_write = sum(1 for outcome in outcomes if outcome.status == "would_write")
    skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    no_match = sum(1 for outcome in outcomes if outcome.status == "no_match")
    errors = sum(1 for outcome in outcomes if outcome.status == "error")

    print()
    summary_parts = [
        f"{len(meeting_summaries)} meetings since {args.cutoff_date}",
    ]
    if args.dry_run:
        summary_parts.append(f"{would_write} would write")
    else:
        summary_parts.append(f"{written} written")
    summary_parts.extend(
        [
            f"{skipped} skipped",
            f"{no_match} unmatched",
        ]
    )
    if errors:
        summary_parts.append(f"{errors} errors")

    print("FETCHED: " + " | ".join(summary_parts))
    return 1 if errors and written == 0 and would_write == 0 and skipped == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
