#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from document_utils import read_file_as_text
from get_facts import parse_json_response
from meetgeek_client import (
    MeetGeekError,
    Meeting,
    Sentence,
    get_meeting,
    get_transcript,
    list_recent_meetings,
)
from paths import (
    deals_base,
    portcos_base,
    resolve_company_folder as resolve_company_folder_path,
)

LOOKBACK_DAYS = 8
MAX_DEAL_DOC_CHARS = 100_000
TRANSCRIPT_EXCERPT_CHARS = 3_000
CONTEXT_BRIEF_CHARS = 2_000
MEETING_LINK_PREFIX = "https://app.meetgeek.ai/meeting/"

TRANSCRIPT_FILENAME_MARKER = "_sentences_"
TRANSCRIPTS_DIR_NAME = "transcripts"
EMAILS_DIR_NAME = "emails"
IDENTITY_FILENAME = "identity.json"
AI_GENERATED_DIR_NAME = "ai-generated"
SHORTLIST_LIMIT = 3
MIN_FOLDER_NAME_MATCH_LEN = 4
MIN_ALIAS_MATCH_LEN = 4
STRONG_SIGNAL_SCORE = 10
WEAK_SIGNAL_SCORE = 1

PROCESSED_MEETINGS_PATH = (
    Path(__file__).resolve().parent / "processed_meetgeek_meetings.json"
)
LEGACY_PROCESSED_MEETINGS_PATH = (
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

IGNORED_EMAIL_DOMAINS = frozenset(
    {
        "antler.co",
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "yahoo.ca",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "icloud.com",
        "me.com",
        "mac.com",
        "aol.com",
        "protonmail.com",
        "proton.me",
        "msn.com",
    }
)

EMAIL_ADDRESS_RE = re.compile(
    r"[A-Z0-9._%+\-]+@([A-Z0-9.\-]+\.[A-Z]{2,})",
    re.IGNORECASE,
)

IDENTITY_EXTRACTION_SYSTEM_PROMPT = """You extract the canonical identity of a startup deal from its deal documents.

Return valid JSON only with this exact shape:
{"company_name": "Acme Inc" or null, "human_names": ["Full Name", ...], "aliases": ["Alt Name", ...], "product_summary": "one-line product blurb" or null}

Include:
- company_name: the startup or company name if one is clearly named; otherwise null
- human_names: full names of founders (not Antler staff)
- aliases: alternate company names, product names, or brand names clearly supported by the documents
- product_summary: one concise sentence describing what the company/product does; otherwise null

Rules:
- These Antler team members appear on many deals and must be excluded from human_names:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Use full names as written in the documents when possible
- The deal folder name is often a founder's first name and may be misspelled; use it only as a weak hint
- Do not invent names, companies, aliases, or product details not supported by the documents
- human_names and aliases must contain only strings; never include explanations or commentary in JSON values
"""

JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return only valid JSON with no commentary inside values or arrays."
)
MAX_JSON_RETRIES = 3

TRANSCRIPT_RELEVANCE_SYSTEM_PROMPT = """You decide whether a MeetGeek meeting belongs to a specific startup deal.

You are given a rich deal identity (company, people, aliases, email domains, product summary, and a context brief from the deal folder) plus meeting metadata and a transcript excerpt.

Return valid JSON only with this exact shape:
{"relevant": true, "reason": "explanation of why the meeting is relevant"}

A meeting is relevant when EITHER of the following is true:

1. Identity evidence: the company name, an alias, a human name, or an email domain from the deal clearly appears in the meeting title, attendee names, participant emails, host email, or transcript text.
2. Context evidence: the discussion clearly aligns with this deal's product/context brief (not generic startup talk).

Conversation nature:
- Diligence-style Q&A counts.
- Founder/company check-ins and progress updates count when identity or context points at this deal.
- Internal Antler syncs about this specific deal count when identity or context clearly points at it.
- Unrelated social catch-ups or meetings about a different company do NOT count.

Hard rules:
- These Antler team members appear on ALL deals and must NEVER alone determine relevance:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Do NOT match similar-sounding or partially similar names (e.g. Chen is not Chan)
- Do NOT mark relevant based on shared generic topics alone
- If the meeting discusses a different company or topic and a deal name appears only coincidentally, set relevant=false
- When evidence is ambiguous, set relevant=false

reason must be one concise sentence. If relevant=true, name the matching evidence and briefly note why it fits this deal.
"""

MEETING_DEAL_MATCH_SYSTEM_PROMPT = """You decide which single startup deal folder a MeetGeek meeting belongs to.

You are given meeting metadata, a transcript excerpt, and a SHORTLIST of candidate deal folders with rich identities (company, people, aliases, email domains, product summary, context brief).

Return valid JSON only with this exact shape:
{"deal_folder": "FolderName" or null, "reason": "short explanation"}

A meeting matches a candidate when EITHER of the following is true for that candidate:

1. Identity evidence: the company name, an alias, a human name, or an email domain from that deal clearly appears in the meeting title, attendee names, participant emails, host email, or transcript text.
2. Context evidence: the discussion clearly aligns with that deal's product/context brief (not generic startup talk).

Conversation nature:
- Diligence-style Q&A counts.
- Founder/company check-ins and progress updates count when identity or context points at that deal.
- Internal Antler syncs about that specific deal count when identity or context clearly points at it.
- Unrelated social catch-ups or meetings about a different company do NOT count.

Hard rules:
- These Antler team members appear on ALL deals and must NEVER alone determine a match:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Do NOT match similar-sounding or partially similar names (e.g. Chen is not Chan)
- Do NOT match based on shared generic topics alone
- Return at most one deal_folder; choose ONLY from the provided shortlist; if none match, return null
- If multiple candidates could match, pick the one with the strongest evidence; if still tied, return null
- When evidence is ambiguous, return null

reason must be one concise sentence. If deal_folder is set, name the matching evidence and briefly note why it fits that deal.
- deal_folder must be a deal folder name string or null; never include explanations in JSON values
"""


@dataclass(frozen=True)
class DealIdentity:
    company_name: str | None
    human_names: list[str]
    aliases: list[str] = field(default_factory=list)
    email_domains: list[str] = field(default_factory=list)
    product_summary: str | None = None
    context_brief: str | None = None


@dataclass(frozen=True)
class RelevanceResult:
    relevant: bool
    reason: str


@dataclass(frozen=True)
class DealMatchTarget:
    folder_name: str
    identity: DealIdentity


@dataclass(frozen=True)
class MatchCandidate:
    folder_name: str
    identity: DealIdentity
    score: int
    strong_hits: int
    weak_hits: int
    reasons: list[str]


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


@dataclass(frozen=True)
class ProcessedMeetingRecord:
    meeting_id: str
    decision: str
    reason: str
    deal_folder: str | None
    title: str
    date: str


def _legacy_processed_record(meeting_id: str) -> ProcessedMeetingRecord:
    return ProcessedMeetingRecord(
        meeting_id=meeting_id,
        decision="unknown",
        reason="unknown",
        deal_folder="unknown",
        title="unknown",
        date="unknown",
    )


def _parse_processed_meeting_line(line: str) -> ProcessedMeetingRecord | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        meeting_id = str(payload.get("meeting_id", "")).strip()
        if not meeting_id:
            return None
        deal_folder_raw = payload.get("deal_folder")
        if deal_folder_raw is None:
            deal_folder: str | None = None
        else:
            deal_folder = str(deal_folder_raw)
        return ProcessedMeetingRecord(
            meeting_id=meeting_id,
            decision=str(payload.get("decision", "unknown")),
            reason=str(payload.get("reason", "unknown")),
            deal_folder=deal_folder,
            title=str(payload.get("title", "unknown")),
            date=str(payload.get("date", "unknown")),
        )

    meeting_id = stripped.split()[0]
    if not meeting_id:
        return None
    return _legacy_processed_record(meeting_id)


def load_processed_meeting_records(
    path: Path = PROCESSED_MEETINGS_PATH,
) -> list[ProcessedMeetingRecord]:
    if not path.is_file():
        return []
    records: list[ProcessedMeetingRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = _parse_processed_meeting_line(line)
        if record is not None:
            records.append(record)
    return records


def load_processed_meeting_ids(
    path: Path = PROCESSED_MEETINGS_PATH,
) -> set[str]:
    return {record.meeting_id for record in load_processed_meeting_records(path)}


def migrate_processed_meetings_log(
    path: Path = PROCESSED_MEETINGS_PATH,
    *,
    legacy_path: Path = LEGACY_PROCESSED_MEETINGS_PATH,
) -> int:
    """Rewrite plain-ID lines to JSONL records. Returns number of legacy lines migrated."""
    if not path.is_file() and legacy_path.is_file():
        legacy_path.replace(path)

    if not path.is_file():
        return 0

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records: list[ProcessedMeetingRecord] = []
    legacy_count = 0
    needs_rewrite = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("{"):
            record = _parse_processed_meeting_line(stripped)
            if record is None:
                needs_rewrite = True
                continue
            records.append(record)
            continue

        meeting_id = stripped.split()[0]
        if not meeting_id:
            needs_rewrite = True
            continue
        records.append(_legacy_processed_record(meeting_id))
        legacy_count += 1
        needs_rewrite = True

    if not needs_rewrite:
        return 0

    # Preserve first-seen meeting_id if duplicates appear during migration.
    seen: set[str] = set()
    unique_records: list[ProcessedMeetingRecord] = []
    for record in records:
        if record.meeting_id in seen:
            continue
        seen.add(record.meeting_id)
        unique_records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in unique_records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return legacy_count


def append_processed_meeting(
    record: ProcessedMeetingRecord,
    path: Path = PROCESSED_MEETINGS_PATH,
    *,
    known_ids: set[str] | None = None,
) -> None:
    cleaned = record.meeting_id.strip()
    if not cleaned:
        return
    if known_ids is not None and cleaned in known_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["meeting_id"] = cleaned
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
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


def transcript_basename(
    title: str,
    timestamp_start_utc: str,
    meeting_id: str,
) -> str:
    return (
        f"{sanitize_title_for_filename(title)}"
        f"{TRANSCRIPT_FILENAME_MARKER}{filename_timestamp(timestamp_start_utc)}"
        f"__{meeting_id}"
    )


def is_meetgeek_transcript(path: Path) -> bool:
    return path.suffix.lower() == ".txt" and TRANSCRIPT_FILENAME_MARKER in path.name


def transcripts_dir(folder: Path) -> Path:
    return folder / TRANSCRIPTS_DIR_NAME


def transcript_relative_path(filename: str) -> str:
    return f"{TRANSCRIPTS_DIR_NAME}/{filename}"


def identity_path(folder: Path) -> Path:
    return folder / AI_GENERATED_DIR_NAME / IDENTITY_FILENAME


def summary_path(folder: Path) -> Path:
    return folder / AI_GENERATED_DIR_NAME / "summary.md"


def emails_dir(folder: Path) -> Path:
    return folder / EMAILS_DIR_NAME


def collect_deal_context(
    folder: Path,
    *,
    summary_only: bool = False,
) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []

    summary = summary_path(folder)
    if summary.is_file():
        summary_text = read_file_as_text(summary)
        if summary_text:
            documents.append((summary, summary_text))
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


def normalize_email_domain(domain: str) -> str | None:
    cleaned = domain.strip().lower().lstrip("@")
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    if not cleaned or "." not in cleaned:
        return None
    if cleaned in IGNORED_EMAIL_DOMAINS:
        return None
    return cleaned


def extract_email_domains_from_text(text: str) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()
    for match in EMAIL_ADDRESS_RE.finditer(text):
        domain = normalize_email_domain(match.group(1))
        if domain is None or domain in seen:
            continue
        seen.add(domain)
        domains.append(domain)
    return domains


def harvest_email_domains(folder: Path) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()
    emails_folder = emails_dir(folder)
    if not emails_folder.is_dir():
        return domains

    for entry in sorted(emails_folder.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() != ".txt":
            continue
        text = read_file_as_text(entry)
        if not text:
            continue
        for domain in extract_email_domains_from_text(text):
            if domain in seen:
                continue
            seen.add(domain)
            domains.append(domain)
    return domains


def truncate_context_brief(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if len(cleaned) <= CONTEXT_BRIEF_CHARS:
        return cleaned
    return (
        cleaned[:CONTEXT_BRIEF_CHARS]
        + "\n\n[Note: context brief truncated due to size limits.]"
    )


def empty_deal_identity() -> DealIdentity:
    return DealIdentity(
        company_name=None,
        human_names=[],
        aliases=[],
        email_domains=[],
        product_summary=None,
        context_brief=None,
    )


def identity_to_dict(identity: DealIdentity) -> dict:
    return {
        "company_name": identity.company_name,
        "human_names": list(identity.human_names),
        "aliases": list(identity.aliases),
        "email_domains": list(identity.email_domains),
        "product_summary": identity.product_summary,
        "context_brief": identity.context_brief,
    }


def parse_identity_payload(payload: dict) -> DealIdentity:
    company_raw = payload.get("company_name")
    company_name = str(company_raw).strip() if company_raw else None
    if company_name and company_name.lower() in {"null", "none", ""}:
        company_name = None

    human_names: list[str] = []
    for name in payload.get("human_names", []) or []:
        cleaned = str(name).strip()
        if cleaned:
            human_names.append(cleaned)

    aliases: list[str] = []
    for alias in payload.get("aliases", []) or []:
        cleaned = str(alias).strip()
        if cleaned:
            aliases.append(cleaned)

    product_raw = payload.get("product_summary")
    product_summary = str(product_raw).strip() if product_raw else None
    if product_summary and product_summary.lower() in {"null", "none", ""}:
        product_summary = None

    domains: list[str] = []
    for domain in payload.get("email_domains", []) or []:
        normalized = normalize_email_domain(str(domain))
        if normalized and normalized not in domains:
            domains.append(normalized)

    context_brief = truncate_context_brief(
        str(payload.get("context_brief") or "").strip() or None
    )

    return DealIdentity(
        company_name=company_name,
        human_names=human_names,
        aliases=aliases,
        email_domains=domains,
        product_summary=product_summary,
        context_brief=context_brief,
    )


def load_identity_json(path: Path) -> DealIdentity | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return parse_identity_payload(payload)


def write_identity_json(folder: Path, identity: DealIdentity) -> Path:
    path = identity_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity_to_dict(identity), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def identity_cache_is_fresh(folder: Path) -> bool:
    path = identity_path(folder)
    if not path.is_file():
        return False
    summary = summary_path(folder)
    if not summary.is_file():
        return True
    return path.stat().st_mtime >= summary.stat().st_mtime


def merge_identity_domains(
    identity: DealIdentity,
    domains: list[str],
) -> DealIdentity:
    merged = list(identity.email_domains)
    seen = set(merged)
    for domain in domains:
        normalized = normalize_email_domain(domain)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    if merged == list(identity.email_domains):
        return identity
    return DealIdentity(
        company_name=identity.company_name,
        human_names=list(identity.human_names),
        aliases=list(identity.aliases),
        email_domains=merged,
        product_summary=identity.product_summary,
        context_brief=identity.context_brief,
    )


def with_context_brief(identity: DealIdentity, brief: str | None) -> DealIdentity:
    truncated = truncate_context_brief(brief)
    if truncated == identity.context_brief:
        return identity
    return DealIdentity(
        company_name=identity.company_name,
        human_names=list(identity.human_names),
        aliases=list(identity.aliases),
        email_domains=list(identity.email_domains),
        product_summary=identity.product_summary,
        context_brief=truncated,
    )


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
    meeting_id: str,
) -> Path | None:
    """Return an existing transcript for this MeetGeek meeting ID, if any.

    Prefer a filename that embeds the ID (go-forward naming). Fall back to
    scanning .txt contents so legacy files (ID only in the Link line) still
    dedupe correctly.
    """
    target = transcripts_dir(folder)
    if not target.is_dir():
        return None

    needle = meeting_id.lower()
    if not needle:
        return None

    for entry in target.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() != ".txt":
            continue
        if needle in entry.name.lower():
            return entry

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


def load_or_build_identity(
    folder: Path,
    *,
    api_key: str,
    model: str,
    refresh: bool = False,
) -> DealIdentity:
    path = identity_path(folder)
    harvested = harvest_email_domains(folder)

    if not refresh and identity_cache_is_fresh(folder):
        cached = load_identity_json(path)
        if cached is not None:
            merged = merge_identity_domains(cached, harvested)
            if merged != cached:
                write_identity_json(folder, merged)
            return merged

    documents = collect_deal_context(folder, summary_only=True)
    if not documents:
        documents = collect_deal_context(folder, summary_only=False)

    if documents:
        deal_payload = build_deal_payload(documents)
        identity = extract_deal_identity(
            deal_payload,
            deal_folder_name=folder.name,
            api_key=api_key,
            model=model,
        )
        brief_source = documents[0][1]
        identity = with_context_brief(identity, brief_source)
    else:
        identity = empty_deal_identity()

    identity = merge_identity_domains(identity, harvested)
    write_identity_json(folder, identity)
    return identity


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
    if identity.aliases:
        print(f"  Aliases: {', '.join(identity.aliases)}")
    if identity.email_domains:
        print(f"  Domains: {', '.join(identity.email_domains)}")
    if identity.product_summary:
        print(f"  Product: {identity.product_summary}")
    print()


def format_identity_for_prompt(
    identity: DealIdentity,
    *,
    include_context_brief: bool = True,
) -> str:
    company = identity.company_name or "(none)"
    people = ", ".join(identity.human_names) or "(none)"
    aliases = ", ".join(identity.aliases) or "(none)"
    domains = ", ".join(identity.email_domains) or "(none)"
    product = identity.product_summary or "(none)"
    lines = [
        f"  company: {company}",
        f"  people: {people}",
        f"  aliases: {aliases}",
        f"  email_domains: {domains}",
        f"  product_summary: {product}",
    ]
    if include_context_brief:
        brief = identity.context_brief or "(none)"
        lines.append(f"  context_brief:\n{brief}")
    return "\n".join(lines)


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
    return (
        "Deal identity:\n"
        f"{format_identity_for_prompt(identity)}\n\n"
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


def meeting_email_domains(meeting: Meeting) -> set[str]:
    domains: set[str] = set()
    emails = list(meeting.participant_emails)
    if meeting.host_email:
        emails.append(meeting.host_email)
    for email in emails:
        if "@" not in email:
            continue
        domain = normalize_email_domain(email.rsplit("@", 1)[-1])
        if domain:
            domains.add(domain)
    return domains


def score_target_against_haystack(
    target: DealMatchTarget,
    haystack: str,
    *,
    email_domains: set[str] | None = None,
) -> MatchCandidate | None:
    strong_hits = 0
    weak_hits = 0
    reasons: list[str] = []
    identity = target.identity

    if (
        len(target.folder_name) >= MIN_FOLDER_NAME_MATCH_LEN
        and word_in_text(target.folder_name, haystack)
    ):
        strong_hits += 1
        reasons.append(f"folder:{target.folder_name}")

    if identity.company_name and word_in_text(identity.company_name, haystack):
        strong_hits += 1
        reasons.append(f"company:{identity.company_name}")

    for alias in identity.aliases:
        if len(alias) >= MIN_ALIAS_MATCH_LEN and word_in_text(alias, haystack):
            strong_hits += 1
            reasons.append(f"alias:{alias}")
            break

    if email_domains:
        for domain in identity.email_domains:
            if domain.lower() in email_domains:
                strong_hits += 1
                reasons.append(f"domain:{domain}")
                break

    for name in identity.human_names:
        if name.lower() in ANTLER_STAFF:
            continue
        if word_in_text(name, haystack):
            strong_hits += 1
            reasons.append(f"person:{name}")
            continue
        name_parts = name.split()
        first_name = name_parts[0] if name_parts else ""
        if len(first_name) >= 3 and word_in_text(first_name, haystack):
            weak_hits += 1
            reasons.append(f"first_name:{first_name}")

    if strong_hits == 0 and weak_hits == 0:
        return None

    score = (strong_hits * STRONG_SIGNAL_SCORE) + (weak_hits * WEAK_SIGNAL_SCORE)
    return MatchCandidate(
        folder_name=target.folder_name,
        identity=identity,
        score=score,
        strong_hits=strong_hits,
        weak_hits=weak_hits,
        reasons=reasons,
    )


def domains_from_haystack(haystack: str) -> set[str]:
    return set(extract_email_domains_from_text(haystack))


def shortlist_deal_matches_from_haystack(
    haystack: str,
    targets: list[DealMatchTarget],
    *,
    email_domains: set[str] | None = None,
    limit: int = SHORTLIST_LIMIT,
) -> list[MatchCandidate]:
    resolved_domains = email_domains if email_domains is not None else domains_from_haystack(haystack)
    candidates: list[MatchCandidate] = []
    for target in targets:
        candidate = score_target_against_haystack(
            target,
            haystack,
            email_domains=resolved_domains,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.strong_hits,
            item.folder_name.lower(),
        ),
        reverse=True,
    )
    return candidates[:limit]


def shortlist_deal_matches(
    meeting: Meeting,
    sentences: list[Sentence],
    targets: list[DealMatchTarget],
    *,
    limit: int = SHORTLIST_LIMIT,
) -> list[MatchCandidate]:
    return shortlist_deal_matches_from_haystack(
        meeting_match_haystack(meeting, sentences),
        targets,
        email_domains=meeting_email_domains(meeting),
        limit=limit,
    )


def unique_strong_match(
    candidates: list[MatchCandidate],
    *,
    source_label: str = "content",
) -> MatchResult | None:
    strong = [candidate for candidate in candidates if candidate.strong_hits > 0]
    if len(strong) != 1:
        return None
    winner = strong[0]
    detail = ", ".join(winner.reasons) or "strong signal"
    return MatchResult(
        deal_folder=winner.folder_name,
        reason=f"Matched {winner.folder_name} by {detail} in {source_label}.",
    )


def find_programmatic_deal_match_from_haystack(
    haystack: str,
    targets: list[DealMatchTarget],
    *,
    source_label: str = "content",
    email_domains: set[str] | None = None,
) -> MatchResult | None:
    """Return a unique strong programmatic match, if any.

    Weak-only hits (e.g. first name) never auto-accept.
    """
    candidates = shortlist_deal_matches_from_haystack(
        haystack,
        targets,
        email_domains=email_domains,
        limit=len(targets) or 1,
    )
    return unique_strong_match(candidates, source_label=source_label)


def find_programmatic_deal_match(
    meeting: Meeting,
    sentences: list[Sentence],
    targets: list[DealMatchTarget],
) -> MatchResult | None:
    return find_programmatic_deal_match_from_haystack(
        meeting_match_haystack(meeting, sentences),
        targets,
        source_label="meeting content",
        email_domains=meeting_email_domains(meeting),
    )


def format_deal_targets_for_prompt(targets: list[DealMatchTarget]) -> str:
    lines: list[str] = []
    for target in targets:
        lines.append(
            f"- folder: {target.folder_name}\n"
            f"{format_identity_for_prompt(target.identity)}"
        )
    return "\n".join(lines)


def format_shortlist_for_prompt(candidates: list[MatchCandidate]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        signal = ", ".join(candidate.reasons) or "unknown"
        lines.append(
            f"- folder: {candidate.folder_name}\n"
            f"  shortlist_signals: {signal}\n"
            f"{format_identity_for_prompt(candidate.identity)}"
        )
    return "\n".join(lines)


def build_meeting_match_prompt(
    meeting: Meeting,
    sentences: list[Sentence],
    candidates: list[MatchCandidate],
) -> str:
    return (
        f"{build_meeting_metadata_prompt(meeting, sentences)}\n\n"
        "Candidate deal folders (shortlist):\n"
        f"{format_shortlist_for_prompt(candidates)}"
    )


def find_matching_deal(
    meeting: Meeting,
    sentences: list[Sentence],
    targets: list[DealMatchTarget],
    *,
    api_key: str,
    model: str,
) -> MatchResult:
    if not targets:
        return MatchResult(
            deal_folder=None,
            reason="No deal identity matched the meeting.",
        )

    candidates = shortlist_deal_matches(meeting, sentences, targets)
    if not candidates:
        return MatchResult(
            deal_folder=None,
            reason="No deal identity signals matched the meeting.",
        )

    known = {candidate.folder_name for candidate in candidates}
    payload = groq_json_chat(
        system_prompt=MEETING_DEAL_MATCH_SYSTEM_PROMPT,
        user_prompt=build_meeting_match_prompt(meeting, sentences, candidates),
        api_key=api_key,
        model=model,
    )
    deal_folder_raw = payload.get("deal_folder")
    deal_folder = str(deal_folder_raw).strip() if deal_folder_raw else None
    if deal_folder and deal_folder.lower() in {"null", "none", ""}:
        deal_folder = None

    if deal_folder and deal_folder not in known:
        return MatchResult(
            deal_folder=None,
            reason=f"Model returned unknown deal folder: {deal_folder}",
        )

    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    return MatchResult(deal_folder=deal_folder, reason=reason)


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
    basename = transcript_basename(
        meeting.title,
        meeting.timestamp_start_utc,
        meeting.meeting_id,
    )
    date_label = meeting_date_label(meeting.timestamp_start_utc)

    existing = find_existing_transcript(folder, meeting.meeting_id)
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
            "into a company transcripts folder."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Path as deals/<folder> or portcos/<folder> (e.g. deals/Tony)",
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
    parser.add_argument(
        "--refresh-identity",
        action="store_true",
        help="Force rebuild of ai-generated/identity.json for this folder.",
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
        folder = resolve_company_folder_path(args.relative_path)
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
        identity = load_or_build_identity(
            folder,
            api_key=api_key,
            model=model,
            refresh=args.refresh_identity,
        )
    except Exception as exc:
        print(f"Error: failed to load deal identity: {exc}", file=sys.stderr)
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

    migrated = migrate_processed_meetings_log()
    if migrated:
        print(f"Migrated {migrated} legacy processed-meeting ID(s) to JSONL.")
    processed_ids = load_processed_meeting_ids()

    outcomes: list[MeetingOutcome] = []
    for summary in meeting_summaries:
        if not args.reprocess and summary.meeting_id in processed_ids:
            outcome = MeetingOutcome(
                status="already_processed",
                title=summary.meeting_id,
                date_label=meeting_date_label(summary.timestamp_start_utc),
                reason="Meeting ID already has a prior decision in processed_meetgeek_meetings.json.",
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
            append_processed_meeting(
                ProcessedMeetingRecord(
                    meeting_id=summary.meeting_id,
                    decision=outcome.status,
                    reason=outcome.reason or "unknown",
                    deal_folder=folder.name,
                    title=outcome.title,
                    date=outcome.date_label,
                ),
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
