#!/usr/bin/env python3
from __future__ import annotations

import argparse
import imaplib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv

from fetch_all_transcripts import (
    DealCatalogEntry,
    catalog_targets,
    deals_base,
    load_combined_catalog,
    print_deal_catalog,
    resolve_catalog_entry,
)
from fetch_transcripts import (
    MatchResult,
    find_programmatic_deal_match_from_haystack,
    groq_json_chat,
)

EMAIL_FILENAME_PREFIX = "email_"
EMAILS_DIR_NAME = "emails"

EMAIL_DEAL_MATCH_SYSTEM_PROMPT = """You decide which single startup folder an email belongs to.

You are given email metadata, body text, and a catalog of deal and portfolio-company folders with their identities.

Return valid JSON only with this exact shape:
{"deal_folder": "FolderName" or null, "reason": "short explanation"}

An email matches a folder when the company name and/or a person's name from that folder appears in the subject, sender, recipients, or body.

Matching rules:
- First-name matches count: "Jad" in a subject matches person "Jad Fadlallah" in folder "Jad"
- Folder names are often a founder's first name; if the folder name appears in the email, that is strong evidence
- Company name matches count when the company name clearly appears in the email
- The catalog includes both active deals and portfolio companies; treat them the same for matching
- These Antler team members appear on ALL folders and must NEVER determine a match:
  Tammer Kamel, Shambhavi Mishra, Alex Wright, Daphne McLarty, Bernie Li
- Do NOT match different similar-sounding names for different people (e.g. Chen is not Chan)
- Do NOT infer company matches from email domains alone
- Do NOT match based on shared generic topics alone
- Return at most one deal_folder; if no folder matches, return null
- If multiple folders could match, pick the one with the strongest name/company evidence; if still tied, return null

reason must be one concise sentence. If deal_folder is set, name the matching company or person and the folder.
- deal_folder must be a folder name string or null; never include explanations in JSON values
"""


@dataclass(frozen=True)
class MailConfig:
    imap_host: str
    address: str
    password: str
    imap_port: int


@dataclass
class MessageOutcome:
    status: str
    uid: str
    subject: str
    sender: str
    deal_folder: str | None = None
    filename: str | None = None
    reason: str = ""


def load_mail_config() -> MailConfig:
    load_dotenv()
    missing = [
        key
        for key in ("MAIL_IMAP_HOST", "MAIL_ADDRESS", "MAIL_PASSWORD")
        if not os.getenv(key)
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return MailConfig(
        imap_host=os.environ["MAIL_IMAP_HOST"],
        address=os.environ["MAIL_ADDRESS"],
        password=os.environ["MAIL_PASSWORD"],
        imap_port=int(os.getenv("MAIL_IMAP_PORT", "993")),
    )


def normalize_address(address: str) -> str:
    _, email_address = parseaddr(address)
    return email_address.strip().lower()


def extract_plain_text(message: EmailMessage) -> str:
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, (bytes, bytearray)):
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="replace"))
        return "\n".join(parts).strip()

    payload = message.get_payload(decode=True)
    if isinstance(payload, (bytes, bytearray)):
        charset = message.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace").strip()
    if isinstance(payload, str):
        return payload.strip()
    return ""


def email_match_haystack(message: EmailMessage) -> str:
    parts = [
        message.get("Subject", ""),
        message.get("From", ""),
        message.get("To", ""),
        message.get("Cc", ""),
        message.get("Reply-To", ""),
        extract_plain_text(message),
    ]
    return " ".join(part for part in parts if part)


def parse_email_date(message: EmailMessage) -> datetime:
    date_header = message.get("Date", "")
    if date_header:
        try:
            return parsedate_to_datetime(date_header).astimezone(timezone.utc)
        except (TypeError, ValueError, IndexError):
            pass
    return datetime.now(timezone.utc)


def sanitize_subject_for_filename(subject: str) -> str:
    safe_subject = subject.strip() or "no-subject"
    for char in ':/\\?*|"<>':
        safe_subject = safe_subject.replace(char, "_")
    return safe_subject.replace(" ", "+")


def email_basename(message: EmailMessage) -> str:
    email_date = parse_email_date(message)
    timestamp = email_date.strftime("%Y%m%d_%H%M%S")
    subject = message.get("Subject", "")
    return f"{EMAIL_FILENAME_PREFIX}{timestamp}_{sanitize_subject_for_filename(subject)}"


def emails_dir(folder: Path) -> Path:
    return folder / EMAILS_DIR_NAME


def email_relative_path(filename: str) -> str:
    return f"{EMAILS_DIR_NAME}/{filename}"


def format_email_text(message: EmailMessage) -> str:
    body = extract_plain_text(message)
    lines = [
        f"Subject: {message.get('Subject', '')}",
        f"From: {message.get('From', '')}",
        f"To: {message.get('To', '')}",
        f"Cc: {message.get('Cc', '')}",
        f"Date: {message.get('Date', '')}",
        f"Message-ID: {message.get('Message-ID', '')}",
        "",
        body,
    ]
    return "\n".join(lines).rstrip() + "\n"


def find_existing_email(folder: Path, message_id: str) -> Path | None:
    if not message_id:
        return None

    target = emails_dir(folder)
    if not target.is_dir():
        return None

    needle = message_id.strip()
    for entry in target.iterdir():
        if not entry.is_file() or not entry.name.startswith(EMAIL_FILENAME_PREFIX):
            continue
        text = entry.read_text(encoding="utf-8", errors="replace")
        if needle in text:
            return entry
    return None


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


def build_email_match_prompt(
    message: EmailMessage,
    catalog: list[DealCatalogEntry],
) -> str:
    body = extract_plain_text(message) or "[no plain-text body]"
    if len(body) > 4_000:
        body = body[:4_000] + "\n\n[Note: email body was truncated due to size limits.]"

    return (
        "Email metadata:\n"
        f"- Subject: {message.get('Subject', '')}\n"
        f"- From: {message.get('From', '')}\n"
        f"- To: {message.get('To', '')}\n"
        f"- Cc: {message.get('Cc', '')}\n"
        f"- Date: {message.get('Date', '')}\n\n"
        "Email body:\n"
        f"{body}\n\n"
        "Folder catalog (deals and portfolio companies):\n"
        f"{format_deal_catalog_for_prompt(catalog)}"
    )


def match_email_to_deal(
    message: EmailMessage,
    catalog: list[DealCatalogEntry],
    *,
    api_key: str,
    model: str,
) -> MatchResult:
    haystack = email_match_haystack(message)
    programmatic = find_programmatic_deal_match_from_haystack(
        haystack,
        catalog_targets(catalog),
        source_label="email content",
    )
    if programmatic is not None:
        return programmatic

    payload = groq_json_chat(
        system_prompt=EMAIL_DEAL_MATCH_SYSTEM_PROMPT,
        user_prompt=build_email_match_prompt(message, catalog),
        api_key=api_key,
        model=model,
    )
    deal_folder_raw = payload.get("deal_folder")
    deal_folder = str(deal_folder_raw).strip() if deal_folder_raw else None
    if deal_folder and deal_folder.lower() in {"null", "none", ""}:
        deal_folder = None

    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    return MatchResult(deal_folder=deal_folder, reason=reason)


def process_message(
    config: MailConfig,
    uid: str,
    message: EmailMessage,
    catalog: list[DealCatalogEntry],
    *,
    api_key: str,
    model: str,
    dry_run: bool,
) -> MessageOutcome:
    subject = message.get("Subject", "(no subject)")
    sender = message.get("From", "(unknown)")

    sender_address = normalize_address(sender)
    if sender_address == normalize_address(config.address):
        return MessageOutcome(
            status="skipped",
            uid=uid,
            subject=subject,
            sender=sender,
            reason="from mailbox itself",
        )

    match = match_email_to_deal(
        message,
        catalog,
        api_key=api_key,
        model=model,
    )

    if not match.deal_folder:
        return MessageOutcome(
            status="no_match",
            uid=uid,
            subject=subject,
            sender=sender,
            reason=match.reason,
        )

    deal = resolve_catalog_entry(catalog, match.deal_folder)
    if deal is None:
        return MessageOutcome(
            status="no_match",
            uid=uid,
            subject=subject,
            sender=sender,
            reason=f"Model returned unknown deal folder: {match.deal_folder}",
        )

    message_id = (message.get("Message-ID") or "").strip()
    existing = find_existing_email(deal.folder, message_id)
    if existing is not None:
        return MessageOutcome(
            status="skipped",
            uid=uid,
            subject=subject,
            sender=sender,
            deal_folder=deal.folder_name,
            filename=email_relative_path(existing.name),
            reason="Email already present in emails folder.",
        )

    filename = f"{email_basename(message)}.txt"
    relative_filename = email_relative_path(filename)
    if dry_run:
        return MessageOutcome(
            status="would_write",
            uid=uid,
            subject=subject,
            sender=sender,
            deal_folder=deal.folder_name,
            filename=relative_filename,
            reason=match.reason,
        )

    output_dir = emails_dir(deal.folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(format_email_text(message), encoding="utf-8")
    return MessageOutcome(
        status="written",
        uid=uid,
        subject=subject,
        sender=sender,
        deal_folder=deal.folder_name,
        filename=relative_filename,
        reason=match.reason,
    )


def fetch_unread_messages(
    config: MailConfig,
) -> tuple[imaplib.IMAP4_SSL, list[tuple[str, EmailMessage]]]:
    mail = imaplib.IMAP4_SSL(config.imap_host, config.imap_port)
    mail.login(config.address, config.password)
    mail.select("INBOX")

    status, data = mail.uid("SEARCH", None, "UNSEEN")
    if status != "OK":
        raise RuntimeError("IMAP SEARCH UNSEEN failed")

    uid_values = data[0].split() if data and data[0] else []
    messages: list[tuple[str, EmailMessage]] = []

    for uid_bytes in uid_values:
        uid = uid_bytes.decode()
        status, fetched = mail.uid("FETCH", uid, "(BODY.PEEK[])")
        if status != "OK" or not fetched or not fetched[0]:
            raise RuntimeError(f"IMAP FETCH failed for UID {uid}")

        raw_message = fetched[0][1]
        if not isinstance(raw_message, (bytes, bytearray)):
            raise RuntimeError(f"IMAP FETCH returned unexpected payload for UID {uid}")

        message = BytesParser(policy=default).parsebytes(raw_message)
        messages.append((uid, message))

    return mail, messages


def mark_as_read(mail: imaplib.IMAP4_SSL, uid: str) -> None:
    status, _ = mail.uid("STORE", uid, "+FLAGS", "\\Seen")
    if status != "OK":
        raise RuntimeError(f"IMAP STORE failed for UID {uid}")


def print_outcome(outcome: MessageOutcome) -> None:
    if outcome.status == "written":
        print(f"WRITTEN: {outcome.deal_folder}/{outcome.filename}")
        print(f"  UID {outcome.uid} | {outcome.sender} | {outcome.subject}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "would_write":
        print(f"WOULD WRITE: {outcome.deal_folder}/{outcome.filename}")
        print(f"  UID {outcome.uid} | {outcome.sender} | {outcome.subject}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "skipped":
        label = f"{outcome.deal_folder}/{outcome.filename}" if outcome.deal_folder else outcome.subject
        print(f"SKIPPED: {label}")
        print(f"  UID {outcome.uid} | {outcome.sender} | {outcome.subject}")
        if outcome.reason:
            print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "no_match":
        print(f"NO MATCH: {outcome.subject}")
        print(f"  UID {outcome.uid} | {outcome.sender}")
        print(f"  Reason: {outcome.reason}")
        return

    if outcome.status == "error":
        print(f"ERROR: {outcome.subject}")
        print(f"  UID {outcome.uid} | {outcome.sender}")
        print(f"  Reason: {outcome.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch unread inbox mail, match to deal folders, and save as text files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matches without writing files or marking messages read.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_mail_config()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        base = deals_base()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.exists() or not base.is_dir():
        print(f"Error: deals base is not a directory: {base}", file=sys.stderr)
        return 1

    try:
        catalog = load_combined_catalog(api_key=api_key, model=model)
    except Exception as exc:
        print(f"Error: failed to load deal catalog: {exc}", file=sys.stderr)
        return 1

    if not catalog:
        print(
            "Error: no folders with usable documents found under deals or portcos",
            file=sys.stderr,
        )
        return 1

    print_deal_catalog(catalog)

    mail: imaplib.IMAP4_SSL | None = None
    outcomes: list[MessageOutcome] = []

    try:
        mail, messages = fetch_unread_messages(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        for uid, message in messages:
            try:
                outcome = process_message(
                    config,
                    uid,
                    message,
                    catalog,
                    api_key=api_key,
                    model=model,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                outcome = MessageOutcome(
                    status="error",
                    uid=uid,
                    subject=message.get("Subject", "(no subject)"),
                    sender=message.get("From", "(unknown)"),
                    reason=str(exc),
                )
                print(f"Error processing UID {uid}: {exc}", file=sys.stderr)

            if outcome.status == "written" and mail is not None:
                try:
                    mark_as_read(mail, uid)
                except Exception as exc:
                    outcome = MessageOutcome(
                        status="error",
                        uid=uid,
                        subject=outcome.subject,
                        sender=outcome.sender,
                        deal_folder=outcome.deal_folder,
                        filename=outcome.filename,
                        reason=f"written but failed to mark read: {exc}",
                    )
                    print(f"Error marking UID {uid} as read: {exc}", file=sys.stderr)

            outcomes.append(outcome)
            print_outcome(outcome)
    finally:
        if mail is not None:
            try:
                mail.logout()
            except Exception:
                pass

    written = sum(1 for outcome in outcomes if outcome.status == "written")
    would_write = sum(1 for outcome in outcomes if outcome.status == "would_write")
    no_match = sum(1 for outcome in outcomes if outcome.status == "no_match")
    skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    errors = sum(1 for outcome in outcomes if outcome.status == "error")

    print()
    print(
        "PROCESSED: "
        f"{len(messages)} unread | "
        f"{written} written | "
        f"{no_match} no match | "
        f"{skipped} skipped"
        + (f" | {would_write} would-write" if would_write else "")
        + (f" | {errors} errors" if errors else "")
    )

    return 1 if errors and written == 0 and would_write == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
