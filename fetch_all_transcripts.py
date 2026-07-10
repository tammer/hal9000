#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from fetch_transcripts import (
    DealIdentity,
    DealMatchTarget,
    build_deal_payload,
    collect_deal_context,
    extract_deal_identity,
    find_existing_transcript,
    find_matching_deal,
    format_transcript_text,
    meeting_date_label,
    transcript_basename,
)
from meetgeek_client import (
    MeetGeekError,
    get_meeting,
    get_transcript,
    list_team_meetings,
)

DEFAULT_LOOKBACK_DAYS = 2


@dataclass(frozen=True)
class DealCatalogEntry:
    folder: Path
    folder_name: str
    identity: DealIdentity


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

        documents = collect_deal_context(entry, summary_only=True)
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


def catalog_targets(catalog: list[DealCatalogEntry]) -> list[DealMatchTarget]:
    return [
        DealMatchTarget(folder_name=deal.folder_name, identity=deal.identity)
        for deal in catalog
    ]


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
    targets: list[DealMatchTarget],
    *,
    api_key: str,
    model: str,
    dry_run: bool,
) -> MeetingOutcome:
    meeting = get_meeting(meeting_id)
    sentences = get_transcript(meeting_id)
    basename = transcript_basename(meeting.title, meeting.timestamp_start_utc)
    date_label = meeting_date_label(meeting.timestamp_start_utc)

    match = find_matching_deal(
        meeting,
        sentences,
        targets,
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
    targets = catalog_targets(catalog)

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
                targets,
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
