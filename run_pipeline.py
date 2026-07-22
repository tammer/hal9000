#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from document_utils import collect_documents
from fetch_all_transcripts import deals_base, folder_has_any_files

REPO_ROOT = Path(__file__).parent


@dataclass
class ClaudeResults:
    ok: list[str] = field(default_factory=list)
    skipped_up_to_date: list[str] = field(default_factory=list)
    skipped_no_docs: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


@dataclass
class PipelineResults:
    fetch: str | None = None
    emails: str | None = None
    meeting_roundup: str | None = None
    daily_summary: str | None = None
    claude: ClaudeResults = field(default_factory=ClaudeResults)
    empty_folders: list[str] = field(default_factory=list)
    no_source_docs: list[str] = field(default_factory=list)
    summarizer: str | None = None
    website: str | None = None
    deploy: str | None = None
    failed_steps: list[str] = field(default_factory=list)


def list_deal_folders(base: Path) -> list[Path]:
    return sorted(
        entry
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def print_banner(step: int, total: int, title: str) -> None:
    print()
    print(f"=== Step {step}/{total}: {title} ===")
    print()


def confirm_step(title: str) -> bool:
    while True:
        answer = input(f"Run step: {title}? [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter y or n.")


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script_path = REPO_ROOT / script_name
    cmd = [sys.executable, str(script_path), *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def scan_folder_issues(deal_folders: list[Path]) -> tuple[list[str], list[str]]:
    empty_folders: list[str] = []
    no_source_docs: list[str] = []

    for folder in deal_folders:
        if not folder_has_any_files(folder):
            empty_folders.append(folder.name)
            print(f"NOTE: {folder.name} is empty (no files)")
            continue

        if not collect_documents(folder, recursive=False):
            no_source_docs.append(folder.name)
            print(f"NOTE: {folder.name} has no readable top-level source documents")

    return empty_folders, no_source_docs


def run_claude_summaries(
    deal_folders: list[Path],
    *,
    no_source_docs: set[str],
) -> ClaudeResults:
    results = ClaudeResults()
    script_path = REPO_ROOT / "claude_summary2.py"

    for folder in deal_folders:
        name = folder.name
        if name in no_source_docs:
            results.skipped_no_docs.append(name)
            continue

        print(f"Summarizing {name}...", file=sys.stderr)
        completed = subprocess.run(
            [sys.executable, str(script_path), name],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        if completed.returncode == 0:
            output = (completed.stdout or "") + (completed.stderr or "")
            if "No new deal.json since the last summary" in output:
                results.skipped_up_to_date.append(name)
            else:
                results.ok.append(name)
            continue

        results.failed.append(name)
        print(f"Error: claude_summary2 failed for {name}", file=sys.stderr)

    return results


def format_claude_summary(claude: ClaudeResults) -> str:
    parts = [f"{len(claude.ok)} ok"]
    if claude.skipped_up_to_date:
        parts.append(f"{len(claude.skipped_up_to_date)} skipped (up to date)")
    if claude.skipped_no_docs:
        parts.append(f"{len(claude.skipped_no_docs)} skipped (no docs)")
    if claude.failed:
        parts.append(f"{len(claude.failed)} failed ({', '.join(claude.failed)})")
    return ", ".join(parts)


def print_pipeline_summary(results: PipelineResults) -> None:
    print()
    print("PIPELINE COMPLETE")

    if results.fetch is not None:
        print(f"  Fetch: {results.fetch}")
    if results.emails is not None:
        print(f"  Emails: {results.emails}")
    if results.meeting_roundup is not None:
        print(f"  Meeting roundup: {results.meeting_roundup}")
    if results.daily_summary is not None:
        print(f"  Daily summary: {results.daily_summary}")
    if (
        results.claude.ok
        or results.claude.skipped_up_to_date
        or results.claude.skipped_no_docs
        or results.claude.failed
    ):
        print(f"  Claude: {format_claude_summary(results.claude)}")
    if results.empty_folders:
        print(f"  Empty folders: {', '.join(results.empty_folders)}")
    if results.no_source_docs:
        print(f"  No source docs: {', '.join(results.no_source_docs)}")
    if results.summarizer is not None:
        print(f"  Summarizer: {results.summarizer}")
    if results.website is not None:
        print(f"  Website: {results.website}")
    if results.deploy is not None:
        print(f"  Deploy: {results.deploy}")
    if results.failed_steps:
        print(f"  Failed steps: {', '.join(results.failed_steps)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full deal pipeline: fetch transcripts, process emails, "
            "meeting roundup, daily summary, Claude summaries, status table, "
            "website generation, and deploy."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass through to fetch_all_transcripts and process_emails.",
    )
    parser.add_argument(
        "--cutoff-date",
        metavar="DATE",
        help="Pass through to fetch_all_transcripts (YYYY-MM-DD).",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Skip step 1.")
    parser.add_argument("--skip-emails", action="store_true", help="Skip step 2.")
    parser.add_argument(
        "--skip-meeting-roundup",
        action="store_true",
        help="Skip step 3.",
    )
    parser.add_argument(
        "--skip-daily-summary",
        action="store_true",
        help="Skip step 4.",
    )
    parser.add_argument("--skip-claude", action="store_true", help="Skip step 5.")
    parser.add_argument(
        "--skip-summarizer",
        action="store_true",
        help="Skip step 6.",
    )
    parser.add_argument("--skip-website", action="store_true", help="Skip step 7.")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip step 8.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Ask yes/no before running each step.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    results = PipelineResults()
    total_steps = 8

    try:
        base = deals_base()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.is_dir():
        print(f"Error: GOOGLE_DRIVE_BASE is not a directory: {base}", file=sys.stderr)
        return 1

    # Step 1: Fetch transcripts
    if not args.skip_fetch:
        if args.confirm and not confirm_step("Fetch transcripts"):
            print("Skipping fetch (declined)", file=sys.stderr)
        else:
            print_banner(1, total_steps, "Fetch transcripts")
            fetch_args: list[str] = []
            if args.cutoff_date:
                fetch_args.extend(["--cutoff-date", args.cutoff_date])
            if args.dry_run:
                fetch_args.append("--dry-run")

            completed = run_script("fetch_all_transcripts.py", *fetch_args)
            if completed.returncode != 0:
                results.fetch = "FAILED"
                results.failed_steps.append("fetch")
                print_pipeline_summary(results)
                return completed.returncode
            results.fetch = "OK"
    else:
        print("Skipping fetch (--skip-fetch)", file=sys.stderr)

    # Step 2: Process emails
    if not args.skip_emails:
        if args.confirm and not confirm_step("Process emails"):
            print("Skipping emails (declined)", file=sys.stderr)
        else:
            print_banner(2, total_steps, "Process emails")
            email_args: list[str] = []
            if args.dry_run:
                email_args.append("--dry-run")

            completed = run_script("process_emails.py", *email_args)
            if completed.returncode != 0:
                results.emails = "FAILED"
                results.failed_steps.append("emails")
                print_pipeline_summary(results)
                return completed.returncode
            results.emails = "OK"
    else:
        print("Skipping emails (--skip-emails)", file=sys.stderr)

    # Step 3: Meeting roundup
    if not args.skip_meeting_roundup:
        if args.confirm and not confirm_step("Meeting roundup"):
            print("Skipping meeting roundup (declined)", file=sys.stderr)
        else:
            print_banner(3, total_steps, "Meeting roundup")
            completed = run_script("meeting_roundup.py")
            if completed.returncode != 0:
                results.meeting_roundup = "FAILED"
                results.failed_steps.append("meeting_roundup")
                print_pipeline_summary(results)
                return completed.returncode
            results.meeting_roundup = "OK"
    else:
        print("Skipping meeting roundup (--skip-meeting-roundup)", file=sys.stderr)

    # Step 4: Daily summary
    if not args.skip_daily_summary:
        if args.confirm and not confirm_step("Daily summary"):
            print("Skipping daily summary (declined)", file=sys.stderr)
        else:
            print_banner(4, total_steps, "Daily summary")
            completed = run_script("daily_summary.py")
            if completed.returncode != 0:
                results.daily_summary = "FAILED"
                results.failed_steps.append("daily_summary")
                print_pipeline_summary(results)
                return completed.returncode
            results.daily_summary = "OK"
    else:
        print("Skipping daily summary (--skip-daily-summary)", file=sys.stderr)

    deal_folders = list_deal_folders(base)

    # Step 5: Claude summaries
    if not args.skip_claude:
        if args.confirm and not confirm_step("Claude summaries"):
            print("Skipping Claude summaries (declined)", file=sys.stderr)
        else:
            print_banner(5, total_steps, "Claude summaries")
            results.empty_folders, results.no_source_docs = scan_folder_issues(
                deal_folders
            )
            print()

            results.claude = run_claude_summaries(
                deal_folders,
                no_source_docs=set(results.no_source_docs)
                | set(results.empty_folders),
            )
            if results.claude.failed:
                results.failed_steps.append("claude")
    else:
        print("Skipping Claude summaries (--skip-claude)", file=sys.stderr)

    # Step 6: Summarizer
    if not args.skip_summarizer:
        if args.confirm and not confirm_step("Summarizer"):
            print("Skipping summarizer (declined)", file=sys.stderr)
        else:
            print_banner(6, total_steps, "Summarizer")
            completed = run_script("summarizer.py")
            if completed.returncode != 0:
                results.summarizer = "FAILED"
                results.failed_steps.append("summarizer")
                print_pipeline_summary(results)
                return completed.returncode
            results.summarizer = "OK"
    else:
        print("Skipping summarizer (--skip-summarizer)", file=sys.stderr)

    # Step 7: Website
    if not args.skip_website:
        if args.confirm and not confirm_step("Website"):
            print("Skipping website (declined)", file=sys.stderr)
        else:
            print_banner(7, total_steps, "Website")
            completed = run_script("generate_website.py")
            if completed.returncode != 0:
                results.website = "FAILED"
                results.failed_steps.append("website")
                print_pipeline_summary(results)
                return completed.returncode
            results.website = "OK"
    else:
        print("Skipping website (--skip-website)", file=sys.stderr)

    # Step 8: Deploy website
    if not args.skip_deploy:
        if args.confirm and not confirm_step("Deploy website"):
            print("Skipping deploy (declined)", file=sys.stderr)
        else:
            print_banner(8, total_steps, "Deploy website")
            completed = run_script("website_deploy.py")
            if completed.returncode != 0:
                results.deploy = "FAILED"
                results.failed_steps.append("deploy")
                print_pipeline_summary(results)
                return completed.returncode
            results.deploy = "OK"
    else:
        print("Skipping deploy (--skip-deploy)", file=sys.stderr)

    print_pipeline_summary(results)
    return 1 if results.failed_steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
