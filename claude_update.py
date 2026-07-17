#!/usr/bin/env python3
import sys
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
    validate_folder,
)
from document_utils import FORBIDDEN_DIR_NAMES, collect_documents

UPDATE_INSTRUCTION = (
    "Below is the current investment report followed by documents that were "
    "added or changed since it was generated. Update the report to reflect any "
    "new information or changes, preserving the required section structure. "
    "Keep sections that have no new information unchanged."
)


def changed_documents(
    folder: Path, summary_mtime: float
) -> list[tuple[Path, str]]:
    documents = collect_documents(
        folder, recursive=True, exclude_dirs=FORBIDDEN_DIR_NAMES
    )
    return [
        (path, content)
        for path, content in documents
        if path.stat().st_mtime >= summary_mtime
    ]


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        (
            "Update an existing investment summary with documents changed "
            "since it was generated, using Claude."
        ),
        "Relative path under Google Drive to the folder to update",
    )

    folder = validate_folder(args.relative_path)
    if folder is None:
        return 1

    summary_path = folder / "ai-generated" / "summary.md"
    if not summary_path.is_file():
        print(
            f"Error: no existing summary found at {summary_path}. "
            "Run claude_summary.py first.",
            file=sys.stderr,
        )
        return 1

    summary_mtime = summary_path.stat().st_mtime

    documents = changed_documents(folder, summary_mtime)
    if not documents:
        print(
            "No documents added or changed since the last report. "
            f"Existing summary: {summary_path}"
        )
        return 0

    api_key = require_api_key()
    if api_key is None:
        return 1

    system_prompt = load_system_prompt()
    if system_prompt is None:
        return 1

    try:
        current_report = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to read {summary_path}: {exc}", file=sys.stderr)
        return 1

    user_content = (
        f"{UPDATE_INSTRUCTION}\n\n"
        f"# Current report\n{current_report}\n\n"
        f"# New or changed documents\n{build_payload(documents)}"
    )

    try:
        updated_report, response = run_claude(
            system_prompt,
            user_content,
            api_key,
            MODEL,
        )
    except Exception as exc:
        print(f"Error: Anthropic API call failed: {exc}", file=sys.stderr)
        return 1

    old_path = summary_path.with_name("summary.md.old")
    try:
        summary_path.replace(old_path)
        summary_path.write_text(updated_report, encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to write updated summary.md: {exc}", file=sys.stderr)
        return 1

    print_usage_report(MODEL, response.usage)
    print(f"Updated summary written to {summary_path}")
    print(f"Previous summary archived to {old_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
