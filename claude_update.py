#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from claude_summary import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    PAYLOAD_WARN_CHARS,
    append_generated_timestamp,
    build_payload,
    extract_response_text,
    load_summary_prompt,
    print_usage_report,
    resolve_folder_path,
    strip_markdown_fences,
)
from document_utils import collect_documents

UPDATE_INSTRUCTION = (
    "Below is the current investment report followed by documents that were "
    "added or changed since it was generated. Update the report to reflect any "
    "new information or changes, preserving the required section structure. "
    "Keep sections that have no new information unchanged."
)


def changed_documents(
    folder: Path, summary_mtime: float
) -> list[tuple[Path, str]]:
    documents = collect_documents(folder, recursive=False)
    return [
        (path, content)
        for path, content in documents
        if path.stat().st_mtime >= summary_mtime
    ]


def generate_update(
    system_prompt: str,
    current_report: str,
    documents: list[tuple[Path, str]],
    api_key: str,
    model: str,
) -> tuple[str, object]:
    client = Anthropic(api_key=api_key)
    payload = build_payload(documents)

    user_content = (
        f"{UPDATE_INSTRUCTION}\n\n"
        f"# Current report\n{current_report}\n\n"
        f"# New or changed documents\n{payload}"
    )

    if len(user_content) > PAYLOAD_WARN_CHARS:
        print(
            f"Warning: update payload is large ({len(user_content):,} chars)",
            file=sys.stderr,
        )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )

    text = append_generated_timestamp(
        strip_markdown_fences(extract_response_text(response))
    )
    return text, response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update an existing investment summary with documents changed "
            "since it was generated, using Claude."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the folder to update",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

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

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    try:
        system_prompt = load_summary_prompt()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        current_report = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to read {summary_path}: {exc}", file=sys.stderr)
        return 1

    try:
        updated_report, response = generate_update(
            system_prompt,
            current_report,
            documents,
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
