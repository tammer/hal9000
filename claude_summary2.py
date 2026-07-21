#!/usr/bin/env python3
import json
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

DEAL_JSON_NAME = "deal.json"


def deal_json_path(folder: Path) -> Path:
    return folder / "ai-generated" / DEAL_JSON_NAME


def existing_summary_if_current(folder: Path, deal_path: Path) -> Path | None:
    summary_path = folder / "ai-generated" / "summary.md"
    if not summary_path.is_file() or not deal_path.is_file():
        return None

    if deal_path.stat().st_mtime >= summary_path.stat().st_mtime:
        return None

    return summary_path


def write_summary(folder: Path, content: str) -> Path:
    ai_generated_dir = folder / "ai-generated"
    ai_generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_generated_dir / "summary.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def strip_claude_stats(entries: list) -> list:
    """Return entries with claude_stats removed from each object."""
    cleaned: list = []
    for item in entries:
        if isinstance(item, dict):
            cleaned.append(
                {key: value for key, value in item.items() if key != "claude_stats"}
            )
        else:
            cleaned.append(item)
    return cleaned


def load_deal_json(deal_path: Path) -> tuple[str, list] | None:
    """Read deal.json, strip claude_stats, return payload text and entries."""
    if not deal_path.is_file():
        print(f"Error: deal.json not found at {deal_path}", file=sys.stderr)
        return None

    try:
        raw = deal_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to read {deal_path}: {exc}", file=sys.stderr)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {deal_path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, list):
        print(
            f"Error: {deal_path} is not a JSON list",
            file=sys.stderr,
        )
        return None

    entries = strip_claude_stats(data)
    text = json.dumps(entries, indent=2, ensure_ascii=False)
    return text, entries


def print_deal_list(folder: Path, deal_path: Path, entries: list) -> None:
    rel = deal_path.relative_to(folder)
    print(f"Documents (1):")
    print(f"  {rel} ({len(entries)} entries)")
    for item in entries:
        if isinstance(item, dict):
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                print(f"    - {filename}")


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        "Generate an investment summary from ai-generated/deal.json using Claude.",
        "Relative path under Google Drive to the folder to summarize",
        with_dry_run=True,
    )

    folder = validate_folder(args.relative_path)
    if folder is None:
        return 1

    deal_path = deal_json_path(folder)

    if not args.dry_run:
        current_summary = existing_summary_if_current(folder, deal_path)
        if current_summary is not None:
            print(
                "No new deal.json since the last summary was generated. "
                f"Existing summary: {current_summary}"
            )
            return 0

    loaded = load_deal_json(deal_path)
    if loaded is None:
        return 1

    text, entries = loaded
    documents = [(deal_path, text)]

    print_deal_list(folder, deal_path, entries)
    if args.dry_run:
        return 0

    api_key = require_api_key()
    if api_key is None:
        return 1

    system_prompt = load_system_prompt()
    if system_prompt is None:
        return 1

    user_content = f"Documents:\n{build_payload(documents)}"

    try:
        summary, response = run_claude(
            system_prompt,
            user_content,
            api_key,
            MODEL,
        )
    except Exception as exc:
        print(f"Error: Anthropic API call failed: {exc}", file=sys.stderr)
        return 1

    try:
        output_path = write_summary(folder, summary)
    except OSError as exc:
        print(f"Error: failed to write summary.md: {exc}", file=sys.stderr)
        return 1

    print_usage_report(MODEL, response.usage)
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
