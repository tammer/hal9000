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
from document_utils import (
    FORBIDDEN_DIR_NAMES,
    collect_documents,
    list_candidate_files,
)


def existing_summary_if_current(folder: Path) -> Path | None:
    summary_path = folder / "ai-generated" / "summary.md"
    if not summary_path.is_file():
        return None

    source_files = list_candidate_files(
        folder, recursive=True, exclude_dirs=FORBIDDEN_DIR_NAMES
    )
    if not source_files:
        return None

    summary_mtime = summary_path.stat().st_mtime
    for source in source_files:
        if source.stat().st_mtime >= summary_mtime:
            return None

    return summary_path


def write_summary(folder: Path, content: str) -> Path:
    ai_generated_dir = folder / "ai-generated"
    ai_generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_generated_dir / "summary.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        "Generate an investment summary from deal documents using Claude.",
        "Relative path under Google Drive to the folder to summarize",
    )

    folder = validate_folder(args.relative_path)
    if folder is None:
        return 1

    current_summary = existing_summary_if_current(folder)
    if current_summary is not None:
        print(
            "No new source documents since the last summary was generated. "
            f"Existing summary: {current_summary}"
        )
        return 0

    api_key = require_api_key()
    if api_key is None:
        return 1

    system_prompt = load_system_prompt()
    if system_prompt is None:
        return 1

    documents = collect_documents(
        folder, recursive=True, exclude_dirs=FORBIDDEN_DIR_NAMES
    )
    if not documents:
        print(
            f"Error: no readable documents found in {folder}",
            file=sys.stderr,
        )
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
