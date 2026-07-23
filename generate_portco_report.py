#!/usr/bin/env python3
"""Generate a portfolio-company report from ai-generated/portco.json using Claude."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from claude_common import parse_relative_path_args
from company_report import JsonReportConfig, run_json_report
from process_portco import PORTCO_JSON_NAME, process_portco, resolve_portco_folder

PORTCO_REPORT_PROMPT_PATH = Path(__file__).parent / "portco_report_prompt.md"


def validate_portco_folder(relative_path: str) -> Path | None:
    """Resolve and validate a portco folder. Prints errors and returns None on failure."""
    try:
        folder = resolve_portco_folder(relative_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None

    if not folder.exists():
        print(f"Error: path does not exist: {folder}", file=sys.stderr)
        return None

    if not folder.is_dir():
        print(f"Error: path is not a directory: {folder}", file=sys.stderr)
        return None

    return folder


PORTCO_REPORT_CONFIG = JsonReportConfig(
    metadata_json_name=PORTCO_JSON_NAME,
    prompt_path=PORTCO_REPORT_PROMPT_PATH,
    resolve_folder=validate_portco_folder,
    refresh_metadata=process_portco,
    metadata_label=PORTCO_JSON_NAME,
)


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        "Generate a portfolio-company report from ai-generated/portco.json using Claude.",
        "Folder name under the sibling portcos/ directory (e.g. Central-Agent)",
        with_dry_run=True,
    )
    return run_json_report(
        PORTCO_REPORT_CONFIG,
        args.relative_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
