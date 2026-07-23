#!/usr/bin/env python3
"""Generate an investment summary from ai-generated/deal.json using Claude."""

from __future__ import annotations

from dotenv import load_dotenv

from claude_common import SUMMARY_PROMPT_PATH, parse_relative_path_args, validate_folder
from company_report import JsonReportConfig, run_json_report
from process_deal import DEAL_JSON_NAME, process_deal

DEAL_REPORT_CONFIG = JsonReportConfig(
    metadata_json_name=DEAL_JSON_NAME,
    prompt_path=SUMMARY_PROMPT_PATH,
    resolve_folder=validate_folder,
    refresh_metadata=process_deal,
    metadata_label=DEAL_JSON_NAME,
)


def main() -> int:
    load_dotenv()
    args = parse_relative_path_args(
        "Generate an investment summary from ai-generated/deal.json using Claude.",
        "Relative path under Google Drive to the folder to summarize",
        with_dry_run=True,
    )
    return run_json_report(
        DEAL_REPORT_CONFIG,
        args.relative_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
