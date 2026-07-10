#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from get_facts import parse_json_response

DEFAULT_MODEL = "llama-3.3-70b-versatile"

EXTRACTOR_SYSTEM_PROMPT = """You extract structured deal information from an investment summary markdown document.

Return valid JSON only with this exact shape:
{
  "product": "1-2 sentence product summary",
  "founders": "founder name(s), brief",
  "status": "brief deal status, 25-40 words max. Get this information from the # State section. Include data of last interaction if available in the state section"
}

Extraction rules:
- product: summarize from the # Product section
- founders: from the Company table Founders row; use plain names only (no markdown links)
- status: from the # State section; keep concise
- If a field is missing from the summary, use an empty string
"""


@dataclass(frozen=True)
class DealRow:
    deal_name: str
    product: str
    founders: str
    status: str


def resolve_google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    return Path(base_raw).expanduser().resolve()


def list_deal_folders(base: Path) -> list[Path]:
    return sorted(
        entry
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def summary_path_for_deal(deal_folder: Path) -> Path:
    return deal_folder / "ai-generated" / "summary.md"


def escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_status_table(rows: list[DealRow]) -> str:
    header = "| Deal Name | Product | Founder(s) | Status |"
    separator = "|-----------|---------|------------|--------|"
    body_lines = [
        "| "
        + " | ".join(
            escape_table_cell(value)
            for value in (
                row.deal_name,
                row.product,
                row.founders,
                row.status,
            )
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body_lines]) + "\n"


def extract_deal_row(
    client: Groq,
    model: str,
    deal_name: str,
    summary_text: str,
) -> DealRow:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": summary_text},
        ],
    )
    payload = parse_json_response(response.choices[0].message.content or "")
    return DealRow(
        deal_name=deal_name,
        product=str(payload.get("product", "")).strip(),
        founders=str(payload.get("founders", "")).strip(),
        status=str(payload.get("status", "")).strip(),
    )


def main() -> int:
    load_dotenv()

    try:
        base = resolve_google_drive_base()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.is_dir():
        print(f"Error: GOOGLE_DRIVE_BASE is not a directory: {base}", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)

    rows: list[DealRow] = []
    for deal_folder in list_deal_folders(base):
        summary_path = summary_path_for_deal(deal_folder)
        if not summary_path.is_file():
            continue

        print(f"Processing {deal_folder.name}...", file=sys.stderr)
        try:
            summary_text = summary_path.read_text(encoding="utf-8")
            row = extract_deal_row(client, model, deal_folder.name, summary_text)
            rows.append(row)
        except Exception as exc:
            print(
                f"Warning: failed to extract data for {deal_folder.name}: {exc}",
                file=sys.stderr,
            )

    rows.sort(key=lambda row: row.deal_name.lower())
    output_path = base / "status.md"
    output_path.write_text(render_status_table(rows), encoding="utf-8")
    print(f"Wrote {output_path} ({len(rows)} deals)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
