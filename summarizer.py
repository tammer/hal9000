#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from get_facts import parse_json_response
from paths import deals_base, list_company_folders, shared_ai_dir

DEFAULT_MODEL = "openai/gpt-oss-120b"

KNOWN_STATUSES = (
    "IN RESIDENCY",
    "COURTING",
    "PIPELINE",
    "MONITOR",
    "REJECTED",
)
STATE_SECTION_RE = re.compile(
    r"^##[ \t]+State\b[^\n]*\n(.*?)(?=^#{1,2}[ \t]|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
STATUS_LINE_RE = re.compile(r"^Status:\s*(.+?)\s*$", re.IGNORECASE)

EXTRACTOR_SYSTEM_PROMPT = """You extract structured deal information from an investment summary markdown document.

Return valid JSON only with this exact shape:
{
  "product": "1-2 sentence product summary",
  "founders": "founder name(s), brief",
  "notes": "brief deal status, 25-40 words max. Get this information from the # State section. Include date of last interaction if available in the state section"
}

Extraction rules:
- product: summarize from the # Product section
- founders: from the Company table Founders row; use plain names only (no markdown links)
- notes: from the # State section; keep concise. Do not copy the enumerated Status line (IN RESIDENCY / COURTING / PIPELINE / MONITOR / REJECTED); summarize the surrounding state narrative.
- If a field is missing from the summary, use an empty string
"""


@dataclass(frozen=True)
class DealRow:
    deal_name: str
    status: str
    product: str
    founders: str
    notes: str


def summary_path_for_deal(deal_folder: Path) -> Path:
    return deal_folder / "ai-generated" / "summary.md"


def _canonical_status(raw: str) -> str | None:
    cleaned = re.sub(r"[*`_\"']+", " ", raw)
    cleaned = " ".join(cleaned.upper().split())
    if not cleaned:
        return None
    for status in sorted(KNOWN_STATUSES, key=len, reverse=True):
        if cleaned == status:
            return status
        if cleaned.startswith(status) and not cleaned[len(status)].isalnum():
            return status
    return None


def parse_deal_status(summary_text: str) -> str:
    section_match = STATE_SECTION_RE.search(summary_text)
    haystack = section_match.group(1) if section_match else summary_text
    for line in haystack.splitlines():
        normalized = re.sub(r"[*_`]", "", line.strip())
        match = STATUS_LINE_RE.match(normalized)
        if not match:
            continue
        status = _canonical_status(match.group(1))
        if status:
            return status
    return "unknown"


def escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_status_table(rows: list[DealRow]) -> str:
    header = "| Deal Name | Status | Product | Founder(s) | Notes |"
    separator = "|-----------|--------|---------|------------|-------|"
    body_lines = [
        "| "
        + " | ".join(
            escape_table_cell(value)
            for value in (
                row.deal_name,
                row.status,
                row.product,
                row.founders,
                row.notes,
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
        status=parse_deal_status(summary_text),
        product=str(payload.get("product", "")).strip(),
        founders=str(payload.get("founders", "")).strip(),
        notes=str(payload.get("notes") or payload.get("status") or "").strip(),
    )


def main() -> int:
    load_dotenv()

    try:
        base = deals_base()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not base.is_dir():
        print(f"Error: deals base is not a directory: {base}", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)

    rows: list[DealRow] = []
    for deal_folder in list_company_folders(base):
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
    output_path = shared_ai_dir() / "status.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_status_table(rows), encoding="utf-8")
    print(f"Wrote {output_path} ({len(rows)} deals)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
