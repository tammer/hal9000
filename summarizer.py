#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
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
STATUS_CACHE_NAME = "status-cache.json"

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


def status_cache_path() -> Path:
    return shared_ai_dir() / STATUS_CACHE_NAME


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_sha256() -> str:
    return sha256_hex(EXTRACTOR_SYSTEM_PROMPT.encode("utf-8"))


def load_status_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: ignoring unreadable status cache {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(payload, dict):
        print(
            f"Warning: ignoring status cache {path} (expected an object)",
            file=sys.stderr,
        )
        return {}
    return payload


def save_status_cache(path: Path, cache: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cache_strings(entry: dict[str, object]) -> tuple[str, str, str] | None:
    product = entry.get("product")
    founders = entry.get("founders")
    notes = entry.get("notes")
    if isinstance(product, str) and isinstance(founders, str) and isinstance(notes, str):
        return product, founders, notes
    return None


def matching_cache_entry(
    entry: object,
    summary_hash: str,
    model: str,
    prompt_hash: str,
) -> dict[str, object] | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("summary_sha256") != summary_hash:
        return None
    if entry.get("model") != model:
        return None
    if entry.get("prompt_sha256") != prompt_hash:
        return None
    if _cache_strings(entry) is None:
        return None
    return entry


def cache_entry_for_row(
    summary_hash: str,
    model: str,
    prompt_hash: str,
    row: DealRow,
) -> dict[str, str]:
    return {
        "summary_sha256": summary_hash,
        "model": model,
        "prompt_sha256": prompt_hash,
        "product": row.product,
        "founders": row.founders,
        "notes": row.notes,
    }


def row_from_cache(
    deal_name: str,
    summary_text: str,
    entry: dict[str, object],
) -> DealRow:
    product, founders, notes = _cache_strings(entry) or ("", "", "")
    return DealRow(
        deal_name=deal_name,
        status=parse_deal_status(summary_text),
        product=product,
        founders=founders,
        notes=notes,
    )


def _copy_cache_entry(entry: object) -> dict[str, object] | None:
    if isinstance(entry, dict):
        return dict(entry)
    return None


def assemble_status(
    summaries: list[tuple[str, str, str]],
    previous_cache: dict[str, object],
    client: Groq,
    model: str,
    *,
    preserve_deals: set[str] | None = None,
) -> tuple[list[DealRow], dict[str, dict[str, object]], int, int]:
    """Build status rows and the cache to write.

    ``summaries`` is ``(deal_name, summary_text, summary_sha256)``. A hit reuses
    product, founders, and notes. Status is always parsed from the current
    summary. Failed extractions keep the previous cache entry. Deals absent
    from ``summaries`` are dropped unless listed in ``preserve_deals``.
    """
    prompt_hash = prompt_sha256()
    rows: list[DealRow] = []
    new_cache: dict[str, dict[str, object]] = {}
    cached = 0
    extracted = 0

    for deal_name, summary_text, summary_hash in summaries:
        previous = previous_cache.get(deal_name)
        hit = matching_cache_entry(previous, summary_hash, model, prompt_hash)
        if hit is not None:
            print(f"Cached {deal_name}", file=sys.stderr)
            copied = _copy_cache_entry(hit)
            if copied is None:
                continue
            rows.append(row_from_cache(deal_name, summary_text, copied))
            new_cache[deal_name] = copied
            cached += 1
            continue

        print(f"Extracting {deal_name}...", file=sys.stderr)
        try:
            row = extract_deal_row(client, model, deal_name, summary_text)
        except Exception as exc:
            print(
                f"Warning: failed to extract data for {deal_name}: {exc}",
                file=sys.stderr,
            )
            copied = _copy_cache_entry(previous)
            if copied is not None:
                new_cache[deal_name] = copied
            continue

        rows.append(row)
        new_cache[deal_name] = cache_entry_for_row(
            summary_hash, model, prompt_hash, row
        )
        extracted += 1

    for deal_name in preserve_deals or ():
        if deal_name in new_cache:
            continue
        copied = _copy_cache_entry(previous_cache.get(deal_name))
        if copied is not None:
            new_cache[deal_name] = copied

    rows.sort(key=lambda row: row.deal_name.lower())
    ordered_cache = dict(
        sorted(new_cache.items(), key=lambda item: item[0].lower())
    )
    return rows, ordered_cache, cached, extracted


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
    messages: list[dict[str, str]] = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": summary_text},
    ]
    last_error: json.JSONDecodeError | None = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        try:
            payload = parse_json_response(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == 0:
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return only valid JSON with no commentary."
                        ),
                    }
                )
                continue
            raise

        return DealRow(
            deal_name=deal_name,
            status=parse_deal_status(summary_text),
            product=str(payload.get("product", "")).strip(),
            founders=str(payload.get("founders", "")).strip(),
            notes=str(payload.get("notes") or payload.get("status") or "").strip(),
        )

    assert last_error is not None
    raise last_error


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

    cache_path = status_cache_path()
    previous_cache = load_status_cache(cache_path)
    summaries: list[tuple[str, str, str]] = []
    preserve_deals: set[str] = set()
    for deal_folder in list_company_folders(base):
        summary_path = summary_path_for_deal(deal_folder)
        if not summary_path.is_file():
            continue
        try:
            raw = summary_path.read_bytes()
            summary_text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(
                f"Warning: failed to read {summary_path}: {exc}",
                file=sys.stderr,
            )
            preserve_deals.add(deal_folder.name)
            continue
        summaries.append((deal_folder.name, summary_text, sha256_hex(raw)))

    rows, new_cache, cached, extracted = assemble_status(
        summaries,
        previous_cache,
        client,
        model,
        preserve_deals=preserve_deals,
    )

    output_path = shared_ai_dir() / "status.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_status_table(rows), encoding="utf-8")
    save_status_cache(cache_path, new_cache)
    print(
        f"Wrote {output_path} ({len(rows)} deals, {cached} cached, {extracted} extracted)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
