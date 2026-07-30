#!/usr/bin/env python3
"""Identify founders and LinkedIn profile URLs for a deal or portco folder.

Usage:
  python founders.py deals/Mobi
  python founders.py portcos/Central-Agent
  python founders.py --all
  python founders.py --all --refresh

Reads summary.md / identity.json / top-level materials (excluding Founders.md),
extracts founders with Groq JSON mode, then resolves missing LinkedIn URLs via
groq/compound web search. Writes Founders.md at the company folder root.

Skips when Founders.md is already complete. Incomplete files are retried with
fill-blanks-only merging so human edits are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from document_utils import collect_documents
from fetch_transcripts import groq_json_chat
from get_facts import parse_json_response
from paths import (
    deals_base,
    list_company_folders,
    portcos_base,
    resolve_company_folder,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
COMPOUND_MODEL = "groq/compound"
AI_GENERATED_DIR = "ai-generated"
SUMMARY_NAME = "summary.md"
IDENTITY_NAME = "identity.json"
FOUNDERS_MD_NAME = "Founders.md"
MAX_MATERIAL_CHARS = 80_000
LINKEDIN_UNKNOWN = "unknown"
LINKEDIN_IN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?",
    re.IGNORECASE,
)
COMPANY_LINE_RE = re.compile(r"^Company:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
STATUS_LINE_RE = re.compile(r"^Status:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
FOUNDER_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
LINKEDIN_LINE_RE = re.compile(
    r"^-\s*LinkedIn:\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

ANTLER_STAFF = (
    "Tammer Kamel",
    "Shambhavi Mishra",
    "Alex Wright",
    "Daphne McLarty",
    "Bernie Li",
)

EXTRACT_SYSTEM_PROMPT = f"""You extract founders and LinkedIn profile URLs from startup deal or portfolio-company materials.

Return valid JSON only with this exact shape:
{{
  "company_name": "Acme Inc" or null,
  "founders": [
    {{
      "first_name": "Jane" or null,
      "last_name": "Doe" or null,
      "full_name": "Jane Doe",
      "linkedin_url": "https://www.linkedin.com/in/jane-doe" or null
    }}
  ]
}}

Rules:
- Include only founders / co-founders of the startup, not investors, advisors, or employees unless clearly labeled as founders.
- These Antler team members must be excluded: {", ".join(ANTLER_STAFF)}
- Prefer full names as written in the materials. Split into first_name / last_name when possible; otherwise leave the unknown part null and still set full_name.
- linkedin_url must be a personal linkedin.com/in/... profile URL found in the materials, or null. Never invent a URL. Prefer https://www.linkedin.com/in/... form.
- Do not invent names not supported by the materials.
- If identity hints list human_names, treat those as strong founder candidates when consistent with the documents.
- founders must be an array (possibly empty). company_name may be null.
"""

COMPOUND_SYSTEM_PROMPT = """You find public LinkedIn profile URLs for startup founders using web search.

Return valid JSON only with this exact shape:
{
  "profiles": [
    {
      "full_name": "Jane Doe",
      "linkedin_url": "https://www.linkedin.com/in/jane-doe" or null,
      "confidence": "high" | "medium" | "low" | "none"
    }
  ]
}

Rules:
- For each person, search for their public LinkedIn personal profile (linkedin.com/in/...), not a company page, search page, or posts.
- Prefer profiles clearly associated with the given company / startup.
- Only return a linkedin_url when you are reasonably confident it is the correct person. Otherwise set linkedin_url to null and confidence to "none" or "low".
- Never invent or guess a slug. The URL must come from search results.
- Return one entry per requested person, preserving full_name.
"""


@dataclass
class Founder:
    full_name: str
    first_name: str | None = None
    last_name: str | None = None
    # LinkedIn value: URL string, "unknown", or None/empty (missing)
    linkedin: str | None = None

    def has_first_and_last(self) -> bool:
        parts = [p for p in self.full_name.split() if p]
        return len(parts) >= 2

    def linkedin_is_set(self) -> bool:
        if not self.linkedin:
            return False
        value = self.linkedin.strip()
        if not value:
            return False
        if value.lower() == LINKEDIN_UNKNOWN:
            return True
        return normalize_linkedin_url(value) is not None


@dataclass
class FoundersDoc:
    company_name: str | None = None
    status: str = "incomplete"
    founders: list[Founder] = field(default_factory=list)


@dataclass
class FolderOutcome:
    path: str
    status: str  # ok | skipped_complete | failed
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify founders and LinkedIn URLs for a deal or portco folder "
            "using Groq (JSON extract + Compound web search). Writes Founders.md."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Company path: deals/<folder> or portcos/<folder> (required unless --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every folder under deals/ and portcos/",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run even when Founders.md is complete (still fill-blanks only)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Groq model for extraction (default: GROQ_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--compound-model",
        default=COMPOUND_MODEL,
        help=f"Groq Compound model for LinkedIn search (default: {COMPOUND_MODEL})",
    )
    parser.add_argument(
        "--skip-web-search",
        action="store_true",
        help="Only extract from materials; do not call Compound for missing LinkedIns",
    )
    return parser.parse_args()


def relative_company_path(path_arg: str) -> str:
    cleaned = path_arg.strip().lstrip("/")
    parts = Path(cleaned).parts
    if len(parts) < 2:
        return cleaned
    return f"{parts[0]}/{parts[1]}"


def is_founders_md_filename(name: str) -> bool:
    return name.lower() == FOUNDERS_MD_NAME.lower()


def find_founders_md_path(folder: Path) -> Path | None:
    """Return existing Founders.md path (case-insensitive), or None."""
    try:
        entries = list(folder.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.is_file() and is_founders_md_filename(entry.name):
            return entry
    return None


def founders_md_write_path(folder: Path) -> Path:
    existing = find_founders_md_path(folder)
    return existing if existing is not None else folder / FOUNDERS_MD_NAME


def normalize_linkedin_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return None
    match = LINKEDIN_IN_RE.search(cleaned)
    if not match:
        return None
    found = match.group(0).rstrip("/")
    if found.lower().startswith("http://"):
        found = "https://" + found[7:]
    if "://www.linkedin.com/" not in found.lower():
        found = re.sub(
            r"^(https://)linkedin\.com/",
            r"\1www.linkedin.com/",
            found,
            flags=re.IGNORECASE,
        )
    return found


def normalize_linkedin_value(raw: str | None) -> str | None:
    """Normalize a LinkedIn field to URL, 'unknown', or None."""
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    if cleaned.lower() == LINKEDIN_UNKNOWN:
        return LINKEDIN_UNKNOWN
    return normalize_linkedin_url(cleaned)


def split_name(full_name: str) -> tuple[str | None, str | None]:
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def parse_founders_md(text: str) -> FoundersDoc | None:
    """Parse Founders.md template. Returns None if unparseable / not our format."""
    if not text or not text.strip():
        return None

    company_match = COMPANY_LINE_RE.search(text)
    status_match = STATUS_LINE_RE.search(text)
    headers = list(FOUNDER_HEADER_RE.finditer(text))

    # Require at least the # Founders heading or a Company/Status line to accept.
    if "# Founders" not in text and company_match is None and not headers:
        return None

    company_name = None
    if company_match:
        company_name = company_match.group(1).strip() or None

    status = "incomplete"
    if status_match:
        status = status_match.group(1).strip().lower() or "incomplete"

    founders: list[Founder] = []
    for i, header in enumerate(headers):
        name = header.group(1).strip()
        if not name:
            continue
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        linkedin_raw: str | None = None
        linkedin_match = LINKEDIN_LINE_RE.search(block)
        if linkedin_match:
            linkedin_raw = linkedin_match.group(1).strip()
        first, last = split_name(name)
        founders.append(
            Founder(
                full_name=name,
                first_name=first,
                last_name=last,
                linkedin=normalize_linkedin_value(linkedin_raw),
            )
        )

    return FoundersDoc(
        company_name=company_name,
        status=status,
        founders=founders,
    )


def is_complete(doc: FoundersDoc | None) -> bool:
    if doc is None:
        return False
    if doc.status.strip().lower() == "complete":
        return True
    if not doc.founders:
        return False
    if not any(f.has_first_and_last() for f in doc.founders):
        return False
    return all(f.linkedin_is_set() for f in doc.founders)


def compute_status(doc: FoundersDoc) -> str:
    if is_complete(FoundersDoc(
        company_name=doc.company_name,
        status="incomplete",
        founders=doc.founders,
    )):
        return "complete"
    return "incomplete"


def render_founders_md(doc: FoundersDoc) -> str:
    status = compute_status(doc)
    company = doc.company_name or ""
    lines = [
        "# Founders",
        "",
        f"Company: {company}",
        f"Status: {status}",
        "",
    ]
    for founder in doc.founders:
        linkedin = founder.linkedin or ""
        lines.append(f"## {founder.full_name}")
        lines.append(f"- LinkedIn: {linkedin}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_founders_md(folder: Path, doc: FoundersDoc) -> Path:
    path = founders_md_write_path(folder)
    path.write_text(render_founders_md(doc), encoding="utf-8")
    return path


def load_existing_founders_doc(folder: Path) -> FoundersDoc | None:
    path = find_founders_md_path(folder)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    return parse_founders_md(text)


def load_identity_hints(folder: Path) -> dict[str, Any] | None:
    path = folder / AI_GENERATED_DIR / IDENTITY_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def collect_materials(folder: Path) -> tuple[str, list[str]]:
    """Return (combined text, sources). Excludes Founders.md from the corpus."""
    chunks: list[tuple[str, str]] = []
    sources: list[str] = []

    summary_path = folder / AI_GENERATED_DIR / SUMMARY_NAME
    if summary_path.is_file():
        try:
            text = summary_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"Warning: could not read {summary_path}: {exc}", file=sys.stderr)
            text = ""
        if text:
            chunks.append(("summary.md", text))
            sources.append("ai-generated/summary.md")

    identity = load_identity_hints(folder)
    if identity:
        hints = {
            "company_name": identity.get("company_name"),
            "human_names": identity.get("human_names") or [],
            "aliases": identity.get("aliases") or [],
            "product_summary": identity.get("product_summary"),
        }
        chunks.append(("identity.json hints", json.dumps(hints, indent=2)))
        sources.append("ai-generated/identity.json")

    for path, text in collect_documents(folder, recursive=False):
        if is_founders_md_filename(path.name):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        label = path.name
        chunks.append((label, cleaned))
        sources.append(label)

    if not chunks:
        return "", sources

    parts: list[str] = []
    total = 0
    truncated = False
    for label, text in chunks:
        header = f"===== {label} =====\n"
        remaining = MAX_MATERIAL_CHARS - total
        if remaining <= 0:
            truncated = True
            break
        body = text
        if len(header) + len(body) > remaining:
            body = body[: max(0, remaining - len(header))]
            truncated = True
        parts.append(header + body)
        total += len(header) + len(body)
        if truncated:
            break

    combined = "\n\n".join(parts)
    if truncated:
        combined += (
            "\n\n[Note: materials truncated due to size limits "
            f"({MAX_MATERIAL_CHARS} chars).]"
        )
        print(
            f"Warning: materials truncated to {MAX_MATERIAL_CHARS} chars",
            file=sys.stderr,
        )

    return combined, sources


def find_linkedin_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in LINKEDIN_IN_RE.finditer(text):
        url = normalize_linkedin_url(match.group(0))
        if not url or url.lower() in seen:
            continue
        seen.add(url.lower())
        urls.append(url)
    return urls


def empty_founder_dict(full_name: str = "") -> dict[str, Any]:
    first, last = split_name(full_name) if full_name else (None, None)
    return {
        "first_name": first,
        "last_name": last,
        "full_name": full_name.strip() or None,
        "linkedin_url": None,
    }


def normalize_founder(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    full = str(raw.get("full_name") or "").strip()
    first = str(raw.get("first_name") or "").strip() or None
    last = str(raw.get("last_name") or "").strip() or None

    if not full:
        if first and last:
            full = f"{first} {last}"
        elif first:
            full = first
        elif last:
            full = last
        else:
            return None

    if not first and not last:
        first, last = split_name(full)

    linkedin = normalize_linkedin_url(
        str(raw.get("linkedin_url")) if raw.get("linkedin_url") else None
    )

    full_lower = full.lower()
    for staff in ANTLER_STAFF:
        if full_lower == staff.lower():
            return None

    return {
        "first_name": first,
        "last_name": last,
        "full_name": full,
        "linkedin_url": linkedin,
    }


def extract_founders(
    materials: str,
    *,
    folder_name: str,
    identity_hints: dict[str, Any] | None,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    hint_block = ""
    if identity_hints:
        hint_block = (
            "\nIdentity hints (may help):\n"
            f"{json.dumps({'company_name': identity_hints.get('company_name'), 'human_names': identity_hints.get('human_names') or []}, indent=2)}\n"
        )

    user_prompt = (
        f"Folder name: {folder_name}\n"
        f"{hint_block}\n"
        "Materials:\n"
        f"{materials}"
    )

    payload = groq_json_chat(
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        api_key=api_key,
        model=model,
    )

    company_name = payload.get("company_name")
    if company_name is not None:
        company_name = str(company_name).strip() or None

    founders: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw in payload.get("founders") or []:
        founder = normalize_founder(raw)
        if founder is None:
            continue
        key = founder["full_name"].lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        founders.append(founder)

    if not founders and identity_hints:
        for name in identity_hints.get("human_names") or []:
            cleaned = str(name).strip()
            if not cleaned:
                continue
            founder = empty_founder_dict(cleaned)
            if founder["full_name"] is None:
                continue
            key = founder["full_name"].lower()
            if any(s.lower() == key for s in ANTLER_STAFF):
                continue
            if key in seen_names:
                continue
            seen_names.add(key)
            founders.append(founder)

    if company_name is None and identity_hints:
        hint_company = identity_hints.get("company_name")
        if hint_company:
            company_name = str(hint_company).strip() or None

    return {"company_name": company_name, "founders": founders}


def drop_linkedin_urls_not_in_materials(
    founders: list[dict[str, Any]],
    materials: str,
) -> None:
    """Clear extract LinkedIn URLs that do not appear in materials (anti-hallucination)."""
    allowed = {url.lower() for url in find_linkedin_urls(materials)}
    for founder in founders:
        url = founder.get("linkedin_url")
        if not url:
            continue
        normalized = normalize_linkedin_url(str(url))
        if not normalized or normalized.lower() not in allowed:
            founder["linkedin_url"] = None


def merge_regex_linkedin_urls(
    founders: list[dict[str, Any]],
    materials: str,
) -> None:
    urls = find_linkedin_urls(materials)
    if not urls:
        return

    assigned = {
        f["linkedin_url"].lower()
        for f in founders
        if f.get("linkedin_url")
    }

    for url in urls:
        if url.lower() in assigned:
            continue
        slug = url.rstrip("/").rsplit("/", 1)[-1].lower().replace("-", " ")
        slug_tokens = set(slug.split())
        best: dict[str, Any] | None = None
        best_score = 0
        for founder in founders:
            if founder.get("linkedin_url"):
                continue
            name_tokens = {
                t.lower()
                for t in re.split(r"\W+", founder["full_name"])
                if t
            }
            score = len(slug_tokens & name_tokens)
            if score > best_score:
                best_score = score
                best = founder
        if best is not None and best_score >= 1:
            best["linkedin_url"] = url
            assigned.add(url.lower())


def compound_resolve_linkedin(
    *,
    company_name: str | None,
    founders_missing: list[Founder],
    api_key: str,
    model: str,
) -> dict[str, str | None]:
    if not founders_missing:
        return {}

    people = [
        {
            "full_name": f.full_name,
            "first_name": f.first_name,
            "last_name": f.last_name,
        }
        for f in founders_missing
    ]
    company = company_name or "(unknown company)"
    user_prompt = (
        f"Company / startup: {company}\n\n"
        "Find LinkedIn personal profile URLs for these founders:\n"
        f"{json.dumps(people, indent=2)}\n\n"
        "Search the web. Return JSON as specified."
    )

    client = Groq(api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": COMPOUND_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    payload: dict[str, Any] | None = None
    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        try:
            payload = parse_json_response(content)
            break
        except Exception as exc:
            last_error = exc
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

    if payload is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Compound LinkedIn resolve failed without a response")

    results: dict[str, str | None] = {}
    for entry in payload.get("profiles") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("full_name") or "").strip()
        if not name:
            continue
        confidence = str(entry.get("confidence") or "").strip().lower()
        url = normalize_linkedin_url(
            str(entry.get("linkedin_url")) if entry.get("linkedin_url") else None
        )
        # Drop only explicit low/none. Missing confidence still accepts a valid URL
        # (Compound often omits the field even when the profile is correct).
        if url and confidence in {"none", "low"}:
            url = None
        results[name] = url

    for founder in founders_missing:
        name = founder.full_name
        if name not in results:
            matched = None
            for key, value in results.items():
                if key.lower() == name.lower():
                    matched = value
                    break
            results[name] = matched

    return results


def fill_blanks_merge(
    seed: FoundersDoc | None,
    extracted_company: str | None,
    extracted_founders: list[dict[str, Any]],
) -> FoundersDoc:
    """Merge extraction into seed: keep existing LinkedIn values; add new names."""
    by_name: dict[str, Founder] = {}
    order: list[str] = []

    company_name = seed.company_name if seed else None
    if extracted_company:
        company_name = extracted_company

    if seed:
        for founder in seed.founders:
            key = founder.full_name.lower()
            by_name[key] = Founder(
                full_name=founder.full_name,
                first_name=founder.first_name,
                last_name=founder.last_name,
                linkedin=founder.linkedin,
            )
            order.append(key)

    for raw in extracted_founders:
        name = str(raw["full_name"]).strip()
        key = name.lower()
        linkedin = normalize_linkedin_value(raw.get("linkedin_url"))
        first = raw.get("first_name")
        last = raw.get("last_name")

        if key in by_name:
            existing = by_name[key]
            # Prefer fuller name split if seed lacked last name
            if not existing.has_first_and_last() and first and last:
                existing.first_name = first
                existing.last_name = last
                existing.full_name = name
            # Fill blank LinkedIn only
            if not existing.linkedin_is_set() and linkedin:
                existing.linkedin = linkedin
        else:
            by_name[key] = Founder(
                full_name=name,
                first_name=first,
                last_name=last,
                linkedin=linkedin,
            )
            order.append(key)

    return FoundersDoc(
        company_name=company_name,
        status="incomplete",
        founders=[by_name[k] for k in order],
    )


def process_company_folder(
    folder: Path,
    *,
    rel_path: str,
    api_key: str,
    model: str,
    compound_model: str,
    skip_web_search: bool,
    refresh: bool,
) -> FolderOutcome:
    existing = load_existing_founders_doc(folder)
    if existing is not None and is_complete(existing) and not refresh:
        print(f"Skipping {rel_path}: Founders.md is complete", file=sys.stderr)
        print(f"Skipped (complete): {rel_path}")
        return FolderOutcome(path=rel_path, status="skipped_complete")

    print(f"Resolving founders for {rel_path} ...", file=sys.stderr)

    materials, sources = collect_materials(folder)
    if not materials.strip() and (existing is None or not existing.founders):
        return FolderOutcome(
            path=rel_path,
            status="failed",
            detail="no readable materials",
        )

    if sources:
        print(f"Sources: {', '.join(sources)}", file=sys.stderr)

    identity_hints = load_identity_hints(folder)
    extracted_company: str | None = None
    extracted_founders: list[dict[str, Any]] = []

    if materials.strip():
        try:
            extracted = extract_founders(
                materials,
                folder_name=folder.name,
                identity_hints=identity_hints,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            return FolderOutcome(
                path=rel_path,
                status="failed",
                detail=f"extraction failed: {exc}",
            )
        extracted_company = extracted.get("company_name")
        extracted_founders = extracted["founders"]
        # Reject model-invented LinkedIns; only keep URLs that appear in materials.
        drop_linkedin_urls_not_in_materials(extracted_founders, materials)
        merge_regex_linkedin_urls(extracted_founders, materials)
    elif identity_hints:
        extracted_company = (
            str(identity_hints.get("company_name") or "").strip() or None
        )
        for name in identity_hints.get("human_names") or []:
            cleaned = str(name).strip()
            if cleaned:
                founder = empty_founder_dict(cleaned)
                if founder["full_name"]:
                    extracted_founders.append(founder)

    doc = fill_blanks_merge(existing, extracted_company, extracted_founders)

    missing = [f for f in doc.founders if not f.linkedin_is_set()]
    if missing and not skip_web_search:
        print(
            f"Resolving {len(missing)} LinkedIn URL(s) via {compound_model} ...",
            file=sys.stderr,
        )
        try:
            resolved = compound_resolve_linkedin(
                company_name=doc.company_name,
                founders_missing=missing,
                api_key=api_key,
                model=compound_model,
            )
        except Exception as exc:
            print(
                f"Warning: Compound LinkedIn search failed: {exc}",
                file=sys.stderr,
            )
            resolved = {}

        for founder in missing:
            url = resolved.get(founder.full_name)
            if not url:
                for key, value in resolved.items():
                    if key.lower() == founder.full_name.lower() and value:
                        url = value
                        break
            if url and not founder.linkedin_is_set():
                founder.linkedin = url
    elif missing and skip_web_search:
        print(
            f"Skipping web search for {len(missing)} founder(s) without LinkedIn",
            file=sys.stderr,
        )

    doc.status = compute_status(doc)
    out_path = write_founders_md(folder, doc)
    markdown = render_founders_md(doc)
    print(f"Wrote {out_path} (Status: {doc.status})", file=sys.stderr)
    print(markdown)
    return FolderOutcome(path=rel_path, status="ok", detail=doc.status)


def list_all_company_paths() -> list[tuple[str, Path]]:
    """Return (relative path, absolute folder) for all deals and portcos."""
    items: list[tuple[str, Path]] = []
    deals = deals_base()
    if deals.is_dir():
        for folder in list_company_folders(deals):
            items.append((f"deals/{folder.name}", folder))
    portcos = portcos_base()
    if portcos.is_dir():
        for folder in list_company_folders(portcos):
            items.append((f"portcos/{folder.name}", folder))
    return items


def print_all_summary(outcomes: list[FolderOutcome]) -> None:
    ok = [o for o in outcomes if o.status == "ok"]
    skipped = [o for o in outcomes if o.status == "skipped_complete"]
    failed = [o for o in outcomes if o.status == "failed"]
    print(file=sys.stderr)
    print(
        f"Founders summary: {len(ok)} ok, "
        f"{len(skipped)} skipped (complete), "
        f"{len(failed)} failed",
        file=sys.stderr,
    )
    if failed:
        for o in failed:
            print(f"  failed {o.path}: {o.detail}", file=sys.stderr)


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.all and args.path:
        print("Error: pass either a path or --all, not both", file=sys.stderr)
        return 1
    if not args.all and not args.path:
        print("Error: path is required unless --all is set", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = args.model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
    compound_model = args.compound_model or COMPOUND_MODEL

    if args.all:
        try:
            targets = list_all_company_paths()
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if not targets:
            print("Warning: no deal or portco folders found", file=sys.stderr)
            return 0

        outcomes: list[FolderOutcome] = []
        for rel_path, folder in targets:
            try:
                outcome = process_company_folder(
                    folder,
                    rel_path=rel_path,
                    api_key=api_key,
                    model=model,
                    compound_model=compound_model,
                    skip_web_search=args.skip_web_search,
                    refresh=args.refresh,
                )
            except Exception as exc:
                outcome = FolderOutcome(
                    path=rel_path,
                    status="failed",
                    detail=str(exc),
                )
                print(f"Error: {rel_path}: {exc}", file=sys.stderr)
            outcomes.append(outcome)

        print_all_summary(outcomes)
        return 0

    try:
        folder = resolve_company_folder(args.path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: folder does not exist: {folder}", file=sys.stderr)
        return 1

    rel_path = relative_company_path(args.path)
    outcome = process_company_folder(
        folder,
        rel_path=rel_path,
        api_key=api_key,
        model=model,
        compound_model=compound_model,
        skip_web_search=args.skip_web_search,
        refresh=args.refresh,
    )
    if outcome.status == "failed":
        print(f"Error: {outcome.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
