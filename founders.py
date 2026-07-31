#!/usr/bin/env python3
"""Identify founders and LinkedIn profile URLs for a deal or portco folder.

Usage:
  python founders.py deals/Mobi
  python founders.py portcos/Central-Agent
  python founders.py --all
  python founders.py --all --refresh

Reads primary materials only (top-level docs, emails/, transcripts/; never
ai-generated/), excluding Founders.md. Extracts founders with Groq JSON mode,
then resolves missing LinkedIn URLs via Brave search + optional Compound
propose, each validated by HTTP title check. Writes Founders.md at the company
folder root.

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
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq

from document_utils import collect_documents
from fetch_transcripts import groq_json_chat
from get_facts import parse_json_response, search_brave
from paths import (
    deals_base,
    list_company_folders,
    portcos_base,
    resolve_company_folder,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
COMPOUND_MODEL = "groq/compound"
AI_GENERATED_DIR = "ai-generated"
FOUNDERS_MD_NAME = "Founders.md"
MAX_MATERIAL_CHARS = 80_000
LINKEDIN_UNKNOWN = "unknown"
LINKEDIN_FETCH_TIMEOUT = 15
LINKEDIN_FETCH_MAX_BYTES = 200_000
LINKEDIN_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
PLACEHOLDER_SLUG_RE = re.compile(
    r"(?:123456789|987654321|0123456789|000000+|111111+|999999+)",
    re.IGNORECASE,
)
ROLEISH_NAME_RE = re.compile(
    r"\b("
    r"co-?founders?|founders?|cto|ceo|cfo|coo|unnamed|unknown|"
    r"not\s+mentioned|name\s+not|based|partner|advisor|employee"
    r")\b",
    re.IGNORECASE,
)
HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
OG_TITLE_RE = re.compile(
    r'property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
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

LinkedInVerdict = Literal["valid", "invalid", "inconclusive"]

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
- founders must be an array (possibly empty). company_name may be null.
"""

COMPOUND_SYSTEM_PROMPT = """You find one public LinkedIn profile URL using web search.

Return valid JSON only with this exact shape:
{
  "full_name": "Jane Doe",
  "linkedin_url": "https://www.linkedin.com/in/jane-doe" or null,
  "confidence": "high" | "medium" | "low" | "none",
  "source_title": "title from the search result that contained the URL" or null
}

Rules:
- Search for the person's public LinkedIn personal profile (linkedin.com/in/...), not a company page, search page, or posts.
- Prefer a profile clearly associated with the given company / startup.
- The linkedin_url MUST appear verbatim in a search result. Never invent or guess a slug.
- Never construct firstname-lastname-######## URLs. If no LinkedIn /in/ URL appears in results, return linkedin_url null and confidence "none".
- Only return confidence "high" when the search result title/snippet clearly matches the person (and ideally the company). Otherwise use "medium", "low", or "none".
- Return the exact full_name you were given.
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
    if looks_like_placeholder_slug(found):
        return None
    return found


def linkedin_slug(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def looks_like_placeholder_slug(url: str) -> bool:
    """Reject obviously invented LinkedIn slugs (e.g. name-123456789)."""
    slug = linkedin_slug(url)
    if PLACEHOLDER_SLUG_RE.search(slug):
        return True
    # Trailing digit runs that are simple ascending/descending sequences.
    m = re.search(r"-(\d{6,})$", slug)
    if not m:
        return False
    digits = m.group(1)
    ascending = "0123456789"
    descending = "9876543210"
    return digits in ascending or digits in descending or digits == digits[0] * len(digits)


def name_tokens(full_name: str) -> set[str]:
    return {
        token.lower()
        for token in re.split(r"\W+", full_name)
        if token and len(token) > 1
    }


def text_matches_name(text: str, full_name: str) -> bool:
    tokens = name_tokens(full_name)
    if not tokens or not text:
        return False
    lowered = text.lower()
    hits = sum(1 for token in tokens if token in lowered)
    if len(tokens) == 1:
        return hits >= 1
    return hits >= 2


def is_searchable_person_name(full_name: str) -> bool:
    """True only for real-looking personal names, not role placeholders."""
    cleaned = full_name.strip()
    if not cleaned:
        return False
    parts = [p for p in cleaned.split() if p]
    if len(parts) < 2 or len(parts) > 5:
        return False
    if ROLEISH_NAME_RE.search(cleaned):
        return False
    # Require at least two alphabetic name tokens (reject "A B" initials-only noise).
    alpha_tokens = [p for p in parts if re.search(r"[A-Za-z]{2,}", p)]
    return len(alpha_tokens) >= 2


def extract_html_title(html: str) -> str:
    match = HTML_TITLE_RE.search(html or "")
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def validate_linkedin_profile(url: str, full_name: str) -> LinkedInVerdict:
    """Check a LinkedIn profile URL via HTTP + page title name match."""
    normalized = normalize_linkedin_url(url)
    if not normalized:
        return "invalid"

    try:
        request = Request(
            normalized,
            headers={
                "User-Agent": LINKEDIN_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(request, timeout=LINKEDIN_FETCH_TIMEOUT) as response:
            status = getattr(response, "status", 200) or 200
            body = response.read(LINKEDIN_FETCH_MAX_BYTES).decode(
                "utf-8", errors="replace"
            )
    except HTTPError as exc:
        if exc.code == 404:
            return "invalid"
        # LinkedIn bot wall / rate limit — do not treat as definitive failure.
        if exc.code in {999, 403, 429, 401}:
            return "inconclusive"
        return "inconclusive"
    except (URLError, TimeoutError, OSError):
        return "inconclusive"

    if status in {999, 403, 429, 401}:
        return "inconclusive"
    if status == 404:
        return "invalid"
    if status != 200:
        return "inconclusive"

    title = extract_html_title(body)
    og_match = OG_TITLE_RE.search(body)
    combined = title
    if og_match:
        combined = f"{title} {og_match.group(1)}"

    lower_body_head = body[:3000].lower()
    if (
        not title
        or "authwall" in lower_body_head
        or "sign in" in title.lower()
        or "join linkedin" in title.lower()
    ):
        return "inconclusive"

    if text_matches_name(combined, full_name):
        return "valid"

    # Public profile page with a clear LinkedIn title but wrong person.
    if "| linkedin" in title.lower() or "linkedin" in title.lower():
        return "invalid"
    return "inconclusive"


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


def write_founders_md(folder: Path, doc: FoundersDoc) -> tuple[Path, bool]:
    """Write Founders.md. Returns (path, changed). Skips write if content identical."""
    path = founders_md_write_path(folder)
    new_text = render_founders_md(doc)
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == new_text:
                return path, False
        except OSError:
            pass
    path.write_text(new_text, encoding="utf-8")
    return path, True


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


def is_founders_md_backup(name: str) -> bool:
    """True for Founders.md and renamed backups like Founders.md.bak-..."""
    return name.lower().startswith(FOUNDERS_MD_NAME.lower())


def collect_materials(folder: Path) -> tuple[str, list[str]]:
    """Return (combined text, sources) from primary materials only.

    Includes top-level docs plus nested primary dirs (e.g. emails/, transcripts/).
    Never reads ai-generated/. Excludes Founders.md and Founders.md.* backups.
    """
    chunks: list[tuple[str, str]] = []
    sources: list[str] = []

    for path, text in collect_documents(
        folder,
        recursive=True,
        exclude_dirs={AI_GENERATED_DIR},
    ):
        if is_founders_md_backup(path.name):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        try:
            label = str(path.relative_to(folder))
        except ValueError:
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
    api_key: str,
    model: str,
) -> dict[str, Any]:
    user_prompt = (
        f"Folder name: {folder_name}\n\n"
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


@dataclass
class LinkedInCandidate:
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""  # brave | compound


def brave_linkedin_candidates(
    founder: Founder,
    company_name: str | None,
) -> list[LinkedInCandidate]:
    """Return LinkedIn /in/ URLs found via Brave search (not model-invented)."""
    company = (company_name or "").strip()
    queries: list[str] = []
    if company:
        queries.append(f'{founder.full_name} {company} site:linkedin.com/in')
        queries.append(f'"{founder.full_name}" {company} LinkedIn')
    queries.append(f'"{founder.full_name}" site:linkedin.com/in')

    seen: set[str] = set()
    candidates: list[LinkedInCandidate] = []

    for query in queries:
        try:
            results = search_brave(query, max_results=8)
        except Exception as exc:
            print(
                f"Warning: Brave search failed for {founder.full_name!r}: {exc}",
                file=sys.stderr,
            )
            continue

        for result in results:
            texts = [result.url, result.title, result.snippet]
            for text in texts:
                for match in LINKEDIN_IN_RE.finditer(text or ""):
                    url = normalize_linkedin_url(match.group(0))
                    if not url or url.lower() in seen:
                        continue
                    seen.add(url.lower())
                    candidates.append(
                        LinkedInCandidate(
                            url=url,
                            title=result.title,
                            snippet=result.snippet,
                            source="brave",
                        )
                    )

        # Early stop when we already have name-matching Brave hits.
        if any(
            text_matches_name(f"{c.title} {c.snippet}", founder.full_name)
            for c in candidates
        ):
            break

    # Prefer candidates whose search title/snippet mention the person.
    candidates.sort(
        key=lambda c: (
            0
            if text_matches_name(f"{c.title} {c.snippet}", founder.full_name)
            else 1
        )
    )
    return candidates


def compound_propose_linkedin(
    *,
    founder: Founder,
    company_name: str | None,
    api_key: str,
    model: str,
) -> str | None:
    """Ask Compound for one LinkedIn URL. Caller must validate the result."""
    company = company_name or "(unknown company)"
    user_prompt = (
        f"Company / startup: {company}\n\n"
        "Find the LinkedIn personal profile URL for this founder:\n"
        f"{json.dumps({'full_name': founder.full_name, 'first_name': founder.first_name, 'last_name': founder.last_name}, indent=2)}\n\n"
        "Search the web. Return JSON as specified. "
        "If the URL is not present in search results, return null."
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
        raise RuntimeError("Compound LinkedIn propose failed without a response")

    # Support both single-object and legacy {"profiles":[...]} shapes.
    entry: dict[str, Any] | None = None
    if isinstance(payload.get("profiles"), list) and payload["profiles"]:
        first = payload["profiles"][0]
        if isinstance(first, dict):
            entry = first
    elif isinstance(payload, dict):
        entry = payload

    if not entry:
        return None

    confidence = str(entry.get("confidence") or "").strip().lower()
    url = normalize_linkedin_url(
        str(entry.get("linkedin_url")) if entry.get("linkedin_url") else None
    )
    if not url:
        return None
    # Require explicit high/medium; reject low/none/missing confidence from Compound.
    if confidence not in {"high", "medium"}:
        print(
            f"Ignoring Compound URL for {founder.full_name} "
            f"(confidence={confidence or 'missing'}): {url}",
            file=sys.stderr,
        )
        return None
    return url


def accept_linkedin_candidate(
    candidate: LinkedInCandidate,
    full_name: str,
) -> str | None:
    """Validate a candidate; accept Brave-corroborated inconclusive hits."""
    verdict = validate_linkedin_profile(candidate.url, full_name)
    if verdict == "valid":
        print(
            f"Validated LinkedIn for {full_name} via {candidate.source}: "
            f"{candidate.url}",
            file=sys.stderr,
        )
        return candidate.url
    if verdict == "invalid":
        print(
            f"Rejected LinkedIn for {full_name} ({candidate.source}, invalid): "
            f"{candidate.url}",
            file=sys.stderr,
        )
        return None

    # HTTP inconclusive (bot wall). Trust Brave only when the search hit
    # title/snippet already names the person.
    brave_ok = text_matches_name(
        f"{candidate.title} {candidate.snippet}", full_name
    )
    if candidate.source == "brave" and brave_ok:
        print(
            f"Accepting Brave-corroborated LinkedIn for {full_name} "
            f"(HTTP inconclusive): {candidate.url}",
            file=sys.stderr,
        )
        return candidate.url

    print(
        f"Skipping LinkedIn for {full_name} ({candidate.source}, inconclusive): "
        f"{candidate.url}",
        file=sys.stderr,
    )
    return None


def resolve_one_linkedin(
    *,
    founder: Founder,
    company_name: str | None,
    api_key: str,
    compound_model: str,
) -> str | None:
    """Brave-first LinkedIn resolve with Compound fallback + HTTP validation."""
    if not founder.has_first_and_last() or not is_searchable_person_name(
        founder.full_name
    ):
        print(
            f"Skipping LinkedIn search for non-person/incomplete name: "
            f"{founder.full_name}",
            file=sys.stderr,
        )
        return None

    print(
        f"Resolving LinkedIn for {founder.full_name} "
        f"(company={company_name or 'unknown'}) ...",
        file=sys.stderr,
    )

    for candidate in brave_linkedin_candidates(founder, company_name):
        accepted = accept_linkedin_candidate(candidate, founder.full_name)
        if accepted:
            return accepted

    try:
        proposed = compound_propose_linkedin(
            founder=founder,
            company_name=company_name,
            api_key=api_key,
            model=compound_model,
        )
    except Exception as exc:
        print(
            f"Warning: Compound LinkedIn propose failed for "
            f"{founder.full_name}: {exc}",
            file=sys.stderr,
        )
        proposed = None

    if proposed:
        accepted = accept_linkedin_candidate(
            LinkedInCandidate(url=proposed, source="compound"),
            founder.full_name,
        )
        if accepted:
            return accepted

    print(
        f"No validated LinkedIn URL for {founder.full_name}",
        file=sys.stderr,
    )
    return None


def resolve_missing_linkedins(
    *,
    company_name: str | None,
    founders_missing: list[Founder],
    api_key: str,
    compound_model: str,
) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    for founder in founders_missing:
        results[founder.full_name] = resolve_one_linkedin(
            founder=founder,
            company_name=company_name,
            api_key=api_key,
            compound_model=compound_model,
        )
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

    extracted_company: str | None = None
    extracted_founders: list[dict[str, Any]] = []

    if materials.strip():
        try:
            extracted = extract_founders(
                materials,
                folder_name=folder.name,
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

    doc = fill_blanks_merge(existing, extracted_company, extracted_founders)

    missing = [f for f in doc.founders if not f.linkedin_is_set()]
    if missing and not skip_web_search:
        print(
            f"Resolving {len(missing)} LinkedIn URL(s) "
            f"(Brave → validate → Compound fallback) ...",
            file=sys.stderr,
        )
        try:
            resolved = resolve_missing_linkedins(
                company_name=doc.company_name,
                founders_missing=missing,
                api_key=api_key,
                compound_model=compound_model,
            )
        except Exception as exc:
            print(
                f"Warning: LinkedIn resolve failed: {exc}",
                file=sys.stderr,
            )
            resolved = {}

        for founder in missing:
            url = resolved.get(founder.full_name)
            if url and not founder.linkedin_is_set():
                founder.linkedin = url
    elif missing and skip_web_search:
        print(
            f"Skipping web search for {len(missing)} founder(s) without LinkedIn",
            file=sys.stderr,
        )

    doc.status = compute_status(doc)
    out_path, changed = write_founders_md(folder, doc)
    markdown = render_founders_md(doc)
    action = "Wrote" if changed else "Unchanged"
    print(f"{action} {out_path} (Status: {doc.status})", file=sys.stderr)
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
