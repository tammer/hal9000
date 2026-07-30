#!/usr/bin/env python3
"""Identify founders and LinkedIn profile URLs for a deal or portco folder.

Usage:
  python founders.py deals/Mobi
  python founders.py portcos/Central-Agent

Reads summary.md / identity.json / top-level materials, extracts founders with
Groq JSON mode, then resolves missing LinkedIn URLs via groq/compound web search.
Writes ai-generated/founders.json and prints the same JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from document_utils import collect_documents
from fetch_transcripts import groq_json_chat
from get_facts import parse_json_response
from paths import resolve_company_folder

DEFAULT_MODEL = "llama-3.3-70b-versatile"
COMPOUND_MODEL = "groq/compound"
AI_GENERATED_DIR = "ai-generated"
SUMMARY_NAME = "summary.md"
IDENTITY_NAME = "identity.json"
FOUNDERS_OUTPUT_NAME = "founders.json"
MAX_MATERIAL_CHARS = 80_000
LINKEDIN_IN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?",
    re.IGNORECASE,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify founders and LinkedIn URLs for a deal or portco folder "
            "using Groq (JSON extract + Compound web search)."
        )
    )
    parser.add_argument(
        "path",
        help="Company path: deals/<folder> or portcos/<folder>",
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
    """Return (combined text, list of source labels) for extraction."""
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


def split_name(full_name: str) -> tuple[str | None, str | None]:
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def empty_founder(full_name: str = "") -> dict[str, Any]:
    first, last = split_name(full_name) if full_name else (None, None)
    return {
        "first_name": first,
        "last_name": last,
        "full_name": full_name.strip() or None,
        "linkedin_url": None,
        "linkedin_source": None,
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

    # Exclude Antler staff
    full_lower = full.lower()
    for staff in ANTLER_STAFF:
        if full_lower == staff.lower():
            return None

    return {
        "first_name": first,
        "last_name": last,
        "full_name": full,
        "linkedin_url": linkedin,
        "linkedin_source": "materials" if linkedin else None,
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

    # If extraction returned nobody but identity has human_names, seed from those.
    if not founders and identity_hints:
        for name in identity_hints.get("human_names") or []:
            cleaned = str(name).strip()
            if not cleaned:
                continue
            founder = empty_founder(cleaned)
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


def merge_regex_linkedin_urls(
    founders: list[dict[str, Any]],
    materials: str,
) -> None:
    """Attach regex-found LinkedIn URLs to founders when unambiguous."""
    urls = find_linkedin_urls(materials)
    if not urls:
        return

    # Assign URL already attached by model stays; for unassigned URLs, try
    # matching slug tokens against founder names.
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
            best["linkedin_source"] = "materials"
            assigned.add(url.lower())


def compound_resolve_linkedin(
    *,
    company_name: str | None,
    founders_missing: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> dict[str, str | None]:
    """Return map full_name -> linkedin_url (or None) for missing founders."""
    if not founders_missing:
        return {}

    people = [
        {
            "full_name": f["full_name"],
            "first_name": f.get("first_name"),
            "last_name": f.get("last_name"),
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
        # Drop weak matches to reduce false positives on the POC.
        if url and confidence in {"", "none", "low"}:
            url = None
        results[name] = url

    # Ensure every requested name has a key
    for founder in founders_missing:
        name = founder["full_name"]
        if name not in results:
            # try case-insensitive match
            matched = None
            for key, value in results.items():
                if key.lower() == name.lower():
                    matched = value
                    break
            results[name] = matched

    return results


def write_founders_json(folder: Path, payload: dict[str, Any]) -> Path:
    ai_dir = folder / AI_GENERATED_DIR
    ai_dir.mkdir(parents=True, exist_ok=True)
    out_path = ai_dir / FOUNDERS_OUTPUT_NAME
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = args.model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
    compound_model = args.compound_model or COMPOUND_MODEL

    try:
        folder = resolve_company_folder(args.path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: folder does not exist: {folder}", file=sys.stderr)
        return 1

    rel_path = relative_company_path(args.path)
    print(f"Resolving founders for {rel_path} ...", file=sys.stderr)

    materials, sources = collect_materials(folder)
    if not materials.strip():
        print(
            "Error: no readable materials found (summary.md, identity.json, "
            "or top-level docs)",
            file=sys.stderr,
        )
        return 1

    print(f"Sources: {', '.join(sources)}", file=sys.stderr)

    identity_hints = load_identity_hints(folder)
    try:
        extracted = extract_founders(
            materials,
            folder_name=folder.name,
            identity_hints=identity_hints,
            api_key=api_key,
            model=model,
        )
    except Exception as exc:
        print(f"Error: founder extraction failed: {exc}", file=sys.stderr)
        return 1

    founders: list[dict[str, Any]] = extracted["founders"]
    merge_regex_linkedin_urls(founders, materials)

    missing = [f for f in founders if not f.get("linkedin_url")]
    if missing and not args.skip_web_search:
        print(
            f"Resolving {len(missing)} LinkedIn URL(s) via {compound_model} ...",
            file=sys.stderr,
        )
        try:
            resolved = compound_resolve_linkedin(
                company_name=extracted.get("company_name"),
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
            url = resolved.get(founder["full_name"])
            if not url:
                # case-insensitive fallback
                for key, value in resolved.items():
                    if key.lower() == founder["full_name"].lower() and value:
                        url = value
                        break
            if url:
                founder["linkedin_url"] = url
                founder["linkedin_source"] = "web_search"
    elif missing and args.skip_web_search:
        print(
            f"Skipping web search for {len(missing)} founder(s) without LinkedIn",
            file=sys.stderr,
        )

    result = {
        "path": rel_path,
        "company_name": extracted.get("company_name"),
        "founders": founders,
        "sources": sources,
    }

    out_path = write_founders_json(folder, result)
    print(f"Wrote {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
