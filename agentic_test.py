#!/usr/bin/env python3
import argparse
import getpass
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from textwrap import shorten
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from document_utils import collect_documents
from template_utils import load_markdown_template

if TYPE_CHECKING:
    from groq import Groq

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_CONTEXT_CHUNKS = 4
DEFAULT_MAX_REPAIR_ROUNDS = 3
RESULTS_PER_SEARCH = 8
PAGES_PER_SEARCH = 4
MAX_SECTION_EVIDENCE = 8
MAX_VALIDATION_QUERY_COUNT = 2
LOCAL_CHUNK_CHARS = 1600
LOCAL_CHUNK_OVERLAP = 200
MAX_PAGE_CHARS = 4_000
REQUEST_TIMEOUT_SECONDS = 20
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

RESEARCH_BRIEF_SYSTEM_PROMPT = """You turn a user's instruction and a markdown report template into a concrete research brief.

Return valid JSON only with this exact shape:
{
  "topic": "short topic name",
  "goal": "what success looks like",
  "presentation": "how the result should be written/formatted",
  "must_cover": ["required area 1", "required area 2"],
  "search_angles": ["promising search angle 1", "promising search angle 2"],
  "quality_checks": ["what a good report must verify"]
}

Rules:
- Preserve the user's intent.
- Treat the report template as authoritative guidance for structure, detail, and citation quality.
- If the task is about checking or validating claims, make that explicit in the goal or quality checks.
- Keep lists concise and useful.
"""

PLANNER_SYSTEM_PROMPT = """You are running a hybrid research loop that can inspect local documents and search the web.

Your job is to choose exactly one next step:
1. inspect local document context,
2. run one web search, or
3. write the final report.

Return valid JSON only.

Rules:
- Treat local documents as first-party context when the user's question is about those materials.
- Prefer local document lookup before web search when the task is grounded in provided transcripts, decks, tax files, or similar internal materials.
- Use web search when the user wants outside validation, when local evidence is insufficient, or when you need up-to-date external facts.
- Use the report template and current section coverage to decide what evidence is still missing.
- Avoid repeating the same lookup or search unless you are intentionally narrowing it.
- Stop only when the evidence appears sufficient for a careful template-shaped report.

Return exactly one of these shapes:
{"action":"local_search","reasoning":"short explanation","local_query":"query for local retrieval","focus":"what this local lookup should clarify"}
{"action":"web_search","reasoning":"short explanation","search_query":"query to run on the web","focus":"what this web search should clarify"}
{"action":"report","reasoning":"short explanation","confidence":0.0,"citations":["S1","S2"]}
"""

SECTION_REPORT_SYSTEM_PROMPT = """You write one section of a markdown research report from gathered evidence.

Return valid JSON only with this exact shape:
{"section_markdown":"markdown for this section only, without the heading","citations":["S1","S2"]}

Rules:
- Follow the section instructions exactly.
- Respect the citation/detail expectations described by the template and user instruction.
- Use only the provided evidence.
- Distinguish between internal document evidence and external web validation when relevant.
- Cite sources inline using ids like [S1].
- Do not include the section heading in section_markdown.
- Do not invent facts.
"""

SECTION_VALIDATOR_SYSTEM_PROMPT = """You validate one markdown report section against the template, instruction, and evidence.

Return valid JSON only with this exact shape:
{
  "passes": true,
  "score": 0,
  "missing_requirements": ["requirement not met"],
  "citation_gaps": ["claim lacks citation"],
  "unsupported_claims": ["claim not supported by evidence"],
  "overstatements": ["claim stated too strongly"],
  "needs_more_research": false,
  "local_queries": ["targeted local lookup"],
  "web_queries": ["targeted web lookup"],
  "rewrite_guidance": "specific revision guidance"
}

Rules:
- Validate only against the provided template instructions, user instruction, and evidence.
- Mark passes=false if the section misses important requirements, lacks needed citations, or overstates what the evidence shows.
- Set needs_more_research=true only when the current evidence appears insufficient.
- Suggested queries must be specific and useful.
- Keep feedback concise and actionable.
"""

FINAL_REPORT_VALIDATOR_SYSTEM_PROMPT = """You validate a full markdown report against the template and evidence.

Return valid JSON only with this exact shape:
{
  "passes": true,
  "score": 0,
  "issues": ["report-level issue"],
  "section_titles_needing_revision": ["Section Title"],
  "citation_gaps": ["citation problem"],
  "consistency_problems": ["cross-section inconsistency"],
  "needs_more_research": false,
  "local_queries": ["targeted local lookup"],
  "web_queries": ["targeted web lookup"],
  "rewrite_guidance": "specific report-level revision guidance"
}

Rules:
- Check template compliance, cross-section consistency, citation sufficiency, and whether conclusions are stronger than the evidence supports.
- Set needs_more_research=true only when the current evidence is not enough to repair the report.
- Keep feedback concise and actionable.
"""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class DocumentRecord:
    path: Path
    name: str
    text: str


@dataclass
class DocumentChunk:
    chunk_key: str
    path: Path
    name: str
    chunk_index: int
    text: str


@dataclass
class ReportTemplateSection:
    title: str
    instruction: str


@dataclass
class ReportTemplate:
    path: Path
    raw_text: str
    sections: list[ReportTemplateSection]


@dataclass
class EvidenceSource:
    source_id: str
    source_kind: str
    title: str
    location: str
    excerpt: str
    query: str
    chunk_index: int | None = None


@dataclass
class ResearchBrief:
    topic: str
    goal: str
    presentation: str
    must_cover: list[str]
    search_angles: list[str]
    quality_checks: list[str]


@dataclass
class SectionDraft:
    title: str
    markdown: str
    citations: list[str]


@dataclass
class SectionValidation:
    title: str
    passes: bool
    score: int
    missing_requirements: list[str]
    citation_gaps: list[str]
    unsupported_claims: list[str]
    overstatements: list[str]
    needs_more_research: bool
    local_queries: list[str]
    web_queries: list[str]
    rewrite_guidance: str
    programmatic_issues: list[str]


@dataclass
class ReportValidation:
    passes: bool
    score: int
    issues: list[str]
    section_titles_needing_revision: list[str]
    citation_gaps: list[str]
    consistency_problems: list[str]
    needs_more_research: bool
    local_queries: list[str]
    web_queries: list[str]
    rewrite_guidance: str
    programmatic_issues: list[str]


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._text_parts.append(data)

    def text(self) -> str:
        return normalize_whitespace(" ".join(self._text_parts))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]


def merge_unique(items: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_whitespace(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def format_list(items: list[str], fallback: str) -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {fallback}"


def is_public_web_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    return "search.brave.com" not in parsed.netloc


def fetch_url(url: str) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read()
    return body.decode(charset, errors="replace"), content_type


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        request_headers.update(headers)

    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
    return json.loads(body)


def brave_search_api_key() -> str:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY")
    if api_key:
        return api_key

    if sys.stdin.isatty():
        api_key = getpass.getpass("Brave Search API key: ").strip()
        if api_key:
            return api_key

    raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")


def search_brave(query: str, max_results: int = RESULTS_PER_SEARCH) -> list[SearchResult]:
    params = urlencode(
        {
            "q": query,
            "count": max_results,
            "extra_snippets": "true",
            "safesearch": "moderate",
        }
    )
    payload = fetch_json(
        f"{BRAVE_SEARCH_URL}?{params}",
        headers={"X-Subscription-Token": brave_search_api_key()},
    )

    raw_results = payload.get("web", {}).get("results", [])
    deduped: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in raw_results:
        title = normalize_whitespace(str(result.get("title", "")))
        url = str(result.get("url", "")).split("#", 1)[0]
        snippets = []
        description = normalize_whitespace(str(result.get("description", "")))
        if description:
            snippets.append(description)
        for extra in result.get("extra_snippets", []) or []:
            extra_text = normalize_whitespace(str(extra))
            if extra_text:
                snippets.append(extra_text)

        if not title or not url or not is_public_web_url(url):
            continue
        if url in seen_urls:
            continue

        seen_urls.add(url)
        deduped.append(SearchResult(title=title, url=url, snippet=" ".join(snippets)))

    return deduped


def extract_visible_text(html_content: str) -> str:
    parser = VisibleTextParser()
    parser.feed(html_content)
    text = parser.text()
    if len(text) > MAX_PAGE_CHARS:
        return text[:MAX_PAGE_CHARS].rsplit(" ", 1)[0] + " ..."
    return text


def read_result_page(result: SearchResult) -> str:
    try:
        content, content_type = fetch_url(result.url)
    except Exception as exc:
        return f"[Failed to fetch page: {exc}]"

    if "html" not in content_type:
        return f"[Skipped non-HTML content type: {content_type}]"

    text = extract_visible_text(content)
    return text or "[Page returned little or no readable text]"


def combine_result_evidence(result: SearchResult, page_excerpt: str) -> str:
    parts: list[str] = []
    if result.snippet:
        parts.append(f"Search snippets: {result.snippet}")
    if page_excerpt:
        parts.append(f"Page text: {page_excerpt}")
    return "\n".join(parts).strip()


def groq_client() -> "Groq":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    from groq import Groq

    return Groq(api_key=api_key)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def escape_control_characters_in_json(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                result.append(char)
                escape = False
                continue

            if char == "\\":
                result.append(char)
                escape = True
                continue

            if char == '"':
                result.append(char)
                in_string = False
                continue

            if ord(char) < 32:
                replacements = {
                    "\n": "\\n",
                    "\r": "\\r",
                    "\t": "\\t",
                    "\b": "\\b",
                    "\f": "\\f",
                }
                result.append(replacements.get(char, f"\\u{ord(char):04x}"))
                continue

            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True

    return "".join(result)


def parse_json_response(content: str) -> dict[str, Any]:
    text = strip_code_fences(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = extract_json_object(text)
        if not candidate:
            raise

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = escape_control_characters_in_json(candidate)
        return json.loads(repaired)


def llm_json(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    client = groq_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_json_response(content)


def sanitize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = normalize_whitespace(str(item))
        if text:
            cleaned.append(text)
    return cleaned


def sanitize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(score, 100))


def extract_inline_citations(text: str) -> list[str]:
    return re.findall(r"\[(S\d+)\]", text)


def instructions_require_citations(text: str) -> bool:
    lowered = text.lower()
    keywords = ("cite", "citation", "source", "citable", "exact source")
    return any(keyword in lowered for keyword in keywords)


def parse_section_validation(
    title: str,
    payload: dict[str, Any],
    *,
    programmatic_issues: list[str],
) -> SectionValidation:
    return SectionValidation(
        title=title,
        passes=bool(payload.get("passes")),
        score=sanitize_score(payload.get("score")),
        missing_requirements=sanitize_list(payload.get("missing_requirements")),
        citation_gaps=sanitize_list(payload.get("citation_gaps")),
        unsupported_claims=sanitize_list(payload.get("unsupported_claims")),
        overstatements=sanitize_list(payload.get("overstatements")),
        needs_more_research=bool(payload.get("needs_more_research")),
        local_queries=sanitize_list(payload.get("local_queries"))[:MAX_VALIDATION_QUERY_COUNT],
        web_queries=sanitize_list(payload.get("web_queries"))[:MAX_VALIDATION_QUERY_COUNT],
        rewrite_guidance=normalize_whitespace(str(payload.get("rewrite_guidance", ""))),
        programmatic_issues=programmatic_issues,
    )


def parse_report_validation(
    payload: dict[str, Any],
    *,
    programmatic_issues: list[str],
    template_titles: list[str],
) -> ReportValidation:
    allowed_titles = {title.lower(): title for title in template_titles}
    raw_titles = sanitize_list(payload.get("section_titles_needing_revision"))
    section_titles = []
    for title in raw_titles:
        mapped = allowed_titles.get(title.lower())
        if mapped and mapped not in section_titles:
            section_titles.append(mapped)

    return ReportValidation(
        passes=bool(payload.get("passes")),
        score=sanitize_score(payload.get("score")),
        issues=sanitize_list(payload.get("issues")),
        section_titles_needing_revision=section_titles,
        citation_gaps=sanitize_list(payload.get("citation_gaps")),
        consistency_problems=sanitize_list(payload.get("consistency_problems")),
        needs_more_research=bool(payload.get("needs_more_research")),
        local_queries=sanitize_list(payload.get("local_queries"))[:MAX_VALIDATION_QUERY_COUNT],
        web_queries=sanitize_list(payload.get("web_queries"))[:MAX_VALIDATION_QUERY_COUNT],
        rewrite_guidance=normalize_whitespace(str(payload.get("rewrite_guidance", ""))),
        programmatic_issues=programmatic_issues,
    )


def section_validation_issue_count(validation: SectionValidation) -> int:
    return sum(
        len(items)
        for items in (
            validation.missing_requirements,
            validation.citation_gaps,
            validation.unsupported_claims,
            validation.overstatements,
            validation.programmatic_issues,
        )
    )


def report_validation_issue_count(validation: ReportValidation) -> int:
    return sum(
        len(items)
        for items in (
            validation.issues,
            validation.section_titles_needing_revision,
            validation.citation_gaps,
            validation.consistency_problems,
            validation.programmatic_issues,
        )
    )


def programmatic_section_issues(
    section: ReportTemplateSection,
    section_markdown: str,
    evidence: list[EvidenceSource],
) -> list[str]:
    issues: list[str] = []
    if not section_markdown.strip():
        issues.append("section is empty")

    relevant_evidence_exists = bool(evidence)
    has_inline_citations = bool(extract_inline_citations(section_markdown))
    if instructions_require_citations(section.instruction) and not has_inline_citations:
        issues.append("section is missing inline citations required by the template")
    elif relevant_evidence_exists and not has_inline_citations:
        issues.append("section has evidence available but no inline citations")

    return issues


def programmatic_report_issues(
    report_template: ReportTemplate,
    section_drafts: dict[str, SectionDraft],
    citation_ids: list[str],
) -> list[str]:
    issues: list[str] = []
    missing_sections = [
        section.title
        for section in report_template.sections
        if section.title not in section_drafts
    ]
    if missing_sections:
        issues.append(
            "missing required sections: " + ", ".join(missing_sections)
        )

    for section in report_template.sections:
        draft = section_drafts.get(section.title)
        if draft and not draft.markdown.strip():
            issues.append(f"section '{section.title}' is empty")

    if not citation_ids:
        issues.append("report has no citations")

    return issues


def display_path(path: Path, root: Path | None) -> str:
    if root and root.is_dir():
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def load_context_documents(
    context_path: str | None,
    *,
    exclude_paths: set[Path] | None = None,
) -> tuple[Path | None, list[DocumentRecord]]:
    if not context_path:
        return None, []

    path = Path(context_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"context path does not exist: {path}")

    excluded = {excluded_path.resolve() for excluded_path in (exclude_paths or set())}
    raw_documents = collect_documents(path, recursive=path.is_dir())
    documents = [
        DocumentRecord(path=document_path, name=document_path.name, text=text)
        for document_path, text in raw_documents
        if document_path.resolve() not in excluded
        if text.strip()
    ]
    return path, documents


def load_report_template(report_template_path: str) -> ReportTemplate:
    path = Path(report_template_path).expanduser().resolve()
    sections_data = load_markdown_template(path, source_name="report template")
    raw_text = path.read_text(encoding="utf-8")
    sections = [
        ReportTemplateSection(title=section["title"], instruction=section["instruction"])
        for section in sections_data
    ]
    return ReportTemplate(path=path, raw_text=raw_text, sections=sections)


def split_text_into_chunks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(text_length, start + LOCAL_CHUNK_CHARS)
        if end < text_length:
            split_at = normalized.rfind("\n\n", start + LOCAL_CHUNK_CHARS // 2, end)
            if split_at == -1:
                split_at = normalized.rfind("\n", start + LOCAL_CHUNK_CHARS // 2, end)
            if split_at != -1 and split_at > start:
                end = split_at

        if end <= start:
            end = min(text_length, start + LOCAL_CHUNK_CHARS)

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - LOCAL_CHUNK_OVERLAP
        start = end if next_start <= start else next_start

    return chunks


def build_document_chunks(documents: list[DocumentRecord]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        for chunk_index, chunk_text in enumerate(split_text_into_chunks(document.text), start=1):
            chunks.append(
                DocumentChunk(
                    chunk_key=f"{document.path}::{chunk_index}",
                    path=document.path,
                    name=document.name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
    return chunks


def lexical_score(query_terms: Counter[str], candidate_terms: Counter[str], phrase_bonus: bool = False) -> float:
    overlap = set(query_terms) & set(candidate_terms)
    if not overlap:
        return 0.0

    frequency_score = sum(min(query_terms[token], candidate_terms[token]) for token in overlap)
    unique_score = len(overlap) * 3
    return float(frequency_score + unique_score + (5 if phrase_bonus else 0))


def score_chunk(query: str, chunk: DocumentChunk) -> float:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return 0.0

    chunk_terms = Counter(tokenize(f"{chunk.name} {chunk.text}"))
    filename_bonus = sum(1 for token in set(query_terms) if token in tokenize(chunk.name)) * 2
    phrase_bonus = normalize_whitespace(query).lower() in f"{chunk.name} {chunk.text}".lower()
    return lexical_score(query_terms, chunk_terms, phrase_bonus) + filename_bonus


def score_evidence_for_section(section: ReportTemplateSection, source: EvidenceSource) -> float:
    section_query = f"{section.title} {section.instruction}"
    section_terms = Counter(tokenize(section_query))
    if not section_terms:
        return 0.0

    evidence_terms = Counter(tokenize(f"{source.title} {source.query} {source.excerpt}"))
    phrase_bonus = section.title.lower() in f"{source.title} {source.excerpt}".lower()
    return lexical_score(section_terms, evidence_terms, phrase_bonus)


def retrieve_local_chunks(
    chunks: list[DocumentChunk],
    query: str,
    max_chunks: int,
    exclude_keys: set[str] | None = None,
) -> list[DocumentChunk]:
    if not query or not chunks or max_chunks < 1:
        return []

    excluded = exclude_keys or set()
    scored = []
    for chunk in chunks:
        if chunk.chunk_key in excluded:
            continue
        score = score_chunk(query, chunk)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].name.lower(), item[1].chunk_index))
    return [chunk for _, chunk in scored[:max_chunks]]


def build_document_inventory_text(documents: list[DocumentRecord], root: Path | None) -> str:
    if not documents:
        return "No local documents provided."

    lines = [f"{len(documents)} readable local document(s) available."]
    preview_count = min(len(documents), 15)
    for document in documents[:preview_count]:
        lines.append(
            f"- {display_path(document.path, root)} ({len(document.text):,} chars)"
        )
    if len(documents) > preview_count:
        lines.append(f"- ... and {len(documents) - preview_count} more document(s)")
    return "\n".join(lines)


def format_local_candidates(
    chunks: list[DocumentChunk],
    root: Path | None,
    query: str,
) -> str:
    if not chunks:
        return "No relevant local context candidates were found for the current query."

    blocks = [f"Current local retrieval query: {query}"]
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"C{index}",
                    f"Document: {display_path(chunk.path, root)}",
                    f"Chunk: {chunk.chunk_index}",
                    f"Excerpt: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_evidence(evidence: list[EvidenceSource]) -> str:
    if not evidence:
        return "No evidence gathered yet."

    blocks = []
    for source in evidence:
        lines = [
            source.source_id,
            f"Type: {source.source_kind}",
            f"Query: {source.query}",
            f"Title: {source.title}",
        ]
        if source.source_kind == "local":
            lines.append(f"Path: {source.location}")
            if source.chunk_index is not None:
                lines.append(f"Chunk: {source.chunk_index}")
        else:
            lines.append(f"URL: {source.location}")
        lines.append(f"Excerpt: {source.excerpt}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_report_template_text(report_template: ReportTemplate) -> str:
    blocks = []
    for index, section in enumerate(report_template.sections, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Section {index}: {section.title}",
                    f"Instructions: {section.instruction}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_section_coverage_text(
    report_template: ReportTemplate,
    evidence: list[EvidenceSource],
) -> str:
    if not report_template.sections:
        return "No report template sections available."

    blocks = []
    for index, section in enumerate(report_template.sections, start=1):
        scored: list[tuple[float, EvidenceSource]] = []
        for source in evidence:
            score = score_evidence_for_section(section, source)
            if score > 0:
                scored.append((score, source))
        scored.sort(key=lambda item: (-item[0], item[1].source_id))
        top_sources = [source.source_id for _, source in scored[:3]]
        if not top_sources:
            status = "under-supported"
        elif len(top_sources) == 1:
            status = f"partially supported by {top_sources[0]}"
        else:
            status = f"supported by {', '.join(top_sources)}"
        blocks.append(
            "\n".join(
                [
                    f"{index}. {section.title}",
                    f"Instructions: {section.instruction}",
                    f"Status: {status}",
                ]
            )
        )
    return "\n\n".join(blocks)


def select_section_evidence(
    section: ReportTemplateSection,
    evidence: list[EvidenceSource],
    *,
    max_sources: int = MAX_SECTION_EVIDENCE,
) -> list[EvidenceSource]:
    scored: list[tuple[float, EvidenceSource]] = []
    for source in evidence:
        score = score_evidence_for_section(section, source)
        if score > 0:
            scored.append((score, source))

    scored.sort(key=lambda item: (-item[0], item[1].source_id))
    selected = [source for _, source in scored[:max_sources]]
    if selected:
        return selected
    return evidence[:max_sources]


def build_research_brief_text(brief: ResearchBrief) -> str:
    return f"""Topic: {brief.topic}
Goal: {brief.goal}
Presentation: {brief.presentation}

Must cover:
{format_list(brief.must_cover, "No explicit required coverage areas provided.")}

Search angles:
{format_list(brief.search_angles, "Use sensible local and web research angles.")}

Quality checks:
{format_list(brief.quality_checks, "Ensure the final report is useful, accurate, and well-cited.")}
"""


def create_research_brief(
    instruction: str,
    report_template: ReportTemplate,
    model: str,
) -> ResearchBrief:
    payload = llm_json(
        RESEARCH_BRIEF_SYSTEM_PROMPT,
        "\n\n".join(
            [
                f"User instruction:\n{instruction}",
                f"Report template:\n{build_report_template_text(report_template)}",
            ]
        ),
        model,
    )
    topic = normalize_whitespace(str(payload.get("topic", ""))) or "Research task"
    goal = normalize_whitespace(str(payload.get("goal", ""))) or instruction
    presentation = (
        normalize_whitespace(str(payload.get("presentation", "")))
        or "Write a polished markdown report that follows the report template."
    )
    must_cover = merge_unique(
        sanitize_list(payload.get("must_cover"))
        + [section.title for section in report_template.sections]
    )
    search_angles = merge_unique(sanitize_list(payload.get("search_angles")))
    quality_checks = merge_unique(sanitize_list(payload.get("quality_checks")))

    return ResearchBrief(
        topic=topic,
        goal=goal,
        presentation=presentation,
        must_cover=must_cover,
        search_angles=search_angles,
        quality_checks=quality_checks,
    )


def build_planner_prompt(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    document_inventory: str,
    local_preview_query: str,
    local_candidates: list[DocumentChunk],
    local_search_history: list[str],
    web_search_history: list[str],
    evidence: list[EvidenceSource],
    iterations_left: int,
    root: Path | None,
) -> str:
    local_history = "\n".join(f"- {query}" for query in local_search_history) or "- none yet"
    web_history = "\n".join(f"- {query}" for query in web_search_history) or "- none yet"

    return f"""User instruction:
{instruction}

Report template:
{build_report_template_text(report_template)}

Current template section coverage:
{build_section_coverage_text(report_template, evidence)}

Research brief:
{build_research_brief_text(brief)}

Local document inventory:
{document_inventory}

Previous local lookups:
{local_history}

Previous web searches:
{web_history}

Iterations left: {iterations_left}

Current local context candidates:
{format_local_candidates(local_candidates, root, local_preview_query)}

Evidence gathered so far:
{format_evidence(evidence)}
"""


def build_report_section_prompt(
    instruction: str,
    report_template: ReportTemplate,
    section: ReportTemplateSection,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    rewrite_guidance: str,
) -> str:
    revision_block = (
        f"\nRevision guidance:\n{rewrite_guidance}\n"
        if rewrite_guidance
        else ""
    )
    return f"""User instruction:
{instruction}

Full report template:
{build_report_template_text(report_template)}

Current section:
Title: {section.title}
Instructions: {section.instruction}

Research brief:
{build_research_brief_text(brief)}

Local document inventory:
{document_inventory}
{revision_block}

Evidence:
{format_evidence(evidence)}
"""


def build_section_validator_prompt(
    instruction: str,
    report_template: ReportTemplate,
    section: ReportTemplateSection,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    section_markdown: str,
) -> str:
    return f"""User instruction:
{instruction}

Full report template:
{build_report_template_text(report_template)}

Current section:
Title: {section.title}
Instructions: {section.instruction}

Research brief:
{build_research_brief_text(brief)}

Local document inventory:
{document_inventory}

Evidence:
{format_evidence(evidence)}

Draft section markdown:
{section_markdown}
"""


def build_final_validator_prompt(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    document_inventory: str,
    report_markdown: str,
    evidence: list[EvidenceSource],
) -> str:
    return f"""User instruction:
{instruction}

Full report template:
{build_report_template_text(report_template)}

Research brief:
{build_research_brief_text(brief)}

Local document inventory:
{document_inventory}

Evidence:
{format_evidence(evidence)}

Draft report markdown:
{report_markdown}
"""


def choose_next_action(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    document_inventory: str,
    local_preview_query: str,
    local_candidates: list[DocumentChunk],
    local_search_history: list[str],
    web_search_history: list[str],
    evidence: list[EvidenceSource],
    iterations_left: int,
    model: str,
    root: Path | None,
) -> dict[str, Any]:
    return llm_json(
        PLANNER_SYSTEM_PROMPT,
        build_planner_prompt(
            instruction=instruction,
            report_template=report_template,
            brief=brief,
            document_inventory=document_inventory,
            local_preview_query=local_preview_query,
            local_candidates=local_candidates,
            local_search_history=local_search_history,
            web_search_history=web_search_history,
            evidence=evidence,
            iterations_left=iterations_left,
            root=root,
        ),
        model,
    )


def generate_report_section(
    instruction: str,
    report_template: ReportTemplate,
    section: ReportTemplateSection,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    rewrite_guidance: str,
    model: str,
) -> dict[str, Any]:
    return llm_json(
        SECTION_REPORT_SYSTEM_PROMPT,
        build_report_section_prompt(
            instruction=instruction,
            report_template=report_template,
            section=section,
            brief=brief,
            document_inventory=document_inventory,
            evidence=evidence,
            rewrite_guidance=rewrite_guidance,
        ),
        model,
    )


def validate_section_draft(
    instruction: str,
    report_template: ReportTemplate,
    section: ReportTemplateSection,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    section_markdown: str,
    model: str,
) -> SectionValidation:
    programmatic_issues = programmatic_section_issues(section, section_markdown, evidence)
    payload = llm_json(
        SECTION_VALIDATOR_SYSTEM_PROMPT,
        build_section_validator_prompt(
            instruction=instruction,
            report_template=report_template,
            section=section,
            brief=brief,
            document_inventory=document_inventory,
            evidence=evidence,
            section_markdown=section_markdown,
        ),
        model,
    )
    validation = parse_section_validation(
        section.title,
        payload,
        programmatic_issues=programmatic_issues,
    )
    if programmatic_issues:
        validation.passes = False
    return validation


def validate_full_report(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    document_inventory: str,
    report_markdown: str,
    evidence: list[EvidenceSource],
    section_drafts: dict[str, SectionDraft],
    citation_ids: list[str],
    model: str,
) -> ReportValidation:
    programmatic_issues = programmatic_report_issues(
        report_template,
        section_drafts,
        citation_ids,
    )
    payload = llm_json(
        FINAL_REPORT_VALIDATOR_SYSTEM_PROMPT,
        build_final_validator_prompt(
            instruction=instruction,
            report_template=report_template,
            brief=brief,
            document_inventory=document_inventory,
            report_markdown=report_markdown,
            evidence=evidence,
        ),
        model,
    )
    validation = parse_report_validation(
        payload,
        programmatic_issues=programmatic_issues,
        template_titles=[section.title for section in report_template.sections],
    )
    if programmatic_issues:
        validation.passes = False
    return validation


def build_report_from_section_drafts(
    report_template: ReportTemplate,
    section_drafts: dict[str, SectionDraft],
) -> dict[str, Any]:
    sections_markdown: list[str] = []
    citations: list[str] = []
    seen_citations: set[str] = set()

    for section in report_template.sections:
        draft = section_drafts.get(section.title)
        section_markdown = (
            draft.markdown if draft else "Insufficient evidence to complete this section."
        )
        sections_markdown.append(f"# {section.title}\n\n{section_markdown}")
        if not draft:
            continue
        for citation in draft.citations:
            if citation in seen_citations:
                continue
            seen_citations.add(citation)
            citations.append(citation)

    return {
        "report_markdown": "\n\n".join(sections_markdown),
        "citations": citations,
    }


def repair_metric(
    section_validations: dict[str, SectionValidation],
    report_validation: ReportValidation | None,
) -> tuple[int, int]:
    issue_count = sum(
        section_validation_issue_count(validation)
        for validation in section_validations.values()
    )
    score_total = sum(validation.score for validation in section_validations.values())
    if report_validation is not None:
        issue_count += report_validation_issue_count(report_validation)
        score_total += report_validation.score
    return issue_count, -score_total


def run_targeted_repair_research(
    *,
    local_queries: list[str],
    web_queries: list[str],
    chunks: list[DocumentChunk],
    evidence: list[EvidenceSource],
    seen_local_keys: set[str],
    evidence_by_url: set[str],
    source_counter: int,
    context_root: Path | None,
    max_context_chunks: int,
    local_search_history: list[str],
    web_search_history: list[str],
) -> tuple[int, int]:
    added_count = 0

    for query in merge_unique(local_queries)[:MAX_VALIDATION_QUERY_COUNT]:
        if query in local_search_history:
            continue
        print(f"Repair local lookup: {query}")
        local_search_history.append(query)
        retrieved = retrieve_local_chunks(
            chunks=chunks,
            query=query,
            max_chunks=max_context_chunks,
            exclude_keys=seen_local_keys,
        )
        before = source_counter
        source_counter = add_local_evidence(
            local_query=query,
            chunks=retrieved,
            evidence=evidence,
            seen_local_keys=seen_local_keys,
            source_counter=source_counter,
            root=context_root,
        )
        added_count += source_counter - before

    for query in merge_unique(web_queries)[:MAX_VALIDATION_QUERY_COUNT]:
        if query in web_search_history:
            continue
        print(f"Repair web search: {query}")
        web_search_history.append(query)
        try:
            results = search_brave(query)
        except Exception as exc:
            print(f"Repair search error: {exc}", file=sys.stderr)
            continue

        for result in results[:PAGES_PER_SEARCH]:
            if result.url in evidence_by_url:
                continue
            excerpt = combine_result_evidence(result, read_result_page(result))
            evidence.append(
                EvidenceSource(
                    source_id=f"S{source_counter}",
                    source_kind="web",
                    title=result.title,
                    location=result.url,
                    excerpt=excerpt,
                    query=query,
                )
            )
            evidence_by_url.add(result.url)
            source_counter += 1
            added_count += 1

    return source_counter, added_count


def build_section_rewrite_guidance(validation: SectionValidation) -> str:
    guidance: list[str] = []
    guidance.extend(validation.missing_requirements)
    guidance.extend(validation.citation_gaps)
    guidance.extend(validation.unsupported_claims)
    guidance.extend(validation.overstatements)
    guidance.extend(validation.programmatic_issues)
    if validation.rewrite_guidance:
        guidance.append(validation.rewrite_guidance)
    return " ".join(guidance).strip()


def synthesize_report(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    model: str,
    *,
    chunks: list[DocumentChunk],
    context_root: Path | None,
    seen_local_keys: set[str],
    evidence_by_url: set[str],
    source_counter: int,
    max_context_chunks: int,
    local_search_history: list[str],
    web_search_history: list[str],
    max_repair_rounds: int,
) -> tuple[dict[str, Any], int]:
    section_drafts: dict[str, SectionDraft] = {}
    rewrite_guidance_by_title: dict[str, str] = {}
    sections_to_redraft = {section.title for section in report_template.sections}
    best_payload: dict[str, Any] | None = None
    best_metric: tuple[int, int] | None = None
    previous_round_metric: tuple[int, int] | None = None

    for repair_round in range(1, max_repair_rounds + 1):
        print(f"\n[Repair round {repair_round}/{max_repair_rounds}]")
        for section in report_template.sections:
            if section.title not in sections_to_redraft and section.title in section_drafts:
                continue

            section_evidence = select_section_evidence(section, evidence)
            payload = generate_report_section(
                instruction=instruction,
                report_template=report_template,
                section=section,
                brief=brief,
                document_inventory=document_inventory,
                evidence=section_evidence,
                rewrite_guidance=rewrite_guidance_by_title.get(section.title, ""),
                model=model,
            )
            section_markdown = str(payload.get("section_markdown", "")).strip()
            if not section_markdown:
                section_markdown = "Insufficient evidence to complete this section."
            section_drafts[section.title] = SectionDraft(
                title=section.title,
                markdown=section_markdown,
                citations=safe_citations(payload, evidence),
            )

        section_validations: dict[str, SectionValidation] = {}
        failing_titles: set[str] = set()
        local_queries: list[str] = []
        web_queries: list[str] = []

        for section in report_template.sections:
            draft = section_drafts[section.title]
            section_evidence = select_section_evidence(section, evidence)
            validation = validate_section_draft(
                instruction=instruction,
                report_template=report_template,
                section=section,
                brief=brief,
                document_inventory=document_inventory,
                evidence=section_evidence,
                section_markdown=draft.markdown,
                model=model,
            )
            section_validations[section.title] = validation
            if not validation.passes or section_validation_issue_count(validation) > 0:
                failing_titles.add(section.title)
                rewrite_guidance_by_title[section.title] = build_section_rewrite_guidance(validation)
            else:
                rewrite_guidance_by_title.pop(section.title, None)

            if validation.needs_more_research:
                local_queries.extend(validation.local_queries)
                web_queries.extend(validation.web_queries)

        report_payload = build_report_from_section_drafts(report_template, section_drafts)
        report_validation: ReportValidation | None = None

        if not failing_titles:
            report_validation = validate_full_report(
                instruction=instruction,
                report_template=report_template,
                brief=brief,
                document_inventory=document_inventory,
                report_markdown=report_payload["report_markdown"],
                evidence=evidence,
                section_drafts=section_drafts,
                citation_ids=report_payload["citations"],
                model=model,
            )
            if not report_validation.passes or report_validation_issue_count(report_validation) > 0:
                local_queries.extend(report_validation.local_queries)
                web_queries.extend(report_validation.web_queries)
                failing_titles.update(report_validation.section_titles_needing_revision)
                if report_validation.rewrite_guidance:
                    titles = (
                        report_validation.section_titles_needing_revision
                        or [section.title for section in report_template.sections]
                    )
                    for title in titles:
                        existing = rewrite_guidance_by_title.get(title, "")
                        rewrite_guidance_by_title[title] = (
                            (existing + " " if existing else "")
                            + report_validation.rewrite_guidance
                        ).strip()
                if not failing_titles and report_template.sections:
                    failing_titles.update(section.title for section in report_template.sections)

        current_metric = repair_metric(section_validations, report_validation)
        if best_metric is None or current_metric < best_metric:
            best_metric = current_metric
            best_payload = report_payload

        if not failing_titles and report_validation and report_validation.passes:
            return report_payload, source_counter

        if repair_round == max_repair_rounds:
            break

        source_counter, added_count = run_targeted_repair_research(
            local_queries=local_queries,
            web_queries=web_queries,
            chunks=chunks,
            evidence=evidence,
            seen_local_keys=seen_local_keys,
            evidence_by_url=evidence_by_url,
            source_counter=source_counter,
            context_root=context_root,
            max_context_chunks=max_context_chunks,
            local_search_history=local_search_history,
            web_search_history=web_search_history,
        )
        if (
            added_count == 0
            and previous_round_metric is not None
            and current_metric >= previous_round_metric
        ):
            print("Repair loop stopped because validation did not improve and no new evidence was found.")
            break

        previous_round_metric = current_metric
        sections_to_redraft = failing_titles or {section.title for section in report_template.sections}

    if best_payload is None:
        best_payload = build_report_from_section_drafts(report_template, section_drafts)
    return best_payload, source_counter


def prompt_for_instruction() -> str:
    print("What would you like researched?")
    print("Include any guidance on how you want the report template applied.")
    return input("> ").strip()


def safe_citations(answer_payload: dict[str, Any], evidence: list[EvidenceSource]) -> list[str]:
    known_ids = {source.source_id for source in evidence}
    raw_citations = answer_payload.get("citations", [])
    if not isinstance(raw_citations, list):
        return []
    return [citation for citation in raw_citations if citation in known_ids]


def print_sources(citation_ids: list[str], evidence: list[EvidenceSource]) -> None:
    if not citation_ids:
        return

    print("\nSources:")
    by_id = {source.source_id: source for source in evidence}
    for citation_id in citation_ids:
        source = by_id[citation_id]
        if source.source_kind == "local":
            chunk_suffix = (
                f", chunk {source.chunk_index}" if source.chunk_index is not None else ""
            )
            print(f"- {citation_id}: {source.title} [local document{chunk_suffix}]")
            print(f"  {source.location}")
        else:
            print(f"- {citation_id}: {source.title} [web]")
            print(f"  {source.location}")


def append_source_appendix(
    report_markdown: str,
    citation_ids: list[str],
    evidence: list[EvidenceSource],
) -> str:
    if not citation_ids:
        return report_markdown

    by_id = {source.source_id: source for source in evidence}
    lines = [report_markdown.rstrip(), "", "## Sources"]
    for citation_id in citation_ids:
        source = by_id[citation_id]
        if source.source_kind == "local":
            label = f"{citation_id}: {source.title} (local document"
            if source.chunk_index is not None:
                label += f", chunk {source.chunk_index}"
            label += ")"
        else:
            label = f"{citation_id}: {source.title} (web)"
        lines.append(f"- {label} - {source.location}")
    return "\n".join(lines).rstrip() + "\n"


def write_output(path: str, report_markdown: str) -> None:
    output_path = Path(path)
    output_path.write_text(report_markdown, encoding="utf-8")


def print_brief_summary(
    brief: ResearchBrief,
    report_template: ReportTemplate,
    documents: list[DocumentRecord],
    chunks: list[DocumentChunk],
) -> None:
    print("\nResearch brief:")
    print(f"- Topic: {brief.topic}")
    print(f"- Goal: {brief.goal}")
    print(f"- Presentation: {brief.presentation}")
    print(f"- Report template: {report_template.path.name}")
    print(f"- Template sections: {len(report_template.sections)}")
    if brief.must_cover:
        print("- Must cover:")
        for item in brief.must_cover:
            print(f"  - {item}")
    if documents:
        print(f"- Local documents loaded: {len(documents)}")
        print(f"- Local chunks indexed: {len(chunks)}")


def add_local_evidence(
    local_query: str,
    chunks: list[DocumentChunk],
    evidence: list[EvidenceSource],
    seen_local_keys: set[str],
    source_counter: int,
    root: Path | None,
) -> int:
    for chunk in chunks:
        if chunk.chunk_key in seen_local_keys:
            continue
        evidence.append(
            EvidenceSource(
                source_id=f"S{source_counter}",
                source_kind="local",
                title=chunk.name,
                location=display_path(chunk.path, root),
                excerpt=chunk.text,
                query=local_query,
                chunk_index=chunk.chunk_index,
            )
        )
        seen_local_keys.add(chunk.chunk_key)
        source_counter += 1
    return source_counter


def ensure_local_report_context(
    instruction: str,
    report_template: ReportTemplate,
    brief: ResearchBrief,
    chunks: list[DocumentChunk],
    evidence: list[EvidenceSource],
    seen_local_keys: set[str],
    source_counter: int,
    max_context_chunks: int,
    root: Path | None,
) -> int:
    added = 0
    queries = [instruction, brief.goal, *brief.must_cover]
    for section in report_template.sections:
        queries.append(section.title)
        queries.append(f"{section.title} {section.instruction}")

    for query in queries:
        if added >= max_context_chunks:
            break
        retrieved = retrieve_local_chunks(
            chunks=chunks,
            query=query,
            max_chunks=1,
            exclude_keys=seen_local_keys,
        )
        before = source_counter
        source_counter = add_local_evidence(
            local_query=query,
            chunks=retrieved,
            evidence=evidence,
            seen_local_keys=seen_local_keys,
            source_counter=source_counter,
            root=root,
        )
        added += source_counter - before
    return source_counter


def run_research(
    instruction: str,
    report_template_path: str,
    model: str,
    max_iterations: int,
    max_repair_rounds: int,
    output_path: str | None,
    context_path: str | None,
    max_context_chunks: int,
) -> int:
    try:
        report_template = load_report_template(report_template_path)
    except Exception as exc:
        print(f"Report template error: {exc}", file=sys.stderr)
        return 1

    try:
        context_root, documents = load_context_documents(
            context_path,
            exclude_paths={report_template.path},
        )
    except Exception as exc:
        print(f"Context loading error: {exc}", file=sys.stderr)
        return 1

    chunks = build_document_chunks(documents)
    document_inventory = build_document_inventory_text(documents, context_root)

    try:
        brief = create_research_brief(instruction, report_template, model)
    except Exception as exc:
        print(f"Research brief error: {exc}", file=sys.stderr)
        return 1

    print_brief_summary(brief, report_template, documents, chunks)

    local_search_history: list[str] = []
    web_search_history: list[str] = []
    evidence: list[EvidenceSource] = []
    seen_local_keys: set[str] = set()
    evidence_by_url: set[str] = set()
    source_counter = 1
    local_preview_query = f"{instruction} {report_template.sections[0].title}"

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_iterations}]")
        local_candidates = retrieve_local_chunks(
            chunks=chunks,
            query=local_preview_query,
            max_chunks=max_context_chunks,
        )

        try:
            decision = choose_next_action(
                instruction=instruction,
                report_template=report_template,
                brief=brief,
                document_inventory=document_inventory,
                local_preview_query=local_preview_query,
                local_candidates=local_candidates,
                local_search_history=local_search_history,
                web_search_history=web_search_history,
                evidence=evidence,
                iterations_left=max_iterations - iteration + 1,
                model=model,
                root=context_root,
            )
        except Exception as exc:
            print(f"Planner error: {exc}", file=sys.stderr)
            return 1

        action = decision.get("action")
        if action == "report":
            print("Evidence looks sufficient. Drafting report.")
            break

        if action == "local_search":
            local_query = normalize_whitespace(str(decision.get("local_query", "")))
            focus = normalize_whitespace(str(decision.get("focus", "")))
            if not local_query:
                print("Planner requested a local lookup but did not provide a query.", file=sys.stderr)
                return 1

            print(f"Inspecting local context for: {local_query}")
            if focus:
                print(f"Focus: {focus}")
            local_search_history.append(local_query)
            local_preview_query = local_query

            retrieved = retrieve_local_chunks(
                chunks=chunks,
                query=local_query,
                max_chunks=max_context_chunks,
                exclude_keys=seen_local_keys,
            )
            if not retrieved:
                print("No new relevant local context found.")
                continue

            for chunk in retrieved:
                label = f"{chunk.name} [chunk {chunk.chunk_index}]"
                print(f"  Local: {shorten(label, width=90, placeholder='...')}")

            source_counter = add_local_evidence(
                local_query=local_query,
                chunks=retrieved,
                evidence=evidence,
                seen_local_keys=seen_local_keys,
                source_counter=source_counter,
                root=context_root,
            )
            continue

        if action != "web_search":
            print(f"Planner returned unexpected action: {action!r}", file=sys.stderr)
            return 1

        query = normalize_whitespace(str(decision.get("search_query", "")))
        focus = normalize_whitespace(str(decision.get("focus", "")))
        if not query:
            print("Planner requested a web search but did not provide a query.", file=sys.stderr)
            return 1

        print(f"Searching Brave for: {query}")
        if focus:
            print(f"Focus: {focus}")
        web_search_history.append(query)

        try:
            results = search_brave(query)
        except Exception as exc:
            print(f"Search error: {exc}", file=sys.stderr)
            return 1

        if not results:
            print("No search results found. Trying another iteration.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"  {index}. {shorten(result.title, width=90, placeholder='...')}")

        for result in results[:PAGES_PER_SEARCH]:
            if result.url in evidence_by_url:
                continue

            print(f"Reading: {shorten(result.title, width=90, placeholder='...')}")
            excerpt = combine_result_evidence(result, read_result_page(result))
            evidence.append(
                EvidenceSource(
                    source_id=f"S{source_counter}",
                    source_kind="web",
                    title=result.title,
                    location=result.url,
                    excerpt=excerpt,
                    query=query,
                )
            )
            evidence_by_url.add(result.url)
            source_counter += 1

    if chunks:
        source_counter = ensure_local_report_context(
            instruction=instruction,
            report_template=report_template,
            brief=brief,
            chunks=chunks,
            evidence=evidence,
            seen_local_keys=seen_local_keys,
            source_counter=source_counter,
            max_context_chunks=max_context_chunks,
            root=context_root,
        )

    print("\nGenerating final report.")
    try:
        report_payload, source_counter = synthesize_report(
            instruction=instruction,
            report_template=report_template,
            brief=brief,
            document_inventory=document_inventory,
            evidence=evidence,
            model=model,
            chunks=chunks,
            context_root=context_root,
            seen_local_keys=seen_local_keys,
            evidence_by_url=evidence_by_url,
            source_counter=source_counter,
            max_context_chunks=max_context_chunks,
            local_search_history=local_search_history,
            web_search_history=web_search_history,
            max_repair_rounds=max_repair_rounds,
        )
    except Exception as exc:
        print(f"Final report synthesis error: {exc}", file=sys.stderr)
        return 1

    report_markdown = str(report_payload.get("report_markdown", "")).strip()
    if not report_markdown:
        print("Final report synthesis returned an empty report.", file=sys.stderr)
        return 1

    citations = safe_citations(report_payload, evidence)
    printable_report = append_source_appendix(report_markdown, citations, evidence)

    print("\nReport:\n")
    print(printable_report.rstrip())
    print_sources(citations, evidence)

    if output_path:
        write_output(output_path, printable_report)
        print(f"\nSaved report to {output_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a template-driven research loop using local document context, Brave search, and Groq synthesis."
    )
    parser.add_argument(
        "--instruction",
        "--question",
        dest="instruction",
        help="Research instruction, including any run-specific details for the analysis.",
    )
    parser.add_argument(
        "--report-template",
        required=True,
        help="Path to a markdown report template file that defines the report sections and quality expectations.",
    )
    parser.add_argument(
        "--context-path",
        help="Optional path to a document folder or file to use as local research context.",
    )
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHUNKS,
        help=f"Maximum local context chunks to surface per retrieval step (default: {DEFAULT_MAX_CONTEXT_CHUNKS}).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the final markdown report.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        help="Groq model to use for planning and report synthesis.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of planning iterations (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=DEFAULT_MAX_REPAIR_ROUNDS,
        help=f"Maximum number of validator repair rounds (default: {DEFAULT_MAX_REPAIR_ROUNDS}).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    instruction = (args.instruction or "").strip()
    if not instruction:
        try:
            instruction = prompt_for_instruction()
        except EOFError:
            print("No research instruction provided.", file=sys.stderr)
            return 1

    if not instruction:
        print("Research instruction cannot be empty.", file=sys.stderr)
        return 1

    if args.max_iterations < 1:
        print("--max-iterations must be at least 1.", file=sys.stderr)
        return 1

    if args.max_context_chunks < 1:
        print("--max-context-chunks must be at least 1.", file=sys.stderr)
        return 1

    if args.max_repair_rounds < 1:
        print("--max-repair-rounds must be at least 1.", file=sys.stderr)
        return 1

    try:
        return run_research(
            instruction=instruction,
            report_template_path=args.report_template,
            model=args.model,
            max_iterations=args.max_iterations,
            max_repair_rounds=args.max_repair_rounds,
            output_path=args.output,
            context_path=args.context_path,
            max_context_chunks=args.max_context_chunks,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
