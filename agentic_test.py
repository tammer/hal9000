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

if TYPE_CHECKING:
    from groq import Groq

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_CONTEXT_CHUNKS = 4
RESULTS_PER_SEARCH = 8
PAGES_PER_SEARCH = 4
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

RESEARCH_BRIEF_SYSTEM_PROMPT = """You turn a user's research instruction into a concrete brief for a research agent.

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
- Infer a sensible presentation style when the user did not specify one.
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
- Avoid repeating the same lookup or search unless you are intentionally narrowing it.
- Stop only when the evidence appears sufficient for a careful report.

Return exactly one of these shapes:
{"action":"local_search","reasoning":"short explanation","local_query":"query for local retrieval","focus":"what this local lookup should clarify"}
{"action":"web_search","reasoning":"short explanation","search_query":"query to run on the web","focus":"what this web search should clarify"}
{"action":"report","reasoning":"short explanation","confidence":0.0,"citations":["S1","S2"]}
"""

REPORT_SYSTEM_PROMPT = """You write high-quality research reports from gathered evidence.

Return valid JSON only with this exact shape:
{"report_markdown":"full markdown report","citations":["S1","S2"]}

Rules:
- Follow the user's requested presentation style.
- If the user did not specify a structure, choose a clear professional structure.
- Use only the provided evidence.
- Distinguish between internal document evidence and external web validation.
- If the task is claim validation, state whether the claim is supported, contradicted, or still unresolved.
- Include uncertainty, gaps, or conflicting evidence when relevant.
- Cite sources inline using source ids like [S1] where claims rely on evidence.
- Do not invent facts.
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


def create_research_brief(instruction: str, model: str) -> ResearchBrief:
    payload = llm_json(
        RESEARCH_BRIEF_SYSTEM_PROMPT,
        f"Research instruction:\n{instruction}",
        model,
    )
    topic = normalize_whitespace(str(payload.get("topic", ""))) or "Research task"
    goal = normalize_whitespace(str(payload.get("goal", ""))) or instruction
    presentation = (
        normalize_whitespace(str(payload.get("presentation", "")))
        or "Write a polished markdown report."
    )
    must_cover = sanitize_list(payload.get("must_cover"))
    search_angles = sanitize_list(payload.get("search_angles"))
    quality_checks = sanitize_list(payload.get("quality_checks"))

    return ResearchBrief(
        topic=topic,
        goal=goal,
        presentation=presentation,
        must_cover=must_cover,
        search_angles=search_angles,
        quality_checks=quality_checks,
    )


def display_path(path: Path, root: Path | None) -> str:
    if root and root.is_dir():
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def load_context_documents(context_path: str | None) -> tuple[Path | None, list[DocumentRecord]]:
    if not context_path:
        return None, []

    path = Path(context_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"context path does not exist: {path}")

    raw_documents = collect_documents(path, recursive=path.is_dir())
    documents = [
        DocumentRecord(path=document_path, name=document_path.name, text=text)
        for document_path, text in raw_documents
        if text.strip()
    ]
    return path, documents


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


def score_chunk(query: str, chunk: DocumentChunk) -> float:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return 0.0

    chunk_terms = Counter(tokenize(f"{chunk.name} {chunk.text}"))
    overlap = set(query_terms) & set(chunk_terms)
    if not overlap:
        return 0.0

    frequency_score = sum(min(query_terms[token], chunk_terms[token]) for token in overlap)
    unique_score = len(overlap) * 3
    filename_bonus = sum(1 for token in overlap if token in tokenize(chunk.name)) * 2
    phrase_bonus = 0
    lowered_query = normalize_whitespace(query).lower()
    if lowered_query and lowered_query in f"{chunk.name} {chunk.text}".lower():
        phrase_bonus = 5

    return float(frequency_score + unique_score + filename_bonus + phrase_bonus)


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


def build_planner_prompt(
    instruction: str,
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


def build_report_prompt(
    instruction: str,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
) -> str:
    return f"""User instruction:
{instruction}

Research brief:
{build_research_brief_text(brief)}

Local document inventory:
{document_inventory}

Evidence:
{format_evidence(evidence)}
"""


def choose_next_action(
    instruction: str,
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


def synthesize_report(
    instruction: str,
    brief: ResearchBrief,
    document_inventory: str,
    evidence: list[EvidenceSource],
    model: str,
) -> dict[str, Any]:
    return llm_json(
        REPORT_SYSTEM_PROMPT,
        build_report_prompt(
            instruction=instruction,
            brief=brief,
            document_inventory=document_inventory,
            evidence=evidence,
        ),
        model,
    )


def prompt_for_instruction() -> str:
    print("What would you like researched?")
    print("Include any guidance on how you want the research presented.")
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
    documents: list[DocumentRecord],
    chunks: list[DocumentChunk],
) -> None:
    print("\nResearch brief:")
    print(f"- Topic: {brief.topic}")
    print(f"- Goal: {brief.goal}")
    print(f"- Presentation: {brief.presentation}")
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
    brief: ResearchBrief,
    chunks: list[DocumentChunk],
    evidence: list[EvidenceSource],
    seen_local_keys: set[str],
    source_counter: int,
    max_context_chunks: int,
    root: Path | None,
) -> int:
    added = 0
    for query in [instruction, brief.goal, *brief.must_cover]:
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
    model: str,
    max_iterations: int,
    output_path: str | None,
    context_path: str | None,
    max_context_chunks: int,
) -> int:
    try:
        context_root, documents = load_context_documents(context_path)
    except Exception as exc:
        print(f"Context loading error: {exc}", file=sys.stderr)
        return 1

    chunks = build_document_chunks(documents)
    document_inventory = build_document_inventory_text(documents, context_root)

    try:
        brief = create_research_brief(instruction, model)
    except Exception as exc:
        print(f"Research brief error: {exc}", file=sys.stderr)
        return 1

    print_brief_summary(brief, documents, chunks)

    local_search_history: list[str] = []
    web_search_history: list[str] = []
    evidence: list[EvidenceSource] = []
    seen_local_keys: set[str] = set()
    evidence_by_url: set[str] = set()
    source_counter = 1
    local_preview_query = instruction

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
        report_payload = synthesize_report(
            instruction=instruction,
            brief=brief,
            document_inventory=document_inventory,
            evidence=evidence,
            model=model,
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
        description="Run a general-purpose research loop using local document context, Brave search, and Groq synthesis."
    )
    parser.add_argument(
        "--instruction",
        "--question",
        dest="instruction",
        help="Research instruction, including how the result should be presented.",
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

    try:
        return run_research(
            instruction=instruction,
            model=args.model,
            max_iterations=args.max_iterations,
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
