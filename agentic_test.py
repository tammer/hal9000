#!/usr/bin/env python3
import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

if TYPE_CHECKING:
    from groq import Groq

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_ITERATIONS = 5
RESULTS_PER_SEARCH = 5
PAGES_PER_SEARCH = 3
MAX_PAGE_CHARS = 3_000
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

PLANNER_SYSTEM_PROMPT = """You are running a web research loop.

Your job is to decide either:
1. one next web search query, or
2. that there is enough evidence to answer the user's question now.

Rules:
- Return valid JSON only.
- Prefer precise, high-signal searches.
- If the question is time-sensitive, prefer recent or authoritative sources.
- Do not claim certainty unless the evidence supports it.
- If the evidence is conflicting, search again unless the answer can still be stated carefully.

Return exactly one of these shapes:
{"action":"search","reasoning":"short explanation","search_query":"query to run next"}
{"action":"answer","reasoning":"short explanation","confidence":0.0,"answer":"final answer in plain text","citations":["S1","S2"]}
"""

ANSWER_SYSTEM_PROMPT = """You answer the user's question using gathered web evidence.

Rules:
- Return valid JSON only.
- Keep the answer clear and direct.
- Use only the provided evidence.
- If the evidence is incomplete or conflicting, say so briefly.
- Cite source ids that materially support the answer.

Return this exact shape:
{"answer":"plain-text answer","confidence":0.0,"citations":["S1","S2"]}
"""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class EvidenceSource:
    source_id: str
    query: str
    title: str
    url: str
    excerpt: str


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

        canonical = url
        if canonical in seen_urls:
            continue
        if not title or not canonical or not is_public_web_url(canonical):
            continue
        seen_urls.add(canonical)
        deduped.append(
            SearchResult(
                title=title,
                url=canonical,
                snippet=" ".join(snippets),
            )
        )

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


def parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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


def build_planner_prompt(
    question: str,
    search_history: list[str],
    evidence: list[EvidenceSource],
    iterations_left: int,
) -> str:
    history_lines = (
        "\n".join(f"- {query}" for query in search_history) if search_history else "- none yet"
    )

    evidence_lines = []
    for source in evidence:
        evidence_lines.append(
            "\n".join(
                [
                    f"{source.source_id}",
                    f"Query: {source.query}",
                    f"Title: {source.title}",
                    f"URL: {source.url}",
                    f"Excerpt: {source.excerpt}",
                ]
            )
        )
    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No evidence gathered yet."

    return f"""Question: {question}
Iterations left: {iterations_left}

Previous searches:
{history_lines}

Evidence:
{evidence_block}
"""


def build_answer_prompt(question: str, evidence: list[EvidenceSource]) -> str:
    evidence_lines = []
    for source in evidence:
        evidence_lines.append(
            "\n".join(
                [
                    f"{source.source_id}",
                    f"Title: {source.title}",
                    f"URL: {source.url}",
                    f"Excerpt: {source.excerpt}",
                ]
            )
        )
    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No evidence gathered."

    return f"""Question: {question}

Evidence:
{evidence_block}
"""


def choose_next_action(
    question: str,
    search_history: list[str],
    evidence: list[EvidenceSource],
    iterations_left: int,
    model: str,
) -> dict[str, Any]:
    return llm_json(
        PLANNER_SYSTEM_PROMPT,
        build_planner_prompt(question, search_history, evidence, iterations_left),
        model,
    )


def synthesize_answer(question: str, evidence: list[EvidenceSource], model: str) -> dict[str, Any]:
    return llm_json(
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt(question, evidence),
        model,
    )


def prompt_for_question() -> str:
    print("What's your question?")
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
        print(f"- {citation_id}: {source.title}")
        print(f"  {source.url}")


def run_agentic_search(question: str, model: str, max_iterations: int) -> int:
    search_history: list[str] = []
    evidence: list[EvidenceSource] = []
    evidence_by_url: set[str] = set()
    source_counter = 1

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_iterations}]")

        try:
            decision = choose_next_action(
                question=question,
                search_history=search_history,
                evidence=evidence,
                iterations_left=max_iterations - iteration + 1,
                model=model,
            )
        except Exception as exc:
            print(f"Planner error: {exc}", file=sys.stderr)
            return 1

        action = decision.get("action")
        if action == "answer":
            citations = safe_citations(decision, evidence)
            print("\nAnswer:")
            print(decision.get("answer", "").strip() or "No answer was produced.")
            print_sources(citations, evidence)
            return 0

        if action != "search":
            print(f"Planner returned unexpected action: {action!r}", file=sys.stderr)
            return 1

        query = str(decision.get("search_query", "")).strip()
        if not query:
            print("Planner requested a search but did not provide a query.", file=sys.stderr)
            return 1

        print(f"Searching Brave for: {query}")
        search_history.append(query)

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
                    query=query,
                    title=result.title,
                    url=result.url,
                    excerpt=excerpt,
                )
            )
            evidence_by_url.add(result.url)
            source_counter += 1

    print("\nReached the search limit. Generating the best answer from the gathered evidence.")
    try:
        answer_payload = synthesize_answer(question, evidence, model)
    except Exception as exc:
        print(f"Final answer synthesis error: {exc}", file=sys.stderr)
        return 1

    citations = safe_citations(answer_payload, evidence)
    print("\nAnswer:")
    print(answer_payload.get("answer", "").strip() or "No answer was produced.")
    print_sources(citations, evidence)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question and let an agentic loop search the web with Brave until it can answer."
    )
    parser.add_argument(
        "--question",
        help="Optional question to answer. If omitted, the program prompts interactively.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        help="Groq model to use for planning and answer synthesis.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of search-planning iterations (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    question = (args.question or "").strip()
    if not question:
        try:
            question = prompt_for_question()
        except EOFError:
            print("No question provided.", file=sys.stderr)
            return 1

    if not question:
        print("Question cannot be empty.", file=sys.stderr)
        return 1

    if args.max_iterations < 1:
        print("--max-iterations must be at least 1.", file=sys.stderr)
        return 1

    try:
        return run_agentic_search(
            question=question,
            model=args.model,
            max_iterations=args.max_iterations,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
