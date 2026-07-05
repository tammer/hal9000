#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

# Pick a model by index: SEARCH_PLANNER_MODEL = MODELS[0], etc.
MODELS = [
    "claude-opus-4-8",       # 0
    "claude-opus-4-7",       # 1
    "claude-opus-4-6",       # 2
    "claude-opus-4-5",       # 3
    "claude-sonnet-5",       # 4
    "claude-sonnet-4-6",     # 5
    "claude-sonnet-4-5",     # 6
    "claude-haiku-4-5",      # 7
    "claude-fable-5",        # 8
    "llama-3.3-70b-versatile",  # 9
]

SEARCH_PLANNER_MODEL = MODELS[6]
FACT_EXTRACTOR_MODEL = MODELS[6]

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MAX_SEARCHES = 4
MIN_FACTS_BEFORE_RETRY = 2
RESULTS_PER_SEARCH = 8
PAGES_PER_SEARCH = 4
MAX_PAGE_CHARS = 4_000
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

SEARCH_PLANNER_SYSTEM_PROMPT = """You plan web searches to find quantitative factual information.

Return valid JSON only with this exact shape:
{"queries": ["search query 1", "search query 2"]}

Rules:
- The user prompt asks for quantitative facts (counts, amounts, percentages, etc.).
- Include queries that directly answer the question.
- Include queries for quantitative information that could help answer the question.
- Keep queries concise and web-search friendly.
- Return 2-4 queries.
"""

FACT_EXTRACTOR_SYSTEM_PROMPT = """You extract quantitative factual claims from web evidence.

Return valid JSON only with this exact shape:
{
  "facts": [
    {"name": "descriptive metric name", "value": "321,033", "source": "https://example.com/path"}
  ],
  "needs_more_research": false,
  "follow_up_queries": ["optional targeted search query"]
}

Rules:
- Only include quantitative facts: counts, dollar amounts, percentages, numeric dates, ranges.
- Each fact must cite exactly one source URL that appears in the provided evidence.
- Prefer facts that answer the user's question; also include closely related quantitative facts.
- If multiple sources report different values for the same metric, include separate entries.
- Do not invent facts not supported by the evidence.
- value must always be a string (preserve formatting like "321,033" or "$12.5B").
- Set needs_more_research=true only when the evidence is clearly insufficient.
- follow_up_queries should be specific gap-filling searches (0-2 items).
"""

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class EvidenceItem:
    title: str
    url: str
    query: str
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


def log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def is_public_web_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    return "search.brave.com" not in parsed.netloc


def normalize_source_url(url: str) -> str:
    return str(url).split("#", 1)[0].rstrip("/")


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
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")
    return api_key


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
        url = normalize_source_url(str(result.get("url", "")))
        snippets: list[str] = []
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


def is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    from groq import Groq

    return Groq(api_key=api_key)


def anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


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
    if is_anthropic_model(model):
        client = anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text
    else:
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


def sanitize_queries(value: Any, *, max_count: int) -> list[str]:
    if not isinstance(value, list):
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for item in value:
        query = normalize_whitespace(str(item))
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_count:
            break
    return queries


def plan_searches(prompt: str, model: str, max_searches: int) -> list[str]:
    payload = llm_json(
        SEARCH_PLANNER_SYSTEM_PROMPT,
        f"User prompt:\n{prompt}",
        model,
    )
    return sanitize_queries(payload.get("queries"), max_count=max_searches)


def build_evidence_text(evidence: list[EvidenceItem]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[E{index}] {item.title}",
                    f"URL: {item.url}",
                    f"Search query: {item.query}",
                    item.excerpt,
                ]
            )
        )
    return "\n\n".join(blocks)


def gather_evidence_for_queries(
    queries: list[str],
    evidence: list[EvidenceItem],
    seen_urls: set[str],
    *,
    verbose: bool,
) -> None:
    for query in queries:
        log(f"Searching: {query}", verbose=verbose)
        try:
            results = search_brave(query)
        except Exception as exc:
            log(f"  Search failed: {exc}", verbose=verbose)
            continue

        if not results:
            log("  No search results found.", verbose=verbose)
            continue

        for index, result in enumerate(results, start=1):
            log(f"  {index}. {shorten(result.title, width=90, placeholder='...')}", verbose=verbose)

        for result in results[:PAGES_PER_SEARCH]:
            if result.url in seen_urls:
                continue

            log(f"  Reading: {shorten(result.title, width=90, placeholder='...')}", verbose=verbose)
            page_excerpt = read_result_page(result)
            excerpt = combine_result_evidence(result, page_excerpt)
            evidence.append(
                EvidenceItem(
                    title=result.title,
                    url=result.url,
                    query=query,
                    excerpt=excerpt,
                )
            )
            seen_urls.add(result.url)


def extract_facts(
    prompt: str,
    evidence: list[EvidenceItem],
    model: str,
) -> tuple[list[dict[str, str]], bool, list[str]]:
    if not evidence:
        return [], True, []

    payload = llm_json(
        FACT_EXTRACTOR_SYSTEM_PROMPT,
        "\n\n".join(
            [
                f"User prompt:\n{prompt}",
                "Evidence:",
                build_evidence_text(evidence),
            ]
        ),
        model,
    )

    raw_facts = payload.get("facts", [])
    facts: list[dict[str, str]] = []
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            name = normalize_whitespace(str(item.get("name", "")))
            value = normalize_whitespace(str(item.get("value", "")))
            source = normalize_source_url(str(item.get("source", "")))
            if name and value and source:
                facts.append({"name": name, "value": value, "source": source})

    needs_more = bool(payload.get("needs_more_research"))
    follow_up = sanitize_queries(payload.get("follow_up_queries"), max_count=2)
    return facts, needs_more, follow_up


def evidence_url_set(evidence: list[EvidenceItem]) -> set[str]:
    urls: set[str] = set()
    for item in evidence:
        urls.add(item.url)
        urls.add(normalize_source_url(item.url))
    return urls


def validate_facts(
    facts: list[dict[str, str]],
    allowed_sources: set[str],
) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for fact in facts:
        name = normalize_whitespace(fact.get("name", ""))
        value = normalize_whitespace(fact.get("value", ""))
        source = normalize_source_url(fact.get("source", ""))

        if not name or not value or not source:
            continue
        if not is_public_web_url(source):
            continue
        if source not in allowed_sources:
            continue

        key = (name.lower(), value, source)
        if key in seen:
            continue
        seen.add(key)
        validated.append({"name": name, "value": value, "source": source})

    return validated


def get_facts(
    prompt: str,
    planner_model: str,
    extractor_model: str,
    max_searches: int,
    *,
    verbose: bool,
) -> list[dict[str, str]]:
    evidence: list[EvidenceItem] = []
    seen_urls: set[str] = set()

    log(f"Planning searches (model: {planner_model})...", verbose=verbose)
    queries = plan_searches(prompt, planner_model, max_searches)
    if not queries:
        raise RuntimeError("Search planner returned no queries")

    gather_evidence_for_queries(queries, evidence, seen_urls, verbose=verbose)

    log(f"Extracting facts (model: {extractor_model})...", verbose=verbose)
    facts, needs_more, follow_up = extract_facts(prompt, evidence, extractor_model)
    validated = validate_facts(facts, evidence_url_set(evidence))

    should_retry = len(validated) < MIN_FACTS_BEFORE_RETRY or (
        needs_more and follow_up
    )
    if should_retry and follow_up:
        log("Running follow-up searches...", verbose=verbose)
        gather_evidence_for_queries(follow_up, evidence, seen_urls, verbose=verbose)
        facts, _, _ = extract_facts(prompt, evidence, extractor_model)
        validated = validate_facts(facts, evidence_url_set(evidence))
    elif should_retry and not follow_up:
        log("Too few facts found; no follow-up queries suggested.", verbose=verbose)

    log(f"Extracted {len(validated)} facts", verbose=verbose)
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find quantitative facts for a prompt using web search and an LLM."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt requesting quantitative factual information",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_flag",
        help="Prompt requesting quantitative factual information",
    )
    parser.add_argument(
        "--max-searches",
        type=int,
        default=DEFAULT_MAX_SEARCHES,
        help=f"Maximum number of search queries to run (default: {DEFAULT_MAX_SEARCHES})",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log progress to stderr (default: on when stderr is a TTY)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    prompt = args.prompt_flag or args.prompt
    if not prompt or not prompt.strip():
        print("Error: a prompt is required", file=sys.stderr)
        return 1

    verbose = args.verbose if args.verbose is not None else sys.stderr.isatty()
    max_searches = max(1, args.max_searches)

    try:
        facts = get_facts(
            prompt.strip(),
            SEARCH_PLANNER_MODEL,
            FACT_EXTRACTOR_MODEL,
            max_searches,
            verbose=verbose,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(facts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
