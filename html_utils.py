from __future__ import annotations

import re
from pathlib import Path

import markdown as markdown_lib

STYLES_PATH = Path(__file__).parent / "styles.css"

ANCHOR_TAG_RE = re.compile(r"<a\s+([^>]+)>", re.IGNORECASE)
ORDERED_LIST_ITEM_RE = re.compile(r"^\s*\d+\.\s")
UNORDERED_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s")
MARKDOWN_FENCE_RE = re.compile(
    r"^```(?:markdown)?\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def strip_markdown_fences(text: str) -> str:
    match = MARKDOWN_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


HREF_ATTR_RE = re.compile(r"""href\s*=\s*(['"])(.*?)\1""", re.IGNORECASE)


def is_external_href(href: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*:", href, re.IGNORECASE))


def add_target_blank_to_links(html: str) -> str:
    def add_target(match: re.Match[str]) -> str:
        attrs = match.group(1)
        if re.search(r"\btarget\s*=", attrs, re.IGNORECASE):
            return match.group(0)

        href_match = HREF_ATTR_RE.search(attrs)
        if href_match and not is_external_href(href_match.group(2)):
            return match.group(0)

        return f'<a {attrs.rstrip()} target="_blank">'

    return ANCHOR_TAG_RE.sub(add_target, html)


def demote_h1_to_h3(html_content: str) -> str:
    html_content = re.sub(r"<\s*h1\b", "<h3", html_content, flags=re.IGNORECASE)
    return re.sub(r"<\s*/\s*h1\s*>", "</h3>", html_content, flags=re.IGNORECASE)


def normalize_markdown_lists(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []
    in_list = False

    for line in lines:
        is_list_item = bool(
            ORDERED_LIST_ITEM_RE.match(line) or UNORDERED_LIST_ITEM_RE.match(line)
        )
        if is_list_item and not in_list and normalized and normalized[-1].strip():
            normalized.append("")
        in_list = is_list_item
        normalized.append(line)

    return "\n".join(normalized)


def markdown_to_html(content: str, *, demote_h1: bool = True) -> str:
    text = strip_markdown_fences(content)
    if not text:
        return text

    html_output = markdown_lib.markdown(
        normalize_markdown_lists(text),
        extensions=["tables", "sane_lists", "nl2br"],
    )
    html_output = add_target_blank_to_links(html_output)
    if demote_h1:
        html_output = demote_h1_to_h3(html_output)
    return html_output


def load_styles() -> str:
    if not STYLES_PATH.exists():
        raise FileNotFoundError(f"styles file not found: {STYLES_PATH}")
    return STYLES_PATH.read_text(encoding="utf-8")
