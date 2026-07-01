#!/usr/bin/env python3
import argparse
import html
import os
import re
import string
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
import markdown as markdown_lib

GOOGLE_DRIVE_BASE = Path(
    "/Users/tammerkamel/Library/CloudStorage/GoogleDrive-tammer.kamel@antler.co/My Drive"
)
TEMPLATE_PATH = Path(__file__).parent / "template.md"
STYLES_PATH = Path(__file__).parent / "styles.css"

H1_HEADING_RE = re.compile(r"^#\s+(.+)$")

MAX_CONTENT_CHARS = 100_000
BINARY_SAMPLE_SIZE = 8192
PRINTABLE_THRESHOLD = 0.75

SECTION_SYSTEM_PROMPT = (
    "You are analyzing startup deal documents. Follow the user's instruction "
    "for this section. Return only HTML suitable for embedding inside a <div> "
    "(use <p>, <ul>, <a href=\"...\">, etc.). Do not include <html>, <body>, "
    "or heading tags."
    "You should be concise."
    "If you can't provide the information, just say 'not available'."
    "any html link you create should open in a new tab."
    "never provide more information that you are asked for."
    "do not state opinions of assessments. just facts."
)


HTML_BLOCK_TAG_RE = re.compile(
    r"<\s*(p|table|ul|ol|li|tr|td|th|thead|tbody|a|div|strong|em)\b",
    re.IGNORECASE,
)
ANCHOR_TAG_RE = re.compile(r"<a\s+([^>]+)>", re.IGNORECASE)
ORDERED_LIST_ITEM_RE = re.compile(r"^\s*\d+\.\s")
UNORDERED_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s")
COMPANY_NAME_ROW_RE = re.compile(
    r"<tr>\s*<td>\s*Name\s*</td>\s*<td>(.*?)</td>\s*</tr>",
    re.IGNORECASE | re.DOTALL,
)


def add_target_blank_to_links(html: str) -> str:
    def add_target(match: re.Match[str]) -> str:
        attrs = match.group(1)
        if re.search(r"\btarget\s*=", attrs, re.IGNORECASE):
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


def ensure_html(content: str) -> str:
    text = content.strip()
    if not text:
        return text

    if HTML_BLOCK_TAG_RE.search(text):
        return demote_h1_to_h3(add_target_blank_to_links(text))

    html = markdown_lib.markdown(
        normalize_markdown_lists(text),
        extensions=["tables", "sane_lists", "nl2br"],
    )
    return demote_h1_to_h3(add_target_blank_to_links(html))


def is_likely_binary(data: bytes) -> bool:
    if b"\x00" in data[:BINARY_SAMPLE_SIZE]:
        return True
    if not data:
        return False
    sample = data[:BINARY_SAMPLE_SIZE]
    printable = sum(
        1 for b in sample if chr(b) in string.printable or b in (9, 10, 13)
    )
    return printable / len(sample) < PRINTABLE_THRESHOLD


def extract_docx_text(path: Path) -> str | None:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
    except (OSError, KeyError, UnicodeDecodeError):
        return None

    text = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"</w:p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text or None


def read_file_as_text(path: Path) -> str | None:
    if path.suffix.lower() == ".docx":
        return extract_docx_text(path)

    data = path.read_bytes()
    if is_likely_binary(data):
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def collect_documents(folder: Path) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []

    for entry in sorted(folder.iterdir()):
        if (
            not entry.is_file()
            or entry.name.startswith(".")
            or entry.name.startswith("~$")
        ):
            continue

        text = read_file_as_text(entry)
        if text is None:
            continue

        documents.append((entry.name, text))

    return documents


def build_payload(documents: list[tuple[str, str]]) -> str:
    sections = [f"### {name}\n{content}" for name, content in documents]
    payload = "\n\n".join(sections)

    if len(payload) > MAX_CONTENT_CHARS:
        truncated = payload[:MAX_CONTENT_CHARS]
        payload = (
            truncated
            + "\n\n[Note: content was truncated due to size limits.]"
        )

    return payload


def parse_template_markdown(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return

        instruction = "\n".join(current_lines).strip()
        if not instruction:
            raise ValueError(
                f"section '{current_title}' in template.md must have instruction text"
            )

        sections.append({"title": current_title, "instruction": instruction})
        current_title = None
        current_lines = []

    for line in text.splitlines():
        heading_match = H1_HEADING_RE.match(line)
        if heading_match:
            flush_section()
            current_title = heading_match.group(1).strip()
            continue

        if current_title is not None:
            current_lines.append(line)

    flush_section()

    if not sections:
        raise ValueError(
            "template.md must define at least one section using '# Title' headings"
        )

    return sections


def load_template() -> list[dict[str, str]]:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"template file not found: {TEMPLATE_PATH}")

    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    return parse_template_markdown(text)


def generate_section_content(
    section: dict[str, str],
    documents: list[tuple[str, str]],
    api_key: str,
    model: str,
) -> str:
    client = Groq(api_key=api_key)
    payload = build_payload(documents)
    user_message = f"{section['instruction']}\n\nDocuments:\n{payload}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return ensure_html(response.choices[0].message.content or "")


def load_styles() -> str:
    if not STYLES_PATH.exists():
        raise FileNotFoundError(f"styles file not found: {STYLES_PATH}")
    return STYLES_PATH.read_text(encoding="utf-8")


def extract_company_name(sections_content: list[tuple[str, str]]) -> str | None:
    for title, content in sections_content:
        if title.lower() != "company":
            continue
        match = COMPANY_NAME_ROW_RE.search(content)
        if not match:
            return None
        name = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if not name or name.lower() == "not available":
            return None
        return name
    return None


def write_deal_html(
    folder: Path, sections_content: list[tuple[str, str]]
) -> Path:
    analysis_dir = folder / "analysis"
    if not analysis_dir.exists():
        analysis_dir.mkdir()

    updated_at = datetime.now().astimezone()
    updated_iso = updated_at.isoformat(timespec="seconds")
    updated_display = updated_at.strftime("%B %-d, %Y at %-I:%M %p %Z")
    styles = load_styles()

    section_html = "\n".join(
        f"    <section>\n      <h1>{title}</h1>\n      <div>{content}</div>\n    </section>"
        for title, content in sections_content
    )

    company_name = extract_company_name(sections_content)
    page_title = html.escape(company_name if company_name else "Deal")

    document_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
{styles}
  </style>
</head>
<body>
  <main>
    <section id="last-updated">
      <h1>Last Updated</h1>
      <p>
        <time id="last-updated-time" datetime="{updated_iso}">{updated_display}</time>
        (<span id="last-updated-relative"></span>)
      </p>
    </section>
{section_html}
  </main>
  <script>
    (function () {{
      const el = document.getElementById("last-updated-time");
      const relativeEl = document.getElementById("last-updated-relative");
      if (!el || !relativeEl) return;

      const updatedAt = new Date(el.getAttribute("datetime"));
      const now = new Date();
      const diffMs = now - updatedAt;
      const diffDays = Math.floor(
        (Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) -
          Date.UTC(updatedAt.getFullYear(), updatedAt.getMonth(), updatedAt.getDate())) /
          86400000
      );

      let label;
      if (diffMs < 60000) {{
        label = "just now";
      }} else if (diffDays === 0) {{
        label = "today";
      }} else if (diffDays === 1) {{
        label = "yesterday";
      }} else if (diffDays < 7) {{
        label = diffDays + " days ago";
      }} else if (diffDays < 30) {{
        const weeks = Math.floor(diffDays / 7);
        label = weeks === 1 ? "1 week ago" : weeks + " weeks ago";
      }} else if (diffDays < 365) {{
        const months = Math.floor(diffDays / 30);
        label = months === 1 ? "1 month ago" : months + " months ago";
      }} else {{
        const years = Math.floor(diffDays / 365);
        label = years === 1 ? "1 year ago" : years + " years ago";
      }}

      relativeEl.textContent = label;
    }})();
  </script>
</body>
</html>
"""

    output_path = analysis_dir / "deal.html"
    output_path.write_text(document_html, encoding="utf-8")
    return output_path


def resolve_folder_path(relative_path: str) -> Path:
    folder = (GOOGLE_DRIVE_BASE / relative_path.lstrip("/")).resolve()
    base = GOOGLE_DRIVE_BASE.resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes Google Drive root: {relative_path}")

    return folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deal analysis HTML from folder documents using an LLM."
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the folder to analyze",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()

    args = parse_args()

    try:
        folder = resolve_folder_path(args.relative_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not folder.exists():
        print(f"Error: path does not exist: {folder}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: path is not a directory: {folder}", file=sys.stderr)
        return 1

    try:
        template_sections = load_template()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    documents = collect_documents(folder)
    if not documents:
        print(
            f"Error: no readable top-level files found in {folder}",
            file=sys.stderr,
        )
        return 1

    sections_content: list[tuple[str, str]] = []
    for section in template_sections:
        try:
            content = generate_section_content(section, documents, api_key, model)
        except Exception as exc:
            print(
                f"Error: Groq API call failed for section '{section['title']}': {exc}",
                file=sys.stderr,
            )
            return 1
        sections_content.append((section["title"], content))

    try:
        output_path = write_deal_html(folder, sections_content)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: failed to write deal.html: {exc}", file=sys.stderr)
        return 1

    print(f"Deal analysis written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
