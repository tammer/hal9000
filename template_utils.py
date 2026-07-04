from __future__ import annotations

import re
from pathlib import Path


H1_HEADING_RE = re.compile(r"^#\s+(.+)$")


def parse_markdown_template(
    text: str,
    *,
    source_name: str = "template",
) -> list[dict[str, str]]:
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
                f"section '{current_title}' in {source_name} must have instruction text"
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
            f"{source_name} must define at least one section using '# Title' headings"
        )

    return sections


def load_markdown_template(path: Path, *, source_name: str = "template") -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{source_name} file not found: {path}")

    text = path.read_text(encoding="utf-8")
    return parse_markdown_template(text, source_name=source_name)
