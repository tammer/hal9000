from __future__ import annotations

import re
from pathlib import Path


H1_HEADING_RE = re.compile(r"^#\s+(.+)$")
NUMERIC_PREFIX_RE = re.compile(r"^\d+-(.+)\.md$")


def slug_from_filename(filename: str) -> str:
    match = NUMERIC_PREFIX_RE.match(filename)
    if match:
        return match.group(1)
    if filename.endswith(".md"):
        return filename[:-3]
    return filename


def title_from_slug(slug: str) -> str:
    return slug.replace("-", " ").title()


def title_from_instruction(text: str) -> str | None:
    for line in text.splitlines():
        match = H1_HEADING_RE.match(line)
        if match:
            return match.group(1).strip()
    return None


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


def load_template_directory(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"template directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"template path is not a directory: {path}")

    template_files = sorted(path.glob("*.md"))
    if not template_files:
        raise ValueError(f"template directory has no .md files: {path}")

    sections: list[dict[str, str]] = []
    seen_slugs: dict[str, Path] = {}

    for template_path in template_files:
        instruction = template_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"template file is empty: {template_path.name}")

        slug = slug_from_filename(template_path.name)
        if slug in seen_slugs:
            raise ValueError(
                f"duplicate template slug '{slug}' in {template_path.name} "
                f"and {seen_slugs[slug].name}"
            )
        seen_slugs[slug] = template_path

        title = title_from_instruction(instruction) or title_from_slug(slug)
        sections.append(
            {
                "title": title,
                "slug": slug,
                "instruction": instruction,
                "path": str(template_path),
            }
        )

    return sections
