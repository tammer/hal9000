from __future__ import annotations

import re
import string
from collections.abc import Collection
from pathlib import Path


BINARY_SAMPLE_SIZE = 8192
PRINTABLE_THRESHOLD = 0.75

# Directory names skipped during recursive document discovery.
FORBIDDEN_DIR_NAMES = frozenset({"ai-generated"})


def is_likely_binary(data: bytes) -> bool:
    if b"\x00" in data[:BINARY_SAMPLE_SIZE]:
        return True
    if not data:
        return False

    sample = data[:BINARY_SAMPLE_SIZE]
    printable = sum(
        1 for byte in sample if chr(byte) in string.printable or byte in (9, 10, 13)
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


def extract_pdf_text(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ModuleNotFoundError:
        return None

    try:
        reader = PdfReader(path)
        parts = [
            text
            for page in reader.pages
            if (text := page.extract_text()) and text.strip()
        ]
    except (OSError, PdfReadError):
        return None

    text = "\n\n".join(parts).strip()
    return text or None


def read_file_as_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)

    data = path.read_bytes()
    if is_likely_binary(data):
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def _is_skipped_file(path: Path) -> bool:
    return path.name.startswith(".") or path.name.startswith("~$")


def list_candidate_files(
    path: Path,
    *,
    recursive: bool = False,
    exclude_dirs: Collection[str] | None = None,
) -> list[Path]:
    """Return readable-looking source file paths under ``path``.

    When ``recursive`` is True, descend into subdirectories, pruning any
    directory whose name is in ``exclude_dirs`` (defaults to none).
    """
    excluded = frozenset(exclude_dirs or ())

    if path.is_file():
        return [] if _is_skipped_file(path) else [path]

    if not path.is_dir():
        return []

    candidates: list[Path] = []

    if not recursive:
        candidates = sorted(
            entry
            for entry in path.iterdir()
            if entry.is_file() and not _is_skipped_file(entry)
        )
        return candidates

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name in excluded:
                    continue
                walk(entry)
            elif entry.is_file() and not _is_skipped_file(entry):
                candidates.append(entry)

    walk(path)
    return candidates


def collect_documents(
    path: Path,
    recursive: bool = False,
    exclude_dirs: Collection[str] | None = None,
) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for entry in list_candidate_files(
        path, recursive=recursive, exclude_dirs=exclude_dirs
    ):
        text = read_file_as_text(entry)
        if text is None:
            continue
        documents.append((entry, text))

    return documents
