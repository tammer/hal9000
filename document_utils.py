from __future__ import annotations

import re
import string
from pathlib import Path


BINARY_SAMPLE_SIZE = 8192
PRINTABLE_THRESHOLD = 0.75


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


def collect_documents(path: Path, recursive: bool = False) -> list[tuple[Path, str]]:
    candidates: list[Path]
    if path.is_file():
        candidates = [path]
    else:
        iterator = path.rglob("*") if recursive else path.iterdir()
        candidates = sorted(entry for entry in iterator if entry.is_file())

    documents: list[tuple[Path, str]] = []
    for entry in candidates:
        if entry.name.startswith(".") or entry.name.startswith("~$"):
            continue

        text = read_file_as_text(entry)
        if text is None:
            continue

        documents.append((entry, text))

    return documents
