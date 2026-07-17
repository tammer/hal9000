from __future__ import annotations

from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]
SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent / "service_account.json"


def _extract_text(document: dict) -> str:
    lines: list[str] = []
    for element in document.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        parts: list[str] = []
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun")
            if text_run and "content" in text_run:
                parts.append(text_run["content"])
        if parts:
            lines.append("".join(parts))
    return "".join(lines)


def read_google_doc(document_id: str) -> str:
    """Fetch a Google Doc by ID and return title + body as plain text."""
    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(f"Missing key file: {SERVICE_ACCOUNT_FILE}")

    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES,
    )
    service = build("docs", "v1", credentials=credentials, cache_discovery=False)
    document = service.documents().get(documentId=document_id).execute()

    title = document.get("title", "(untitled)")
    text = _extract_text(document)
    return f"Title: {title}\n---\n{text}"
