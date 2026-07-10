from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
DEFAULT_API_BASE = "https://api.meetgeek.ai"


@dataclass(frozen=True)
class MeetingSummary:
    meeting_id: str
    timestamp_start_utc: str
    timestamp_end_utc: str


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    title: str
    host_email: str
    participant_emails: tuple[str, ...]
    timestamp_start_utc: str
    timestamp_end_utc: str
    join_link: str
    timezone: str = ""
    source: str = ""


@dataclass(frozen=True)
class Sentence:
    id: int
    speaker: str
    timestamp: str
    transcript: str


class MeetGeekError(RuntimeError):
    pass


def api_base() -> str:
    return os.getenv("MEETGEEK_API_BASE", DEFAULT_API_BASE).rstrip("/")


def api_key() -> str:
    key = os.getenv("MEETGEEK_API_KEY")
    if not key:
        raise MeetGeekError("MEETGEEK_API_KEY is not set")
    return key


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers=_auth_headers())
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise MeetGeekError(f"MeetGeek API error {exc.code} for {url}: {detail}") from exc

    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise MeetGeekError(f"Unexpected MeetGeek response for {url}")
    return payload


def _build_url(path: str, query: dict[str, str | int] | None = None) -> str:
    url = f"{api_base()}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    return url


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def list_recent_meetings(days: int = 4) -> list[MeetingSummary]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    meetings: list[MeetingSummary] = []
    cursor: str | None = None

    while True:
        query: dict[str, str | int] = {"limit": 500}
        if cursor:
            query["cursor"] = cursor

        payload = _fetch_json(_build_url("/v1/meetings", query))
        page_meetings = payload.get("meetings", [])
        if not isinstance(page_meetings, list):
            raise MeetGeekError("MeetGeek meetings response missing meetings list")

        if not page_meetings:
            break

        page_has_recent = False
        for item in page_meetings:
            if not isinstance(item, dict):
                continue
            meeting_id = str(item.get("meeting_id", "")).strip()
            start = str(item.get("timestamp_start_utc", "")).strip()
            end = str(item.get("timestamp_end_utc", "")).strip()
            if not meeting_id or not end:
                continue

            end_dt = _parse_iso_datetime(end)
            if end_dt >= cutoff:
                meetings.append(
                    MeetingSummary(
                        meeting_id=meeting_id,
                        timestamp_start_utc=start,
                        timestamp_end_utc=end,
                    )
                )
                page_has_recent = True

        pagination = payload.get("pagination", {})
        next_cursor = ""
        if isinstance(pagination, dict):
            next_cursor = str(pagination.get("next_cursor", "")).strip()

        if not page_has_recent or not next_cursor:
            break
        cursor = next_cursor

    return meetings


def list_team_meetings(team_id: str, cutoff: datetime) -> list[MeetingSummary]:
    meetings: list[MeetingSummary] = []
    cursor: str | None = None

    while True:
        query: dict[str, str | int] = {"limit": 500}
        if cursor:
            query["cursor"] = cursor

        payload = _fetch_json(
            _build_url(f"/v1/teams/{team_id}/meetings", query)
        )
        page_meetings = payload.get("meetings", [])
        if not isinstance(page_meetings, list):
            raise MeetGeekError("MeetGeek team meetings response missing meetings list")

        if not page_meetings:
            break

        page_has_recent = False
        for item in page_meetings:
            if not isinstance(item, dict):
                continue
            meeting_id = str(item.get("meeting_id", "")).strip()
            start = str(item.get("timestamp_start_utc", "")).strip()
            end = str(item.get("timestamp_end_utc", "")).strip()
            if not meeting_id or not start:
                continue

            start_dt = _parse_iso_datetime(start)
            if start_dt >= cutoff:
                meetings.append(
                    MeetingSummary(
                        meeting_id=meeting_id,
                        timestamp_start_utc=start,
                        timestamp_end_utc=end,
                    )
                )
                page_has_recent = True

        pagination = payload.get("pagination", {})
        next_cursor = ""
        if isinstance(pagination, dict):
            next_cursor = str(pagination.get("next_cursor", "")).strip()

        if not page_has_recent or not next_cursor:
            break
        cursor = next_cursor

    return meetings


def get_meeting(meeting_id: str) -> Meeting:
    payload = _fetch_json(_build_url(f"/v1/meetings/{meeting_id}"))
    title = str(payload.get("title", "")).strip() or "Untitled Meeting"
    host_email = str(payload.get("host_email", "")).strip()
    participant_emails_raw = payload.get("participant_emails", [])
    participant_emails: tuple[str, ...]
    if isinstance(participant_emails_raw, list):
        participant_emails = tuple(
            str(email).strip()
            for email in participant_emails_raw
            if str(email).strip()
        )
    else:
        participant_emails = ()

    return Meeting(
        meeting_id=str(payload.get("meeting_id", meeting_id)).strip() or meeting_id,
        title=title,
        host_email=host_email,
        participant_emails=participant_emails,
        timestamp_start_utc=str(payload.get("timestamp_start_utc", "")).strip(),
        timestamp_end_utc=str(payload.get("timestamp_end_utc", "")).strip(),
        join_link=str(payload.get("join_link", "")).strip(),
        timezone=str(payload.get("timezone", "")).strip(),
        source=str(payload.get("source", "")).strip(),
    )


def get_transcript(meeting_id: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    cursor: str | None = None

    while True:
        query: dict[str, str | int] = {"limit": 500}
        if cursor:
            query["cursor"] = cursor

        payload = _fetch_json(
            _build_url(f"/v1/meetings/{meeting_id}/transcript", query)
        )
        page_sentences = payload.get("sentences", [])
        if not isinstance(page_sentences, list):
            raise MeetGeekError(
                f"MeetGeek transcript response missing sentences for {meeting_id}"
            )

        for item in page_sentences:
            if not isinstance(item, dict):
                continue
            transcript = str(item.get("transcript", "")).strip()
            if not transcript:
                continue
            sentences.append(
                Sentence(
                    id=int(item.get("id", 0) or 0),
                    speaker=str(item.get("speaker", "")).strip() or "Unknown",
                    timestamp=str(item.get("timestamp", "")).strip(),
                    transcript=transcript,
                )
            )

        pagination = payload.get("pagination", {})
        next_cursor = ""
        if isinstance(pagination, dict):
            next_cursor = str(pagination.get("next_cursor", "")).strip()

        if not next_cursor:
            break
        cursor = next_cursor

    return sentences
