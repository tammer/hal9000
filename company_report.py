#!/usr/bin/env python3
"""Shared JSON-metadata → Claude summary.md report runner."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from claude_common import (
    MODEL,
    build_payload,
    load_system_prompt,
    print_usage_report,
    require_api_key,
    run_claude,
)

SUMMARY_NAME = "summary.md"


@dataclass(frozen=True)
class JsonReportConfig:
    metadata_json_name: str
    prompt_path: Path
    resolve_folder: Callable[[str], Path | None]
    refresh_metadata: Callable[[str], Path]
    metadata_label: str


def metadata_json_path(folder: Path, json_name: str) -> Path:
    return folder / "ai-generated" / json_name


def existing_summary_if_current(folder: Path, metadata_path: Path) -> Path | None:
    summary_path = folder / "ai-generated" / SUMMARY_NAME
    if not summary_path.is_file() or not metadata_path.is_file():
        return None

    if metadata_path.stat().st_mtime >= summary_path.stat().st_mtime:
        return None

    return summary_path


def write_summary(folder: Path, content: str) -> Path:
    ai_generated_dir = folder / "ai-generated"
    ai_generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_generated_dir / SUMMARY_NAME
    output_path.write_text(content, encoding="utf-8")
    return output_path


def strip_claude_stats(entries: list) -> list:
    """Return entries with claude_stats removed from each object."""
    cleaned: list = []
    for item in entries:
        if isinstance(item, dict):
            cleaned.append(
                {key: value for key, value in item.items() if key != "claude_stats"}
            )
        else:
            cleaned.append(item)
    return cleaned


def load_metadata_json(metadata_path: Path) -> tuple[str, list] | None:
    """Read metadata JSON, strip claude_stats, return payload text and entries."""
    if not metadata_path.is_file():
        print(f"Error: {metadata_path.name} not found at {metadata_path}", file=sys.stderr)
        return None

    try:
        raw = metadata_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to read {metadata_path}: {exc}", file=sys.stderr)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {metadata_path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, list):
        print(
            f"Error: {metadata_path} is not a JSON list",
            file=sys.stderr,
        )
        return None

    entries = strip_claude_stats(data)
    text = json.dumps(entries, indent=2, ensure_ascii=False)
    return text, entries


def print_entry_list(folder: Path, metadata_path: Path, entries: list) -> None:
    rel = metadata_path.relative_to(folder)
    print("Documents (1):")
    print(f"  {rel} ({len(entries)} entries)")
    for item in entries:
        if isinstance(item, dict):
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                print(f"    - {filename}")


def run_json_report(
    config: JsonReportConfig,
    relative_path: str,
    *,
    dry_run: bool,
) -> int:
    folder = config.resolve_folder(relative_path)
    if folder is None:
        return 1

    if not dry_run:
        try:
            config.refresh_metadata(relative_path)
        except (ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(
                f"Error: metadata refresh failed: {exc}",
                file=sys.stderr,
            )
            return 1

    metadata_path = metadata_json_path(folder, config.metadata_json_name)

    if not dry_run:
        current_summary = existing_summary_if_current(folder, metadata_path)
        if current_summary is not None:
            print(
                f"No new {config.metadata_label} since the last summary was generated. "
                f"Existing summary: {current_summary}"
            )
            return 0

    loaded = load_metadata_json(metadata_path)
    if loaded is None:
        return 1

    text, entries = loaded
    documents = [(metadata_path, text)]

    print_entry_list(folder, metadata_path, entries)
    if dry_run:
        return 0

    api_key = require_api_key()
    if api_key is None:
        return 1

    system_prompt = load_system_prompt(config.prompt_path)
    if system_prompt is None:
        return 1

    user_content = f"Documents:\n{build_payload(documents)}"

    try:
        summary, response = run_claude(
            system_prompt,
            user_content,
            api_key,
            MODEL,
        )
    except Exception as exc:
        print(f"Error: Anthropic API call failed: {exc}", file=sys.stderr)
        return 1

    try:
        output_path = write_summary(folder, summary)
    except OSError as exc:
        print(f"Error: failed to write {SUMMARY_NAME}: {exc}", file=sys.stderr)
        return 1

    print_usage_report(MODEL, response.usage)
    print(f"Summary written to {output_path}")
    return 0
