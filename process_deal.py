#!/usr/bin/env python3
"""Process a deal folder into ai-generated/deal.json via generate_metadata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from claude_common import resolve_folder_path
from consolidator import DATETIME_FMT, file_mtime
from document_utils import FORBIDDEN_DIR_NAMES, list_candidate_files
from generate_metadata import generate_metadata

DEAL_JSON_NAME = "deal.json"


def deal_json_path(folder: Path) -> Path:
    return folder / "ai-generated" / DEAL_JSON_NAME


def load_deal_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing deal.json and index entries by filename."""
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
        return {}

    if not isinstance(raw, list):
        print(
            f"Warning: {path} is not a JSON list; starting fresh",
            file=sys.stderr,
        )
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if isinstance(filename, str) and filename:
            cache[filename] = item
    return cache


def relative_to_drive_base(path: Path) -> str:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).resolve()
    return path.resolve().relative_to(base).as_posix()


def is_cache_fresh(entry: dict[str, Any], path: Path) -> bool:
    timestamp_raw = entry.get("timestamp")
    if not isinstance(timestamp_raw, str) or not timestamp_raw.strip():
        return False
    try:
        generated_at = datetime.strptime(timestamp_raw.strip(), DATETIME_FMT)
    except ValueError:
        return False
    return file_mtime(path) <= generated_at


def write_deal_json(folder: Path, entries: list[dict[str, Any]]) -> tuple[Path, bool]:
    """Write deal.json if content changed. Returns (path, wrote)."""
    ai_dir = folder / "ai-generated"
    ai_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_dir / DEAL_JSON_NAME
    new_text = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
    if output_path.is_file():
        try:
            if output_path.read_text(encoding="utf-8") == new_text:
                return output_path, False
        except OSError:
            pass
    output_path.write_text(new_text, encoding="utf-8")
    return output_path, True


def process_deal(relative_path: str) -> Path:
    folder = resolve_folder_path(relative_path)
    if not folder.exists():
        raise FileNotFoundError(f"path does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"path is not a directory: {folder}")

    output_path = deal_json_path(folder)
    cache = load_deal_cache(output_path)
    candidates = list_candidate_files(
        folder, recursive=True, exclude_dirs=FORBIDDEN_DIR_NAMES
    )

    results: list[dict[str, Any]] = []
    reused = 0
    regenerated = 0

    for path in candidates:
        rel = relative_to_drive_base(path)
        cached = cache.get(rel)

        if cached is not None and is_cache_fresh(cached, path):
            print(f"Cached: {rel}", file=sys.stderr)
            results.append(cached)
            reused += 1
            continue

        print(f"Generating: {rel}", file=sys.stderr)
        try:
            entry = dict(generate_metadata(rel))
            results.append(entry)
            regenerated += 1
        except Exception as exc:
            print(
                f"Warning: failed to generate metadata for {rel}: {exc}",
                file=sys.stderr,
            )
            if cached is not None:
                print(f"Keeping stale cache for {rel}", file=sys.stderr)
                results.append(cached)
                reused += 1

    results.sort(key=lambda entry: str(entry.get("created_at", "")))
    written, changed = write_deal_json(folder, results)
    action = "Wrote" if changed else "Unchanged"
    print(
        f"{action} {written} ({len(results)} entries; "
        f"{regenerated} generated, {reused} reused)",
        file=sys.stderr,
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or refresh ai-generated/deal.json for a deal folder "
            "using per-file metadata from generate_metadata."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under GOOGLE_DRIVE_BASE to the deal folder",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        process_deal(args.relative_path)
    except (ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: process_deal failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
