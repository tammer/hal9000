#!/usr/bin/env python3
"""Process a portfolio-company folder into ai-generated/portco.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from fetch_transcripts import portcos_base
from process_deal import process_company_folder

PORTCO_JSON_NAME = "portco.json"


def resolve_portco_folder(relative_path: str) -> Path:
    cleaned = relative_path.strip().lstrip("/")
    if not cleaned:
        raise ValueError("relative_path must be a non-empty portco folder name")

    base = portcos_base()
    folder = (base / cleaned).resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes portcos root: {relative_path}")

    return folder


def process_portco(relative_path: str) -> Path:
    folder = resolve_portco_folder(relative_path)
    if not folder.exists():
        raise FileNotFoundError(f"path does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"path is not a directory: {folder}")

    return process_company_folder(
        folder,
        path_base=portcos_base(),
        json_name=PORTCO_JSON_NAME,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or refresh ai-generated/portco.json for a portfolio-company "
            "folder using per-file metadata from generate_metadata."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Folder name under the sibling portcos/ directory (e.g. Central-Agent)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        process_portco(args.relative_path)
    except (ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: process_portco failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
