#!/usr/bin/env python3
"""Central Google Drive path helpers for the Canada shared-drive root.

``GOOGLE_DRIVE_BASE`` points at the Canada root, which contains peer ``deals/``
and ``portcos/`` trees plus shared ``ai-generated/`` outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

COMPANY_ROOTS = frozenset({"deals", "portcos"})

__all__ = [
    "COMPANY_ROOTS",
    "deals_base",
    "facts_md",
    "google_drive_base",
    "list_company_folders",
    "portcos_base",
    "resolve_company_folder",
    "resolve_deal_folder",
    "resolve_portco_folder",
    "shared_ai_dir",
]


def google_drive_base() -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    return Path(base_raw).expanduser().resolve()


def deals_base() -> Path:
    return google_drive_base() / "deals"


def portcos_base() -> Path:
    return google_drive_base() / "portcos"


def shared_ai_dir() -> Path:
    return google_drive_base() / "ai-generated"


def facts_md() -> Path:
    return google_drive_base() / "facts.md"


def list_company_folders(root: Path) -> list[Path]:
    """Return child company folders under ``root``, skipping dotdirs and ai-generated."""
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name != "ai-generated"
    )


def _resolve_under(base: Path, relative_path: str, *, label: str) -> Path:
    cleaned = relative_path.strip().lstrip("/")
    if not cleaned:
        raise ValueError(f"relative_path must be a non-empty {label} folder name")

    folder = (base / cleaned).resolve()
    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes {label} root: {relative_path}")
    return folder


def resolve_deal_folder(relative_path: str) -> Path:
    """Resolve a folder name (or relative path) under ``deals/``."""
    return _resolve_under(deals_base(), relative_path, label="deals")


def resolve_portco_folder(relative_path: str) -> Path:
    """Resolve a folder name (or relative path) under ``portcos/``."""
    return _resolve_under(portcos_base(), relative_path, label="portcos")


def resolve_company_folder(relative_path: str) -> Path:
    """Resolve ``deals/<folder>`` or ``portcos/<folder>`` to an absolute path."""
    cleaned = relative_path.strip().lstrip("/")
    parts = Path(cleaned).parts
    if not parts or parts[0] not in COMPANY_ROOTS:
        raise ValueError(
            "path must start with 'deals/' or 'portcos/' "
            f"(e.g. deals/Tony or portcos/Central-Agent); got {relative_path!r}"
        )
    if len(parts) < 2:
        raise ValueError(
            f"path must include a folder under {parts[0]}/; got {relative_path!r}"
        )

    root = parts[0]
    rest = Path(*parts[1:])
    base = deals_base() if root == "deals" else portcos_base()
    folder = (base / rest).resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes {root} root: {relative_path}")

    return folder
