#!/usr/bin/env python3
"""Shared helpers for Claude summary / update CLI scripts."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-sonnet-5"

MAX_OUTPUT_TOKENS = 16_384
PAYLOAD_WARN_CHARS = 500_000

SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt.md"

MARKDOWN_FENCE_RE = re.compile(
    r"^```(?:markdown)?\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5": ModelPricing(10.0, 50.0),
    "claude-opus-4-8": ModelPricing(5.0, 25.0),
    "claude-opus-4-7": ModelPricing(5.0, 25.0),
    "claude-opus-4-6": ModelPricing(5.0, 25.0),
    "claude-opus-4-5": ModelPricing(5.0, 25.0),
    "claude-sonnet-5": ModelPricing(2.0, 10.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0),
}


def strip_markdown_fences(text: str) -> str:
    match = MARKDOWN_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def append_generated_timestamp(markdown: str) -> str:
    stamp = datetime.now().strftime("Generated at %H:%M on %b-%d")
    return f"{markdown.rstrip()}\n\n{stamp}\n"


def build_payload(documents: list[tuple[Path, str]]) -> str:
    sections = [f"### {path.name}\n{content}" for path, content in documents]
    return "\n\n".join(sections)


def resolve_folder_path(relative_path: str) -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).resolve()
    folder = (base / relative_path.lstrip("/")).resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes Google Drive root: {relative_path}")

    return folder


def load_prompt_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def load_summary_prompt() -> str:
    return load_prompt_file(SUMMARY_PROMPT_PATH)


def extract_response_text(response) -> str:
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def estimate_cost(model: str, usage) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    input_cost = input_tokens * pricing.input_per_mtok / 1_000_000
    output_cost = output_tokens * pricing.output_per_mtok / 1_000_000
    cache_write_cost = cache_write_tokens * pricing.input_per_mtok / 1_000_000
    cache_read_cost = cache_read_tokens * (pricing.input_per_mtok * 0.1) / 1_000_000

    return input_cost + output_cost + cache_write_cost + cache_read_cost


def print_usage_report(model: str, usage) -> None:
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    print(
        f"Tokens: {input_tokens:,} in / {output_tokens:,} out",
        file=sys.stderr,
    )

    cost = estimate_cost(model, usage)
    if cost is None:
        print(
            f"Estimated cost: unavailable (no pricing for {model})",
            file=sys.stderr,
        )
        return

    print(f"Estimated cost: ${cost:.4f} USD ({model})", file=sys.stderr)


def parse_relative_path_args(
    description: str, path_help: str, *, with_dry_run: bool = False
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("relative_path", help=path_help)
    if with_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List documents that would be summarized without calling the API",
        )
    return parser.parse_args()


def validate_folder(relative_path: str) -> Path | None:
    """Resolve and validate a deal folder. Prints errors and returns None on failure."""
    try:
        folder = resolve_folder_path(relative_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None

    if not folder.exists():
        print(f"Error: path does not exist: {folder}", file=sys.stderr)
        return None

    if not folder.is_dir():
        print(f"Error: path is not a directory: {folder}", file=sys.stderr)
        return None

    return folder


def require_api_key() -> str | None:
    """Return ANTHROPIC_API_KEY, or print an error and return None."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return None
    return api_key


def load_system_prompt(prompt_path: Path | None = None) -> str | None:
    """Load a system prompt file, or print an error and return None."""
    path = prompt_path if prompt_path is not None else SUMMARY_PROMPT_PATH
    try:
        return load_prompt_file(path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None


def run_claude(
    system_prompt: str,
    user_content: str,
    api_key: str,
    model: str,
) -> tuple[str, object]:
    client = Anthropic(api_key=api_key)

    if len(user_content) > PAYLOAD_WARN_CHARS:
        print(
            f"Warning: request payload is large ({len(user_content):,} chars)",
            file=sys.stderr,
        )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )

    text = append_generated_timestamp(
        strip_markdown_fences(extract_response_text(response))
    )
    return text, response
