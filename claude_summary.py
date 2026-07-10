#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from document_utils import collect_documents

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


def list_top_level_source_files(folder: Path) -> list[Path]:
    return sorted(
        entry
        for entry in folder.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and not entry.name.startswith("~$")
    )


def existing_summary_if_current(folder: Path) -> Path | None:
    summary_path = folder / "ai-generated" / "summary.md"
    if not summary_path.is_file():
        return None

    source_files = list_top_level_source_files(folder)
    if not source_files:
        return None

    summary_mtime = summary_path.stat().st_mtime
    for source in source_files:
        if source.stat().st_mtime >= summary_mtime:
            return None

    return summary_path


def resolve_folder_path(relative_path: str) -> Path:
    base_raw = os.getenv("GOOGLE_DRIVE_BASE")
    if not base_raw:
        raise ValueError("GOOGLE_DRIVE_BASE is not set")
    base = Path(base_raw).resolve()
    folder = (base / relative_path.lstrip("/")).resolve()

    if base not in folder.parents and folder != base:
        raise ValueError(f"path escapes Google Drive root: {relative_path}")

    return folder


def load_summary_prompt() -> str:
    if not SUMMARY_PROMPT_PATH.exists():
        raise FileNotFoundError(f"summary prompt not found: {SUMMARY_PROMPT_PATH}")
    return SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")


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


def generate_summary(
    system_prompt: str,
    documents: list[tuple[Path, str]],
    api_key: str,
    model: str,
) -> tuple[str, object]:
    client = Anthropic(api_key=api_key)
    payload = build_payload(documents)

    if len(payload) > PAYLOAD_WARN_CHARS:
        print(
            f"Warning: document payload is large ({len(payload):,} chars)",
            file=sys.stderr,
        )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Documents:\n{payload}",
            }
        ],
    )

    text = append_generated_timestamp(
        strip_markdown_fences(extract_response_text(response))
    )
    return text, response


def write_summary(folder: Path, content: str) -> Path:
    ai_generated_dir = folder / "ai-generated"
    ai_generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = ai_generated_dir / "summary.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an investment summary from deal documents using Claude."
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the folder to summarize",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        folder = resolve_folder_path(args.relative_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not folder.exists():
        print(f"Error: path does not exist: {folder}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: path is not a directory: {folder}", file=sys.stderr)
        return 1

    current_summary = existing_summary_if_current(folder)
    if current_summary is not None:
        print(
            "No new source documents since the last summary was generated. "
            f"Existing summary: {current_summary}"
        )
        return 0

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    try:
        system_prompt = load_summary_prompt()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    documents = collect_documents(folder, recursive=False)
    if not documents:
        print(
            f"Error: no readable top-level files found in {folder}",
            file=sys.stderr,
        )
        return 1

    try:
        summary, response = generate_summary(
            system_prompt,
            documents,
            api_key,
            MODEL,
        )
    except Exception as exc:
        print(f"Error: Anthropic API call failed: {exc}", file=sys.stderr)
        return 1

    try:
        output_path = write_summary(folder, summary)
    except OSError as exc:
        print(f"Error: failed to write summary.md: {exc}", file=sys.stderr)
        return 1

    print_usage_report(MODEL, response.usage)
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
