#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from main import resolve_folder_path

ACTIONS_SYSTEM_PROMPT = (
    "You are a deal-flow assistant for the Antler investment team. "
    "The team is Tammer Kamel (TK), Shambhavi Mishra (SM), Alex Wright (AW), "
    "and Daphne McLarty (DM). When the report says 'us', 'we', or 'our team', "
    "it means this Antler team; 'the other party' means the founder or company.\n\n"
    "You are given today's date and an investment summary for one deal. "
    "Report only actions that our Antler team needs to take now. An action is "
    "needed only when:\n"
    "1. We promised or owe an action by a date, and today's date is near, at, "
    "or past that date.\n"
    "2. We were expecting something from the other party and they have not done "
    "it yet.\n\n"
    "Ignore commitments that appear already completed. For each action, cite the "
    "relevant next-step or line from the summary and any date involved, and be "
    "concise. If there are no outstanding actions for us, respond with exactly "
    "'No actions needed.' and nothing else."
)


def build_user_message(today: str, summary: str) -> str:
    return (
        f"Today's date is {today}.\n\n"
        f"Investment summary:\n{summary}"
    )


def generate_actions_report(
    today: str,
    summary: str,
    api_key: str,
    model: str,
) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ACTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(today, summary)},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report outstanding actions the Antler team needs to take for a "
            "deal, based on its summary and today's date, using Groq."
        )
    )
    parser.add_argument(
        "relative_path",
        help="Relative path under Google Drive to the deal folder (the deal name)",
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

    summary_path = folder / "ai-generated" / "summary.md"
    if not summary_path.is_file():
        print(
            f"Error: no summary found at {summary_path}. "
            "Run claude_summary.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        summary = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: failed to read {summary_path}: {exc}", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    today = datetime.now().strftime("%B %d, %Y")

    try:
        report = generate_actions_report(today, summary, api_key, model)
    except Exception as exc:
        print(f"Error: Groq API call failed: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
