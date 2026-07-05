#!/usr/bin/env python3
import argparse
import os
import sys

from dotenv import load_dotenv
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple terminal chat with an LLM via Groq."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        help=f"Groq model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the assistant",
    )
    return parser.parse_args()


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help   Show this help\n"
        "  /clear  Clear conversation history\n"
        "  /quit   Exit the chat\n"
        "  quit    Exit the chat\n"
    )


def chat_loop(client: Groq, model: str, system_prompt: str) -> int:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    print("Chat started. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.")
            return 0
        if lowered == "/help":
            print_help()
            continue
        if lowered == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            messages.pop()
            continue

        reply = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")

    return 0


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1

    client = Groq(api_key=api_key)
    return chat_loop(client, args.model, args.system)


if __name__ == "__main__":
    raise SystemExit(main())
