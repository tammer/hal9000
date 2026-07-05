#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from groq import Groq

Provider = Literal["groq", "anthropic"]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_TOKENS = 4096

ANTHROPIC_MODELS = (
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)


def infer_provider(model: str) -> Provider:
    if model.startswith("claude-"):
        return "anthropic"
    return "groq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple terminal chat with Groq or Anthropic models."
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "groq", "anthropic"),
        default="auto",
        help="API provider (default: auto-detect from model name)",
    )
    parser.add_argument(
        "--model",
        help=(
            "Model name. Anthropic examples: "
            + ", ".join(ANTHROPIC_MODELS)
            + f". Groq default: {DEFAULT_GROQ_MODEL}"
        ),
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the assistant",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens for Anthropic models (default: {DEFAULT_MAX_TOKENS})",
    )
    return parser.parse_args()


def resolve_model_and_provider(args: argparse.Namespace) -> tuple[str, Provider]:
    model = args.model
    if model is None:
        if args.provider == "anthropic":
            model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        elif args.provider == "groq":
            model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        else:
            model = os.getenv(
                "ANTHROPIC_MODEL",
                os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            )

    provider: Provider
    if args.provider == "auto":
        provider = infer_provider(model)
    else:
        provider = args.provider

    return model, provider


def print_help(provider: Provider) -> None:
    print(
        "\nCommands:\n"
        "  /help            Show this help\n"
        "  /clear           Clear conversation history\n"
        "  /models          List models for the current provider\n"
        "  /model <name>    Switch model (and provider if needed)\n"
        "  /quit            Exit the chat\n"
        "  quit             Exit the chat\n"
    )
    if provider == "anthropic":
        print("Anthropic models:")
        for name in ANTHROPIC_MODELS:
            print(f"  {name}")
        print()


def print_models(provider: Provider) -> None:
    if provider == "anthropic":
        print("\nAnthropic models:")
        for name in ANTHROPIC_MODELS:
            print(f"  {name}")
    else:
        print("\nGroq: pass any model name supported by your Groq account.")
        print(f"  default: {DEFAULT_GROQ_MODEL}")
    print()


def complete_groq(
    client: Groq,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, *messages],
    )
    return response.choices[0].message.content or ""


def complete_anthropic(
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def complete(
    provider: Provider,
    client: Any,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    if provider == "anthropic":
        return complete_anthropic(
            client, model, system_prompt, messages, max_tokens
        )
    return complete_groq(client, model, system_prompt, messages)


def chat_loop(
    groq_client: Groq | None,
    anthropic_client: Anthropic | None,
    model: str,
    provider: Provider,
    system_prompt: str,
    max_tokens: int,
) -> int:
    messages: list[dict[str, str]] = []

    print(
        f"Chat started ({provider}, {model}). "
        "Type /help for commands.\n"
    )

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
            print_help(provider)
            continue
        if lowered == "/clear":
            messages = []
            print("Conversation cleared.\n")
            continue
        if lowered == "/models":
            print_models(provider)
            continue
        if lowered.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /model <name>\n")
                continue
            model = parts[1].strip()
            provider = infer_provider(model)
            print(f"Switched to {provider} / {model}\n")
            continue

        messages.append({"role": "user", "content": user_input})

        client = anthropic_client if provider == "anthropic" else groq_client
        if client is None:
            print(
                f"Error: no client configured for provider '{provider}'",
                file=sys.stderr,
            )
            messages.pop()
            continue

        try:
            reply = complete(
                provider,
                client,
                model,
                system_prompt,
                messages,
                max_tokens,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")

    return 0


def main() -> int:
    load_dotenv()
    args = parse_args()
    model, provider = resolve_model_and_provider(args)

    groq_client: Groq | None = None
    anthropic_client: Anthropic | None = None

    if provider == "groq" or args.provider == "auto":
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            groq_client = Groq(api_key=api_key)

    if provider == "anthropic" or args.provider == "auto":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            anthropic_client = Anthropic(api_key=api_key)

    if provider == "groq" and groq_client is None:
        print("Error: GROQ_API_KEY is not set", file=sys.stderr)
        return 1
    if provider == "anthropic" and anthropic_client is None:
        print("Error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    return chat_loop(
        groq_client,
        anthropic_client,
        model,
        provider,
        args.system,
        args.max_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
