#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from get_facts import FACT_EXTRACTOR_MODEL, SEARCH_PLANNER_MODEL, get_facts

# Supervisor model (answer synthesis + stop/continue decisions)
SUPERVISOR_MODEL = "claude-sonnet-4-6"
# get_facts uses its own models (SEARCH_PLANNER_MODEL / FACT_EXTRACTOR_MODEL in get_facts.py)

MAX_ITERATIONS = 8
MAX_SEARCHES_PER_ROUND = 4
MAX_TOKENS = 4096
CONFIDENCE_THRESHOLD = 0.85

Provider = Literal["groq", "anthropic"]

ANSWER_SYSTEM_PROMPT = """You synthesize a research answer from quantitative facts gathered from the web.

Return valid JSON only with this exact shape:
{"answer_markdown":"markdown answer text","confidence":0.0}

Rules:
- The user prompt contains both the research question and success criteria. Address every criterion.
- Use only the provided facts. Do not invent information beyond them.
- Cite sources inline as markdown links using each fact's source URL.
- If a criterion cannot be met from the available facts, say so clearly.
- confidence is 0.0 to 1.0 reflecting how well the answer meets all criteria.
"""

DECISION_SYSTEM_PROMPT = """You decide whether research should stop or continue.

Return valid JSON only with this exact shape:
{
  "should_stop": false,
  "confidence": 0.0,
  "reasoning": "why stop or continue",
  "unmet_criteria": ["criterion not yet satisfied"],
  "research_gaps": ["what information is still missing"],
  "next_focus": "specific guidance for the next round of fact-finding"
}

Rules:
- Evaluate against the user's full prompt (question + criteria), the accumulated facts, and the current best answer.
- Set should_stop=true only when the answer adequately meets all criteria.
- confidence is 0.0 to 1.0 for how confident you are that criteria are met.
- reasoning must explain the stop/continue decision clearly.
- If continuing, next_focus should be actionable guidance for targeted web research.
- research_gaps and unmet_criteria should be empty when should_stop=true.
"""


@dataclass
class AnswerDraft:
    answer_markdown: str
    confidence: float


@dataclass
class ResearchDecision:
    should_stop: bool
    confidence: float
    reasoning: str
    unmet_criteria: list[str]
    research_gaps: list[str]
    next_focus: str


@dataclass
class IterationResult:
    iteration: int
    answer: AnswerDraft
    decision: ResearchDecision
    fact_count: int


def infer_provider(model: str) -> Provider:
    if model.startswith("claude-"):
        return "anthropic"
    return "groq"


def groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    from groq import Groq

    return Groq(api_key=api_key)


def anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def escape_control_characters_in_json(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                result.append(char)
                escape = False
                continue

            if char == "\\":
                result.append(char)
                escape = True
                continue

            if char == '"':
                result.append(char)
                in_string = False
                continue

            if ord(char) < 32:
                replacements = {
                    "\n": "\\n",
                    "\r": "\\r",
                    "\t": "\\t",
                    "\b": "\\b",
                    "\f": "\\f",
                }
                result.append(replacements.get(char, f"\\u{ord(char):04x}"))
                continue

            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True

    return "".join(result)


def parse_json_response(content: str) -> dict[str, Any]:
    text = strip_code_fences(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = extract_json_object(text)
        if not candidate:
            raise

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = escape_control_characters_in_json(candidate)
        return json.loads(repaired)


def llm_json(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    if infer_provider(model) == "anthropic":
        client = anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
    else:
        client = groq_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""

    return parse_json_response(content)


def sanitize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def sanitize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def merge_facts(
    existing: list[dict[str, str]],
    new_facts: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged = list(existing)
    seen = {
        (fact["name"].lower(), fact["value"], fact["source"])
        for fact in existing
    }

    for fact in new_facts:
        name = str(fact.get("name", "")).strip()
        value = str(fact.get("value", "")).strip()
        source = str(fact.get("source", "")).strip()
        if not name or not value or not source:
            continue

        key = (name.lower(), value, source)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"name": name, "value": value, "source": source})

    return merged


def format_facts(facts: list[dict[str, str]]) -> str:
    if not facts:
        return "No facts gathered yet."

    lines = []
    for index, fact in enumerate(facts, start=1):
        lines.append(
            f"{index}. {fact['name']}: {fact['value']} (source: {fact['source']})"
        )
    return "\n".join(lines)


def build_follow_up_prompt(user_prompt: str, decision: ResearchDecision) -> str:
    parts = [user_prompt, "", "Additional research guidance for this round:"]
    if decision.next_focus:
        parts.append(f"- Focus: {decision.next_focus}")
    for gap in decision.research_gaps:
        parts.append(f"- Gap: {gap}")
    for criterion in decision.unmet_criteria:
        parts.append(f"- Unmet criterion: {criterion}")
    return "\n".join(parts)


def synthesize_answer(
    user_prompt: str,
    facts: list[dict[str, str]],
    model: str,
) -> AnswerDraft:
    payload = llm_json(
        ANSWER_SYSTEM_PROMPT,
        "\n\n".join(
            [
                f"User prompt (question + criteria):\n{user_prompt}",
                "Accumulated facts:",
                format_facts(facts),
            ]
        ),
        model,
    )
    answer_markdown = str(payload.get("answer_markdown", "")).strip()
    if not answer_markdown:
        answer_markdown = "Insufficient facts to answer the research question."

    return AnswerDraft(
        answer_markdown=answer_markdown,
        confidence=sanitize_confidence(payload.get("confidence")),
    )


def evaluate_stop(
    user_prompt: str,
    facts: list[dict[str, str]],
    answer: AnswerDraft,
    model: str,
    *,
    iterations_left: int,
) -> ResearchDecision:
    payload = llm_json(
        DECISION_SYSTEM_PROMPT,
        "\n\n".join(
            [
                f"User prompt (question + criteria):\n{user_prompt}",
                f"Iterations remaining: {iterations_left}",
                "Accumulated facts:",
                format_facts(facts),
                "Current best answer:",
                answer.answer_markdown,
            ]
        ),
        model,
    )
    return ResearchDecision(
        should_stop=bool(payload.get("should_stop")),
        confidence=sanitize_confidence(payload.get("confidence")),
        reasoning=str(payload.get("reasoning", "")).strip(),
        unmet_criteria=sanitize_list(payload.get("unmet_criteria")),
        research_gaps=sanitize_list(payload.get("research_gaps")),
        next_focus=str(payload.get("next_focus", "")).strip(),
    )


def format_list_items(items: list[str], fallback: str = "none") -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def print_iteration_report(
    iteration: int,
    max_iterations: int,
    answer: AnswerDraft,
    decision: ResearchDecision,
    fact_count: int,
) -> None:
    stop_label = "yes" if decision.should_stop else "no"
    print(f"=== Iteration {iteration}/{max_iterations} ===\n")
    print(f"Facts accumulated: {fact_count}\n")
    print("## Best answer so far\n")
    print(answer.answer_markdown.rstrip())
    print("\n## Decision\n")
    print(f"- **Stop researching:** {stop_label}")
    print(f"- **Confidence:** {decision.confidence:.2f}")
    print(f"- **Reasoning:** {decision.reasoning or 'none'}")
    print(f"- **Unmet criteria:**\n{format_list_items(decision.unmet_criteria)}")
    print(f"- **Research gaps:**\n{format_list_items(decision.research_gaps)}")
    print(f"- **Next focus:** {decision.next_focus or 'none'}")
    print("\n---\n")


def run_research(
    user_prompt: str,
    supervisor_model: str,
    max_iterations: int,
    max_searches_per_round: int,
    *,
    verbose: bool,
) -> tuple[AnswerDraft, ResearchDecision, list[IterationResult]]:
    accumulated_facts: list[dict[str, str]] = []
    research_prompt = user_prompt
    best_answer: AnswerDraft | None = None
    best_decision: ResearchDecision | None = None
    best_confidence = -1.0
    iterations: list[IterationResult] = []

    for iteration in range(1, max_iterations + 1):
        new_facts = get_facts(
            research_prompt,
            SEARCH_PLANNER_MODEL,
            FACT_EXTRACTOR_MODEL,
            max_searches_per_round,
            verbose=verbose,
        )
        accumulated_facts = merge_facts(accumulated_facts, new_facts)

        answer = synthesize_answer(user_prompt, accumulated_facts, supervisor_model)
        decision = evaluate_stop(
            user_prompt,
            accumulated_facts,
            answer,
            supervisor_model,
            iterations_left=max_iterations - iteration,
        )

        result = IterationResult(
            iteration=iteration,
            answer=answer,
            decision=decision,
            fact_count=len(accumulated_facts),
        )
        iterations.append(result)
        print_iteration_report(
            iteration,
            max_iterations,
            answer,
            decision,
            len(accumulated_facts),
        )

        if decision.confidence > best_confidence:
            best_confidence = decision.confidence
            best_answer = answer
            best_decision = decision

        if decision.should_stop:
            return answer, decision, iterations

        if iteration < max_iterations:
            research_prompt = build_follow_up_prompt(user_prompt, decision)

    assert best_answer is not None and best_decision is not None
    return best_answer, best_decision, iterations


def format_research_caveats(decision: ResearchDecision) -> list[str]:
    if decision.should_stop and decision.confidence >= CONFIDENCE_THRESHOLD:
        return []

    lines = ["\n## Research caveats\n"]
    if decision.should_stop:
        lines.append(
            f"Research stopped per evaluator decision, but confidence was below "
            f"threshold ({decision.confidence:.2f} < {CONFIDENCE_THRESHOLD})."
        )
    else:
        lines.append(
            f"Research ended without a stop decision "
            f"(confidence {decision.confidence:.2f}, threshold {CONFIDENCE_THRESHOLD})."
        )
    if decision.unmet_criteria:
        lines.append("\nUnmet criteria:")
        for item in decision.unmet_criteria:
            lines.append(f"- {item}")
    if decision.research_gaps:
        lines.append("\nResearch gaps:")
        for item in decision.research_gaps:
            lines.append(f"- {item}")
    return lines


def prompt_for_research() -> str:
    print("Enter your research question and criteria:")
    return input("> ").strip()


def write_output(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an iterative research loop using get_facts for web search "
            "and fact extraction. Provide the research question and criteria together."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Research question and criteria in one prompt",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_flag",
        help="Research question and criteria in one prompt",
    )
    parser.add_argument(
        "--model",
        default=SUPERVISOR_MODEL,
        help=f"Supervisor LLM model (default: {SUPERVISOR_MODEL})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Maximum research iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max-searches",
        type=int,
        default=MAX_SEARCHES_PER_ROUND,
        help=(
            f"Maximum searches per get_facts call "
            f"(default: {MAX_SEARCHES_PER_ROUND})"
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the final markdown answer",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log get_facts progress to stderr (default: on when stderr is a TTY)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    prompt = (args.prompt_flag or args.prompt or "").strip()
    if not prompt:
        try:
            prompt = prompt_for_research()
        except EOFError:
            print("No research prompt provided.", file=sys.stderr)
            return 1

    if not prompt:
        print("Research prompt cannot be empty.", file=sys.stderr)
        return 1

    if args.max_iterations < 1:
        print("--max-iterations must be at least 1.", file=sys.stderr)
        return 1

    if args.max_searches < 1:
        print("--max-searches must be at least 1.", file=sys.stderr)
        return 1

    supervisor_model = args.model
    verbose = args.verbose if args.verbose is not None else sys.stderr.isatty()

    if verbose:
        print(
            f"Supervisor: {infer_provider(supervisor_model)} / {supervisor_model}",
            file=sys.stderr,
        )
        print(
            f"get_facts: groq / {SEARCH_PLANNER_MODEL} (planner), "
            f"{FACT_EXTRACTOR_MODEL} (extractor)",
            file=sys.stderr,
        )

    try:
        answer, decision, _ = run_research(
            prompt,
            supervisor_model,
            args.max_iterations,
            args.max_searches,
            verbose=verbose,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print("## Final answer\n")
    print(answer.answer_markdown.rstrip())

    caveat_lines = format_research_caveats(decision)
    if caveat_lines:
        print("".join(caveat_lines))

    if args.output:
        output_parts = ["## Final answer\n", answer.answer_markdown.rstrip()]
        output_parts.extend(format_research_caveats(decision))

        write_output(args.output, "\n".join(output_parts).rstrip() + "\n")
        print(f"\nSaved answer to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
