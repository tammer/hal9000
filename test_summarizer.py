#!/usr/bin/env python3
from __future__ import annotations

import unittest

from summarizer import assemble_status, prompt_sha256, sha256_hex

FRESH_JSON = '{"product": "fresh product", "founders": "fresh founders", "notes": "fresh notes"}'


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0

    def create(self, **_kwargs: object) -> _Response:
        content = self.contents[self.calls]
        self.calls += 1
        return _Response(content)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, contents: list[str]) -> None:
        self.chat = _Chat(_Completions(contents))


def _summary_parts(text: str) -> tuple[str, str]:
    return text, sha256_hex(text.encode("utf-8"))


def _entry(
    summary_hash: str,
    *,
    model: str = "test-model",
    prompt_hash: str | None = None,
    product: str = "cached product",
    founders: str = "cached founders",
    notes: str = "cached notes",
    status: str = "COURTING",
) -> dict[str, str]:
    return {
        "summary_sha256": summary_hash,
        "model": model,
        "prompt_sha256": prompt_sha256() if prompt_hash is None else prompt_hash,
        "product": product,
        "founders": founders,
        "notes": notes,
        "status": status,
    }


class AssembleStatusCacheTests(unittest.TestCase):
    def test_hash_hit_skips_groq_and_reparses_status(self) -> None:
        text, digest = _summary_parts("Status: REJECTED\n")
        previous = {"Scadable": _entry(digest)}
        client = _Client([])

        rows, new_cache, cached, extracted = assemble_status(
            [("Scadable", text, digest)],
            previous,
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 0)
        self.assertEqual(cached, 1)
        self.assertEqual(extracted, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "REJECTED")
        self.assertEqual(rows[0].product, "cached product")
        self.assertEqual(rows[0].founders, "cached founders")
        self.assertEqual(rows[0].notes, "cached notes")
        self.assertEqual(new_cache["Scadable"], previous["Scadable"])

    def test_hash_mismatch_calls_groq(self) -> None:
        text, digest = _summary_parts("Status: PIPELINE\n")
        previous = {"Scadable": _entry("stale-hash")}
        client = _Client([FRESH_JSON])

        rows, new_cache, cached, extracted = assemble_status(
            [("Scadable", text, digest)],
            previous,
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 1)
        self.assertEqual(cached, 0)
        self.assertEqual(extracted, 1)
        self.assertEqual(rows[0].product, "fresh product")
        self.assertEqual(rows[0].status, "PIPELINE")
        self.assertEqual(new_cache["Scadable"]["summary_sha256"], digest)
        self.assertEqual(new_cache["Scadable"]["product"], "fresh product")

    def test_model_mismatch_calls_groq(self) -> None:
        text, digest = _summary_parts("Status: MONITOR\n")
        previous = {"Scadable": _entry(digest, model="other-model")}
        client = _Client([FRESH_JSON])

        rows, new_cache, cached, extracted = assemble_status(
            [("Scadable", text, digest)],
            previous,
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 1)
        self.assertEqual(cached, 0)
        self.assertEqual(extracted, 1)
        self.assertEqual(rows[0].notes, "fresh notes")
        self.assertEqual(new_cache["Scadable"]["model"], "test-model")

    def test_prompt_mismatch_calls_groq(self) -> None:
        text, digest = _summary_parts("Status: COURTING\n")
        previous = {"Scadable": _entry(digest, prompt_hash="0" * 64)}
        client = _Client([FRESH_JSON])

        _rows, new_cache, cached, extracted = assemble_status(
            [("Scadable", text, digest)],
            previous,
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 1)
        self.assertEqual(cached, 0)
        self.assertEqual(extracted, 1)
        self.assertEqual(new_cache["Scadable"]["prompt_sha256"], prompt_sha256())

    def test_failed_extraction_leaves_previous_cache_entry_unchanged(self) -> None:
        text, digest = _summary_parts("Status: PIPELINE\n")
        previous_entry = _entry("stale-hash", product="old product")
        previous = {"Scadable": previous_entry}
        client = _Client(["nope", "still nope"])

        rows, new_cache, cached, extracted = assemble_status(
            [("Scadable", text, digest)],
            previous,
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 2)
        self.assertEqual(rows, [])
        self.assertEqual(cached, 0)
        self.assertEqual(extracted, 0)
        self.assertEqual(new_cache["Scadable"], previous_entry)
        self.assertIsNot(new_cache["Scadable"], previous_entry)

    def test_absent_deals_are_dropped_from_cache(self) -> None:
        text, digest = _summary_parts("Status: REJECTED\n")
        kept = _entry(digest)
        gone = _entry("whatever")
        client = _Client([])

        rows, new_cache, cached, extracted = assemble_status(
            [("Kept", text, digest)],
            {"Kept": kept, "Gone": gone},
            client,
            "test-model",
        )

        self.assertEqual(client.chat.completions.calls, 0)
        self.assertEqual([row.deal_name for row in rows], ["Kept"])
        self.assertEqual(cached, 1)
        self.assertEqual(extracted, 0)
        self.assertEqual(set(new_cache), {"Kept"})
        self.assertEqual(new_cache["Kept"], kept)


if __name__ == "__main__":
    unittest.main()
