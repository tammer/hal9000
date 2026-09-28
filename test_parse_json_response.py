#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from get_facts import parse_json_response
from summarizer import extract_deal_row


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


class ParseJsonResponseTests(unittest.TestCase):
    def test_invalid_dollar_escape_becomes_dollar(self) -> None:
        raw = """{
  "product": "compliance platform",
  "founders": "Ali Rahbar, Hossein Molavi",
  "notes": "founders want to hit ~\\$200K ARR by Jan 2027 before raising"
}"""
        notes = parse_json_response(raw)["notes"]
        self.assertIn("~$200K", notes)
        self.assertNotIn("\\", notes)

    def test_valid_escapes_stay_intact(self) -> None:
        raw = '{"notes": "line\\nquote \\"hi\\" end\\\\"}'
        self.assertEqual(parse_json_response(raw)["notes"], 'line\nquote "hi" end\\')

    def test_unicode_escape_stays_intact(self) -> None:
        raw = '{"notes": "Sep\\u202f21"}'
        self.assertEqual(parse_json_response(raw)["notes"], "Sep\u202f21")

    def test_raw_newline_inside_string_still_parses(self) -> None:
        raw = '{"notes": "hello\nworld"}'
        self.assertEqual(parse_json_response(raw)["notes"], "hello\nworld")


class ExtractDealRowRetryTests(unittest.TestCase):
    def test_retries_once_after_invalid_json(self) -> None:
        client = _Client(
            [
                "not json",
                '{"product": "p", "founders": "f", "notes": "n"}',
            ]
        )
        row = extract_deal_row(client, "test-model", "Scadable", "Status: COURTING\n")
        self.assertEqual(row.notes, "n")
        self.assertEqual(row.status, "COURTING")
        self.assertEqual(client.chat.completions.calls, 2)

    def test_dollar_escape_does_not_need_a_retry(self) -> None:
        client = _Client(
            [
                '{"product": "p", "founders": "f", "notes": "hit ~\\$200K ARR"}',
            ]
        )
        row = extract_deal_row(client, "test-model", "Scadable", "Status: COURTING\n")
        self.assertEqual(row.notes, "hit ~$200K ARR")
        self.assertEqual(client.chat.completions.calls, 1)

    def test_second_invalid_response_raises(self) -> None:
        client = _Client(["nope", "still nope"])
        with self.assertRaises(json.JSONDecodeError):
            extract_deal_row(client, "test-model", "Scadable", "Status: COURTING\n")
        self.assertEqual(client.chat.completions.calls, 2)


if __name__ == "__main__":
    unittest.main()
