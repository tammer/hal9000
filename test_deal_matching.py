#!/usr/bin/env python3
from __future__ import annotations

import unittest

from fetch_transcripts import (
    DealIdentity,
    DealMatchTarget,
    find_programmatic_deal_match_from_haystack,
    shortlist_deal_matches_from_haystack,
    unique_strong_match,
)


def make_target(
    folder_name: str,
    *,
    company_name: str | None = None,
    human_names: list[str] | None = None,
    aliases: list[str] | None = None,
    email_domains: list[str] | None = None,
) -> DealMatchTarget:
    return DealMatchTarget(
        folder_name=folder_name,
        identity=DealIdentity(
            company_name=company_name,
            human_names=human_names or [],
            aliases=aliases or [],
            email_domains=email_domains or [],
        ),
    )


class DealMatchingTests(unittest.TestCase):
    def test_company_name_is_strong_hit(self) -> None:
        targets = [
            make_target("Acme", company_name="Acme Robotics"),
            make_target("Other", company_name="OtherCo"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Intro call with Acme Robotics founders",
            targets,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].folder_name, "Acme")
        self.assertGreater(candidates[0].strong_hits, 0)

        match = unique_strong_match(candidates, source_label="test")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.deal_folder, "Acme")

    def test_email_domain_is_strong_hit(self) -> None:
        targets = [
            make_target("Mobi", company_name="Mobi", email_domains=["mobi.ai"]),
            make_target("Other", company_name="OtherCo"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Meeting with founder@mobi.ai and tammer@antler.co",
            targets,
            email_domains={"mobi.ai"},
        )
        self.assertEqual([c.folder_name for c in candidates], ["Mobi"])
        self.assertGreater(candidates[0].strong_hits, 0)

    def test_first_name_only_is_weak_and_not_auto_accepted(self) -> None:
        targets = [
            make_target("Sam", human_names=["Sam Rivera"]),
            make_target("Other", company_name="OtherCo"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Quick sync with Sam about hiring",
            targets,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].folder_name, "Sam")
        self.assertEqual(candidates[0].strong_hits, 0)
        self.assertGreater(candidates[0].weak_hits, 0)

        self.assertIsNone(unique_strong_match(candidates, source_label="test"))
        self.assertIsNone(
            find_programmatic_deal_match_from_haystack(
                "Quick sync with Sam about hiring",
                targets,
                source_label="test",
            )
        )

    def test_full_person_name_is_strong_and_auto_accepted_for_emails(self) -> None:
        targets = [
            make_target("Sam", human_names=["Sam Rivera"]),
            make_target("Other", company_name="OtherCo"),
        ]
        match = find_programmatic_deal_match_from_haystack(
            "Follow-up with Sam Rivera on product",
            targets,
            source_label="email content",
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.deal_folder, "Sam")

    def test_multi_deal_ambiguity_returns_multiple_candidates(self) -> None:
        targets = [
            make_target("Alpha", company_name="Alpha Labs"),
            make_target("Beta", company_name="Beta Labs"),
            make_target("Gamma", company_name="Gamma Inc"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Comparing Alpha Labs and Beta Labs for the residency",
            targets,
        )
        self.assertEqual(
            sorted(c.folder_name for c in candidates),
            ["Alpha", "Beta"],
        )
        self.assertIsNone(unique_strong_match(candidates, source_label="test"))

    def test_short_folder_name_does_not_match(self) -> None:
        targets = [
            make_target("Ed", human_names=["Edward Chen"]),
            make_target("Mobi", company_name="Mobi"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Ed joined the call briefly",
            targets,
        )
        # "Ed" folder name is too short for strong folder match; first-name
        # from "Edward" is "Edward", not "Ed", so no hit unless we only have Ed.
        # The haystack has word "Ed" which is < MIN_FOLDER_NAME_MATCH_LEN.
        self.assertEqual(candidates, [])

    def test_alias_match_is_strong(self) -> None:
        targets = [
            make_target(
                "Tony",
                company_name="TonyCo",
                aliases=["Central Agent"],
            ),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Demo of Central Agent roadmap",
            targets,
        )
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].strong_hits, 0)

    def test_shortlist_limit_keeps_top_scores(self) -> None:
        targets = [
            make_target("A", company_name="Alpha"),
            make_target("B", company_name="Beta"),
            make_target("C", company_name="Charlie"),
            make_target("D", company_name="Delta"),
        ]
        candidates = shortlist_deal_matches_from_haystack(
            "Alpha Beta Charlie Delta all mentioned",
            targets,
            limit=3,
        )
        self.assertEqual(len(candidates), 3)


if __name__ == "__main__":
    unittest.main()
