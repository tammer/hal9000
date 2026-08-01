#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

from founders import (
    Founder,
    LinkedInCandidate,
    accept_linkedin_candidate,
    build_linkedin_search_queries,
    first_name_compatible,
    normalize_linkedin_url,
    resolve_search_location,
    slug_supports_name,
    slug_surname_conflict,
    text_matches_name,
    token_in_text,
)


class NameMatchingTests(unittest.TestCase):
    def test_token_in_text_rejects_substring_prefix(self) -> None:
        self.assertTrue(token_in_text("sol", "Jake Sol-Strozberg"))
        self.assertFalse(token_in_text("sol", "Jake Soloff | LinkedIn"))
        self.assertFalse(token_in_text("sol", "jake-soloff-975283b3"))

    def test_text_matches_name_requires_full_surname(self) -> None:
        self.assertTrue(
            text_matches_name("Jake Sol-Strozberg | LinkedIn", "Jake Sol-Strozberg")
        )
        self.assertFalse(
            text_matches_name("Jake Soloff | LinkedIn", "Jake Sol-Strozberg")
        )
        self.assertFalse(
            text_matches_name("jake-soloff-975283b3", "Jake Sol-Strozberg")
        )

    def test_text_matches_name_allows_jake_jakob_alias(self) -> None:
        self.assertTrue(first_name_compatible("Jake", "Jakob"))
        self.assertTrue(
            text_matches_name(
                "Jakob Sol Strozberg - Co-Founding Engineer",
                "Jake Sol-Strozberg",
            )
        )

    def test_text_matches_name_saumya_wrong_person(self) -> None:
        self.assertTrue(text_matches_name("Saumya Banker | LinkedIn", "Saumya Banker"))
        self.assertFalse(
            text_matches_name("Saumya Singh | LinkedIn", "Saumya Banker")
        )

    def test_slug_surname_conflict_donna_cases(self) -> None:
        self.assertTrue(
            slug_surname_conflict(
                "https://www.linkedin.com/in/saumya-singh-432465316",
                "Saumya Banker",
            )
        )
        self.assertTrue(
            slug_surname_conflict(
                "https://www.linkedin.com/in/jake-soloff-975283b3",
                "Jake Sol-Strozberg",
            )
        )
        self.assertFalse(
            slug_surname_conflict(
                "https://www.linkedin.com/in/sdubey97",
                "Shivam Dubey",
            )
        )
        self.assertFalse(
            slug_surname_conflict(
                "https://www.linkedin.com/in/saumyabanker",
                "Saumya Banker",
            )
        )
        self.assertFalse(
            slug_surname_conflict(
                "https://www.linkedin.com/in/jakob-sol-strozberg-50a974356",
                "Jake Sol-Strozberg",
            )
        )


class LinkedInUrlTests(unittest.TestCase):
    def test_normalizes_country_subdomain(self) -> None:
        self.assertEqual(
            normalize_linkedin_url("https://ca.linkedin.com/in/saumyabanker"),
            "https://www.linkedin.com/in/saumyabanker",
        )
        self.assertEqual(
            normalize_linkedin_url(
                "https://ca.linkedin.com/in/jakob-sol-strozberg-50a974356"
            ),
            "https://www.linkedin.com/in/jakob-sol-strozberg-50a974356",
        )

    def test_slug_supports_name_donna_cases(self) -> None:
        self.assertTrue(
            slug_supports_name(
                "https://ca.linkedin.com/in/saumyabanker",
                "Saumya Banker",
            )
        )
        self.assertTrue(
            slug_supports_name(
                "https://www.linkedin.com/in/jakob-sol-strozberg-50a974356",
                "Jake Sol-Strozberg",
            )
        )
        self.assertFalse(
            slug_supports_name(
                "https://www.linkedin.com/in/saumya-singh-432465316",
                "Saumya Banker",
            )
        )


class SearchQueryTests(unittest.TestCase):
    def test_queries_include_company_and_canada(self) -> None:
        founder = Founder(
            full_name="Saumya Banker",
            first_name="Saumya",
            last_name="Banker",
        )
        queries = build_linkedin_search_queries(founder, "Donna", "Canada")
        self.assertTrue(queries)
        self.assertTrue(any("Donna" in q and "Canada" in q for q in queries))
        self.assertTrue(any("site:linkedin.com/in" in q for q in queries))
        # Company-bearing queries should come before bare-name fallback.
        first_company = next(i for i, q in enumerate(queries) if "Donna" in q)
        first_bare = next(
            i
            for i, q in enumerate(queries)
            if "Donna" not in q and "Canada" not in q
        )
        self.assertLess(first_company, first_bare)

    def test_location_defaults_to_canada(self) -> None:
        self.assertEqual(resolve_search_location(""), "Canada")
        self.assertEqual(
            resolve_search_location("Location: Berlin, Germany"),
            "Berlin, Germany",
        )


class AcceptCandidateTests(unittest.TestCase):
    def test_rejects_wrong_surname_even_with_query_echo(self) -> None:
        # Rejected before HTTP via slug surname conflict.
        accepted = accept_linkedin_candidate(
            LinkedInCandidate(
                url="https://www.linkedin.com/in/saumya-singh-432465316",
                title="Saumya Banker - Donna",
                snippet="Founder at Donna",
                source="brave",
            ),
            "Saumya Banker",
            company_name="Donna",
            location="Canada",
        )
        self.assertIsNone(accepted)

    @patch("founders.validate_linkedin_profile", return_value="inconclusive")
    def test_rejects_inconclusive_even_with_brave_corroboration(
        self, _mock_validate: object
    ) -> None:
        # Dead LinkedIn slugs often still appear in Brave; HTTP must confirm.
        accepted = accept_linkedin_candidate(
            LinkedInCandidate(
                url="https://www.linkedin.com/in/jakob-sol-strozberg-a30abb1a4",
                title="Jakob Sol Strozberg - Co-Founding Engineer/CTO - Viable AI",
                snippet="Hi! My name is Jake",
                source="brave",
            ),
            "Jake Sol-Strozberg",
            company_name="Donna",
            location="Canada",
        )
        self.assertIsNone(accepted)

    @patch("founders.validate_linkedin_profile", return_value="valid")
    def test_accepts_http_validated_profile(self, _mock_validate: object) -> None:
        accepted = accept_linkedin_candidate(
            LinkedInCandidate(
                url="https://ca.linkedin.com/in/saumyabanker",
                title="Saumya B. - Co-founder/CEO, Donna AI",
                snippet="",
                source="brave",
            ),
            "Saumya Banker",
            company_name="Donna",
            location="Canada",
        )
        self.assertEqual(accepted, "https://www.linkedin.com/in/saumyabanker")


if __name__ == "__main__":
    unittest.main()
