"""Unit tests for Principle 8 name matching, geo normalization, and date intervals."""

from __future__ import annotations

from datetime import date
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PartialDate
from src.normalize import (
    GeoNormalizer,
    NameMatcher,
    compare_dob_against_hit_dobs,
    fold_text,
    is_incorporated_after,
    is_strictly_before_date,
    is_younger_than,
    levenshtein,
    parse_partial_date,
)

SAMPLE_VARIANTS = {
    "groups": [
        ["muhammad", "mohammed", "mohammad", "mohamed", "mohd"],
        ["yusuf", "yousef", "josef", "joseph", "joe"],
        ["joseph", "joe", "jose", "giuseppe", "jozef"],
        ["anthony", "antony", "tony", "antonio"],
        ["william", "will", "bill", "billy", "liam"],
    ],
    "company_legal_forms": ["ab", "ag", "gmbh", "inc", "llc", "ltd", "plc", "sa", "sarl", "se", "sl"],
}


class TestNormalize(unittest.TestCase):
    def setUp(self) -> None:
        self.nm = NameMatcher(variants_data=SAMPLE_VARIANTS)
        self.geo = GeoNormalizer()

    def test_pure_levenshtein_transposition_is_two_edits(self) -> None:
        # Swapping two adjacent letters ('john' vs 'jonh') is 2 edits in pure Levenshtein.
        self.assertEqual(levenshtein("john", "jonh"), 2)
        # Length 4 -> max 1 edit allowed -> must NOT match
        self.assertFalse(self.nm.parts_match("john", "jonh"))
        # Length 7 ('morillo' vs 'moriilo' = 1, 'mroillo' = 2) -> max 2 edits -> matches
        self.assertEqual(levenshtein("morillo", "mroillo"), 2)
        self.assertTrue(self.nm.parts_match("morillo", "mroillo"))
        # Length 3 (< 4) -> must be identical
        self.assertFalse(self.nm.parts_match("ali", "aly"))
        self.assertTrue(self.nm.parts_match("ali", "ali"))

    def test_multi_group_variant_membership(self) -> None:
        # "joseph"/"joe" appear in BOTH group 1 and group 2
        self.assertTrue(self.nm.parts_match("yusuf", "joseph"))
        self.assertTrue(self.nm.parts_match("joseph", "giuseppe"))
        self.assertTrue(self.nm.parts_match("joe", "jose"))

    def test_concatenated_split_counts_as_one_part(self) -> None:
        # "Abdulsalam" <-> "Abdul Salam" covers both tokens and counts as 1 matched part
        self.assertTrue(self.nm.names_agree("Abdulsalam", "Abdul Salam"))
        self.assertTrue(self.nm.names_agree("Abdulsalam Khan", "Abdul Salam Khan"))
        # Must NOT double-count Abdul+Salam as 2 matches when surname differs!
        self.assertFalse(self.nm.names_agree("Abdulsalam Khan", "Abdul Salam Smith"))

    def test_unicode_transliteration_non_nfkd(self) -> None:
        self.assertEqual(fold_text("Paweł Łukasz"), "pawel lukasz")
        self.assertEqual(fold_text("Østergaard"), "ostergaard")
        self.assertEqual(fold_text("Đorđević"), "dordevic")
        self.assertEqual(fold_text("Straße"), "strasse")
        self.assertTrue(self.nm.names_agree("Paweł Kowalski", "Pawel Kowalski"))

    def test_brackets_initials_placeholders_and_legal_forms(self) -> None:
        self.assertEqual(
            self.nm.extract_name_parts("Anna K. Petrova (AKA) [PEP]"),
            ["anna", "petrova"],
        )
        self.assertEqual(
            self.nm.extract_name_parts("NADEEM UNKNOWN 0000 N/A TEST XXX NONE"),
            ["nadeem"],
        )
        self.assertEqual(
            self.nm.extract_name_parts("Pescanova, S.A. (AKA)", is_company=True),
            ["pescanova"],
        )
        self.assertEqual(
            self.nm.extract_name_parts("Stadtwerke München GmbH", is_company=True),
            ["stadtwerke", "munchen"],
        )
        self.assertEqual(
            self.nm.extract_name_parts("N26 Bank SE", is_company=True),
            ["n26", "bank"],
        )
        self.assertTrue(self.nm.names_agree("Pescanova S.A.", "Pescanova, SA", is_company=True))
        self.assertTrue(self.nm.names_agree("Li Wei", "Wei Li"))

    def test_geo_territories_and_non_countries(self) -> None:
        self.assertEqual(self.geo.normalize_structured_country("Puerto Rico"), "PR")
        self.assertNotEqual(self.geo.normalize_structured_country("Puerto Rico"), "US")
        self.assertEqual(self.geo.normalize_structured_country("Northern Mariana Islands"), "MP")
        self.assertEqual(self.geo.normalize_structured_country("British Virgin Islands"), "VG")
        self.assertEqual(self.geo.normalize_structured_country("Jersey"), "JE")
        self.assertEqual(self.geo.normalize_structured_country("Macao Special Administrative Region of China"), "MO")
        self.assertEqual(self.geo.normalize_structured_country("Hong Kong"), "HK")
        self.assertEqual(self.geo.normalize_structured_country("Taiwan"), "TW")
        self.assertEqual(self.geo.normalize_structured_country("Kosovo"), "XK")
        self.assertEqual(self.geo.normalize_structured_country("Palestina"), "PS")
        self.assertIsNone(self.geo.normalize_structured_country("International"))
        self.assertIsNone(self.geo.normalize_structured_country("Europe"))

        self.assertEqual(self.geo.normalize_structured_country("Belgrade, Serbia"), "RS")
        self.assertEqual(
            self.geo.extract_countries_from_address("10 Downing Street, London, United Kingdom [PEP]"),
            ["GB"],
        )
        self.assertEqual(
            self.geo.extract_countries_from_pob("Zamora, Spain; Madrid, Spain"),
            ["ES"],
        )

    def test_prose_country_extractor_guards(self) -> None:
        # Must NOT treat 2-letter prepositions/pronouns ("in", "at", "it", "us", "no", "is") as ISO codes
        self.assertEqual(
            self.geo.extract_countries_from_prose("It is at no risk to us in this city."),
            [],
        )
        # Must NOT mistake US states for countries
        self.assertEqual(
            self.geo.extract_countries_from_prose("He lived in Newark, New Jersey and Santa Fe, New Mexico."),
            [],
        )
        self.assertEqual(
            self.geo.extract_countries_from_prose("Born in Atlanta, Georgia, United States."),
            ["US"],
        )
        self.assertEqual(
            self.geo.extract_countries_from_prose("She worked as a nurse in Denmark and was British."),
            ["DK", "GB"],
        )
        # Name masking prevents "Jordan Andrew Chartrand" or "Benin City" or "English" false tags
        self.assertEqual(
            self.geo.extract_countries_from_prose(
                "During August 2012, Jordan Andrew Chartrand, 29, was named on a watchlist in Poland.",
                mask_names=["Jordan Andrew Chartrand"],
            ),
            ["PL"],
        )
        self.assertEqual(
            self.geo.extract_countries_from_prose("Customer lives in Benin City and speaks English."),
            [],
        )

    def test_dates_and_intervals(self) -> None:
        # Bank day-first vs hit note US month-first
        self.assertEqual(parse_partial_date("14/03/1988", day_first=True), PartialDate(1988, 3, 14, "14/03/1988"))
        self.assertEqual(parse_partial_date("03/01/1965", day_first=True), PartialDate(1965, 1, 3, "03/01/1965"))
        self.assertEqual(parse_partial_date("6/20/1955", day_first=False), PartialDate(1955, 6, 20, "6/20/1955"))
        self.assertEqual(parse_partial_date("2022-03").latest, date(2022, 3, 31))
        self.assertEqual(parse_partial_date("2020-02").latest, date(2020, 2, 29))

        # Agreement vs off-by-1 inconclusive vs contradiction
        c = PartialDate(1975, 6, 14)
        self.assertEqual(compare_dob_against_hit_dobs(c, [PartialDate(1975)]), "AGREE")
        self.assertEqual(compare_dob_against_hit_dobs(c, [PartialDate(1974), PartialDate(1975)]), "AGREE")
        self.assertEqual(compare_dob_against_hit_dobs(c, [PartialDate(1974)]), "INCONCLUSIVE")
        self.assertEqual(compare_dob_against_hit_dobs(c, [PartialDate(1975, 7, 14)]), "CONTRADICT")
        self.assertEqual(compare_dob_against_hit_dobs(c, [PartialDate(1973)]), "CONTRADICT")

        # Temporal impossibility & deceased boundaries
        self.assertTrue(is_younger_than(PartialDate(2004), PartialDate(2019, 12, 31), min_age=16))
        self.assertFalse(is_younger_than(PartialDate(2004, 1, 1), PartialDate(2020, 1, 1), min_age=16))
        self.assertTrue(is_incorporated_after(PartialDate(2015, 1, 1), PartialDate(2014, 12, 31)))
        self.assertTrue(is_strictly_before_date(PartialDate(2022, 3), date(2022, 4, 1)))
        self.assertFalse(is_strictly_before_date(PartialDate(2022, 3), date(2022, 3, 15)))


if __name__ == "__main__":
    unittest.main()
