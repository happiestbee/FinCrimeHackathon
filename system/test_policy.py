"""Unit and synthetic integration tests for POLICY.md compliance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from engine import solve_dataset
from normalizer import (
    NameMatcher,
    PartialDate,
    date_strictly_before,
    dates_agree,
    dob_contradicts,
    max_age_at,
    normalize_country,
    parse_date,
)
from text_extractor import (
    _extract_age_and_ref_year_regex,
    _extract_kyc_note_llm,
    extract_countries_in_snippet,
    extract_dated_reports,
    extract_note_facts,
)


DEV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dev"))
_DEFAULT_VARIANTS = {
    "groups": [
        ["muhammad", "mohammed", "mohammad", "mohamed"],
        ["ali", "aly"],
        ["mark", "marc", "marco"],
        ["markus", "marcus", "marcos"],
        ["thomas", "tom", "tommy"],
        ["yusuf", "yousef", "josef", "joseph", "joe"],
    ],
    "company_legal_forms": ["ab", "ag", "gmbh", "inc", "limited", "llc", "ltd", "plc", "sa", "se", "sl"],
}
_CANDIDATE_VARIANTS = os.path.join(DEV_DIR, "name_variants.json")
if os.path.exists(_CANDIDATE_VARIANTS):
    VARIANTS_PATH = _CANDIDATE_VARIANTS
else:
    VARIANTS_PATH = os.path.join(tempfile.gettempdir(), "screening_test_name_variants.json")
    with open(VARIANTS_PATH, "w", encoding="utf-8") as _vf:
        json.dump(_DEFAULT_VARIANTS, _vf)


class TestNormalizerAndPolicy(unittest.TestCase):
    def setUp(self):
        self.matcher = NameMatcher(VARIANTS_PATH)

    def test_mark_vs_markus_strict(self):
        # Policy Principle 8: mark (len 4) vs markus (len 6) -> dist 2 > 1, not in same variant group
        self.assertFalse(self.matcher.names_agree("MARKUS GERHARD MUELLER", "Mark Mueller"))

    def test_joined_and_split_parts(self):
        self.assertTrue(self.matcher.names_agree("Abdulsalam Khan", "Abdul Salam Khan"))
        self.assertTrue(self.matcher.names_agree("Abdulsalam", "Abdul Salam"))
        self.assertTrue(self.matcher.names_agree("ABDULSALAM TARIQ", "Abdul Salam", require_min_two_a=True))
        self.assertFalse(self.matcher.names_agree("Abdulsalam Tariq", "Abdul Salam Hassan", require_min_two_a=True))
        self.assertTrue(self.matcher.names_agree("MEHMET YILMAZ", "Mehmet Yılmaz", require_min_two_a=True))
        self.assertTrue(self.matcher.names_agree("Bluegrass Craft Coffee Roasters LLC", "Blue Grass, LLC", is_company=True))
        self.assertFalse(self.matcher.names_agree("Starling Logistics Ltd", "ST Logistics (UK) Ltd.", is_company=True))

    def test_variants_and_edit_distance(self):
        self.assertTrue(self.matcher.names_agree("Muhammad Ali", "Mohamed Aly"))
        self.assertTrue(self.matcher.names_agree("Martin", "Marton"))  # len 6, dist 1
        self.assertFalse(self.matcher.names_agree("Tom Smith", "Tim Smith"))  # len 3 < 4, must match exactly

    def test_placeholders_and_company_forms(self):
        self.assertEqual(self.matcher.extract_parts("PAWEŁ N/A N/A"), ["pawel"])
        self.assertEqual(self.matcher.extract_parts("AZAM NA NA"), ["azam"])
        self.assertEqual(self.matcher.extract_parts("BUSTAMANTE 0000"), ["bustamante"])
        self.assertTrue(
            self.matcher.names_agree("N26 Bank AG Limited", "N26 Bank SE", is_company=True)
        )
        # Fewer than 2 real parts for individual customer -> blocked when require_min_two_a=True
        self.assertFalse(
            self.matcher.names_agree("UNKNOWN NADEEM", "Nadeem", is_company=False, require_min_two_a=True)
        )

    def test_dob_one_year_buffer(self):
        cust = parse_date("1985-05-01")
        hit_plus1 = parse_date("1986")
        hit_plus2 = parse_date("1987")
        assert cust and hit_plus1 and hit_plus2
        self.assertFalse(dates_agree(cust, hit_plus1))
        self.assertFalse(dob_contradicts(cust, hit_plus1))
        self.assertTrue(dob_contradicts(cust, hit_plus2))

    def test_temporal_and_deceased_bounds(self):
        self.assertEqual(max_age_at(PartialDate(2007, 8, 21), PartialDate(2020, 1, 8)), 12)
        self.assertEqual(max_age_at(PartialDate(2004, 1, 8), PartialDate(2020, 1, 8)), 16)
        self.assertTrue(date_strictly_before(PartialDate(2022, 3, None), PartialDate(2026, 11, 14)))
        self.assertFalse(date_strictly_before(PartialDate(2026, 11, None), PartialDate(2026, 11, 14)))

    def test_countries_and_territories(self):
        self.assertEqual(normalize_country("GB"), "united kingdom")
        self.assertEqual(normalize_country("British"), "united kingdom")
        self.assertEqual(normalize_country("Côte d'Ivoire"), "cote d'ivoire")
        self.assertEqual(normalize_country("Cote D'Ivoire"), "cote d'ivoire")
        self.assertIsNone(normalize_country("International"))
        self.assertIsNone(normalize_country("Europe"))
        self.assertEqual(normalize_country("Puerto Rico"), "puerto rico")

    def test_rejected_kyc_note_ignored(self):
        nf = extract_note_facts({
            "note_id": "NOTE-999999",
            "channel": "kyc",
            "created_at": "2025-01-01",
            "text": "Sumsub rejected the passport for JOHN DOE, DOB 1990-01-01 from Germany.",
        })
        self.assertIsNotNone(nf.kyc)
        self.assertFalse(nf.kyc.approved)


class TestSyntheticRarePolicyCases(unittest.TestCase):
    def _run_synthetic(self, tables):
        tmp = tempfile.mkdtemp(prefix="scr_test_")
        try:
            shutil.copy(VARIANTS_PATH, os.path.join(tmp, "name_variants.json"))
            for name in ("alerts", "hits", "customers", "accounts", "onboarding", "kyc_checks", "notes", "search_results"):
                rows = tables.get(name, [])
                with open(os.path.join(tmp, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            return {r["hit_id"]: r for r in solve_dataset(tmp)}
        finally:
            shutil.rmtree(tmp)

    def test_ambiguous_conflicting_decisions(self):
        res = self._run_synthetic({
            "alerts": [{
                "alert_id": "A-000001", "screened_at": "2026-11-01",
                "customer": {"name": "John Smith", "entity_type": "Individual", "dob": "1980", "country": "United Kingdom", "city": None},
                "account_ref": None, "login_handle": None, "email": None, "hit_ids": ["H-000001"],
            }],
            "hits": [{
                "hit_id": "H-000001", "alert_id": "A-000001", "match_score": 100,
                "entity_type": "Individual", "name": "John Smith", "aliases": [],
                "dobs": ["1980-05-10"], "countries": ["United Kingdom"],
            }],
            "customers": [
                {
                    "record_id": "CUS-100001", "entity_type": "Individual", "full_name": "John Smith",
                    "dob": "1980-05-10", "nationality": "United Kingdom", "residence_country": "United Kingdom",
                    "phone": "+44 1111111", "email": "a@example.com",
                },
                {
                    "record_id": "CUS-100002", "entity_type": "Individual", "full_name": "John Smith",
                    "dob": "1980-09-20", "nationality": "United Kingdom", "residence_country": "United Kingdom",
                    "phone": "+44 2222222", "email": "b@example.com",
                },
            ],
        })
        r = res["H-000001"]
        self.assertEqual(r["decision"], "REVIEW")
        self.assertEqual(r["reason_code"], "AMBIGUOUS_CUSTOMER_RECORD")
        self.assertEqual(r["customer_record_id"], "AMBIGUOUS")

    def test_country_diff_with_name_and_dob_agree_is_review(self):
        res = self._run_synthetic({
            "alerts": [{
                "alert_id": "A-000002", "screened_at": "2026-11-01",
                "customer": {"name": "Maria Silva", "entity_type": "Individual", "dob": "1982-03-04", "country": "Brazil", "city": None},
                "account_ref": None, "login_handle": None, "email": None, "hit_ids": ["H-000002"],
            }],
            "hits": [{
                "hit_id": "H-000002", "alert_id": "A-000002", "match_score": 100,
                "entity_type": "Individual", "name": "Maria Silva", "aliases": [],
                "dobs": ["1982-03-04"], "countries": ["Germany"],
            }],
            "customers": [{
                "record_id": "CUS-200001", "entity_type": "Individual", "full_name": "Maria Silva",
                "dob": "1982-03-04", "nationality": "Brazil", "residence_country": "Brazil",
                "phone": "+55 111", "email": "m@example.com",
            }],
        })
        r = res["H-000002"]
        self.assertEqual(r["decision"], "REVIEW")
        self.assertEqual(r["reason_code"], "INSUFFICIENT_DATA")

    def test_deceased_and_entity_type_mismatch(self):
        res = self._run_synthetic({
            "alerts": [
                {
                    "alert_id": "A-000003", "screened_at": "2026-11-10",
                    "customer": {"name": "Jane Doe", "entity_type": "Individual", "dob": "1970-01-01", "country": "Canada", "city": None},
                    "account_ref": None, "login_handle": None, "email": None, "hit_ids": ["H-000003"],
                },
                {
                    "alert_id": "A-000004", "screened_at": "2026-11-10",
                    "customer": {"name": "Acme Ltd", "entity_type": "Company", "dob": "2010-01-01", "country": "Canada", "city": None},
                    "account_ref": None, "login_handle": None, "email": None, "hit_ids": ["H-000004"],
                },
            ],
            "hits": [
                {
                    "hit_id": "H-000003", "alert_id": "A-000003", "match_score": 100,
                    "entity_type": "Individual", "name": "Jane Doe", "aliases": [],
                    "dobs": ["1970-01-01"], "deceased_date": "2024-05", "countries": ["Canada"],
                },
                {
                    "hit_id": "H-000004", "alert_id": "A-000004", "match_score": 100,
                    "entity_type": "Individual", "name": "Acme Doe", "aliases": [],
                    "dobs": [], "countries": ["Canada"],
                },
            ],
            "customers": [
                {
                    "record_id": "CUS-300001", "entity_type": "Individual", "full_name": "Jane Doe",
                    "dob": "1970-01-01", "nationality": "Canada", "residence_country": "Canada",
                },
                {
                    "record_id": "CUS-300002", "entity_type": "Company", "full_name": "Acme Ltd",
                    "dob": "2010-01-01", "nationality": "Canada", "residence_country": "Canada",
                },
            ],
        })
        self.assertEqual(res["H-000003"]["reason_code"], "DECEASED")
        self.assertEqual(res["H-000004"]["reason_code"], "ENTITY_TYPE_MISMATCH")

    def test_placeholder_individual_cannot_confirm_on_name(self):
        res = self._run_synthetic({
            "alerts": [{
                "alert_id": "A-000005", "screened_at": "2026-11-10",
                "customer": {"name": "UNKNOWN NADEEM", "entity_type": "Individual", "dob": "1988-02-02", "country": "Pakistan", "city": None},
                "account_ref": None, "login_handle": None, "email": None, "hit_ids": ["H-000005"],
            }],
            "hits": [{
                "hit_id": "H-000005", "alert_id": "A-000005", "match_score": 100,
                "entity_type": "Individual", "name": "Nadeem Khan", "aliases": [],
                "dobs": ["1988-02-02"], "countries": ["Pakistan"],
            }],
            "customers": [{
                "record_id": "CUS-500001", "entity_type": "Individual", "full_name": "Nadeem Unknown",
                "dob": "1988-02-02", "nationality": "Pakistan", "residence_country": "Pakistan",
            }],
        })
        self.assertEqual(res["H-000005"]["decision"], "REVIEW")
        self.assertEqual(res["H-000005"]["reason_code"], "INSUFFICIENT_DATA")

    def test_dated_report_multi_semicolon_and_snippet_country_homonym(self):
        dts = extract_dated_reports("According to Regulator A; Regulator B; October 15, 2018: Nordic Cloud GmbH was fined.")
        self.assertEqual(len(dts), 1)
        self.assertEqual(dts[0].format(), "2018-10-15")

        countries = extract_countries_in_snippet(
            "During August 2012, Jordan Miller, 52, was named on a watchlist in Poland within a regulator's notice.",
            title="Jordan Miller - report",
            hit_names=["Jordan Miller"],
        )
        self.assertEqual(countries, {"poland"})

        self.assertIsNone(
            _extract_age_and_ref_year_regex("Federal Register notice: Vol. 74, No. 11 under Section (25) in March 2009")
        )

    def test_kyc_llm_dob_cannot_equal_checked_at(self):
        class FakeLLM:
            enabled = True
            def extract_json(self, tag, sys_p, txt):
                return {
                    "is_kyc": True,
                    "approved": True,
                    "verified_name": "Acme Ltd",
                    "verified_dob": "2024-03-01",
                    "document_country": "United Kingdom",
                    "checked_at": "2024-03-01",
                }

        res = _extract_kyc_note_llm(
            {
                "note_id": "NOTE-999998",
                "created_at": "2024-03-01",
                "channel": "kyc",
                "text": "Sumsub verified Acme Ltd from United Kingdom on 2024-03-01.",
            },
            FakeLLM(),
        )
        self.assertIsNotNone(res)
        self.assertIsNone(res.verified_dob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
