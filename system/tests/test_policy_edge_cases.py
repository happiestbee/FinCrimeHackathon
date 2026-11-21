"""Unit tests covering all 11 reason codes, rare POLICY.md edge cases, and LLM/verifier gates."""

from __future__ import annotations

from datetime import date
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import DataStore
from src.llm_client import OpenRouterClient
from src.models import HitProfile, PartialDate, PersonProfile, ResolvedAlert
from src.normalize import GeoNormalizer, NameMatcher
from src.policy_engine import PolicyEngine
from src.search_shield import (
    SearchShield,
    extract_age_and_year_from_profile_note,
    extract_age_and_year_from_prose,
)
from src.verifier import ExtractionVerifier
from src.watchlist_resolver import WatchlistResolver


VARIANTS = {
    "groups": [["anthony", "tony"], ["muhammad", "mohammed"]],
    "company_legal_forms": ["ltd", "llc", "gmbh", "sa"],
}


def make_person(
    cid: str = "CUS-000001",
    etype: str = "Individual",
    vname: str | None = "Anthony Smith",
    dname: str = "Anthony Smith",
    dob: PartialDate | None = PartialDate(1985, 5, 10),
    countries: dict[str, list[str]] | None = None,
    unresolved: bool = False,
) -> PersonProfile:
    return PersonProfile(
        cluster_id=cid,
        record_ids=[] if unresolved else [cid],
        canonical_record_id=None if unresolved else cid,
        entity_type=etype,
        verified_name=vname,
        verified_name_source_id="KYC-000001" if vname else None,
        declared_names=[(dname, cid)],
        effective_name=vname or dname,
        dob=dob,
        dob_source_id="KYC-000001" if dob else None,
        dob_is_verified=bool(vname),
        country_sources=countries if countries is not None else {"GB": [cid]},
        is_unresolved=unresolved,
    )


def make_hit(
    hid: str = "H-000001",
    aid: str = "A-000001",
    etype: str = "Individual",
    name: str = "Anthony Smith",
    dobs: list[PartialDate] | None = None,
    countries: set[str] | None = None,
    deceased_date: PartialDate | None = None,
    life_status: str | None = None,
    earliest_report: PartialDate | None = None,
    dob_source: str | None = "dobs_field",
) -> HitProfile:
    return HitProfile(
        hit_id=hid,
        alert_id=aid,
        entity_type=etype,
        primary_name=name,
        all_names=[name],
        dobs=dobs if dobs is not None else [],
        dob_source=dob_source if dobs else None,
        dob_evidence_ids=[hid] if dobs else [],
        countries=countries if countries is not None else {"GB"},
        deceased_date=deceased_date,
        life_status=life_status,
        earliest_report_date=earliest_report,
    )


class TestPolicyEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.nm = NameMatcher(variants_data=VARIANTS)
        self.geo = GeoNormalizer()
        self.engine = PolicyEngine(self.nm, self.geo)
        self.screened = date(2026, 11, 15)

    def _eval(self, person: PersonProfile, hit: HitProfile, status: str = "LINKED"):
        ra = ResolvedAlert(
            alert_id=hit.alert_id,
            resolution_status=status,
            customer_record_id=person.canonical_record_id,
            candidates=[person],
        )
        return self.engine.evaluate_hit(ra, hit, self.screened)

    def test_entity_type_mismatch(self) -> None:
        out = self._eval(
            make_person(etype="Company", vname="Acme Ltd", dname="Acme Ltd", dob=PartialDate(2010)),
            make_hit(etype="Individual", name="Acme", dobs=[PartialDate(1980)]),
        )
        self.assertEqual(out.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out.reason_code, "ENTITY_TYPE_MISMATCH")

    def test_deceased_with_and_without_date(self) -> None:
        out_clear = self._eval(
            make_person(),
            make_hit(deceased_date=PartialDate(2022, 3), life_status="Deceased"),
        )
        self.assertEqual(out_clear.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_clear.reason_code, "DECEASED")

        # Deceased status WITHOUT deceased_date must NOT clear!
        out_rev = self._eval(
            make_person(),
            make_hit(deceased_date=None, life_status="Deceased"),
        )
        self.assertEqual(out_rev.decision, "REVIEW")
        self.assertEqual(out_rev.reason_code, "INSUFFICIENT_DATA")

    def test_temporal_impossibility_boundary_age_16_and_company(self) -> None:
        # Born 2004-06-01, report 2020-05-31 -> age 15 -> clears!
        out_15 = self._eval(
            make_person(dob=PartialDate(2004, 6, 1)),
            make_hit(earliest_report=PartialDate(2020, 5, 31)),
        )
        self.assertEqual(out_15.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_15.reason_code, "TEMPORAL_IMPOSSIBILITY")

        # Born 2004-06-01, report 2020-06-01 -> age 16 -> does NOT clear!
        out_16 = self._eval(
            make_person(dob=PartialDate(2004, 6, 1)),
            make_hit(earliest_report=PartialDate(2020, 6, 1)),
        )
        self.assertEqual(out_16.decision, "REVIEW")

        # Company incorporated 2020 after report 2018 -> clears!
        out_comp = self._eval(
            make_person(etype="Company", vname="Acme Ltd", dname="Acme Ltd", dob=PartialDate(2020)),
            make_hit(etype="Company", name="Acme Ltd", earliest_report=PartialDate(2018, 9, 10)),
        )
        self.assertEqual(out_comp.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_comp.reason_code, "TEMPORAL_IMPOSSIBILITY")

    def test_dob_mismatch_vs_off_by_one_vs_multi_dob(self) -> None:
        out_mis = self._eval(
            make_person(dob=PartialDate(1985, 5, 10)),
            make_hit(dobs=[PartialDate(1980)]),
        )
        self.assertEqual(out_mis.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_mis.reason_code, "DOB_MISMATCH")

        # Off by 1 year (1985 vs 1986) -> neither agrees nor contradicts -> REVIEW!
        out_off1 = self._eval(
            make_person(dob=PartialDate(1985, 5, 10)),
            make_hit(dobs=[PartialDate(1986)]),
        )
        self.assertEqual(out_off1.decision, "REVIEW")
        self.assertEqual(out_off1.reason_code, "INSUFFICIENT_DATA")

        # Multi-DOB where one agrees (1985) and another differs (1980) -> SAME_ENTITY!
        out_multi = self._eval(
            make_person(dob=PartialDate(1985, 5, 10)),
            make_hit(dobs=[PartialDate(1980), PartialDate(1985)]),
        )
        self.assertEqual(out_multi.decision, "SAME_ENTITY")

    def test_location_mismatch_suppressed_when_name_and_dob_agree_or_unresolved(self) -> None:
        # Name & DOB agree, countries differ -> MUST BE REVIEW, never LOCATION_MISMATCH!
        out_rev = self._eval(
            make_person(countries={"GB": ["CUS-000001"]}),
            make_hit(dobs=[PartialDate(1985, 5, 10)], countries={"DE"}),
        )
        self.assertEqual(out_rev.decision, "REVIEW")
        self.assertEqual(out_rev.reason_code, "INSUFFICIENT_DATA")

        # Unresolved customer -> MUST NOT fire LOCATION_MISMATCH!
        out_unres = self._eval(
            make_person(vname=None, dob=None, countries={"GB": []}, unresolved=True),
            make_hit(countries={"DE"}),
            status="UNRESOLVED",
        )
        self.assertEqual(out_unres.decision, "REVIEW")

        # Different DOB inconclusive or missing, name agrees, countries disjoint -> LOCATION_MISMATCH!
        out_loc = self._eval(
            make_person(countries={"GB": ["CUS-000001"]}),
            make_hit(dobs=[], countries={"DE"}),
        )
        self.assertEqual(out_loc.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_loc.reason_code, "LOCATION_MISMATCH")

    def test_name_mismatch_verified_and_placeholder_guard(self) -> None:
        out_nm = self._eval(
            make_person(vname="David Brown", dname="Anthony Smith"),
            make_hit(dobs=[], countries={"GB"}),
        )
        self.assertEqual(out_nm.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_nm.reason_code, "NAME_MISMATCH_VERIFIED")

        # Placeholder individual ("NADEEM UNKNOWN" -> 1 real part):
        # Cannot confirm SAME_ENTITY nor clear on NAME_MISMATCH_VERIFIED!
        out_ph = self._eval(
            make_person(vname="NADEEM UNKNOWN", dname="NADEEM UNKNOWN", dob=PartialDate(1985, 5, 10)),
            make_hit(name="Nadeem Khan", dobs=[PartialDate(1985)], countries={"GB"}),
        )
        self.assertEqual(out_ph.decision, "REVIEW")
        self.assertEqual(out_ph.reason_code, "INSUFFICIENT_DATA")

        # Placeholder individual CAN still clear on DOB_MISMATCH!
        out_ph_dob = self._eval(
            make_person(vname="NADEEM UNKNOWN", dname="NADEEM UNKNOWN", dob=PartialDate(1985, 5, 10)),
            make_hit(name="Nadeem Khan", dobs=[PartialDate(1960)], countries={"GB"}),
        )
        self.assertEqual(out_ph_dob.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_ph_dob.reason_code, "DOB_MISMATCH")

    def test_same_entity_reason_codes_and_company_rule(self) -> None:
        # NAME_DOB_LOCATION_MATCH
        out1 = self._eval(
            make_person(countries={"GB": ["CUS-000001"]}),
            make_hit(dobs=[PartialDate(1985)], countries={"GB"}),
        )
        self.assertEqual(out1.decision, "SAME_ENTITY")
        self.assertEqual(out1.reason_code, "NAME_DOB_LOCATION_MATCH")

        # NAME_DOB_MATCH (hit has no country)
        out2 = self._eval(
            make_person(countries={"GB": ["CUS-000001"]}),
            make_hit(dobs=[PartialDate(1985)], countries=set()),
        )
        self.assertEqual(out2.decision, "SAME_ENTITY")
        self.assertEqual(out2.reason_code, "NAME_DOB_MATCH")

        # Company NAME_LOCATION_MATCH (even when hit.dobs is empty!)
        out3 = self._eval(
            make_person(etype="Company", vname="Pescanova S.A.", dname="Pescanova S.A.", dob=PartialDate(1960)),
            make_hit(etype="Company", name="Pescanova, Sociedad Anónima", dobs=[], countries={"ES"}),
        )
        # Disjoint countries above? Wait: customer was GB -> set ES!
        out3 = self._eval(
            make_person(etype="Company", vname="Pescanova S.A.", dname="Pescanova S.A.", dob=PartialDate(1960), countries={"ES": ["CUS-000001"]}),
            make_hit(etype="Company", name="Pescanova, Sociedad Anónima", dobs=[], countries={"ES"}),
        )
        self.assertEqual(out3.decision, "SAME_ENTITY")
        self.assertEqual(out3.reason_code, "NAME_LOCATION_MATCH")

    def test_ambiguous_customer_aggregation(self) -> None:
        p1 = make_person(cid="CUS-000001", dob=PartialDate(1990, 1, 1), countries={"PK": ["CUS-000001"]})
        p2 = make_person(cid="CUS-000002", dob=PartialDate(1992, 1, 1), countries={"PK": ["CUS-000002"]})
        hit = make_hit(dobs=[PartialDate(1960)], countries={"PK"})

        # Both clear on DOB_MISMATCH -> DIFFERENT_ENTITY + customer_record_id="AMBIGUOUS"
        ra_unan = ResolvedAlert("A-000001", "AMBIGUOUS", "AMBIGUOUS", [p1, p2])
        out_unan = self.engine.evaluate_hit(ra_unan, hit, self.screened)
        self.assertEqual(out_unan.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out_unan.reason_code, "DOB_MISMATCH")
        self.assertEqual(out_unan.customer_record_id, "AMBIGUOUS")

        # Split: p1 matches (1985), p2 clears (1992) -> REVIEW + AMBIGUOUS_CUSTOMER_RECORD
        p1_match = make_person(cid="CUS-000001", dob=PartialDate(1985, 1, 1), countries={"PK": ["CUS-000001"]})
        hit_1985 = make_hit(dobs=[PartialDate(1985)], countries={"PK"})
        ra_split = ResolvedAlert("A-000001", "AMBIGUOUS", "AMBIGUOUS", [p1_match, p2])
        out_split = self.engine.evaluate_hit(ra_split, hit_1985, self.screened)
        self.assertEqual(out_split.decision, "REVIEW")
        self.assertEqual(out_split.reason_code, "AMBIGUOUS_CUSTOMER_RECORD")
        self.assertEqual(out_split.customer_record_id, "AMBIGUOUS")

    def test_company_ignores_profile_notes_establishment_date_and_non_according_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "name_variants.json"), "w", encoding="utf-8") as fh:
                json.dump(VARIANTS, fh)
            for fn in (
                "alerts.jsonl", "customers.jsonl", "accounts.jsonl",
                "onboarding.jsonl", "kyc_checks.jsonl", "notes.jsonl", "search_results.jsonl",
            ):
                open(os.path.join(tmp, fn), "w").close()
            with open(os.path.join(tmp, "hits.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "hit_id": "H-C00001",
                    "alert_id": "A-C00001",
                    "entity_type": "Company",
                    "name": "Acme Corp",
                    "first_name": None, "middle_name": None, "last_name": None,
                    "best_matching_name": None, "aliases": [],
                    "dobs": [],
                    "deceased_date": None, "life_status": None,
                    "countries": ["United States"], "nationality": [], "citizenship": [],
                    "addresses": [], "place_of_birth": None,
                    "primary_category": "Enforcement", "categories": ["Enforcement"],
                    "profile_notes": "Acme Corp was established on January 1, 1990.\n\nLocal press from 1995 noted violations.",
                }) + "\n")
            store = DataStore(tmp)
            wr = WatchlistResolver(store, SearchShield(store))
            hp = wr.resolve_hit(store.hits[0])
            # Company must NOT extract establishment date from profile_notes; non-"According to" line must NOT count
            self.assertEqual(hp.dobs, [])
            self.assertIsNone(hp.earliest_report_date)

    def test_search_shield_rejects_wrong_category_wrong_country_and_non_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "name_variants.json"), "w", encoding="utf-8") as fh:
                json.dump(VARIANTS, fh)
            for fn in ("alerts.jsonl", "hits.jsonl", "customers.jsonl", "accounts.jsonl", "onboarding.jsonl", "kyc_checks.jsonl", "notes.jsonl"):
                open(os.path.join(tmp, fn), "w").close()
            with open(os.path.join(tmp, "search_results.jsonl"), "w", encoding="utf-8") as fh:
                rows = [
                    # Wrong listing phrase (PEP vs Sanctions)
                    {"result_id": "SR-1", "query": "john doe", "title": "John Doe", "snippet": "John Doe, 40, a politically exposed person in Germany, in March 2020."},
                    # Wrong country (France vs Germany)
                    {"result_id": "SR-2", "query": "john doe", "title": "John Doe", "snippet": "John Doe, 40, a sanctioned individual in France, in March 2020."},
                    # Non-listing hobby article
                    {"result_id": "SR-3", "query": "john doe", "title": "John Doe", "snippet": "John Doe, 40, a marathon runner in Germany, in March 2020."},
                    # Exact match (Sanctions + Germany)
                    {"result_id": "SR-4", "query": "john doe", "title": "John Doe", "snippet": "John Doe, 45, a sanctioned individual in Germany, in March 2019."},
                ]
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            store = DataStore(tmp)
            shield = SearchShield(store)
            dobs, ids, _ = shield.lookup_hit_dobs(["John Doe"], {"DE"}, "a sanctioned individual")
            self.assertEqual(ids, ["SR-4"])
            self.assertEqual([d.year for d in dobs], [1973, 1974])

    def test_verifier_and_openrouter_retries_and_cache(self) -> None:
        verifier = ExtractionVerifier(self.nm, self.geo)
        # Reject PEP family member's DOB (about_subject=False or wrong name in quote)
        bad_rel = {
            "about_subject": False,
            "subject_name_in_quote": "Maria Garcia",
            "birth_date": "1990-01-01",
            "age": None,
            "event_year": None,
            "verbatim_quote": "His daughter Maria Garcia was born in 1990.",
        }
        self.assertEqual(verifier.verify_hit_dob_extraction(bad_rel, "His daughter Maria Garcia was born in 1990.", ["Juan Garcia"]), [])

        # Reject hallucinated age not present in quote
        bad_num = {
            "about_subject": True,
            "subject_name_in_quote": "Juan Garcia",
            "birth_date": None,
            "age": 45,
            "event_year": 2019,
            "verbatim_quote": "Juan Garcia was sentenced in March 2019.",
        }
        self.assertEqual(verifier.verify_hit_dob_extraction(bad_num, "Juan Garcia was sentenced in March 2019.", ["Juan Garcia"]), [])

        # Accept valid quote and compute [1973, 1974] strictly in Python
        good = {
            "about_subject": True,
            "subject_name_in_quote": "Juan Garcia",
            "birth_date": None,
            "age": 45,
            "event_year": 2019,
            "verbatim_quote": "Juan Garcia, aged 45, was sentenced in March 2019.",
        }
        res = verifier.verify_hit_dob_extraction(good, "Juan Garcia, aged 45, was sentenced in March 2019.", ["Juan Garcia"])
        self.assertEqual([d.year for d in res], [1973, 1974])

        # OpenRouterClient retry on HTTP 200 content:null, then success + SHA-256 cache hit
        client = OpenRouterClient(api_key="test-key", max_retries=2)
        responses = [
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": json.dumps(good)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        ]

        class FakeResp:
            def __init__(self, body: dict) -> None:
                self._b = json.dumps(body).encode("utf-8")
            def getcode(self) -> int:
                return 200
            def read(self) -> bytes:
                return self._b
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        with patch("urllib.request.urlopen", side_effect=[FakeResp(r) for r in responses]):
            with patch("time.sleep"):
                out1 = client.extract_hit_first_para_dob("Juan Garcia, aged 45, was sentenced in March 2019.", ["Juan Garcia"])
                out2 = client.extract_hit_first_para_dob("Juan Garcia, aged 45, was sentenced in March 2019.", ["Juan Garcia"])
        self.assertEqual(out1, good)
        self.assertEqual(out2, good)
        self.assertEqual(client.stats["retries"], 1)
        self.assertEqual(client.stats["cache_hits"], 1)

    def test_red_team_hardening_guards(self) -> None:
        # 1. Strict profile_notes age extractor rejects Federal Register Vol. 74, No. 11 and legal clauses
        fed_reg = "Sanction: Export-Import Bank Act\nEntity: Paul Griffin\nDate imposed: 1/9/2009\nFederal Register notice: Vol. 74, No. 11 01/16/09"
        self.assertEqual(extract_age_and_year_from_profile_note(fed_reg), (None, 2009))
        self.assertEqual(
            extract_age_and_year_from_profile_note("Under Section (25) of the Act of 1998, John Smith was sanctioned."),
            (None, 1998),
        )
        self.assertEqual(
            extract_age_and_year_from_prose("The 18 members of the gang were charged in May 2019."),
            (None, 2019),
        )

        # 2. Ambiguous unanimous clearance prefers shared clearance reason code across all candidates
        p1 = make_person(cid="CUS-000001", vname="Anthony Smith", dob=PartialDate(1985, 5, 10), countries={"US": ["CUS-000001"]})
        p2 = make_person(cid="CUS-000002", vname="David Brown", dob=PartialDate(1985, 5, 10), countries={"US": ["CUS-000002"]})
        hit = make_hit(name="Anthony Smith", dobs=[], countries={"GB"})
        ra = ResolvedAlert("A-000001", "AMBIGUOUS", "AMBIGUOUS", [p1, p2])
        out = self.engine.evaluate_hit(ra, hit, self.screened)
        self.assertEqual(out.decision, "DIFFERENT_ENTITY")
        self.assertEqual(out.reason_code, "LOCATION_MISMATCH")
        self.assertEqual(out.customer_record_id, "AMBIGUOUS")

        # 3. Single-part alias in search shield must NOT pull unrelated multi-part queries
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "name_variants.json"), "w", encoding="utf-8") as fh:
                json.dump(VARIANTS, fh)
            for fn in ("alerts.jsonl", "hits.jsonl", "customers.jsonl", "accounts.jsonl", "onboarding.jsonl", "kyc_checks.jsonl", "notes.jsonl"):
                open(os.path.join(tmp, fn), "w").close()
            with open(os.path.join(tmp, "search_results.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "result_id": "SR-OTHER",
                    "query": "hassan ali",
                    "title": "Hassan Ali",
                    "snippet": "Hassan Ali, 55, a sanctioned individual in Egypt, in March 2019.",
                }) + "\n")
                fh.write(json.dumps({
                    "result_id": "SR-JORDAN",
                    "query": "jordan andrew chartrand",
                    "title": "Jordan Andrew Chartrand - profile",
                    "snippet": "During August 2012, Jordan Andrew Chartrand, 29, was named on a watchlist in Poland.",
                }) + "\n")
            store = DataStore(tmp)
            shield = SearchShield(store)
            # Single-part alias "Ali" must NOT pull "Hassan Ali"
            dobs, ids, _ = shield.lookup_hit_dobs(["Mohammed Smith", "Ali"], {"EG"}, "a sanctioned individual")
            self.assertEqual(ids, [])
            # Name "Jordan Andrew Chartrand" must NOT self-anchor to country Jordan (JO)
            dobs2, ids2, _ = shield.lookup_hit_dobs(["Jordan Andrew Chartrand"], {"JO"}, "named on a watchlist")
            self.assertEqual(ids2, [])

        # 4. Double-semicolon "According to Source A; Source B; July, 2021:" parses cleanly
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "name_variants.json"), "w", encoding="utf-8") as fh:
                json.dump(VARIANTS, fh)
            for fn in ("alerts.jsonl", "customers.jsonl", "accounts.jsonl", "onboarding.jsonl", "kyc_checks.jsonl", "notes.jsonl", "search_results.jsonl"):
                open(os.path.join(tmp, fn), "w").close()
            with open(os.path.join(tmp, "hits.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "hit_id": "H-SEMI1", "alert_id": "A-SEMI1", "entity_type": "Company",
                    "name": "Acme Corp", "first_name": None, "middle_name": None, "last_name": None,
                    "best_matching_name": None, "aliases": [], "dobs": [], "deceased_date": None, "life_status": None,
                    "countries": ["United States"], "nationality": [], "citizenship": [], "addresses": [],
                    "place_of_birth": None, "primary_category": "Enforcement", "categories": ["Enforcement"],
                    "profile_notes": "According to Source A; Source B; July, 2021: Acme Corp was fined.",
                }) + "\n")
            store = DataStore(tmp)
            hp = WatchlistResolver(store, SearchShield(store)).resolve_hit(store.hits[0])
            self.assertEqual(hp.earliest_report_date, PartialDate(2021, 7, None, "July, 2021"))

    def test_openrouter_vs_xai_grok_backend_selection(self) -> None:
        # 1. OPENROUTER_API_KEY takes strict priority even if XAI_API_KEY is also set
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "or-key", "XAI_API_KEY": "xai-key"},
            clear=False,
        ):
            c = OpenRouterClient()
            self.assertTrue(c.enabled)
            self.assertEqual(c.backend, "openrouter")
            self.assertEqual(c.base_url, "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(c.provider_order, ["Google"])

        # 2. When OPENROUTER_API_KEY is unset and XAI_API_KEY is set, route to xAI Grok without provider pinning
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "", "XAI_API_KEY": "xai-key"},
            clear=False,
        ):
            c = OpenRouterClient()
            self.assertTrue(c.enabled)
            self.assertEqual(c.backend, "xai")
            self.assertEqual(c.base_url, "https://api.x.ai/v1/chat/completions")
            self.assertEqual(c.model, "grok-4.20-0309-non-reasoning")
            self.assertEqual(c.provider_order, [])


if __name__ == "__main__":
    unittest.main()
