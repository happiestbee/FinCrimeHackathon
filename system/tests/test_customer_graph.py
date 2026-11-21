"""Unit tests for Principle 2, 3, 4 customer graph merging, quarantine, and note mining."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.customer_resolver import CustomerResolver
from src.ingest import DataStore


def write_fixture(tmpdir: str, data: dict) -> str:
    with open(os.path.join(tmpdir, "name_variants.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "groups": [["anthony", "tony"], ["muhammad", "mohammed"]],
                "company_legal_forms": ["ltd", "llc", "gmbh", "sa"],
            },
            fh,
        )
    filenames = (
        "alerts.jsonl",
        "hits.jsonl",
        "customers.jsonl",
        "accounts.jsonl",
        "onboarding.jsonl",
        "kyc_checks.jsonl",
        "notes.jsonl",
        "search_results.jsonl",
    )
    for fn in filenames:
        rows = data.get(fn, [])
        with open(os.path.join(tmpdir, fn), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return tmpdir


class TestCustomerGraph(unittest.TestCase):
    def test_union_find_merge_and_family_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(
                tmp,
                {
                    "customers.jsonl": [
                        # Alice old & new (same phone + same full DOB -> MERGE)
                        {
                            "record_id": "CUS-000001",
                            "entity_type": "Individual",
                            "full_name": "Alice Smith",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "1985-04-12",
                            "nationality": "GB",
                            "residence_country": "GB",
                            "city": "London",
                            "address": "10 High St, London",
                            "email": "alice1@example.com",
                            "phone": "+44 7700900001",
                            "status": "closed",
                        },
                        {
                            "record_id": "CUS-000002",
                            "entity_type": "Individual",
                            "full_name": "Alice Smith",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "12/04/1985",
                            "nationality": "GB",
                            "residence_country": "ES",
                            "city": "Madrid",
                            "address": "10 High St, London",
                            "email": "alice2@example.com",
                            "phone": "+44 7700900001",
                            "status": "active",
                        },
                        # Alice 3rd record via shared approved document_number -> transitive MERGE
                        {
                            "record_id": "CUS-000003",
                            "entity_type": "Individual",
                            "full_name": "Alice J Smith",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "1985-04-12",
                            "nationality": "GB",
                            "residence_country": "FR",
                            "city": "Paris",
                            "address": "5 Rue de la Paix",
                            "email": "alice3@example.com",
                            "phone": "+33 600000000",
                            "status": "dormant",
                        },
                        # Bob Smith: shares SAME phone & address as Alice, DIFFERENT DOB -> QUARANTINE!
                        {
                            "record_id": "CUS-000004",
                            "entity_type": "Individual",
                            "full_name": "Bob Smith",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "1983-01-01",
                            "nationality": "GB",
                            "residence_country": "GB",
                            "city": "London",
                            "address": "10 High St, London",
                            "email": "bob@example.com",
                            "phone": "+44 7700900001",
                            "status": "active",
                        },
                    ],
                    "accounts.jsonl": [
                        {"account_id": "ACC-000001", "customer_id": "CUS-000001", "login_handle": "alice1", "phone": "+44 7700900001"},
                        # Joint account shared by Alice (CUS-000002) and Bob (CUS-000004)
                        {"account_id": "ACC-JOINT1", "customer_id": "CUS-000002", "login_handle": "alice2", "phone": "+44 7700900001"},
                        {"account_id": "ACC-JOINT1", "customer_id": "CUS-000004", "login_handle": "bob1", "phone": "+44 7700900001"},
                    ],
                    "onboarding.jsonl": [
                        {"account_id": "ACC-000001", "customer_id": "CUS-000002", "kyc_applicant_id": "APP-02"},
                        {"account_id": "ACC-000002", "customer_id": "CUS-000003", "kyc_applicant_id": "APP-03"},
                        {"account_id": "ACC-000003", "customer_id": "CUS-000004", "kyc_applicant_id": "APP-04"},
                    ],
                    "kyc_checks.jsonl": [
                        {
                            "check_id": "KYC-000002",
                            "applicant_id": "APP-02",
                            "verified_name": "ALICE MARIE SMITH",
                            "verified_dob": "1985-04-12",
                            "document_number": "P-999-888",
                            "document_country": "United Kingdom",
                            "result": "approved",
                            "checked_at": "2024-01-10",
                        },
                        {
                            "check_id": "KYC-000003",
                            "applicant_id": "APP-03",
                            "verified_name": "ALICE MARIE SMITH",
                            "verified_dob": "1985-04-12",
                            "document_number": "P999888",
                            "document_country": "France",
                            "result": "approved",
                            "checked_at": "2020-01-10",
                        },
                        {
                            "check_id": "KYC-000004",
                            "applicant_id": "APP-04",
                            "verified_name": "WRONG NAME",
                            "verified_dob": "1983-01-01",
                            "document_number": "P999888",  # Rejected check must NOT merge or count!
                            "document_country": "Germany",
                            "result": "rejected",
                            "checked_at": "2025-01-10",
                        },
                    ],
                    "notes.jsonl": [
                        {
                            "note_id": "NOTE-000001",
                            "requester_email": "alice1@example.com",
                            "created_at": "2026-01-01",
                            "channel": "phone",
                            "text": "The client was a resident and worker in Canada from 2010 to 2015.",
                        }
                    ],
                    "alerts.jsonl": [
                        {
                            "alert_id": "A-000001",
                            "screened_at": "2026-11-01",
                            "customer": {"name": "Alice Smith", "entity_type": "Individual", "dob": "1985-04-12", "country": None, "city": None},
                            "account_ref": "ACC-JOINT1",
                            "login_handle": None,
                            "email": None,
                            "hit_ids": [],
                        }
                    ],
                },
            )
            store = DataStore(tmp)
            # Alice's 3 records merged; Bob quarantined!
            self.assertEqual(store.cluster_members[store.cluster_id_by_customer["CUS-000001"]], ["CUS-000001", "CUS-000002", "CUS-000003"])
            self.assertEqual(store.cluster_members[store.cluster_id_by_customer["CUS-000004"]], ["CUS-000004"])

            resolver = CustomerResolver(store)
            ra = resolver.resolve_alert(store.alerts[0])
            self.assertEqual(ra.resolution_status, "LINKED")
            self.assertEqual(len(ra.candidates), 1)
            prof = ra.candidates[0]
            self.assertEqual(prof.canonical_record_id, "CUS-000002")  # directly linked active record
            self.assertEqual(prof.verified_name, "ALICE MARIE SMITH")
            self.assertEqual(prof.verified_name_source_id, "KYC-000002")  # newest approved
            # Countries collected across merged records + approved KYC + residence note in email alice1
            self.assertEqual(set(prof.country_sources.keys()), {"GB", "ES", "FR", "CA"})

    def test_unlinked_zero_one_many_and_preferred_name_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(
                tmp,
                {
                    "customers.jsonl": [
                        {
                            "record_id": "CUS-000010",
                            "entity_type": "Individual",
                            "full_name": "Jessica Jackson",
                            "preferred_name": "Jessica",  # Single word must NOT match "Jessica Doe"
                            "login_name": None,
                            "dob": "1990-05-01",
                            "nationality": "US",
                            "residence_country": "US",
                            "city": "Newark",
                            "address": "1 Main St, Newark",
                            "email": None,
                            "phone": "+1 5550001",
                            "status": "active",
                        },
                        {
                            "record_id": "CUS-000011",
                            "entity_type": "Individual",
                            "full_name": "William Brown",
                            "preferred_name": "Billy",
                            "login_name": None,
                            "dob": "1990-05-01",
                            "nationality": "US",
                            "residence_country": "US",
                            "city": "Newark",
                            "address": "2 Main St, Newark",
                            "email": None,
                            "phone": "+1 5550002",
                            "status": "active",
                        },
                        {
                            "record_id": "CUS-000012",
                            "entity_type": "Individual",
                            "full_name": "Juan Garcia",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "1980-01-01",
                            "nationality": "ES",
                            "residence_country": "ES",
                            "city": "Madrid",
                            "address": "3 Calle Mayor",
                            "email": None,
                            "phone": "+34 600001",
                            "status": "active",
                        },
                        {
                            "record_id": "CUS-000013",
                            "entity_type": "Individual",
                            "full_name": "Juan Garcia",
                            "preferred_name": None,
                            "login_name": None,
                            "dob": "1980-02-02",
                            "nationality": "ES",
                            "residence_country": "ES",
                            "city": "Madrid",
                            "address": "4 Calle Mayor",
                            "email": None,
                            "phone": "+34 600002",
                            "status": "active",
                        },
                    ],
                    "alerts.jsonl": [
                        {
                            "alert_id": "A-ZERO",
                            "screened_at": "2026-11-01",
                            "customer": {"name": "Jessica Doe", "entity_type": "Individual", "dob": "1990-05-01", "country": "US", "city": "Newark"},
                            "account_ref": None, "login_handle": None, "email": None, "hit_ids": [],
                        },
                        {
                            "alert_id": "A-ONE",
                            "screened_at": "2026-11-01",
                            "customer": {"name": "Billy Brown", "entity_type": "Individual", "dob": "1990-05-01", "country": "US", "city": None},
                            "account_ref": None, "login_handle": None, "email": None, "hit_ids": [],
                        },
                        {
                            "alert_id": "A-MANY",
                            "screened_at": "2026-11-01",
                            "customer": {"name": "Juan Garcia", "entity_type": "Individual", "dob": "1980", "country": "Spain", "city": "Madrid"},
                            "account_ref": None, "login_handle": None, "email": None, "hit_ids": [],
                        },
                    ],
                },
            )
            store = DataStore(tmp)
            resolver = CustomerResolver(store)

            r0 = resolver.resolve_alert(store.alerts_by_id["A-ZERO"])
            self.assertEqual(r0.resolution_status, "UNRESOLVED")
            self.assertIsNone(r0.customer_record_id)

            r1 = resolver.resolve_alert(store.alerts_by_id["A-ONE"])
            self.assertEqual(r1.resolution_status, "INFERRED_UNIQUE")
            self.assertEqual(r1.customer_record_id, "CUS-000011")

            rm = resolver.resolve_alert(store.alerts_by_id["A-MANY"])
            self.assertEqual(rm.resolution_status, "AMBIGUOUS")
            self.assertEqual(rm.customer_record_id, "AMBIGUOUS")
            self.assertEqual(len(rm.candidates), 2)


if __name__ == "__main__":
    unittest.main()
