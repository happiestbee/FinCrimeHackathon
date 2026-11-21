"""Determinism test verifying identical byte-for-byte JSONL output across repeated runs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import run_pipeline


class TestDeterminism(unittest.TestCase):
    def test_repeated_pipeline_runs_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            with open(os.path.join(data_dir, "name_variants.json"), "w", encoding="utf-8") as fh:
                json.dump({"groups": [["mohammed", "mohamed"]], "company_legal_forms": ["ltd", "sa"]}, fh)

            fixtures = {
                "customers.jsonl": [
                    {
                        "record_id": "CUS-000001", "entity_type": "Individual",
                        "full_name": "Mohammed Ali", "preferred_name": None, "login_name": None,
                        "dob": "1988-03-14", "nationality": "GB", "residence_country": "GB",
                        "city": "London", "address": "1 High St, London", "email": "m@example.com",
                        "phone": "+44 70001", "status": "active",
                    }
                ],
                "accounts.jsonl": [
                    {"account_id": "ACC-000001", "customer_id": "CUS-000001", "login_handle": "mali", "phone": "+44 70001"}
                ],
                "onboarding.jsonl": [
                    {"account_id": "ACC-000001", "customer_id": "CUS-000001", "kyc_applicant_id": "APP-01"}
                ],
                "kyc_checks.jsonl": [
                    {
                        "check_id": "KYC-000001", "applicant_id": "APP-01",
                        "verified_name": "MOHAMMED ALI", "verified_dob": "14/03/1988",
                        "document_number": "P123", "document_country": "United Kingdom",
                        "result": "approved", "checked_at": "2024-01-01",
                    }
                ],
                "notes.jsonl": [],
                "search_results.jsonl": [
                    {
                        "result_id": "SR-000001", "query": "mohamed ali", "title": "Mohamed Ali",
                        "snippet": "In March 2010, Mohamed Ali, aged 58, was named in adverse media in Egypt.",
                    }
                ],
                "alerts.jsonl": [
                    {
                        "alert_id": "A-000001", "screened_at": "2026-11-01",
                        "customer": {"name": "Mohammed Ali", "entity_type": "Individual", "dob": None, "country": None, "city": None},
                        "account_ref": "ACC-000001", "login_handle": None, "email": None,
                        "hit_ids": ["H-000001", "H-000002"],
                    }
                ],
                "hits.jsonl": [
                    {
                        "hit_id": "H-000001", "alert_id": "A-000001", "entity_type": "Individual",
                        "name": "Mohamed Ali", "first_name": None, "middle_name": None, "last_name": None,
                        "best_matching_name": None, "aliases": [], "dobs": [], "deceased_date": None, "life_status": None,
                        "countries": ["Egypt"], "nationality": [], "citizenship": [], "addresses": [],
                        "place_of_birth": None, "primary_category": "Adverse Media", "categories": ["Adverse Media"],
                        "profile_notes": None,
                    },
                    {
                        "hit_id": "H-000002", "alert_id": "A-000001", "entity_type": "Individual",
                        "name": "Ali, Mohammed", "first_name": None, "middle_name": None, "last_name": None,
                        "best_matching_name": None, "aliases": [], "dobs": ["1988-03-14"], "deceased_date": None, "life_status": None,
                        "countries": ["United Kingdom"], "nationality": [], "citizenship": [], "addresses": [],
                        "place_of_birth": None, "primary_category": "PEP", "categories": ["PEP"],
                        "profile_notes": None,
                    },
                ],
            }
            for fn, rows in fixtures.items():
                with open(os.path.join(data_dir, fn), "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")

            digests = []
            for idx in range(3):
                out_path = os.path.join(tmp, f"out_{idx}.jsonl")
                run_pipeline(data_dir, out_path)
                with open(out_path, "rb") as fh:
                    digests.append(hashlib.sha256(fh.read()).hexdigest())

            self.assertEqual(len(set(digests)), 1)
            with open(os.path.join(tmp, "out_0.jsonl"), encoding="utf-8") as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            self.assertEqual(lines[0]["decision"], "DIFFERENT_ENTITY")
            self.assertEqual(lines[0]["reason_code"], "DOB_MISMATCH")
            self.assertEqual(lines[1]["decision"], "SAME_ENTITY")
            self.assertEqual(lines[1]["reason_code"], "NAME_DOB_LOCATION_MATCH")


if __name__ == "__main__":
    unittest.main()
