"""Zero-dependency web visualizer server for the watchlist entity resolution system."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_DIR = REPO_ROOT / "system"
VISUALIZER_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = VISUALIZER_DIR / "index.html"

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from src.customer_resolver import CustomerResolver  # noqa: E402
from src.ingest import DataStore, read_jsonl  # noqa: E402
from src.llm_client import OpenRouterClient  # noqa: E402
from src.normalize import normalize_phone, parse_partial_date  # noqa: E402
from src.policy_engine import PolicyEngine  # noqa: E402
from src.search_shield import SearchShield  # noqa: E402
from src.verifier import ExtractionVerifier  # noqa: E402
from src.watchlist_resolver import WatchlistResolver  # noqa: E402


COST_MATRIX = {
    "DIFFERENT_ENTITY": {"DIFFERENT_ENTITY": 0, "SAME_ENTITY": 3, "REVIEW": 3},
    "SAME_ENTITY": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 0, "REVIEW": 2},
    "REVIEW": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 2, "REVIEW": 0},
}


def load_dotenv_if_present() -> None:
    """Load key=value pairs from .env at repo root without overriding set env vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and not os.environ.get(k) and v:
                os.environ[k] = v
    except Exception:
        pass


def discover_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []

    sample_dir = SYSTEM_DIR / "tests" / "sample_data"
    if (sample_dir / "hits.jsonl").exists():
        datasets.append(
            {
                "id": "sample",
                "name": "Synthetic Sample Suite (20 curated cases)",
                "data_dir": str(sample_dir),
                "labels_path": str(sample_dir / "expected_labels.jsonl")
                if (sample_dir / "expected_labels.jsonl").exists()
                else None,
                "hit_count_hint": 20,
            }
        )

    adv_dir = SYSTEM_DIR / "tests" / "adversarial"
    if (adv_dir / "hits.jsonl").exists():
        datasets.append(
            {
                "id": "adversarial",
                "name": "Adversarial Stress Suite (180 hits)",
                "data_dir": str(adv_dir),
                "labels_path": str(adv_dir / "practice_labels.jsonl")
                if (adv_dir / "practice_labels.jsonl").exists()
                else None,
                "hit_count_hint": 180,
            }
        )

    dev_dir = REPO_ROOT / "data" / "screening" / "dev"
    if (dev_dir / "hits.jsonl").exists():
        datasets.append(
            {
                "id": "dev",
                "name": "Development Screening Set (819 hits)",
                "data_dir": str(dev_dir),
                "labels_path": str(dev_dir / "practice_labels.jsonl")
                if (dev_dir / "practice_labels.jsonl").exists()
                else None,
                "hit_count_hint": 819,
            }
        )

    return datasets


class DatasetSession:
    """In-memory resolved pipeline session for one screening dataset directory."""

    def __init__(self, spec: dict[str, Any], offline: bool = False) -> None:
        self.spec = spec
        self.id = spec["id"]
        self.name = spec["name"]
        t0 = time.monotonic()

        self.store = DataStore(spec["data_dir"])
        if offline:
            llm_client = OpenRouterClient(api_key="")
            llm_client.api_key = ""
            llm_client.enabled = False
        else:
            llm_client = OpenRouterClient()
        verifier = ExtractionVerifier(self.store.name_matcher, self.store.geo)
        self.customer_resolver = CustomerResolver(
            self.store, llm_client=llm_client, verifier=verifier
        )
        self.search_shield = SearchShield(
            self.store, llm_client=llm_client, verifier=verifier
        )
        self.watchlist_resolver = WatchlistResolver(
            self.store,
            search_shield=self.search_shield,
            llm_client=llm_client,
            verifier=verifier,
        )
        self.engine = PolicyEngine(self.store.name_matcher, self.store.geo)

        # Auxiliary O(1) indexes for UI inspection
        self.sr_by_id: dict[str, dict[str, Any]] = {
            r["result_id"]: r for r in self.store.search_results
        }
        self.kyc_by_id: dict[str, dict[str, Any]] = {
            k["check_id"]: k for k in self.store.kyc_checks
        }
        self.notes_by_id: dict[str, dict[str, Any]] = {
            n["note_id"]: n for n in self.store.notes
        }
        self.onboarding_by_id: dict[str, dict[str, Any]] = {
            o["kyc_applicant_id"]: o for o in self.store.onboarding
        }
        self.phone_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        for c in self.store.customers:
            p = normalize_phone(c.get("phone"))
            if p:
                self.phone_index[(c["entity_type"], p)].append(c["record_id"])

        # Optional ground-truth / practice labels
        self.labels_by_hit: dict[str, dict[str, Any]] = {}
        labels_path = spec.get("labels_path")
        if labels_path and os.path.exists(labels_path):
            for row in read_jsonl(labels_path):
                self.labels_by_hit[row["hit_id"]] = row

        self.hit_cache: dict[str, dict[str, Any]] = {}
        self.summary_hits: list[dict[str, Any]] = []
        decision_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        resolution_counts: Counter[str] = Counter()

        if llm_client.enabled and len(self.store.hits) > 1:
            unique_alerts = {
                h["alert_id"]: self.store.alerts_by_id[h["alert_id"]] for h in self.store.hits
            }
            with ThreadPoolExecutor(max_workers=12) as executor:
                alert_futures = {
                    aid: executor.submit(self.customer_resolver.resolve_alert, alert)
                    for aid, alert in unique_alerts.items()
                }
                hit_futures = [
                    executor.submit(self.watchlist_resolver.resolve_hit, hit)
                    for hit in self.store.hits
                ]
                resolved_pairs = [
                    (alert_futures[hit["alert_id"]].result(), hf.result())
                    for hit, hf in zip(self.store.hits, hit_futures)
                ]
        else:
            resolved_pairs = [
                (
                    self.customer_resolver.resolve_alert(self.store.alerts_by_id[hit["alert_id"]]),
                    self.watchlist_resolver.resolve_hit(hit),
                )
                for hit in self.store.hits
            ]

        for hit, (resolved_alert, hit_profile) in zip(self.store.hits, resolved_pairs):
            hid = hit["hit_id"]
            alert = self.store.alerts_by_id[hit["alert_id"]]
            screened_pd = parse_partial_date(alert.get("screened_at"), day_first=True)
            screened_dt = screened_pd.latest if screened_pd is not None else date.today()

            eval_results = [
                self.engine.evaluate_candidate(cand, hit_profile, screened_dt)
                for cand in resolved_alert.candidates
            ]
            output = self.engine.evaluate_hit(resolved_alert, hit_profile, screened_dt)
            out_dict = output.to_dict()

            decision_counts[output.decision] += 1
            reason_counts[output.reason_code] += 1
            resolution_counts[resolved_alert.resolution_status] += 1

            label = self.labels_by_hit.get(hid)
            label_match: str | None = None
            if label is not None:
                dec_ok = output.decision == label.get("decision")
                code_ok = output.reason_code in (label.get("reason_codes") or [])
                cid_expected = label.get("customer_record_ids") or []
                cid_ok = (output.customer_record_id in cid_expected) or (
                    cid_expected == ["AMBIGUOUS"]
                    and output.customer_record_id in (None, "", "AMBIGUOUS")
                )
                label_match = "MATCH" if (dec_ok and code_ok and cid_ok) else (
                    "DECISION_ONLY" if dec_ok else "MISMATCH"
                )

            primary_cand = resolved_alert.candidates[0] if resolved_alert.candidates else None
            cust_name = (
                (primary_cand.effective_name if primary_cand else None)
                or (alert.get("customer") or {}).get("name")
                or "Unknown"
            )

            self.hit_cache[hid] = {
                "alert": alert,
                "hit": hit,
                "resolved_alert": resolved_alert,
                "hit_profile": hit_profile,
                "screened_at": screened_dt.isoformat(),
                "eval_results": eval_results,
                "output": out_dict,
                "label": label,
                "label_match": label_match,
            }

            self.summary_hits.append(
                {
                    "hit_id": hid,
                    "alert_id": hit["alert_id"],
                    "decision": output.decision,
                    "reason_code": output.reason_code,
                    "customer_record_id": output.customer_record_id,
                    "resolution_status": resolved_alert.resolution_status,
                    "entity_type": hit["entity_type"],
                    "customer_name": cust_name,
                    "hit_name": hit_profile.primary_name,
                    "dob_source": hit_profile.dob_source,
                    "reason": output.reason,
                    "evidence_count": len(output.evidence),
                    "has_label": label is not None,
                    "label_match": label_match,
                }
            )

        self.load_time_ms = round((time.monotonic() - t0) * 1000.0, 1)
        self.label_stats = self._compute_label_stats()
        self.llm_info = {
            "enabled": llm_client.enabled,
            "backend": llm_client.backend if llm_client.enabled else "offline",
            "model": llm_client.model if llm_client.enabled else None,
            "stats": dict(llm_client.stats),
        }
        self.summary_payload = {
            "dataset": {
                "id": self.id,
                "name": self.name,
                "data_dir": self.spec["data_dir"],
            },
            "load_time_ms": self.load_time_ms,
            "llm_info": self.llm_info,
            "totals": {
                "hits": len(self.summary_hits),
                "alerts": len(self.store.alerts),
                "customers": len(self.store.customers),
                "search_results": len(self.store.search_results),
            },
            "decision_counts": dict(decision_counts),
            "reason_counts": dict(reason_counts.most_common()),
            "resolution_counts": dict(resolution_counts),
            "label_stats": self.label_stats,
            "hits": self.summary_hits,
        }

    def _compute_label_stats(self) -> dict[str, Any] | None:
        if not self.labels_by_hit:
            return None
        labels = list(self.labels_by_hit.values())
        cost = 0
        baseline = 0
        right = []
        unsafe = 0
        for row in labels:
            hid = row["hit_id"]
            got_dec = self.hit_cache[hid]["output"]["decision"] if hid in self.hit_cache else "REVIEW"
            exp_dec = row["decision"]
            cost += COST_MATRIX[exp_dec][got_dec]
            baseline += COST_MATRIX[exp_dec]["REVIEW"]
            if got_dec == exp_dec:
                right.append(row)
            elif got_dec == "DIFFERENT_ENTITY":
                unsafe += 1

        codes_ok = sum(
            1
            for row in right
            if self.hit_cache[row["hit_id"]]["output"]["reason_code"] in row["reason_codes"]
        )
        cids_ok = sum(
            1
            for row in right
            if (
                self.hit_cache[row["hit_id"]]["output"]["customer_record_id"]
                in row["customer_record_ids"]
            )
            or (
                row["customer_record_ids"] == ["AMBIGUOUS"]
                and self.hit_cache[row["hit_id"]]["output"]["customer_record_id"]
                in (None, "", "AMBIGUOUS")
            )
        )
        decision_score = round(100.0 * max(0.0, 1.0 - cost / baseline), 1) if baseline else 100.0
        return {
            "total": len(labels),
            "right_decisions": len(right),
            "unsafe_clearances": unsafe,
            "decision_score": decision_score,
            "reason_code_pct": round(100.0 * codes_ok / len(right), 1) if right else 0.0,
            "customer_id_pct": round(100.0 * cids_ok / len(right), 1) if right else 0.0,
        }

    def get_hit_detail(self, hit_id: str) -> dict[str, Any] | None:
        cached = self.hit_cache.get(hit_id)
        if cached is None:
            return None

        alert = cached["alert"]
        hit = cached["hit"]
        resolved_alert = cached["resolved_alert"]
        hit_profile = cached["hit_profile"]
        eval_results = cached["eval_results"]
        out_dict = cached["output"]
        geo = self.store.geo

        records_by_id: dict[str, dict[str, Any]] = {
            alert["alert_id"]: {"kind": "Alert", "id": alert["alert_id"], "data": alert},
            hit["hit_id"]: {"kind": "Watchlist Hit", "id": hit["hit_id"], "data": hit},
        }

        # Direct account links from alert
        direct_accounts: list[dict[str, Any]] = []
        seen_acc: set[str] = set()
        if alert.get("account_ref"):
            for acc in self.store.accounts_by_id.get(alert["account_ref"], ()):
                if acc["account_id"] not in seen_acc:
                    seen_acc.add(acc["account_id"])
                    direct_accounts.append(acc)
                    records_by_id[acc["account_id"]] = {
                        "kind": "Account",
                        "id": acc["account_id"],
                        "data": acc,
                    }
        if alert.get("login_handle"):
            for acc in self.store.accounts_by_login.get(
                alert["login_handle"].strip().lower(), ()
            ):
                if acc["account_id"] not in seen_acc:
                    seen_acc.add(acc["account_id"])
                    direct_accounts.append(acc)
                    records_by_id[acc["account_id"]] = {
                        "kind": "Account",
                        "id": acc["account_id"],
                        "data": acc,
                    }

        eval_by_cluster = {er.person.cluster_id: er for er in eval_results}

        candidates_payload: list[dict[str, Any]] = []
        for idx, cand in enumerate(resolved_alert.candidates):
            er = eval_by_cluster.get(cand.cluster_id)
            crm_rows: list[dict[str, Any]] = []
            cand_accounts: list[dict[str, Any]] = []
            cand_onboarding: list[dict[str, Any]] = []
            cand_kyc: list[dict[str, Any]] = []
            cand_notes: list[dict[str, Any]] = []
            quarantined_records: list[dict[str, Any]] = []

            seen_q: set[str] = set()
            seen_cand_acc: set[str] = set()
            seen_ob: set[str] = set()
            seen_kyc: set[str] = set()
            seen_notes: set[str] = set()
            excluded_set = set(cand.excluded_kyc_ids)

            for cid in cand.record_ids:
                c_row = self.store.customers_by_id.get(cid)
                if c_row:
                    crm_rows.append(c_row)
                    records_by_id[cid] = {"kind": "Customer CRM", "id": cid, "data": c_row}
                    norm_p = normalize_phone(c_row.get("phone"))
                    if norm_p:
                        for qcid in self.phone_index.get((c_row["entity_type"], norm_p), ()):
                            if qcid not in cand.record_ids and qcid not in seen_q:
                                seen_q.add(qcid)
                                q_row = self.store.customers_by_id.get(qcid)
                                if q_row:
                                    quarantined_records.append(q_row)
                                    records_by_id[qcid] = {
                                        "kind": "Quarantined CRM (Family/Shared Phone)",
                                        "id": qcid,
                                        "data": q_row,
                                    }

                for acc in self.store.accounts_by_customer.get(cid, ()):
                    aid = acc["account_id"]
                    if aid not in seen_cand_acc:
                        seen_cand_acc.add(aid)
                        cand_accounts.append(acc)
                        records_by_id[aid] = {"kind": "Account", "id": aid, "data": acc}

                for ob in self.store.onboarding_by_customer.get(cid, ()):
                    oid = ob["kyc_applicant_id"]
                    if oid not in seen_ob:
                        seen_ob.add(oid)
                        cand_onboarding.append(ob)
                        records_by_id[oid] = {"kind": "Onboarding", "id": oid, "data": ob}

                for kyc in self.store.kyc_checks_by_customer.get(cid, ()):
                    kid = kyc["check_id"]
                    if kid not in seen_kyc:
                        seen_kyc.add(kid)
                        kyc_copy = dict(kyc)
                        kyc_copy["_tier_status"] = (
                            "TIER_0_EXCLUDED"
                            if kid in excluded_set
                            or (kyc.get("result") or "").lower() != "approved"
                            else "TIER_1_APPROVED"
                        )
                        cand_kyc.append(kyc_copy)
                        records_by_id[kid] = {"kind": "KYC Check", "id": kid, "data": kyc_copy}

            for em in cand.emails:
                for note in self.store.notes_by_email.get(em.lower(), ()):
                    nid = note["note_id"]
                    if nid not in seen_notes:
                        seen_notes.add(nid)
                        n_copy = dict(note)
                        n_copy["_tier_status"] = (
                            "TIER_0_EXCLUDED" if nid in excluded_set else "ACTIVE_NOTE"
                        )
                        cand_notes.append(n_copy)
                        records_by_id[nid] = {
                            "kind": f"Note ({note.get('channel') or 'note'})",
                            "id": nid,
                            "data": n_copy,
                        }

            # Fallback resolve any KYC/Note IDs referenced directly on profile if not yet indexed
            for extra_id in list(cand.excluded_kyc_ids) + (
                [cand.verified_name_source_id] if cand.verified_name_source_id else []
            ) + ([cand.dob_source_id] if cand.dob_source_id else []):
                if extra_id and extra_id not in records_by_id:
                    if extra_id in self.kyc_by_id:
                        records_by_id[extra_id] = {
                            "kind": "KYC Check",
                            "id": extra_id,
                            "data": self.kyc_by_id[extra_id],
                        }
                    elif extra_id in self.notes_by_id:
                        records_by_id[extra_id] = {
                            "kind": "Note",
                            "id": extra_id,
                            "data": self.notes_by_id[extra_id],
                        }

            country_map = {
                iso: {
                    "iso": iso,
                    "name": geo.display_name(iso),
                    "sources": srcs,
                }
                for iso, srcs in sorted(cand.country_sources.items())
            }

            candidates_payload.append(
                {
                    "index": idx,
                    "cluster_id": cand.cluster_id,
                    "record_ids": cand.record_ids,
                    "canonical_record_id": cand.canonical_record_id,
                    "entity_type": cand.entity_type,
                    "verified_name": cand.verified_name,
                    "verified_name_source_id": cand.verified_name_source_id,
                    "declared_names": [
                        {"name": nm, "source_id": src} for nm, src in cand.declared_names
                    ],
                    "effective_name": cand.effective_name,
                    "dob": cand.dob.iso_str if cand.dob else None,
                    "dob_source_id": cand.dob_source_id,
                    "dob_is_verified": cand.dob_is_verified,
                    "countries": list(country_map.values()),
                    "emails": cand.emails,
                    "excluded_kyc_ids": cand.excluded_kyc_ids,
                    "is_unresolved": cand.is_unresolved,
                    "resolution_notes": cand.resolution_notes,
                    "evaluation": {
                        "decision": er.decision,
                        "reason_code": er.reason_code,
                        "all_applicable_codes": er.all_applicable_codes,
                        "evidence_ids": er.evidence_ids,
                        "facts": {
                            "screened_at": er.facts.get("screened_at"),
                            "name_agrees": er.facts.get("name_agrees"),
                            "short_individual_name": er.facts.get("short_individual_name"),
                            "dob_status": er.facts.get("dob_status"),
                            "shared_countries": sorted(er.facts.get("shared_countries") or ()),
                            "countries_disjoint": er.facts.get("countries_disjoint"),
                            "max_age_at_report": er.facts.get("max_age_at_report"),
                        },
                    }
                    if er
                    else None,
                    "raw_trail": {
                        "customers": crm_rows,
                        "quarantined_customers": quarantined_records,
                        "accounts": cand_accounts,
                        "onboarding": cand_onboarding,
                        "kyc_checks": cand_kyc,
                        "notes": cand_notes,
                    },
                }
            )

        # Trace Anti-Namesake Search Shield candidates
        search_trace = self._trace_search_shield(hit_profile)
        for sr_item in search_trace["candidates"]:
            rid = sr_item["result_id"]
            if rid in self.sr_by_id:
                records_by_id[rid] = {
                    "kind": f"Search Result ({sr_item['status']})",
                    "id": rid,
                    "data": self.sr_by_id[rid],
                }

        # Guarantee every ID listed in output["evidence"] exists in records_by_id
        for evid in out_dict.get("evidence") or []:
            if evid not in records_by_id:
                if evid in self.sr_by_id:
                    records_by_id[evid] = {"kind": "Search Result", "id": evid, "data": self.sr_by_id[evid]}
                elif evid in self.kyc_by_id:
                    records_by_id[evid] = {"kind": "KYC Check", "id": evid, "data": self.kyc_by_id[evid]}
                elif evid in self.notes_by_id:
                    records_by_id[evid] = {"kind": "Note", "id": evid, "data": self.notes_by_id[evid]}
                elif evid in self.store.customers_by_id:
                    records_by_id[evid] = {"kind": "Customer CRM", "id": evid, "data": self.store.customers_by_id[evid]}
                elif evid in self.store.accounts_by_id:
                    records_by_id[evid] = {"kind": "Account", "id": evid, "data": self.store.accounts_by_id[evid][0]}

        hit_countries = [
            {"iso": iso, "name": geo.display_name(iso)}
            for iso in sorted(hit_profile.countries)
        ]

        return {
            "output": out_dict,
            "label": cached["label"],
            "label_match": cached["label_match"],
            "screened_at": cached["screened_at"],
            "alert": alert,
            "hit_raw": hit,
            "direct_accounts": direct_accounts,
            "resolved_alert": {
                "alert_id": resolved_alert.alert_id,
                "resolution_status": resolved_alert.resolution_status,
                "customer_record_id": resolved_alert.customer_record_id,
                "candidates": candidates_payload,
            },
            "hit_profile": {
                "hit_id": hit_profile.hit_id,
                "entity_type": hit_profile.entity_type,
                "primary_name": hit_profile.primary_name,
                "all_names": hit_profile.all_names,
                "dobs": [d.iso_str for d in hit_profile.dobs],
                "dob_source": hit_profile.dob_source,
                "dob_evidence_ids": hit_profile.dob_evidence_ids,
                "dob_detail": hit_profile.dob_detail,
                "countries": hit_countries,
                "deceased_date": hit_profile.deceased_date.iso_str
                if hit_profile.deceased_date
                else None,
                "life_status": hit_profile.life_status,
                "earliest_report_date": hit_profile.earliest_report_date.iso_str
                if hit_profile.earliest_report_date
                else None,
                "listing_phrase": hit_profile.listing_phrase,
            },
            "search_shield_trace": search_trace,
            "records_by_id": records_by_id,
        }

    def _trace_search_shield(self, hit_profile: Any) -> dict[str, Any]:
        is_comp = hit_profile.entity_type == "Company"
        if is_comp:
            return {
                "executed": False,
                "skip_reason": "Skipped: Policy Principle 5 restricts Companies to structured hit.dobs only.",
                "listing_phrase": hit_profile.listing_phrase,
                "candidates": [],
            }

        nm = self.store.name_matcher
        geo = self.store.geo
        hit_names = hit_profile.all_names
        hit_countries = set(hit_profile.countries)

        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for name in hit_names:
            parts = nm.extract_name_parts(name, is_company=False)
            if not parts:
                continue
            for row in self.store.sr_by_query_tokens.get(frozenset(parts), ()):
                rid = row["result_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    candidates.append(row)

        multi_part_hit_names = [
            hn for hn in hit_names if nm.real_part_count(hn, is_company=False) >= 2
        ]
        if not candidates and multi_part_hit_names:
            token_pool: list[dict[str, Any]] = []
            pool_ids: set[str] = set()
            for name in multi_part_hit_names:
                for tok in nm.index_tokens(name, is_company=False):
                    for row in self.store.sr_by_token.get(tok, ()):
                        rid = row["result_id"]
                        if rid not in pool_ids:
                            pool_ids.add(rid)
                            token_pool.append(row)
            for row in token_pool:
                q = row.get("query") or ""
                if nm.real_part_count(q, is_company=False) < 2:
                    continue
                if any(nm.names_agree(hn, q, is_company=False) for hn in multi_part_hit_names):
                    if row["result_id"] not in seen_ids:
                        seen_ids.add(row["result_id"])
                        candidates.append(row)

        phrase = hit_profile.listing_phrase
        phrase_lc = phrase.lower()
        used_ids = set(hit_profile.dob_evidence_ids) if hit_profile.dob_source == "search_results" else set()
        executed = hit_profile.dob_source not in ("dobs_field", "profile_notes")

        traced_rows: list[dict[str, Any]] = []
        for row in sorted(candidates, key=lambda r: r["result_id"]):
            snip = row.get("snippet") or ""
            phrase_matched = phrase_lc in snip.lower()
            title_name = (row.get("title") or "").split(" - ")[0].strip()
            mask_names = list(hit_names)
            if row.get("query"):
                mask_names.append(row["query"])
            if title_name:
                mask_names.append(title_name)

            snip_isos = set(geo.extract_countries_from_prose(snip, mask_names=mask_names))
            matched_isos = sorted(snip_isos & hit_countries)
            country_matched = bool(matched_isos)
            anchored = bool(hit_countries) and phrase_matched and country_matched
            used_for_dob = row["result_id"] in used_ids

            block_reasons: list[str] = []
            if not hit_countries:
                block_reasons.append("Hit has no country anchor")
            if not phrase_matched:
                block_reasons.append(f'Missing listing phrase "{phrase}"')
            if hit_countries and not country_matched:
                found_names = ", ".join(geo.display_name(i) for i in sorted(snip_isos)) or "none"
                expected_names = ", ".join(geo.display_name(i) for i in sorted(hit_countries))
                block_reasons.append(
                    f"Country anchor mismatch (snippet: {found_names}; expected: {expected_names})"
                )

            if not executed and hit_profile.dob_source in ("dobs_field", "profile_notes"):
                status = "NOT_NEEDED"
            elif used_for_dob:
                status = "USED_FOR_DOB"
            elif anchored:
                status = "ANCHORED_NO_DOB"
            else:
                status = "BLOCKED"

            traced_rows.append(
                {
                    "result_id": row["result_id"],
                    "query": row.get("query"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "snippet": snip,
                    "phrase_matched": phrase_matched,
                    "country_matched": country_matched,
                    "found_countries": [
                        {"iso": i, "name": geo.display_name(i)} for i in sorted(snip_isos)
                    ],
                    "matched_countries": [
                        {"iso": i, "name": geo.display_name(i)} for i in matched_isos
                    ],
                    "anchored": anchored,
                    "used_for_dob": used_for_dob,
                    "status": status,
                    "block_reason": "; ".join(block_reasons) if block_reasons else None,
                }
            )

        skip_reason = None
        if not executed:
            skip_reason = f"Waterfall stopped earlier at {hit_profile.dob_source}; Stage A candidates shown for inspection."

        return {
            "executed": executed,
            "skip_reason": skip_reason,
            "listing_phrase": phrase,
            "candidates": traced_rows,
        }


class VisualizerAppState:
    """Thread-safe registry managing lazy loading and caching of datasets."""

    def __init__(self, preload_sample: bool = True, offline: bool = False) -> None:
        self.offline = offline
        self.specs = {d["id"]: d for d in discover_datasets()}
        self.sessions: dict[str, DatasetSession] = {}
        self._lock = threading.Lock()
        if preload_sample and "sample" in self.specs:
            self.get_session("sample")

    def list_datasets(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for did, spec in self.specs.items():
            out.append(
                {
                    "id": did,
                    "name": spec["name"],
                    "loaded": did in self.sessions,
                    "hit_count_hint": spec.get("hit_count_hint"),
                }
            )
        return out

    def get_session(self, dataset_id: str) -> DatasetSession | None:
        if dataset_id not in self.specs:
            return None
        with self._lock:
            if dataset_id not in self.sessions:
                self.sessions[dataset_id] = DatasetSession(
                    self.specs[dataset_id], offline=self.offline
                )
            return self.sessions[dataset_id]


def make_handler(app_state: VisualizerAppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_bytes: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html_bytes)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                if not INDEX_HTML_PATH.exists():
                    self._send_json({"error": "index.html not found"}, status=500)
                    return
                self._send_html(INDEX_HTML_PATH.read_bytes())
                return

            if path == "/api/datasets":
                self._send_json({"datasets": app_state.list_datasets()})
                return

            parts = [unquote(p) for p in path.strip("/").split("/") if p]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "datasets":
                dataset_id = parts[2]
                session = app_state.get_session(dataset_id)
                if session is None:
                    self._send_json({"error": f"Unknown dataset: {dataset_id}"}, status=404)
                    return

                if len(parts) == 4 and parts[3] == "summary":
                    self._send_json(session.summary_payload)
                    return

                if len(parts) == 5 and parts[3] == "hit":
                    hit_id = parts[4]
                    detail = session.get_hit_detail(hit_id)
                    if detail is None:
                        self._send_json({"error": f"Unknown hit: {hit_id}"}, status=404)
                        return
                    self._send_json(detail)
                    return

            self._send_json({"error": "Not found"}, status=404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(description="Start the Screening Entity Resolution Visualizer.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Do not preload sample dataset on startup",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable LLM extraction fallback even if an API key is configured",
    )
    args = parser.parse_args(argv)

    app_state = VisualizerAppState(preload_sample=not args.no_preload, offline=args.offline)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app_state))
    print(f"[visualizer] listening on http://{args.host}:{args.port} (datasets={list(app_state.specs)}, offline={args.offline})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
