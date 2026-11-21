"""Fast in-memory ingestion, hash indexing, and Principle 3 person clustering."""

from __future__ import annotations

from collections import defaultdict
import json
import os
import re
from typing import Any

from .normalize import (
    GeoNormalizer,
    NameMatcher,
    normalize_doc_number,
    normalize_phone,
    parse_partial_date,
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class DataStore:
    """In-memory relational & inverted index store over a screening data directory."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.name_matcher = NameMatcher(os.path.join(data_dir, "name_variants.json"))
        self.geo = GeoNormalizer()

        self.alerts = read_jsonl(os.path.join(data_dir, "alerts.jsonl"))
        self.hits = read_jsonl(os.path.join(data_dir, "hits.jsonl"))
        self.customers = read_jsonl(os.path.join(data_dir, "customers.jsonl"))
        self.accounts = read_jsonl(os.path.join(data_dir, "accounts.jsonl"))
        self.onboarding = read_jsonl(os.path.join(data_dir, "onboarding.jsonl"))
        self.kyc_checks = read_jsonl(os.path.join(data_dir, "kyc_checks.jsonl"))
        self.notes = read_jsonl(os.path.join(data_dir, "notes.jsonl"))
        self.search_results = read_jsonl(os.path.join(data_dir, "search_results.jsonl"))

        self.alerts_by_id: dict[str, dict[str, Any]] = {
            a["alert_id"]: a for a in self.alerts
        }
        self.hits_by_id: dict[str, dict[str, Any]] = {
            h["hit_id"]: h for h in self.hits
        }
        self.customers_by_id: dict[str, dict[str, Any]] = {
            c["record_id"]: c for c in self.customers
        }

        self.customers_by_email: dict[str, list[str]] = defaultdict(list)
        for c in self.customers:
            em = (c.get("email") or "").strip().lower()
            if em:
                self.customers_by_email[em].append(c["record_id"])

        self.accounts_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.accounts_by_login: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.accounts_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for acc in self.accounts:
            self.accounts_by_id[acc["account_id"]].append(acc)
            lh = (acc.get("login_handle") or "").strip()
            if lh:
                self.accounts_by_login[lh.lower()].append(acc)
            self.accounts_by_customer[acc["customer_id"]].append(acc)

        self.onboarding_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.onboarding_by_applicant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ob in self.onboarding:
            self.onboarding_by_customer[ob["customer_id"]].append(ob)
            self.onboarding_by_applicant[ob["kyc_applicant_id"]].append(ob)

        self.kyc_by_applicant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for k in self.kyc_checks:
            self.kyc_by_applicant[k["applicant_id"]].append(k)

        self.kyc_checks_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cid, obs in self.onboarding_by_customer.items():
            seen_checks: set[str] = set()
            for ob in obs:
                for k in self.kyc_by_applicant.get(ob["kyc_applicant_id"], ()):
                    if k["check_id"] not in seen_checks:
                        seen_checks.add(k["check_id"])
                        self.kyc_checks_by_customer[cid].append(k)

        self.notes_by_email: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for n in self.notes:
            em = (n.get("requester_email") or "").strip().lower()
            if em:
                self.notes_by_email[em].append(n)

        # Build Principle 3 Union-Find clusters
        self.cluster_id_by_customer: dict[str, str] = {}
        self.cluster_members: dict[str, list[str]] = {}
        self._build_person_clusters()

        # Build inverted indexes for unlinked customer lookup
        self.cust_by_type_year: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.cust_by_type_country: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.cust_by_type_token: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._build_customer_inverted_indexes()

        # Index search_results.jsonl by canonical query name parts
        self.sr_by_query_tokens: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
        self.sr_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._build_search_indexes()

    def _build_person_clusters(self) -> None:
        parent: dict[str, str] = {c["record_id"]: c["record_id"] for c in self.customers}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        # 1. Phone + full DOB (from CRM dob or approved KYC check)
        phone_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for c in self.customers:
            p = normalize_phone(c.get("phone"))
            if p:
                phone_groups[(c["entity_type"], p)].append(c["record_id"])

        for (etype, p), cids in phone_groups.items():
            if len(cids) < 2:
                continue
            by_full_dob: dict[str, list[str]] = defaultdict(list)
            for cid in cids:
                c = self.customers_by_id[cid]
                dobs_for_cid: set[str] = set()
                pd = parse_partial_date(c.get("dob"), day_first=True)
                if pd and pd.is_full:
                    dobs_for_cid.add(pd.iso_str)
                for k in self.kyc_checks_by_customer.get(cid, ()):
                    if (k.get("result") or "").lower() == "approved":
                        kd = parse_partial_date(k.get("verified_dob"), day_first=True)
                        if kd and kd.is_full:
                            dobs_for_cid.add(kd.iso_str)
                em = (c.get("email") or "").strip().lower()
                if em:
                    for note in self.notes_by_email.get(em, ()):
                        if note.get("channel") == "kyc":
                            nd = _extract_approved_kyc_note_full_dob(
                                note.get("text") or "", note.get("created_at") or ""
                            )
                            if nd:
                                dobs_for_cid.add(nd)
                for iso in dobs_for_cid:
                    by_full_dob[iso].append(cid)
            for group in by_full_dob.values():
                for other in group[1:]:
                    union(group[0], other)

        # 2. Same normalized document_number on approved KYC checks
        doc_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for cid, ks in self.kyc_checks_by_customer.items():
            etype = self.customers_by_id[cid]["entity_type"]
            for k in ks:
                if (k.get("result") or "").lower() == "approved":
                    doc = normalize_doc_number(k.get("document_number"))
                    if doc:
                        doc_groups[(etype, doc)].append(cid)
        for group in doc_groups.values():
            for other in group[1:]:
                union(group[0], other)

        clusters: dict[str, list[str]] = defaultdict(list)
        for c in self.customers:
            cid = c["record_id"]
            root = find(cid)
            clusters[root].append(cid)

        for root, members in clusters.items():
            members.sort()
            canonical_cluster_id = members[0]
            self.cluster_members[canonical_cluster_id] = members
            for cid in members:
                self.cluster_id_by_customer[cid] = canonical_cluster_id

    def _build_customer_inverted_indexes(self) -> None:
        for c in self.customers:
            cid = c["record_id"]
            etype = c["entity_type"]
            is_comp = etype == "Company"

            years: set[int] = set()
            pd = parse_partial_date(c.get("dob"), day_first=True)
            if pd:
                years.add(pd.year)
            for k in self.kyc_checks_by_customer.get(cid, ()):
                if (k.get("result") or "").lower() == "approved":
                    kd = parse_partial_date(k.get("verified_dob"), day_first=True)
                    if kd:
                        years.add(kd.year)
            for y in years:
                self.cust_by_type_year[(etype, y)].append(cid)

            countries: set[str] = set()
            for field_name in ("residence_country", "nationality"):
                iso = self.geo.normalize_structured_country(c.get(field_name))
                if iso:
                    countries.add(iso)
            for iso in countries:
                self.cust_by_type_country[(etype, iso)].append(cid)

            tok_set: set[str] = set()
            for nf in ("full_name", "preferred_name", "login_name"):
                for tok in self.name_matcher.index_tokens(c.get(nf), is_company=is_comp):
                    tok_set.add(tok)
            for tok in tok_set:
                self.cust_by_type_token[(etype, tok)].append(cid)

    def _build_search_indexes(self) -> None:
        for row in self.search_results:
            q = row.get("query") or ""
            parts = self.name_matcher.extract_name_parts(q, is_company=False)
            if not parts:
                continue
            fset = frozenset(parts)
            self.sr_by_query_tokens[fset].append(row)
            for tok in self.name_matcher.index_tokens(q, is_company=False):
                self.sr_by_token[tok].append(row)


_KYC_NEG_RE = re.compile(
    r"\b(?:reject(?:ed)?|expir(?:ed|es|ing)?|fail(?:ed|s|ing)?|declin(?:ed)?|invalid|"
    r"denied|refus(?:ed)?|unsuccessful|not\s+verif(?:ied)?|unable\s+to\s+verif(?:y|ied)|not\s+approved)\b",
    re.I,
)
_KYC_POS_RE = re.compile(
    r"\b(?:approv(?:ed)?|verif(?:ied)?|validat(?:ed)?|pass(?:ed)?|clear(?:ed)?|confirm(?:ed)?|accept(?:ed)?)\b",
    re.I,
)
_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_KYC_NOTE_DOB_RE = re.compile(
    rf"\b(?:DOB|date\s+of\s+birth|birth\s+date|born)\b[^0-9A-Za-z\n]{{0,18}}"
    rf"(?:on\s+|in\s+|is\s+|as\s+|of\s+)?"
    rf"(\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}\.\d{{1,2}}\.\d{{4}}"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?,?\s+\d{{4}}|{_MONTH_PAT}\.?\s+\d{{1,2}},?\s+\d{{4}})",
    re.I,
)
_KYC_DATE_FALLBACK_RE = re.compile(
    rf"\b(\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?,?\s+\d{{4}}|{_MONTH_PAT}\.?\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.I,
)


def _extract_approved_kyc_note_full_dob(text: str, created_at: str = "") -> str | None:
    """Lightweight helper used during person clustering for shared-phone customers."""
    if not text or (_KYC_NEG_RE.search(text) and not _KYC_POS_RE.search(text)):
        return None
    if _KYC_NEG_RE.search(text):
        return None
    m = _KYC_NOTE_DOB_RE.search(text)
    if m:
        pd = parse_partial_date(m.group(1), day_first=True)
        if pd and pd.is_full:
            return pd.iso_str
    checked_yr = int(created_at[:4]) if len(created_at) >= 4 and created_at[:4].isdigit() else 2026
    for m2 in _KYC_DATE_FALLBACK_RE.finditer(text):
        pd = parse_partial_date(m2.group(1), day_first=True)
        if pd and pd.is_full and pd.iso_str != created_at and pd.year <= checked_yr - 14:
            return pd.iso_str
    return None
