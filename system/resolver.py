"""Multi-table indexing, same-person Union-Find clustering, and customer profile resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from llm_client import LLMClient
from normalizer import (
    NameMatcher,
    PartialDate,
    dates_agree,
    display_country,
    normalize_country,
    parse_date,
    strip_accents,
)
from text_extractor import NoteFacts, NoteKYC, extract_note_facts


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    cleaned = re.sub(r"[\s\-().]+", "", str(phone))
    return cleaned or None


@dataclass
class VerifiedIdentity:
    """Most recent approved identity check (from kyc_checks.jsonl or notes.jsonl)."""

    source_id: str
    checked_at: Optional[PartialDate]
    verified_name: Optional[str]
    verified_dob: Optional[PartialDate]
    document_country: Optional[str]
    from_structured_kyc: bool


@dataclass
class CustomerProfile:
    """Resolved picture of one natural person or company across all linked records."""

    primary_record_id: str
    record_ids: List[str]
    entity_type: str
    records: List[Dict[str, Any]]
    latest_verified: Optional[VerifiedIdentity]
    all_approved_kyc_ids: List[str]
    declared_names: List[str]
    declared_dobs: List[PartialDate]
    # country -> list of record IDs (CUS-*, KYC-*, NOTE-*) supporting it
    country_sources: Dict[str, List[str]] = field(default_factory=dict)
    # note_id -> NoteFacts for cited residence/KYC notes
    note_facts: Dict[str, NoteFacts] = field(default_factory=dict)

    @property
    def is_company(self) -> bool:
        return self.entity_type.lower() == "company"

    def best_dob(self) -> Tuple[Optional[PartialDate], Optional[str]]:
        """Authoritative DOB: latest approved KYC check > declared customer DOB."""
        if self.latest_verified and self.latest_verified.verified_dob:
            return self.latest_verified.verified_dob, self.latest_verified.source_id
        for rec in self.records:
            dt = parse_date(rec.get("dob"))
            if dt:
                return dt, rec["record_id"]
        return None, None

    def all_candidate_names(self) -> List[Tuple[str, str]]:
        """All names associated with this person: (name, source_id), verified first."""
        out: List[Tuple[str, str]] = []
        if self.latest_verified and self.latest_verified.verified_name:
            out.append((self.latest_verified.verified_name, self.latest_verified.source_id))
        for rec in self.records:
            for key in ("full_name", "preferred_name", "login_name"):
                val = rec.get(key)
                if val:
                    out.append((val, rec["record_id"]))
        return out

    def countries(self) -> Set[str]:
        return set(self.country_sources.keys())

    def sorted_display_countries(self) -> str:
        return ", ".join(sorted(display_country(c) for c in self.country_sources.keys()))


class UnionFind:
    """Disjoint-set structure for merging records belonging to the same person."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class DataRepository:
    """Loads, indexes, and clusters all screening tables from a data directory."""

    def __init__(self, data_dir: str, llm_client: Optional[LLMClient] = None):
        self.data_dir = data_dir
        self.llm_client = llm_client
        self.matcher = NameMatcher(os.path.join(data_dir, "name_variants.json"))

        self.alerts = read_jsonl(os.path.join(data_dir, "alerts.jsonl"))
        self.hits = read_jsonl(os.path.join(data_dir, "hits.jsonl"))
        self.customers = read_jsonl(os.path.join(data_dir, "customers.jsonl"))
        self.accounts = read_jsonl(os.path.join(data_dir, "accounts.jsonl"))
        self.onboarding = read_jsonl(os.path.join(data_dir, "onboarding.jsonl"))
        self.kyc_checks = read_jsonl(os.path.join(data_dir, "kyc_checks.jsonl"))
        self.notes = read_jsonl(os.path.join(data_dir, "notes.jsonl"))
        self.search_results = read_jsonl(os.path.join(data_dir, "search_results.jsonl"))

        self.alerts_by_id: Dict[str, Dict[str, Any]] = {a["alert_id"]: a for a in self.alerts}
        self.cust_idx_by_id: Dict[str, int] = {
            c["record_id"]: i for i, c in enumerate(self.customers)
        }

        # Account & email lookup indexes
        self.accounts_by_acc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.accounts_by_login: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.accounts_by_cust: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for acc in self.accounts:
            self.accounts_by_acc[acc["account_id"]].append(acc)
            if acc.get("login_handle"):
                self.accounts_by_login[acc["login_handle"]].append(acc)
            self.accounts_by_cust[acc["customer_id"]].append(acc)

        self.custs_by_email: Dict[str, List[int]] = defaultdict(list)
        self.custs_by_country_year: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        self.custs_by_country: Dict[str, List[int]] = defaultdict(list)
        self.custs_by_year: Dict[int, List[int]] = defaultdict(list)
        self.cust_parsed_dobs: List[Optional[PartialDate]] = []
        self.cust_countries: List[Set[str]] = []

        for idx, c in enumerate(self.customers):
            em = (c.get("email") or "").strip().lower()
            if em:
                self.custs_by_email[em].append(idx)
            dt = parse_date(c.get("dob"))
            self.cust_parsed_dobs.append(dt)
            c_countries: Set[str] = set()
            for k in ("residence_country", "nationality"):
                nc = normalize_country(c.get(k))
                if nc:
                    c_countries.add(nc)
            self.cust_countries.append(c_countries)
            for nc in c_countries:
                self.custs_by_country[nc].append(idx)
                if dt:
                    self.custs_by_country_year[(nc, dt.year)].append(idx)
            if dt:
                self.custs_by_year[dt.year].append(idx)

        # Onboarding & KYC indexes
        self.applicants_by_cust: Dict[str, Set[str]] = defaultdict(set)
        self.custs_by_applicant: Dict[str, Set[str]] = defaultdict(set)
        for ob in self.onboarding:
            cid = ob.get("customer_id")
            aid = ob.get("kyc_applicant_id")
            if cid and aid:
                self.applicants_by_cust[cid].add(aid)
                self.custs_by_applicant[aid].add(cid)

        self.kyc_by_applicant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for kc in self.kyc_checks:
            self.kyc_by_applicant[kc["applicant_id"]].append(kc)

        # Notes & Search results indexes
        self.notes_by_email: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for nt in self.notes:
            em = (nt.get("requester_email") or "").strip().lower()
            if em:
                self.notes_by_email[em].append(nt)

        self.search_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sr in self.search_results:
            q = re.sub(r"\s+", " ", strip_accents(sr.get("query") or "").lower().strip())
            if q:
                self.search_by_query[q].append(sr)

        # Build same-person Union-Find clusters (Principle 3)
        self.uf = UnionFind(len(self.customers))
        self._build_person_clusters()
        self.clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self.customers)):
            self.clusters[self.uf.find(idx)].append(idx)

        # Cache parsed NoteFacts by note_id
        self._note_facts_cache: Dict[str, NoteFacts] = {}
        self._profile_cache: Dict[Tuple[int, str], CustomerProfile] = {}

    def _build_person_clusters(self) -> None:
        # 1. Same normalized phone + same full DOB
        phone_dob_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, c in enumerate(self.customers):
            ph = normalize_phone(c.get("phone"))
            dt = self.cust_parsed_dobs[idx]
            if ph and dt and dt.is_full():
                phone_dob_groups[(ph, dt.format())].append(idx)
        for idxs in phone_dob_groups.values():
            if len(idxs) > 1:
                base = idxs[0]
                for other in idxs[1:]:
                    self.uf.union(base, other)

        # 2. Same document_number on approved identity checks
        doc_groups: Dict[str, List[int]] = defaultdict(list)
        for kc in self.kyc_checks:
            if (kc.get("result") or "").lower() != "approved":
                continue
            doc_num = (kc.get("document_number") or "").strip().upper()
            if not doc_num:
                continue
            for cid in self.custs_by_applicant.get(kc["applicant_id"], ()):
                idx = self.cust_idx_by_id.get(cid)
                if idx is not None:
                    doc_groups[doc_num].append(idx)
        for idxs in doc_groups.values():
            if len(idxs) > 1:
                base = idxs[0]
                for other in idxs[1:]:
                    self.uf.union(base, other)

    def _record_fits_alert(self, cust_idx: int, alert: Dict[str, Any]) -> bool:
        """Check Principle 2 criteria for matching an alert's screened details."""
        c = self.customers[cust_idx]
        ac = alert["customer"]
        is_comp = (ac.get("entity_type") or "").lower() == "company"
        if (c.get("entity_type") or "").lower() != (ac.get("entity_type") or "").lower():
            return False

        # Name match against full_name, preferred_name, or login_name
        alert_name = ac.get("name")
        name_ok = any(
            self.matcher.names_agree(c.get(field), alert_name, is_company=is_comp, require_min_two_a=False)
            for field in ("full_name", "preferred_name", "login_name")
            if c.get(field)
        )
        if not name_ok:
            return False

        # DOB equal in every part both give
        adob = parse_date(ac.get("dob"))
        if adob is not None:
            cdob = self.cust_parsed_dobs[cust_idx]
            if cdob is None or not dates_agree(cdob, adob):
                return False

        # Country matches residence or nationality
        acountry = normalize_country(ac.get("country"))
        if acountry is not None:
            if acountry not in self.cust_countries[cust_idx]:
                return False

        # City appears in record's city or address
        acity = strip_accents(ac.get("city") or "").lower().strip()
        if acity:
            ccity = strip_accents(c.get("city") or "").lower()
            caddr = strip_accents(c.get("address") or "").lower()
            if acity not in ccity and acity not in caddr:
                return False

        return True

    def find_matching_customer_clusters(self, alert: Dict[str, Any]) -> List[Tuple[int, str]]:
        """Resolve an alert to list of (cluster_root_idx, preferred_cus_id) pairs."""
        direct_idxs: List[int] = []
        if alert.get("account_ref"):
            for acc in self.accounts_by_acc.get(alert["account_ref"], ()):
                idx = self.cust_idx_by_id.get(acc["customer_id"])
                if idx is not None:
                    direct_idxs.append(idx)
        elif alert.get("login_handle"):
            for acc in self.accounts_by_login.get(alert["login_handle"], ()):
                idx = self.cust_idx_by_id.get(acc["customer_id"])
                if idx is not None:
                    direct_idxs.append(idx)
        elif alert.get("email"):
            em = alert["email"].strip().lower()
            direct_idxs.extend(self.custs_by_email.get(em, ()))

        if direct_idxs:
            # Group direct indices by person cluster
            by_root: Dict[int, List[int]] = defaultdict(list)
            for idx in direct_idxs:
                by_root[self.uf.find(idx)].append(idx)
            if len(by_root) == 1:
                root = next(iter(by_root))
                pref_id = self.customers[by_root[root][0]]["record_id"]
                return [(root, pref_id)]
            # Multiple people on same link (e.g. joint account): filter by screened details
            fitted = [idx for idx in direct_idxs if self._record_fits_alert(idx, alert)]
            if not fitted:
                # Fallback if screened details had minor discrepancy on linked joint holders
                fitted = direct_idxs
            out_roots: Dict[int, str] = {}
            for idx in fitted:
                root = self.uf.find(idx)
                out_roots.setdefault(root, self.customers[idx]["record_id"])
            return list(out_roots.items())

        # NO_LINK alert: narrow search via country + DOB year buckets before Principle 2 verification
        ac = alert["customer"]
        acountry = normalize_country(ac.get("country"))
        adob = parse_date(ac.get("dob"))

        if acountry and adob:
            candidates = self.custs_by_country_year.get((acountry, adob.year), [])
        elif acountry:
            candidates = self.custs_by_country.get(acountry, [])
        elif adob:
            candidates = self.custs_by_year.get(adob.year, [])
        else:
            candidates = range(len(self.customers))

        out_roots = {}
        for idx in candidates:
            if self._record_fits_alert(idx, alert):
                root = self.uf.find(idx)
                out_roots.setdefault(root, self.customers[idx]["record_id"])
        return list(out_roots.items())

    def get_note_facts(self, note: Dict[str, Any]) -> NoteFacts:
        nid = note["note_id"]
        if nid not in self._note_facts_cache:
            self._note_facts_cache[nid] = extract_note_facts(note, llm_client=self.llm_client)
        return self._note_facts_cache[nid]

    def build_customer_profile(self, cluster_root: int, preferred_cus_id: str) -> CustomerProfile:
        """Assemble the full authoritative CustomerProfile for a same-person cluster."""
        key = (cluster_root, preferred_cus_id)
        if key in self._profile_cache:
            return self._profile_cache[key]

        idxs = self.clusters[cluster_root]
        records = [self.customers[i] for i in idxs]
        # Order so preferred_cus_id comes first
        records.sort(key=lambda r: (0 if r["record_id"] == preferred_cus_id else 1, r["record_id"]))
        record_ids = [r["record_id"] for r in records]
        entity_type = records[0].get("entity_type") or "Individual"

        country_sources: Dict[str, List[str]] = defaultdict(list)
        declared_names: List[str] = []
        declared_dobs: List[PartialDate] = []
        emails: Set[str] = set()
        applicant_ids: Set[str] = set()

        for rec in records:
            rid = rec["record_id"]
            for k in ("full_name", "preferred_name", "login_name"):
                if rec.get(k):
                    declared_names.append(rec[k])
            dt = parse_date(rec.get("dob"))
            if dt:
                declared_dobs.append(dt)
            for k in ("residence_country", "nationality"):
                nc = normalize_country(rec.get(k))
                if nc and rid not in country_sources[nc]:
                    country_sources[nc].append(rid)
            em = (rec.get("email") or "").strip().lower()
            if em:
                emails.add(em)
            applicant_ids |= self.applicants_by_cust.get(rid, set())

        # Gather all approved KYC checks (structured + from notes)
        verified_candidates: List[VerifiedIdentity] = []
        all_approved_kyc_ids: List[str] = []

        for aid in sorted(applicant_ids):
            for kc in self.kyc_by_applicant.get(aid, ()):
                if (kc.get("result") or "").lower() != "approved":
                    continue
                kid = kc["check_id"]
                all_approved_kyc_ids.append(kid)
                doc_c = normalize_country(kc.get("document_country"))
                if doc_c and kid not in country_sources[doc_c]:
                    country_sources[doc_c].append(kid)
                verified_candidates.append(
                    VerifiedIdentity(
                        source_id=kid,
                        checked_at=parse_date(kc.get("checked_at")),
                        verified_name=kc.get("verified_name"),
                        verified_dob=parse_date(kc.get("verified_dob")),
                        document_country=doc_c,
                        from_structured_kyc=True,
                    )
                )

        # Gather notes across all emails in cluster
        note_facts_map: Dict[str, NoteFacts] = {}
        for em in sorted(emails):
            for nt in self.notes_by_email.get(em, ()):
                nf = self.get_note_facts(nt)
                note_facts_map[nf.note_id] = nf
                if nf.kyc and nf.kyc.approved:
                    all_approved_kyc_ids.append(nf.note_id)
                    if nf.kyc.document_country and nf.note_id not in country_sources[nf.kyc.document_country]:
                        country_sources[nf.kyc.document_country].append(nf.note_id)
                    verified_candidates.append(
                        VerifiedIdentity(
                            source_id=nf.note_id,
                            checked_at=nf.kyc.checked_at,
                            verified_name=nf.kyc.verified_name,
                            verified_dob=nf.kyc.verified_dob,
                            document_country=nf.kyc.document_country,
                            from_structured_kyc=False,
                        )
                    )
                for lc in nf.lived_countries:
                    if nf.note_id not in country_sources[lc]:
                        country_sources[lc].append(nf.note_id)

        latest_verified: Optional[VerifiedIdentity] = None
        if verified_candidates:
            verified_candidates.sort(
                key=lambda v: (
                    v.checked_at.tuple_key() if v.checked_at else (0, 0, 0),
                    1 if v.from_structured_kyc else 0,
                    v.source_id,
                ),
                reverse=True,
            )
            latest_verified = verified_candidates[0]

        profile = CustomerProfile(
            primary_record_id=preferred_cus_id,
            record_ids=record_ids,
            entity_type=entity_type,
            records=records,
            latest_verified=latest_verified,
            all_approved_kyc_ids=sorted(set(all_approved_kyc_ids)),
            declared_names=declared_names,
            declared_dobs=declared_dobs,
            country_sources=dict(country_sources),
            note_facts=note_facts_map,
        )
        self._profile_cache[key] = profile
        return profile
