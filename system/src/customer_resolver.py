"""Alert-level customer resolution, Principle 3 merging, and defensive note harvesting."""

from __future__ import annotations

from collections import defaultdict
import re
import threading
from typing import Any, Callable, TypeVar

from .ingest import DataStore
from .models import (
    Claim,
    KycRecord,
    PersonProfile,
    PartialDate,
    ResolvedAlert,
    TIER_0_EXCLUDED,
    TIER_1_VERIFIED,
    TIER_2_DECLARED,
    TIER_3_CONTEXTUAL,
)
from .normalize import (
    MONTH_MAP,
    fold_text,
    parse_partial_date,
)

_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_DATE_SPAN_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}\.\d{{1,2}}\.\d{{4}}"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?,?\s+\d{{4}}|{_MONTH_PAT}\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|{_MONTH_PAT}\.?,?\s+\d{{4}})\b",
    re.I,
)

_DOB_CUE_RE = re.compile(
    rf"\b(?:DOB|date\s+of\s+birth|birth\s+date|born|incorporated|established|founded)\b[^0-9A-Za-z\n]{{0,18}}"
    rf"(?:on\s+|in\s+|is\s+|as\s+|of\s+)?"
    rf"(\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}\.\d{{1,2}}\.\d{{4}}"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?,?\s+\d{{4}}|{_MONTH_PAT}\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|{_MONTH_PAT}\.?,?\s+\d{{4}}|\d{{4}})",
    re.I,
)

_POSSESSIVE_BIRTH_RE = re.compile(
    rf"\b(\d{{1,2}}\s+{_MONTH_PAT}\s+\d{{4}}|{_MONTH_PAT}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})\s+birth\s+date\b",
    re.I,
)

_KYC_NEGATIVE_RE = re.compile(
    r"\b(?:reject(?:ed)?|expir(?:ed|es|ing)?|fail(?:ed|s|ing)?|declin(?:ed)?|invalid"
    r"|denied|refus(?:ed)?|unsuccessful|not\s+verif(?:ied)?|unable\s+to\s+verif(?:y|ied)|not\s+approved)\b",
    re.I,
)
_KYC_POSITIVE_RE = re.compile(
    r"\b(?:approv(?:ed)?|verif(?:ied)?|validat(?:ed)?|pass(?:ed)?|clear(?:ed)?|confirm(?:ed)?|accept(?:ed)?)\b",
    re.I,
)

_ALLCAPS_SPAN_RE = re.compile(
    r"(?<![A-Za-z])((?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ]{2,})(?:[\s'\-]+(?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ]{2,})){1,6})(?='s\b|[^A-Za-z0-9]|$)"
)

_TITLE_NAME_PATTERNS = [
    re.compile(
        r"\b(?:for|of|matching|matches|held\s+by|belonging\s+to|named)\s+"
        r"((?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][^\s,;()\n]+)"
        r"(?:\s+(?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][^\s,;()\n]+)){1,5}?)"
        r"(?='s\b|\s*(?:,|\(|\bon\s+\d|\bborn\b|\bDOB\b|\busing\b|\bvia\b|\bthrough\b|\bfrom\b|\bwith\b|\.|$))"
    ),
    re.compile(
        r"^((?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][^\s,;()\n]+)"
        r"(?:\s+(?:0000|N/A|[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][^\s,;()\n]+)){1,5}?)"
        r"(?='s\b|\s*(?:\(|,\s*(?:born|DOB)|\b(?:passed|was|had|has|provided|cleared|submitted)\b))"
    ),
]

_STOP_UPPER_TOKENS = frozenset(
    {"DOB", "ID", "KYC", "USA", "UK", "UAE", "SUMSUB", "VERIFF", "PERSONA", "ONFIDO"}
)

_STRONG_RESIDENCE_CUE_RE = re.compile(
    r"\b(?:liv(?:e|ed|ing)|resid(?:e|ed|ing|ence|ences|ent|ents|ential|ency)|"
    r"dwell(?:ed|ing)?|household|housing|habitation|domicil(?:e|ed)|relocat(?:e|ed|ing)|moved\s+to)\b",
    re.I,
)

_RESIDENCE_CUE_RE = re.compile(
    r"\b(?:liv(?:e|ed|ing)|resid(?:e|ed|ing|ence|ences|ent|ents|ential|ency)|"
    r"work(?:ed|ing|er|ers|force)?|employ(?:ed|ment|ee|ees)?|"
    r"home|house|housing|household|habitation|dwell(?:ed|ing)?|occup(?:y|ied|ying)|"
    r"bas(?:e|ed)|career|tenure|relocat(?:e|ed|ing)|stay(?:ed|ing)?|"
    r"spent\s+time|time\s+spent|domicil(?:e|ed)|domestic|personal\s+life|"
    r"private\s+life|life\s+and\s+labor|personal\s+ties|professional|"
    r"presence|situated|located|address|moved\s+to|grew\s+up|settled|job|jobs|established|professionally|personally)\b",
    re.I,
)

_TRANSFER_ONLY_RE = re.compile(
    r"\b(?:transfer|remittance|payment|wire|sent\s+money)\b",
    re.I,
)

_NOISE_CONTEXT_RE = re.compile(
    r"\b(?:transfer|remittance|payment|paid|wire|sent\s+money|payee|merchant|hotel|holiday|vacation|card|employer|headquartered|branch)\b",
    re.I,
)


_T = TypeVar("_T")


class CustomerResolver:
    """Resolves alerts to customer PersonProfiles and mines KYC & residence notes."""

    def __init__(self, store: DataStore, llm_client: Any = None, verifier: Any = None) -> None:
        self.store = store
        self.llm_client = llm_client
        self.verifier = verifier
        self._lock = threading.Lock()
        self._alert_cache: dict[str, ResolvedAlert] = {}
        self._alert_inflight: dict[str, threading.Event] = {}
        self._kyc_note_cache: dict[str, KycRecord] = {}
        self._kyc_note_inflight: dict[str, threading.Event] = {}
        self._res_note_cache: dict[str, list[str]] = {}
        self._res_note_inflight: dict[str, threading.Event] = {}
        self.stats = {
            "alerts_resolved": 0,
            "kyc_notes_parsed": 0,
            "kyc_notes_llm": 0,
            "res_notes_parsed": 0,
            "res_notes_llm": 0,
        }

    def _cached_single_flight(
        self,
        cache: dict[str, _T],
        inflight: dict[str, threading.Event],
        key: str,
        compute_fn: Callable[[], _T],
    ) -> _T:
        wait_evt: threading.Event | None = None
        my_evt: threading.Event | None = None
        with self._lock:
            if key in cache:
                return cache[key]
            if key in inflight:
                wait_evt = inflight[key]
            else:
                my_evt = threading.Event()
                inflight[key] = my_evt

        if wait_evt is not None:
            wait_evt.wait()
            with self._lock:
                return cache[key]

        try:
            result = compute_fn()
        except BaseException:
            with self._lock:
                inflight.pop(key, None)
            if my_evt is not None:
                my_evt.set()
            raise

        with self._lock:
            cache[key] = result
            inflight.pop(key, None)
        if my_evt is not None:
            my_evt.set()
        return result

    def _inc_stat(self, stat_key: str) -> None:
        with self._lock:
            self.stats[stat_key] += 1

    def resolve_alert(self, alert: dict[str, Any]) -> ResolvedAlert:
        aid = alert["alert_id"]
        return self._cached_single_flight(
            self._alert_cache,
            self._alert_inflight,
            aid,
            lambda: self._resolve_alert_uncached(alert),
        )

    def _resolve_alert_uncached(self, alert: dict[str, Any]) -> ResolvedAlert:
        aid = alert["alert_id"]
        self._inc_stat("alerts_resolved")
        screened = alert["customer"]
        link_sets: list[list[str]] = []
        if alert.get("account_ref"):
            ids = [acc["customer_id"] for acc in self.store.accounts_by_id.get(alert["account_ref"], ())]
            if ids:
                link_sets.append(ids)
        if alert.get("login_handle"):
            ids = [
                acc["customer_id"]
                for acc in self.store.accounts_by_login.get(alert["login_handle"].strip().lower(), ())
            ]
            if ids:
                link_sets.append(ids)
        if alert.get("email"):
            ids = list(self.store.customers_by_email.get(alert["email"].strip().lower(), ()))
            if ids:
                link_sets.append(ids)

        direct_cids: list[str] = []
        if len(link_sets) == 1:
            direct_cids = list(dict.fromkeys(link_sets[0]))
        elif len(link_sets) > 1:
            common = set(link_sets[0])
            for s in link_sets[1:]:
                common &= set(s)
            if common:
                direct_cids = [cid for cid in dict.fromkeys(link_sets[0]) if cid in common]
            else:
                seen: set[str] = set()
                for s in link_sets:
                    for cid in s:
                        if cid not in seen:
                            seen.add(cid)
                            direct_cids.append(cid)

        if direct_cids:
            # Group directly linked records by Principle 3 person cluster
            by_cluster: dict[str, list[str]] = defaultdict(list)
            for cid in direct_cids:
                clid = self.store.cluster_id_by_customer[cid]
                by_cluster[clid].append(cid)

            cluster_ids = sorted(by_cluster.keys())
            if len(cluster_ids) > 1:
                # Joint account or shared login/email across multiple distinct people:
                # use screened details to pick between them per Principle 2.
                fitting = [
                    clid
                    for clid in cluster_ids
                    if self._cluster_fits_screened(clid, screened)
                ]
                if len(fitting) == 1:
                    chosen_clid = fitting[0]
                    profile = self._build_person_profile(
                        chosen_clid,
                        direct_cids=set(by_cluster[chosen_clid]),
                        alert=alert,
                    )
                    res = ResolvedAlert(
                        alert_id=aid,
                        resolution_status="LINKED",
                        customer_record_id=profile.canonical_record_id,
                        candidates=[profile],
                    )
                    return res
                use_clusters = fitting if fitting else cluster_ids
                profiles = [
                    self._build_person_profile(
                        clid,
                        direct_cids=set(by_cluster[clid]),
                        alert=alert,
                    )
                    for clid in use_clusters
                ]
                res = ResolvedAlert(
                    alert_id=aid,
                    resolution_status="AMBIGUOUS",
                    customer_record_id="AMBIGUOUS",
                    candidates=profiles,
                )
                return res

            only_clid = cluster_ids[0]
            profile = self._build_person_profile(
                only_clid,
                direct_cids=set(by_cluster[only_clid]),
                alert=alert,
            )
            res = ResolvedAlert(
                alert_id=aid,
                resolution_status="LINKED",
                customer_record_id=profile.canonical_record_id,
                candidates=[profile],
            )
            return res

        # Unlinked alert: search CRM by Principle 2 screened criteria
        matching_clusters = self._find_unlinked_clusters(screened)
        if not matching_clusters:
            fallback_profile = self._build_unresolved_profile(alert)
            res = ResolvedAlert(
                alert_id=aid,
                resolution_status="UNRESOLVED",
                customer_record_id=None,
                candidates=[fallback_profile],
            )
            return res

        if len(matching_clusters) == 1:
            only_clid = matching_clusters[0]
            profile = self._build_person_profile(only_clid, direct_cids=set(), alert=alert)
            res = ResolvedAlert(
                alert_id=aid,
                resolution_status="INFERRED_UNIQUE",
                customer_record_id=profile.canonical_record_id,
                candidates=[profile],
            )
            return res

        profiles = [
            self._build_person_profile(clid, direct_cids=set(), alert=alert)
            for clid in matching_clusters
        ]
        res = ResolvedAlert(
            alert_id=aid,
            resolution_status="AMBIGUOUS",
            customer_record_id="AMBIGUOUS",
            candidates=profiles,
        )
        return res

    def _find_unlinked_clusters(self, screened: dict[str, Any]) -> list[str]:
        s_type = screened["entity_type"]
        is_comp = s_type == "Company"
        s_name = screened.get("name") or ""
        s_dob = parse_partial_date(screened.get("dob"), day_first=True)
        s_country = self.store.geo.normalize_structured_country(screened.get("country"))
        s_city = screened.get("city")

        # Gather candidate record IDs via inverted index
        idx_tokens = self.store.name_matcher.index_tokens(s_name, is_company=is_comp)
        name_pool: set[str] = set()
        for tok in idx_tokens:
            name_pool.update(self.store.cust_by_type_token.get((s_type, tok), ()))

        # If country & year are both given, also include intersection of (type, year) & (type, country)
        # so multi-typo names aren't missed.
        if s_dob and s_country:
            year_set = set(self.store.cust_by_type_year.get((s_type, s_dob.year), ()))
            country_set = set(self.store.cust_by_type_country.get((s_type, s_country), ()))
            candidate_ids = name_pool | (year_set & country_set)
        elif s_dob:
            year_set = set(self.store.cust_by_type_year.get((s_type, s_dob.year), ()))
            candidate_ids = name_pool | year_set
        elif s_country:
            country_set = set(self.store.cust_by_type_country.get((s_type, s_country), ()))
            candidate_ids = country_set
        else:
            candidate_ids = name_pool

        matched_clusters: set[str] = set()
        for cid in candidate_ids:
            rec = self.store.customers_by_id[cid]
            if self._record_fits_screened(rec, screened, s_dob, s_country, s_city):
                matched_clusters.add(self.store.cluster_id_by_customer[cid])
        return sorted(matched_clusters)

    def _record_fits_screened(
        self,
        rec: dict[str, Any],
        screened: dict[str, Any],
        s_dob: PartialDate | None,
        s_country: str | None,
        s_city: str | None,
    ) -> bool:
        s_type = screened["entity_type"]
        if rec["entity_type"] != s_type:
            return False
        is_comp = s_type == "Company"
        s_name = screened.get("name") or ""

        # Name check: full_name, full_name+preferred_name, preferred_name (if >=2 real parts), login_name
        nm = self.store.name_matcher
        name_ok = nm.names_agree(s_name, rec.get("full_name"), is_company=is_comp)
        if not name_ok and rec.get("preferred_name"):
            pref = rec["preferred_name"]
            if nm.real_part_count(pref, is_company=is_comp) >= 2:
                name_ok = nm.names_agree(s_name, pref, is_company=is_comp)
            if not name_ok:
                name_ok = nm.names_agree(
                    rec.get("full_name"),
                    s_name,
                    is_company=is_comp,
                    extra_parts_a=[pref],
                )
        if not name_ok and rec.get("login_name"):
            ln = rec["login_name"]
            name_ok = nm.names_agree(s_name, ln, is_company=is_comp)
            if not name_ok and not is_comp:
                ln_nodigits = re.sub(r"\d+$", "", ln).strip()
                if ln_nodigits and ln_nodigits != ln:
                    name_ok = nm.names_agree(s_name, ln_nodigits, is_company=False)
        if not name_ok:
            return False

        # DOB check: equal in every part both give
        if s_dob is not None:
            r_dob = parse_partial_date(rec.get("dob"), day_first=True)
            if r_dob is None:
                # Check if record has approved KYC DOB in kyc_checks.jsonl
                for k in self.store.kyc_checks_by_customer.get(rec["record_id"], ()):
                    if (k.get("result") or "").lower() == "approved":
                        kd = parse_partial_date(k.get("verified_dob"), day_first=True)
                        if kd is not None:
                            r_dob = kd
                            break
            if r_dob is None and rec.get("email"):
                # Also check approved KYC notes filed under this record's email
                for note in self.store.notes_by_email.get(rec["email"].strip().lower(), ()):
                    if note.get("channel") == "kyc":
                        kr = self.parse_kyc_note(note)
                        if kr.result == "approved" and kr.verified_dob is not None:
                            r_dob = kr.verified_dob
                            break
            if r_dob is not None:
                if not _partial_dates_equal_shared_parts(s_dob, r_dob):
                    return False

        # Country check: screened country appears as residence or nationality
        if s_country is not None:
            r_res = self.store.geo.normalize_structured_country(rec.get("residence_country"))
            r_nat = self.store.geo.normalize_structured_country(rec.get("nationality"))
            if s_country != r_res and s_country != r_nat:
                return False

        # City check: screened city in record's city or address
        if s_city:
            if not self.store.geo.city_matches(s_city, rec.get("city"), rec.get("address")):
                return False

        return True

    def _cluster_fits_screened(self, cluster_id: str, screened: dict[str, Any]) -> bool:
        s_dob = parse_partial_date(screened.get("dob"), day_first=True)
        s_country = self.store.geo.normalize_structured_country(screened.get("country"))
        s_city = screened.get("city")
        for cid in self.store.cluster_members[cluster_id]:
            rec = self.store.customers_by_id[cid]
            if self._record_fits_screened(rec, screened, s_dob, s_country, s_city):
                return True
        return False

    def _build_unresolved_profile(self, alert: dict[str, Any]) -> PersonProfile:
        sc = alert["customer"]
        s_dob = parse_partial_date(sc.get("dob"), day_first=True)
        s_country = self.store.geo.normalize_structured_country(sc.get("country"))
        country_sources: dict[str, list[str]] = {}
        if s_country:
            country_sources[s_country] = []
        return PersonProfile(
            cluster_id="UNRESOLVED",
            record_ids=[],
            canonical_record_id=None,
            entity_type=sc["entity_type"],
            verified_name=None,
            verified_name_source_id=None,
            declared_names=[(sc["name"], alert["alert_id"])] if sc.get("name") else [],
            effective_name=sc.get("name"),
            dob=s_dob,
            dob_source_id=None,
            dob_is_verified=False,
            country_sources=country_sources,
            emails=[alert["email"]] if alert.get("email") else [],
            excluded_kyc_ids=[],
            claims=[],
            is_unresolved=True,
        )

    def _build_person_profile(
        self,
        cluster_id: str,
        direct_cids: set[str],
        alert: dict[str, Any],
    ) -> PersonProfile:
        record_ids = list(self.store.cluster_members[cluster_id])
        recs = [self.store.customers_by_id[cid] for cid in record_ids]
        etype = recs[0]["entity_type"]

        emails: list[str] = []
        seen_emails: set[str] = set()
        for r in recs:
            em = (r.get("email") or "").strip().lower()
            if em and em not in seen_emails:
                seen_emails.add(em)
                emails.append(em)
        alert_em = (alert.get("email") or "").strip().lower()
        if alert_em and alert_em not in seen_emails and (direct_cids & set(record_ids)):
            seen_emails.add(alert_em)
            emails.append(alert_em)

        approved_checks: list[KycRecord] = []
        excluded_kyc_ids: list[str] = []
        claims: list[Claim] = []

        # 1. Gather structured KYC checks across all records in cluster
        for r in recs:
            cid = r["record_id"]
            for k in self.store.kyc_checks_by_customer.get(cid, ()):
                res_str = (k.get("result") or "").lower()
                kd = parse_partial_date(k.get("verified_dob"), day_first=True)
                kc = self.store.geo.normalize_structured_country(k.get("document_country"))
                if res_str == "approved":
                    approved_checks.append(
                        KycRecord(
                            source_id=k["check_id"],
                            source_kind="kyc_check",
                            checked_at=k.get("checked_at") or "",
                            result="approved",
                            verified_name=k.get("verified_name"),
                            verified_dob=kd,
                            document_country=kc,
                            document_number=k.get("document_number"),
                            customer_id=cid,
                        )
                    )
                else:
                    excluded_kyc_ids.append(k["check_id"])
                    claims.append(
                        Claim(
                            side="CUSTOMER",
                            attribute="EXCLUDED_KYC",
                            raw_value=res_str,
                            normalized_value=res_str,
                            trust_tier=TIER_0_EXCLUDED,
                            source_id=k["check_id"],
                        )
                    )

        # 2. Gather all notes for all emails in cluster
        country_sources: dict[str, list[str]] = defaultdict(list)
        for em in emails:
            for note in self.store.notes_by_email.get(em, ()):
                nid = note["note_id"]
                if note.get("channel") == "kyc" or self._looks_like_kyc_note(note["text"]):
                    kr = self.parse_kyc_note(note)
                    if kr.result == "approved":
                        approved_checks.append(kr)
                    elif kr.result in ("rejected", "expired"):
                        excluded_kyc_ids.append(nid)
                        claims.append(
                            Claim(
                                side="CUSTOMER",
                                attribute="EXCLUDED_KYC",
                                raw_value=kr.result,
                                normalized_value=kr.result,
                                trust_tier=TIER_0_EXCLUDED,
                                source_id=nid,
                            )
                        )
                else:
                    past_isos = self.parse_residence_note(note)
                    for iso in past_isos:
                        if nid not in country_sources[iso]:
                            country_sources[iso].append(nid)
                        claims.append(
                            Claim(
                                side="CUSTOMER",
                                attribute="RESIDENCE_HISTORICAL",
                                raw_value=note["text"],
                                normalized_value=iso,
                                trust_tier=TIER_3_CONTEXTUAL,
                                source_id=nid,
                                verbatim_quote=note["text"],
                            )
                        )

        # Sort approved checks newest-first deterministically (prefer structured kyc_check on ties)
        approved_checks.sort(
            key=lambda x: (x.checked_at or "", 1 if x.source_kind == "kyc_check" else 0, x.source_id),
            reverse=True,
        )

        # Principle 4: Most recent approved check is most reliable for legal name, DOB, doc country
        verified_name: str | None = None
        verified_name_src: str | None = None
        verified_dob: PartialDate | None = None
        verified_dob_src: str | None = None
        latest_kyc_cid: str | None = None

        for idx, chk in enumerate(approved_checks):
            if idx == 0 and chk.customer_id:
                latest_kyc_cid = chk.customer_id
            if verified_name is None and chk.verified_name:
                verified_name = chk.verified_name
                verified_name_src = chk.source_id
                claims.append(
                    Claim(
                        side="CUSTOMER",
                        attribute="NAME",
                        raw_value=chk.verified_name,
                        normalized_value=chk.verified_name,
                        trust_tier=TIER_1_VERIFIED,
                        source_id=chk.source_id,
                    )
                )
            if verified_dob is None and chk.verified_dob is not None:
                verified_dob = chk.verified_dob
                verified_dob_src = chk.source_id
                claims.append(
                    Claim(
                        side="CUSTOMER",
                        attribute="DOB",
                        raw_value=chk.verified_dob.raw or chk.verified_dob.iso_str,
                        normalized_value=chk.verified_dob,
                        trust_tier=TIER_1_VERIFIED,
                        source_id=chk.source_id,
                    )
                )
            if chk.document_country:
                if chk.source_id not in country_sources[chk.document_country]:
                    country_sources[chk.document_country].append(chk.source_id)
                claims.append(
                    Claim(
                        side="CUSTOMER",
                        attribute="DOC_COUNTRY",
                        raw_value=chk.document_country,
                        normalized_value=chk.document_country,
                        trust_tier=TIER_1_VERIFIED,
                        source_id=chk.source_id,
                    )
                )

        # Declared CRM records: names, DOB fallback, residence/nationality countries
        declared_names: list[tuple[str, str]] = []
        crm_dob: PartialDate | None = None
        crm_dob_src: str | None = None
        for r in recs:
            cid = r["record_id"]
            if r.get("full_name"):
                declared_names.append((r["full_name"], cid))
                claims.append(
                    Claim(
                        side="CUSTOMER",
                        attribute="NAME",
                        raw_value=r["full_name"],
                        normalized_value=r["full_name"],
                        trust_tier=TIER_2_DECLARED,
                        source_id=cid,
                    )
                )
            if r.get("preferred_name"):
                pref = r["preferred_name"]
                if self.store.name_matcher.real_part_count(
                    pref, is_company=(etype == "Company")
                ) >= 2:
                    declared_names.append((pref, cid))
                elif r.get("full_name"):
                    declared_names.append((f"{r['full_name']} {pref}", cid))
            if r.get("login_name"):
                declared_names.append((r["login_name"], cid))

            rdob = parse_partial_date(r.get("dob"), day_first=True)
            if rdob is not None and (
                crm_dob is None or (rdob.is_full and not crm_dob.is_full)
            ):
                crm_dob = rdob
                crm_dob_src = cid

            for field_name, attr in (("residence_country", "RESIDENCE_CURRENT"), ("nationality", "NATIONALITY")):
                iso = self.store.geo.normalize_structured_country(r.get(field_name))
                if iso:
                    if cid not in country_sources[iso]:
                        country_sources[iso].append(cid)
                    claims.append(
                        Claim(
                            side="CUSTOMER",
                            attribute=attr,
                            raw_value=str(r.get(field_name)),
                            normalized_value=iso,
                            trust_tier=TIER_2_DECLARED,
                            source_id=cid,
                        )
                    )

        # Fallback to screened alert DOB only if neither KYC nor CRM provided one
        dob = verified_dob or crm_dob
        dob_src = verified_dob_src if verified_dob is not None else crm_dob_src
        dob_is_verified = verified_dob is not None
        if dob is None and alert["customer"].get("dob"):
            sd = parse_partial_date(alert["customer"]["dob"], day_first=True)
            if sd is not None:
                dob = sd
                dob_src = recs[0]["record_id"]

        effective_name = verified_name or (declared_names[0][0] if declared_names else alert["customer"]["name"])

        # Deterministic canonical customer_record_id selection
        canonical_id = sorted(
            recs,
            key=lambda r: (
                0 if r["record_id"] in direct_cids else 1,
                0 if r["record_id"] == latest_kyc_cid else 1,
                0 if (r.get("status") or "").lower() == "active" else 1,
                r["record_id"],
            ),
        )[0]["record_id"]

        for iso in country_sources:
            country_sources[iso].sort()

        return PersonProfile(
            cluster_id=cluster_id,
            record_ids=record_ids,
            canonical_record_id=canonical_id,
            entity_type=etype,
            verified_name=verified_name,
            verified_name_source_id=verified_name_src,
            declared_names=declared_names,
            effective_name=effective_name,
            dob=dob,
            dob_source_id=dob_src,
            dob_is_verified=dob_is_verified,
            country_sources=dict(country_sources),
            emails=emails,
            excluded_kyc_ids=sorted(set(excluded_kyc_ids)),
            claims=claims,
            is_unresolved=False,
        )

    def _looks_like_kyc_note(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:Sumsub|Veriff|Persona|Onfido|Companies\s+registry\s+lookup|registry\s+extract|"
                r"identity\s+check|identity\s+verification|passport|national\s+ID|driving\s+licence)\b",
                text,
                re.I,
            )
        )

    def parse_kyc_note(self, note: dict[str, Any]) -> KycRecord:
        nid = note["note_id"]
        return self._cached_single_flight(
            self._kyc_note_cache,
            self._kyc_note_inflight,
            nid,
            lambda: self._parse_kyc_note_uncached(note),
        )

    def _parse_kyc_note_uncached(self, note: dict[str, Any]) -> KycRecord:
        nid = note["note_id"]
        self._inc_stat("kyc_notes_parsed")
        text = note["text"]
        created_at = note.get("created_at") or ""

        has_neg = bool(_KYC_NEGATIVE_RE.search(text))
        has_pos = bool(_KYC_POSITIVE_RE.search(text))
        if has_neg and not has_pos:
            return KycRecord(
                source_id=nid,
                source_kind="kyc_note",
                checked_at=created_at,
                result="rejected",
            )

        # Extract DOB via explicit DOB/born cue
        verified_dob: PartialDate | None = None
        m_dob = _DOB_CUE_RE.search(text)
        if m_dob:
            verified_dob = parse_partial_date(m_dob.group(1), day_first=True)
        if verified_dob is None:
            m_poss = _POSSESSIVE_BIRTH_RE.search(text)
            if m_poss:
                verified_dob = parse_partial_date(m_poss.group(1), day_first=True)

        # Extract explicit verification date if stated in text; otherwise fall back to created_at
        checked_at = created_at
        m_chk_dt = re.search(
            r"\b(?:on|as\s+of|dated|checked\s+on)\s+(\d{4}-\d{2}-\d{2})\b",
            text,
            re.I,
        )
        if m_chk_dt:
            cand_chk = parse_partial_date(m_chk_dt.group(1), day_first=True)
            if cand_chk and cand_chk.is_full and (
                verified_dob is None or cand_chk.iso_str != verified_dob.iso_str
            ):
                checked_at = cand_chk.iso_str

        # Fallback: find non-checked_at date in note whose year is clearly a birth year
        if verified_dob is None:
            checked_yr = int(checked_at[:4]) if len(checked_at) >= 4 and checked_at[:4].isdigit() else 2026
            for m_dt in _DATE_SPAN_RE.finditer(text):
                cand = parse_partial_date(m_dt.group(0), day_first=True)
                if cand and cand.iso_str != checked_at and cand.year <= checked_yr - 14:
                    verified_dob = cand
                    break

        # Extract verified_name: ALL-CAPS span first, then syntactic Title-Case fallback
        verified_name: str | None = None
        for m_cap in _ALLCAPS_SPAN_RE.finditer(text):
            span = m_cap.group(1).strip()
            toks = [t for t in span.split() if t not in _STOP_UPPER_TOKENS]
            if len(toks) >= 2:
                verified_name = " ".join(toks)
                break
        if verified_name is None:
            for pat in _TITLE_NAME_PATTERNS:
                m_tn = pat.search(text)
                if m_tn:
                    cand_name = m_tn.group(1).strip(" ,.;:()")
                    # Strip leading boilerplate words if matched
                    cand_words = cand_name.split()
                    while cand_words and cand_words[0].lower() in {
                        "the", "on", "per", "via", "as", "using", "through", "for", "of", "subject"
                    }:
                        cand_words.pop(0)
                    if len(cand_words) >= 2:
                        verified_name = " ".join(cand_words)
                        break

        # Extract document country via prose geo extractor (masking verified_name to prevent collisions)
        mask = [verified_name] if verified_name else []
        countries = self.store.geo.extract_countries_from_prose(text, mask_names=mask)
        doc_country = countries[0] if countries else None

        # Trigger-gated LLM fallback when deterministic extractor missed key fields or status is mixed
        if (
            self.llm_client is not None
            and self.llm_client.enabled
            and (
                (has_neg and has_pos)
                or verified_name is None
                or verified_dob is None
                or doc_country is None
            )
        ):
            llm_res = self.llm_client.extract_kyc_note(text)
            if llm_res and self.verifier:
                verified_rec = self.verifier.verify_kyc_note_extraction(
                    llm_res, text, nid, checked_at
                )
                if verified_rec is not None:
                    self._inc_stat("kyc_notes_llm")
                    return verified_rec

        if has_neg and has_pos:
            # Allow approval only if a clear contrastive re-verification succeeded after a prior failure
            contrast_ok = bool(
                re.search(
                    r"\b(?:but|however|subsequently|after(?:ward)?s?|later)\b[^.\n]{0,60}"
                    r"\b(?:approv(?:ed)?|verif(?:ied)?|validat(?:ed)?|pass(?:ed)?|clear(?:ed)?|accept(?:ed)?)\b",
                    text,
                    re.I,
                )
            ) and not bool(
                re.search(
                    r"\b(?:not|never|unable\s+to|could\s+not|failed\s+to)\s+"
                    r"(?:be\s+)?(?:approv(?:ed)?|verif(?:ied)?|validat(?:ed)?|pass|clear|accept)\b",
                    text,
                    re.I,
                )
            )
            result = "approved" if contrast_ok else "rejected"
        else:
            result = "rejected" if has_neg else "approved"
        return KycRecord(
            source_id=nid,
            source_kind="kyc_note",
            checked_at=checked_at,
            result=result,
            verified_name=verified_name,
            verified_dob=verified_dob,
            document_country=doc_country,
        )

    def parse_residence_note(self, note: dict[str, Any]) -> list[str]:
        nid = note["note_id"]
        return self._cached_single_flight(
            self._res_note_cache,
            self._res_note_inflight,
            nid,
            lambda: self._parse_residence_note_uncached(note),
        )

    def _parse_residence_note_uncached(self, note: dict[str, Any]) -> list[str]:
        self._inc_stat("res_notes_parsed")
        text = note["text"]
        countries = self.store.geo.extract_countries_from_prose(text)
        if not countries:
            return []

        has_strong = bool(_STRONG_RESIDENCE_CUE_RE.search(text))
        has_noise = bool(_NOISE_CONTEXT_RE.search(text))
        has_res_cue = has_strong or (bool(_RESIDENCE_CUE_RE.search(text)) and not has_noise)
        if has_res_cue:
            return countries

        # If note mentions a country ambiguously or alongside transaction/noise terms, verify via LLM
        if self.llm_client is not None and self.llm_client.enabled and (
            not _TRANSFER_ONLY_RE.search(text) or bool(_RESIDENCE_CUE_RE.search(text))
        ):
            llm_res = self.llm_client.extract_residence_note(text)
            if llm_res and self.verifier:
                self._inc_stat("res_notes_llm")
                return self.verifier.verify_residence_note_extraction(llm_res, text)

        return []


def _partial_dates_equal_shared_parts(a: PartialDate, b: PartialDate) -> bool:
    if a.year != b.year:
        return False
    if a.month is not None and b.month is not None and a.month != b.month:
        return False
    if a.day is not None and b.day is not None and a.day != b.day:
        return False
    return True
