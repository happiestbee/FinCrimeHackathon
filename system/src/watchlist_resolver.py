"""Hit-level watchlist resolution: typing, countries, dated reports, and DOB waterfall."""

from __future__ import annotations

import re
import threading
from typing import Any

from .models import Claim, HitProfile, PartialDate, TIER_2_DECLARED, TIER_3_CONTEXTUAL
from .normalize import parse_partial_date
from .search_shield import SearchShield, extract_age_and_year_from_profile_note

LISTING_PHRASES: dict[str, str] = {
    "Sanctions": "a sanctioned individual",
    "PEP": "a politically exposed person",
    "Enforcement": "the subject of enforcement action",
    "Adverse Media": "named in adverse media",
    "SOE": "a state-owned enterprise official",
}

_ACCORDING_TO_LINE_RE = re.compile(
    r"^\s*According to\s+[^\n:]+;\s*([^;:\n]+):",
    re.MULTILINE | re.IGNORECASE,
)

_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_EXPLICIT_DOB_LINE_RE = re.compile(r"(?im)^DOB\s*:\s*([^\n]+)")
_BORN_RE = re.compile(
    rf"\bborn\s+(?:on\s+|in\s+)?(\d{{1,2}}\s+{_MONTH_PAT}\s+\d{{4}}|{_MONTH_PAT}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{4}})\b",
    re.I,
)
_DOB_TRIGGER_RE = re.compile(
    r"\b(?:born\b|aged\b|age\s+(?:of|is|was)\b|\bage\s+\d{2}\b|\d{2}\s*-?\s*years?\s*-?\s*old\b|"
    r"dob\b|date\s+of\s+birth\b|reached\s+(?:the\s+age\s+of\s+)?\d{2}\b|"
    r"(?:was|were|then)\s+\d{2}\b|at\s+\d{2}\s*(?:,|\.|when\b|while\b|in\b|on\b|$))|"
    r"(?-i:\b(?!(?:Section|Article|Paragraph|Clause|Rule|Order|Item|Part|Case|Title|Chapter|Regulation)\b)"
    r"[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][a-zà-öø-ÿłśźżćńęąó']+\s+\([1-9]\d\))",
    re.I,
)
_NON_SUBJECT_HEADER_RE = re.compile(
    r"^(?:Career:|Sanctions ownership|Associated with|Political Party:|Position:|Description:|Type of Facility:)",
    re.I,
)
_FAMILY_RELATION_CUE_RE = re.compile(
    r"\b(?:wife|husband|spouse|son|daughter|father|mother|brother|sister|uncle|aunt|cousin|nephew|niece|relative)\s+of\b",
    re.I,
)


class WatchlistResolver:
    """Extracts structured and prose facts from hits.jsonl and search_results.jsonl."""

    def __init__(
        self,
        store: Any,
        search_shield: SearchShield,
        llm_client: Any = None,
        verifier: Any = None,
    ) -> None:
        self.store = store
        self.search_shield = search_shield
        self.llm_client = llm_client
        self.verifier = verifier
        self._stats_lock = threading.Lock()
        self.stats = {
            "hits_resolved": 0,
            "dobs_from_field": 0,
            "dobs_from_notes": 0,
            "dobs_from_notes_llm": 0,
            "dobs_from_search": 0,
        }

    def resolve_hit(self, hit: dict[str, Any]) -> HitProfile:
        with self._stats_lock:
            self.stats["hits_resolved"] += 1
        hid = hit["hit_id"]
        etype = hit["entity_type"]
        is_comp = etype == "Company"

        # 1. Gather all names & aliases
        all_names: list[str] = []
        seen_names: set[str] = set()
        candidates_raw: list[str | None] = [
            hit.get("name"),
            hit.get("best_matching_name"),
        ]
        fml = " ".join(
            p for p in (hit.get("first_name"), hit.get("middle_name"), hit.get("last_name")) if p
        ).strip()
        if fml:
            candidates_raw.append(fml)
        candidates_raw.extend(hit.get("aliases") or [])

        for nm in candidates_raw:
            if nm and nm.strip() and nm.strip() not in seen_names:
                seen_names.add(nm.strip())
                all_names.append(nm.strip())

        primary_name = hit.get("name") or (all_names[0] if all_names else "")

        # 2. Gather countries per Principle 5
        countries: set[str] = set()
        geo = self.store.geo
        for field_name in ("countries", "nationality", "citizenship"):
            for raw_val in hit.get(field_name) or []:
                iso = geo.normalize_structured_country(raw_val)
                if iso:
                    countries.add(iso)
        for addr in hit.get("addresses") or []:
            for iso in geo.extract_countries_from_address(addr):
                countries.add(iso)
        if hit.get("place_of_birth"):
            for iso in geo.extract_countries_from_pob(hit["place_of_birth"]):
                countries.add(iso)

        # 3. Deceased date
        deceased_date = parse_partial_date(hit.get("deceased_date"), day_first=False)

        # 4. Earliest dated report ("According to …; Month DD, YYYY: …")
        earliest_report: PartialDate | None = None
        pn = hit.get("profile_notes") or ""
        for m in _ACCORDING_TO_LINE_RE.finditer(pn):
            dt = parse_partial_date(m.group(1).strip(), day_first=False)
            if dt is not None:
                if earliest_report is None or (dt.earliest, dt.latest) < (
                    earliest_report.earliest,
                    earliest_report.latest,
                ):
                    earliest_report = dt

        # 5. Listing phrase for search shield
        cat = hit.get("primary_category") or (
            hit["categories"][0] if hit.get("categories") else None
        )
        listing_phrase = LISTING_PHRASES.get(cat, "named on a watchlist")

        # 6. Strict Principle 5 DOB waterfall
        dobs: list[PartialDate] = []
        dob_source: str | None = None
        dob_evidence_ids: list[str] = []
        dob_detail = ""
        claims: list[Claim] = []

        # Step 1: structured hit["dobs"] (only source allowed for companies)
        for raw_d in hit.get("dobs") or []:
            pd = parse_partial_date(raw_d, day_first=False)
            if pd is not None and pd not in dobs:
                dobs.append(pd)
        if dobs:
            dob_source = "dobs_field"
            dob_evidence_ids = [hid]
            dob_detail = ", ".join(d.iso_str for d in dobs)
            with self._stats_lock:
                self.stats["dobs_from_field"] += 1

        # Step 2 (Individuals only): first paragraph of profile_notes
        if not dobs and not is_comp and pn.strip():
            first_para = re.split(r"\n\s*\n", pn.strip())[0].strip()
            extracted_notes, detail_notes, used_llm = self._extract_dob_from_first_para(
                first_para, all_names
            )
            if extracted_notes:
                dobs.extend(extracted_notes)
                dob_source = "profile_notes"
                dob_evidence_ids = [hid]
                dob_detail = detail_notes
                with self._stats_lock:
                    self.stats["dobs_from_notes"] += 1
                    if used_llm:
                        self.stats["dobs_from_notes_llm"] += 1

        # Step 3 (Individuals only): search_results.jsonl via Anti-Namesake Search Shield
        if not dobs and not is_comp:
            sr_dobs, sr_ids, sr_detail = self.search_shield.lookup_hit_dobs(
                all_names, countries, listing_phrase
            )
            if sr_dobs:
                dobs.extend(sr_dobs)
                dob_source = "search_results"
                dob_evidence_ids = sr_ids
                dob_detail = sr_detail
                with self._stats_lock:
                    self.stats["dobs_from_search"] += 1

        for d in dobs:
            claims.append(
                Claim(
                    side="WATCHLIST",
                    attribute="DOB",
                    raw_value=d.raw or d.iso_str,
                    normalized_value=d,
                    trust_tier=TIER_2_DECLARED if dob_source == "dobs_field" else TIER_3_CONTEXTUAL,
                    source_id=dob_evidence_ids[0] if dob_evidence_ids else hid,
                )
            )
        for iso in sorted(countries):
            claims.append(
                Claim(
                    side="WATCHLIST",
                    attribute="COUNTRY",
                    raw_value=iso,
                    normalized_value=iso,
                    trust_tier=TIER_2_DECLARED,
                    source_id=hid,
                )
            )

        return HitProfile(
            hit_id=hid,
            alert_id=hit["alert_id"],
            entity_type=etype,
            primary_name=primary_name,
            all_names=all_names,
            dobs=dobs,
            dob_source=dob_source,
            dob_evidence_ids=dob_evidence_ids,
            dob_detail=dob_detail,
            countries=countries,
            deceased_date=deceased_date,
            life_status=hit.get("life_status"),
            earliest_report_date=earliest_report,
            listing_phrase=listing_phrase,
            claims=claims,
        )

    def _extract_dob_from_first_para(
        self, first_para: str, hit_names: list[str]
    ) -> tuple[list[PartialDate], str, bool]:
        if _NON_SUBJECT_HEADER_RE.match(first_para):
            return [], "", False

        # Guard: if paragraph starts with "According to ...", only extract if it explicitly names
        # the subject with a birth/age sentence without family-relation or multi-subject confusion.
        is_according = first_para.lower().startswith("according to")

        # 1. Explicit DOB: field
        m_dob_line = _EXPLICIT_DOB_LINE_RE.search(first_para)
        if m_dob_line:
            pd = parse_partial_date(m_dob_line.group(1).strip(), day_first=False)
            if pd is not None:
                return [pd], pd.iso_str, False

        # Guard against relative's DOB when relation cues exist
        if _FAMILY_RELATION_CUE_RE.search(first_para):
            # Delegate to LLM verifier if available so relative's DOB is never misattributed
            if self.llm_client is not None and self.llm_client.enabled and _DOB_TRIGGER_RE.search(first_para):
                llm_res = self.llm_client.extract_hit_first_para_dob(first_para, hit_names)
                if llm_res and self.verifier:
                    verified = self.verifier.verify_hit_dob_extraction(
                        llm_res, first_para, hit_names
                    )
                    if verified:
                        return verified, ", ".join(d.iso_str for d in verified), True
            return [], "", False

        # Check subject presence in first paragraph (avoid extracting from unrelated boilerplate)
        nm = self.store.name_matcher
        has_subject_mention = any(
            nm.names_agree(first_para, hn, is_company=False) for hn in hit_names
        )

        # 2. Explicit "born ..." sentence
        m_born = _BORN_RE.search(first_para)
        if m_born and (has_subject_mention or not is_according):
            pd = parse_partial_date(m_born.group(1), day_first=False)
            if pd is not None:
                return [pd], pd.iso_str, False

        # 3. Age at event date sentence (strict extractor to guard against legal clause/count numbers)
        if not is_according:
            age, ev_year = extract_age_and_year_from_profile_note(first_para)
            if age is not None and ev_year is not None:
                y1, y2 = ev_year - age - 1, ev_year - age
                return (
                    [PartialDate(y1, raw=f"{age} in {ev_year}"), PartialDate(y2, raw=f"{age} in {ev_year}")],
                    f"{y1} or {y2} (aged {age} in {ev_year})",
                    False,
                )

        # 4. Trigger-gated LLM fallback for unseen phrasings
        if (
            self.llm_client is not None
            and self.llm_client.enabled
            and _DOB_TRIGGER_RE.search(first_para)
            and re.search(r"\b(?:19|20)\d\d\b", first_para)
        ):
            llm_res = self.llm_client.extract_hit_first_para_dob(first_para, hit_names)
            if llm_res and self.verifier:
                verified = self.verifier.verify_hit_dob_extraction(
                    llm_res, first_para, hit_names
                )
                if verified:
                    return verified, ", ".join(d.iso_str for d in verified), True

        return [], "", False
