"""Watchlist hit enrichment and POLICY.md decision engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_client import LLMClient
from normalizer import (
    PartialDate,
    date_strictly_after,
    date_strictly_before,
    dates_agree,
    display_country,
    dob_contradicts,
    extract_country_from_location,
    max_age_at,
    normalize_country,
    parse_date,
)
from resolver import CustomerProfile, DataRepository
from text_extractor import (
    extract_countries_in_snippet,
    extract_countries_in_text,
    extract_dated_reports,
    extract_profile_p1_dobs,
    extract_search_snippet_dobs,
)


CATEGORY_PHRASES: Dict[str, str] = {
    "Sanctions": "a sanctioned individual",
    "PEP": "a politically exposed person",
    "Enforcement": "the subject of enforcement action",
    "Adverse Media": "named in adverse media",
    "SOE": "a state-owned enterprise official",
}


@dataclass
class HitFacts:
    """Normalized representation of a watchlist hit and its external search evidence."""

    hit_id: str
    alert_id: str
    is_company: bool
    display_name: str
    names: List[str]
    countries: Set[str]
    dobs: List[PartialDate]
    dob_source: str  # "dobs", "profile_notes", "search_results", or "none"
    sr_ids: List[str]
    deceased_date: Optional[PartialDate]
    earliest_report: Optional[PartialDate]

    def sorted_display_countries(self) -> str:
        return ", ".join(sorted(display_country(c) for c in self.countries))

    def format_dobs(self) -> str:
        yrs = sorted({d.format() for d in self.dobs})
        return ", ".join(yrs)


@dataclass
class EvalOutcome:
    """Decision outcome for a single candidate person against a hit."""

    decision: str
    reason_code: str
    reason: str
    evidence: List[str]


def extract_hit_countries(hit: Dict[str, Any]) -> Set[str]:
    """Gather all countries tied to a hit per Principle 5."""
    out: Set[str] = set()
    for c in hit.get("countries") or []:
        nc = normalize_country(c)
        if nc:
            out.add(nc)
    for field in ("nationality", "citizenship", "addresses"):
        for item in hit.get(field) or []:
            nc = extract_country_from_location(item)
            if nc:
                out.add(nc)
    pob = hit.get("place_of_birth")
    if pob:
        for seg in str(pob).split(";"):
            nc = extract_country_from_location(seg)
            if nc:
                out.add(nc)
    return out


def listing_phrase_for_hit(hit: Dict[str, Any]) -> str:
    """Map primary/first category to the search snippet listing phrase (Principle 5)."""
    cat = hit.get("primary_category")
    if not cat:
        cats = hit.get("categories") or []
        cat = cats[0] if cats else ""
    return CATEGORY_PHRASES.get(cat, "named on a watchlist")


def build_hit_facts(repo: DataRepository, hit: Dict[str, Any]) -> HitFacts:
    """Gather all structured and unstructured facts for a single hit."""
    is_company = (hit.get("entity_type") or "").lower() == "company"
    disp_name = hit.get("name") or ""
    names = [disp_name] if disp_name else []
    for al in hit.get("aliases") or []:
        if al:
            names.append(al)

    countries = extract_hit_countries(hit)
    deceased_date = parse_date(hit.get("deceased_date"))
    dated_reports = extract_dated_reports(hit.get("profile_notes"))
    earliest_report = min(dated_reports, key=lambda d: d.tuple_key()) if dated_reports else None

    dobs: List[PartialDate] = []
    dob_source = "none"
    sr_ids: List[str] = []

    for raw_d in hit.get("dobs") or []:
        dt = parse_date(raw_d)
        if dt:
            dobs.append(dt)
    if dobs:
        dob_source = "dobs"

    if not dobs and not is_company:
        p1_dobs = extract_profile_p1_dobs(hit.get("profile_notes"), is_company=False, llm_client=repo.llm_client)
        if p1_dobs:
            dobs = p1_dobs
            dob_source = "profile_notes"

    if not dobs and not is_company:
        phrase = listing_phrase_for_hit(hit)
        q = repo.matcher.search_query_key(disp_name)
        for sr in repo.search_by_query.get(q, ()):
            snip = sr.get("snippet") or ""
            if phrase not in snip.lower():
                continue
            snip_countries = extract_countries_in_snippet(
                snip, title=sr.get("title"), hit_names=names
            )
            if not (snip_countries & countries):
                continue
            sr_dobs = extract_search_snippet_dobs(snip, llm_client=repo.llm_client)
            if sr_dobs:
                dobs.extend(sr_dobs)
                sr_ids.append(sr["result_id"])
        if dobs:
            dob_source = "search_results"

    return HitFacts(
        hit_id=hit["hit_id"],
        alert_id=hit["alert_id"],
        is_company=is_company,
        display_name=disp_name,
        names=names,
        countries=countries,
        dobs=dobs,
        dob_source=dob_source,
        sr_ids=sorted(set(sr_ids)),
        deceased_date=deceased_date,
        earliest_report=earliest_report,
    )


def _dedup_evidence(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _country_evidence(cust: CustomerProfile) -> List[str]:
    """Return all customer record IDs plus KYC/Note IDs supporting customer countries."""
    ids = list(cust.record_ids)
    if cust.latest_verified and cust.latest_verified.document_country:
        ids.append(cust.latest_verified.source_id)
    for nf in cust.note_facts.values():
        if nf.lived_countries:
            ids.append(nf.note_id)
    return _dedup_evidence(ids)


def evaluate_person_against_hit(
    repo: DataRepository,
    cust: CustomerProfile,
    hit: HitFacts,
    alert: Dict[str, Any],
) -> EvalOutcome:
    """Apply POLICY.md Principles 6, 7, 8, and 9 for one resolved customer entity."""
    matcher = repo.matcher

    # 1. Entity type mismatch
    if cust.is_company != hit.is_company:
        c_type = "a company" if cust.is_company else "an individual"
        h_type = "a company" if hit.is_company else "an individual"
        return EvalOutcome(
            decision="DIFFERENT_ENTITY",
            reason_code="ENTITY_TYPE_MISMATCH",
            reason=(
                f"The customer record {cust.primary_record_id} is {c_type}, whereas watchlist hit "
                f"{hit.hit_id} ({hit.display_name}) is {h_type}."
            ),
            evidence=_dedup_evidence(cust.record_ids + [hit.hit_id]),
        )

    # 2. Deceased before screened_at
    screened_at = parse_date(alert.get("screened_at"))
    if hit.deceased_date and screened_at and date_strictly_before(hit.deceased_date, screened_at):
        return EvalOutcome(
            decision="DIFFERENT_ENTITY",
            reason_code="DECEASED",
            reason=(
                f"Watchlist hit {hit.hit_id} ({hit.display_name}) reports a date of death of "
                f"{hit.deceased_date.format()}, which is before the customer ({cust.primary_record_id}) "
                f"was screened on {screened_at.format()}."
            ),
            evidence=_dedup_evidence([f"{hit.hit_id}#deceased_date", hit.alert_id, cust.primary_record_id]),
        )

    cust_dob, cust_dob_src = cust.best_dob()

    # 3. Temporal impossibility
    if hit.earliest_report and cust_dob:
        if cust.is_company:
            if date_strictly_after(cust_dob, hit.earliest_report):
                return EvalOutcome(
                    decision="DIFFERENT_ENTITY",
                    reason_code="TEMPORAL_IMPOSSIBILITY",
                    reason=(
                        f"The customer company was incorporated on {cust_dob.format()} ({cust_dob_src}), "
                        f"after the hit's earliest dated report on {hit.earliest_report.format()}."
                    ),
                    evidence=_dedup_evidence(cust.record_ids + ([cust_dob_src] if cust_dob_src else [])),
                )
        else:
            age = max_age_at(cust_dob, hit.earliest_report)
            if age is not None and age < 16:
                return EvalOutcome(
                    decision="DIFFERENT_ENTITY",
                    reason_code="TEMPORAL_IMPOSSIBILITY",
                    reason=(
                        f"The customer was born in {cust_dob.year} ({cust_dob_src}), so they were at most "
                        f"{max(0, age)} when the hit's profile reports activity on {hit.earliest_report.format()}."
                    ),
                    evidence=_dedup_evidence(cust.record_ids + ([cust_dob_src] if cust_dob_src else [])),
                )

    # 4. Name comparison (Principle 4 & Principle 8)
    verified = cust.latest_verified
    if verified and verified.verified_name:
        vname = verified.verified_name
        v_parts = matcher.extract_parts(vname, is_company=cust.is_company)
        can_judge_name = cust.is_company or len(v_parts) >= 2
        if can_judge_name:
            verified_matches_hit = any(
                matcher.names_agree(vname, hname, is_company=cust.is_company, require_min_two_a=True)
                for hname in hit.names
            )
            if not verified_matches_hit:
                return EvalOutcome(
                    decision="DIFFERENT_ENTITY",
                    reason_code="NAME_MISMATCH_VERIFIED",
                    reason=(
                        f"The customer's verified name is {vname} ({verified.source_id}), "
                        f"which does not match the hit's name {hit.display_name}."
                    ),
                    evidence=[verified.source_id],
                )
            name_agrees = True
        else:
            # Individual verified name has fewer than 2 real parts (placeholders stripped)
            name_agrees = False
    else:
        # No approved verified name; check declared names
        name_agrees = any(
            matcher.names_agree(dname, hname, is_company=cust.is_company, require_min_two_a=True)
            for dname in cust.declared_names
            for hname in hit.names
        )

    # 5. DOB comparison
    dob_agrees = False
    dob_contradicts_all = False
    if cust_dob and hit.dobs:
        dob_agrees = any(dates_agree(cust_dob, hd) for hd in hit.dobs)
        dob_contradicts_all = all(dob_contradicts(cust_dob, hd) for hd in hit.dobs)

    if dob_contradicts_all and cust_dob and cust_dob_src:
        ev = [cust_dob_src]
        if hit.dob_source == "search_results":
            ev.extend(hit.sr_ids)
        src_desc = {
            "dobs": f"the hit's listed date of birth is {hit.format_dobs()}",
            "profile_notes": f"the hit's profile notes put its birth year at {hit.format_dobs()}",
            "search_results": f"the hit's search results put its birth year at {hit.format_dobs()}",
        }.get(hit.dob_source, f"the hit gives {hit.format_dobs()}")
        return EvalOutcome(
            decision="DIFFERENT_ENTITY",
            reason_code="DOB_MISMATCH",
            reason=(
                f"The customer's date of birth is {cust_dob.format()} ({cust_dob_src}) "
                f"but {src_desc}."
            ),
            evidence=_dedup_evidence(ev),
        )

    # 6. Location comparison
    cust_countries = cust.countries()
    hit_countries = hit.countries
    loc_overlap = bool(cust_countries & hit_countries)
    loc_disjoint = bool(cust_countries) and bool(hit_countries) and not loc_overlap

    if loc_disjoint and not (name_agrees and dob_agrees):
        return EvalOutcome(
            decision="DIFFERENT_ENTITY",
            reason_code="LOCATION_MISMATCH",
            reason=(
                f"The customer is linked only to {cust.sorted_display_countries()} "
                f"but the hit is linked only to {hit.sorted_display_countries()}."
            ),
            evidence=_country_evidence(cust),
        )

    # 7. Confirmation (Principle 7)
    if cust.is_company:
        if name_agrees and loc_overlap:
            common = ", ".join(sorted(display_country(c) for c in (cust_countries & hit_countries)))
            ev = _country_evidence(cust)
            return EvalOutcome(
                decision="SAME_ENTITY",
                reason_code="NAME_LOCATION_MATCH",
                reason=(
                    f"Both the customer and the hit are linked to {common}. "
                    f"The customer's name agrees with the hit's name {hit.display_name}."
                ),
                evidence=ev,
            )
    else:
        if name_agrees and dob_agrees and not loc_disjoint:
            if loc_overlap:
                common = ", ".join(sorted(display_country(c) for c in (cust_countries & hit_countries)))
                ev = list(cust.record_ids)
                if cust_dob_src:
                    ev.append(cust_dob_src)
                if hit.dob_source == "search_results":
                    ev.extend(hit.sr_ids)
                for c in (cust_countries & hit_countries):
                    ev.extend(cust.country_sources.get(c, ()))
                return EvalOutcome(
                    decision="SAME_ENTITY",
                    reason_code="NAME_DOB_LOCATION_MATCH",
                    reason=(
                        f"The customer's date of birth {cust_dob.format()} ({cust_dob_src}) agrees with "
                        f"the hit's {hit.format_dobs()}. Both the customer and the hit are linked to {common}. "
                        f"The customer's name agrees with the hit's name {hit.display_name}."
                    ),
                    evidence=_dedup_evidence(ev),
                )
            ev = list(cust.record_ids)
            if cust_dob_src:
                ev.append(cust_dob_src)
            if hit.dob_source == "search_results":
                ev.extend(hit.sr_ids)
            return EvalOutcome(
                decision="SAME_ENTITY",
                reason_code="NAME_DOB_MATCH",
                reason=(
                    f"The customer's date of birth {cust_dob.format()} ({cust_dob_src}) agrees with "
                    f"the hit's {hit.format_dobs()}, and the customer's name agrees with the hit's name {hit.display_name}."
                ),
                evidence=_dedup_evidence(ev),
            )

    # 8. Insufficient data -> REVIEW
    ev: List[str] = []
    if loc_overlap:
        for c in sorted(cust_countries & hit_countries):
            ev.extend(cust.country_sources.get(c, ()))
    if not ev:
        ev = list(cust.record_ids)
        if verified:
            ev.append(verified.source_id)
    ev = _dedup_evidence(ev)

    parts_reason: List[str] = []
    if loc_overlap:
        common = ", ".join(sorted(display_country(c) for c in (cust_countries & hit_countries)))
        parts_reason.append(f"Both the customer and the hit are linked to {common}.")
    elif loc_disjoint:
        parts_reason.append(
            f"Although the customer ({cust.sorted_display_countries()}) and hit ({hit.sorted_display_countries()}) "
            f"have different countries, name and date of birth agree so the hit cannot be cleared."
        )
    if name_agrees:
        parts_reason.append(f"The customer's name agrees with the hit's name {hit.display_name}.")
    parts_reason.append(
        "Nothing in the customer's records contradicts the hit, and there is insufficient matching evidence "
        "to confirm or clear it, so human review is required."
    )
    return EvalOutcome(
        decision="REVIEW",
        reason_code="INSUFFICIENT_DATA",
        reason=" ".join(parts_reason),
        evidence=ev,
    )


def evaluate_no_customer_found(
    repo: DataRepository,
    hit: HitFacts,
    alert: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle alerts that match zero records in customers.jsonl."""
    ac = alert["customer"]
    alert_is_company = (ac.get("entity_type") or "").lower() == "company"

    if alert_is_company != hit.is_company:
        return {
            "hit_id": hit.hit_id,
            "alert_id": hit.alert_id,
            "decision": "DIFFERENT_ENTITY",
            "reason_code": "ENTITY_TYPE_MISMATCH",
            "customer_record_id": None,
            "reason": (
                f"The screened entity in alert {alert['alert_id']} is {ac.get('entity_type')}, "
                f"whereas hit {hit.hit_id} ({hit.display_name}) is {'a company' if hit.is_company else 'an individual'}."
            ),
            "evidence": [alert["alert_id"], hit.hit_id],
        }

    screened_at = parse_date(alert.get("screened_at"))
    if hit.deceased_date and screened_at and date_strictly_before(hit.deceased_date, screened_at):
        return {
            "hit_id": hit.hit_id,
            "alert_id": hit.alert_id,
            "decision": "DIFFERENT_ENTITY",
            "reason_code": "DECEASED",
            "customer_record_id": None,
            "reason": (
                f"Hit {hit.hit_id} ({hit.display_name}) reports date of death {hit.deceased_date.format()}, "
                f"before screening on {screened_at.format()}."
            ),
            "evidence": [f"{hit.hit_id}#deceased_date", alert["alert_id"]],
        }

    return {
        "hit_id": hit.hit_id,
        "alert_id": hit.alert_id,
        "decision": "REVIEW",
        "reason_code": "INSUFFICIENT_DATA",
        "customer_record_id": None,
        "reason": (
            f"No matching customer record was found in the system of record for screened name "
            f"{ac.get('name')} in alert {alert['alert_id']}, so hit {hit.hit_id} ({hit.display_name}) "
            f"cannot be cleared or confirmed without human review."
        ),
        "evidence": [alert["alert_id"]],
    }


def solve_dataset(data_dir: str, llm_client: Optional[LLMClient] = None) -> List[Dict[str, Any]]:
    """Run the end-to-end screening resolution pipeline on a data folder."""
    repo = DataRepository(data_dir, llm_client=llm_client)

    # 1. Resolve alerts -> matching customer clusters first
    alert_clusters: Dict[str, List[Tuple[int, str]]] = {}
    for alert in repo.alerts:
        alert_clusters[alert["alert_id"]] = repo.find_matching_customer_clusters(alert)

    # 2. Pre-warm notes extraction concurrently when LLM is enabled (only for candidate customer clusters)
    if llm_client and llm_client.enabled:
        notes_to_parse: Dict[str, Dict[str, Any]] = {}
        for pairs in alert_clusters.values():
            for root_idx, _ in pairs:
                for idx in repo.clusters[root_idx]:
                    em = (repo.customers[idx].get("email") or "").strip().lower()
                    for nt in repo.notes_by_email.get(em, ()):
                        notes_to_parse[nt["note_id"]] = nt
        if notes_to_parse:
            with ThreadPoolExecutor(max_workers=12) as pool:
                list(pool.map(repo.get_note_facts, notes_to_parse.values()))

    # 3. Build hit facts (concurrently if LLM enabled)
    if llm_client and llm_client.enabled:
        with ThreadPoolExecutor(max_workers=12) as pool:
            hit_facts_list = list(pool.map(lambda h: build_hit_facts(repo, h), repo.hits))
    else:
        hit_facts_list = [build_hit_facts(repo, h) for h in repo.hits]

    results: List[Dict[str, Any]] = []
    for hf in hit_facts_list:
        alert = repo.alerts_by_id[hf.alert_id]
        pairs = alert_clusters.get(hf.alert_id, [])
        if not pairs:
            results.append(evaluate_no_customer_found(repo, hf, alert))
            continue

        profiles = [repo.build_customer_profile(root_idx, pref_id) for root_idx, pref_id in pairs]
        outcomes = [evaluate_person_against_hit(repo, prof, hf, alert) for prof in profiles]

        if len(profiles) == 1:
            out = outcomes[0]
            results.append({
                "hit_id": hf.hit_id,
                "alert_id": hf.alert_id,
                "decision": out.decision,
                "reason_code": out.reason_code,
                "customer_record_id": profiles[0].primary_record_id,
                "reason": out.reason,
                "evidence": out.evidence,
            })
        else:
            # Principle 3: multiple people fit the alert -> decide separately, customer_record_id = AMBIGUOUS
            distinct_decisions = {o.decision for o in outcomes}
            combined_ev = _dedup_evidence([e for o in outcomes for e in o.evidence])
            if len(distinct_decisions) == 1:
                chosen = outcomes[0]
                results.append({
                    "hit_id": hf.hit_id,
                    "alert_id": hf.alert_id,
                    "decision": chosen.decision,
                    "reason_code": chosen.reason_code,
                    "customer_record_id": "AMBIGUOUS",
                    "reason": chosen.reason,
                    "evidence": combined_ev,
                })
            else:
                cust_list = ", ".join(p.primary_record_id for p in profiles)
                results.append({
                    "hit_id": hf.hit_id,
                    "alert_id": hf.alert_id,
                    "decision": "REVIEW",
                    "reason_code": "AMBIGUOUS_CUSTOMER_RECORD",
                    "customer_record_id": "AMBIGUOUS",
                    "reason": (
                        f"Multiple customer profiles ({cust_list}) match alert {hf.alert_id} and lead to "
                        f"different decisions against hit {hf.hit_id} ({hf.display_name}), so human review is required."
                    ),
                    "evidence": combined_ev,
                })

    return results
