"""Deterministic policy narrator producing factual 2-4 sentence explanations."""

from __future__ import annotations

from typing import Any

from .models import EvaluationResult, HitProfile, PersonProfile, ResolvedAlert
from .normalize import GeoNormalizer


def _fmt_countries(isos: set[str] | list[str], geo: GeoNormalizer) -> str:
    names = sorted({geo.display_name(i) for i in isos})
    return ", ".join(names)


def _fmt_hit_dobs(hit: HitProfile) -> str:
    if hit.dob_detail:
        return hit.dob_detail
    return ", ".join(d.iso_str for d in hit.dobs)


def _fmt_excluded(person: PersonProfile) -> str:
    if not person.excluded_kyc_ids:
        return ""
    return f"; rejected or expired check(s) {', '.join(person.excluded_kyc_ids)} were excluded"


def narrate_decision(
    resolved_alert: ResolvedAlert,
    hit: HitProfile,
    decision: str,
    reason_code: str,
    eval_results: list[EvaluationResult],
    geo: GeoNormalizer,
) -> str:
    """Compile a deterministic 2-4 sentence factual explanation citing exact deciding facts."""
    if resolved_alert.resolution_status == "AMBIGUOUS" and len(eval_results) > 1:
        return _narrate_ambiguous(resolved_alert, hit, decision, reason_code, eval_results, geo)

    res = eval_results[0]
    p = res.person
    f = res.facts
    s_cust = _sentence_customer_resolution(resolved_alert, p)

    if reason_code == "ENTITY_TYPE_MISMATCH":
        c_article = "an" if p.entity_type == "Individual" else "a"
        h_article = "an" if hit.entity_type == "Individual" else "a"
        s_dec = (
            f"The customer is registered as {c_article} {p.entity_type} ({p.effective_name or 'unnamed'}), "
            f"whereas hit {hit.hit_id} ({hit.primary_name}) is {h_article} {hit.entity_type}."
        )
        s_end = "Under the screening policy, a person and a company cannot be the same entity, so the hit is cleared."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "DECEASED":
        dec_str = hit.deceased_date.iso_str if hit.deceased_date else "unknown"
        scr_str = f.get("screened_at", "")
        s_dec = (
            f"Hit {hit.hit_id} ({hit.primary_name}) records a date of death of {dec_str}, "
            f"which precedes the customer's screening date of {scr_str}."
        )
        s_end = f"Because the listed subject died before screening, the hit is cleared as a different entity{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "TEMPORAL_IMPOSSIBILITY":
        dob_str = p.dob.iso_str if p.dob else "unknown"
        dob_src = f" ({p.dob_source_id})" if p.dob_source_id else ""
        rep_str = hit.earliest_report_date.iso_str if hit.earliest_report_date else "unknown"
        if p.entity_type == "Company":
            s_dec = (
                f"The customer company was incorporated on {dob_str}{dob_src}, which is after "
                f"the hit profile's earliest dated report on {rep_str}."
            )
            s_end = f"Because the entity did not yet exist when the report was dated, the hit is cleared{_fmt_excluded(p)}."
        else:
            max_age = f.get("max_age_at_report")
            s_dec = (
                f"The customer's date of birth is {dob_str}{dob_src}, making them at most {max_age} years old "
                f"at the hit profile's earliest dated report on {rep_str}."
            )
            s_end = f"Because the customer was younger than 16 at the time of the report, this hit is cleared{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "DOB_MISMATCH":
        c_dob_str = p.dob.iso_str if p.dob else "unknown"
        c_src = f" ({p.dob_source_id})" if p.dob_source_id else ""
        if hit.dob_source == "dobs_field":
            h_desc = f"the hit's listed date(s) of birth ({_fmt_hit_dobs(hit)})"
        elif hit.dob_source == "profile_notes":
            h_desc = f"the hit's profile notes ({_fmt_hit_dobs(hit)})"
        else:
            h_desc = f"anchored search results ({_fmt_hit_dobs(hit)})"
        label = "incorporation date" if hit.entity_type == "Company" else "date of birth"
        s_dec = f"The customer's {label} is {c_dob_str}{c_src}, which contradicts {h_desc}."
        s_end = f"Under the policy date-of-birth rule, this contradiction clears the hit as a different entity{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "NAME_MISMATCH_VERIFIED":
        v_src = f" ({p.verified_name_source_id})" if p.verified_name_source_id else ""
        s_dec = (
            f"The customer's approved identity check records a verified legal name of "
            f"{p.verified_name}{v_src}, which does not agree with the hit's name {hit.primary_name} or its aliases."
        )
        s_end = f"Because verified legal identity outranks declared/screened names, the hit is cleared{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "LOCATION_MISMATCH":
        c_countries = _fmt_countries(set(p.country_sources.keys()), geo)
        h_countries = _fmt_countries(hit.countries, geo)
        s_dec = (
            f"After checking CRM records, approved KYC documents, and support notes, the customer is linked only to {c_countries}, "
            f"whereas hit {hit.hit_id} ({hit.primary_name}) is linked only to {h_countries}."
        )
        s_end = f"Because the two country sets are completely disjoint and DOB does not confirm a match, the hit is cleared{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "NAME_DOB_LOCATION_MATCH":
        c_dob_str = p.dob.iso_str if p.dob else "unknown"
        c_src = f" ({p.dob_source_id})" if p.dob_source_id else ""
        shared = _fmt_countries(f.get("shared_countries", set()), geo)
        s_dec = (
            f"The customer's name ({p.effective_name}) agrees with hit {hit.primary_name}, "
            f"the customer's date of birth {c_dob_str}{c_src} agrees with the hit's {_fmt_hit_dobs(hit)}, "
            f"and both sides are linked to {shared}."
        )
        s_end = f"No verified or declared facts contradict this match, so the hit is confirmed as the same entity{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "NAME_DOB_MATCH":
        c_dob_str = p.dob.iso_str if p.dob else "unknown"
        c_src = f" ({p.dob_source_id})" if p.dob_source_id else ""
        s_dec = (
            f"The customer's name ({p.effective_name}) agrees with hit {hit.primary_name}, "
            f"and the customer's date of birth {c_dob_str}{c_src} agrees with the hit's {_fmt_hit_dobs(hit)}."
        )
        gap = _describe_missing_country(p, hit)
        s_end = f"{gap} Nothing contradicts the match, so the hit is confirmed as the same entity{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    if reason_code == "NAME_LOCATION_MATCH":
        shared = _fmt_countries(f.get("shared_countries", set()), geo)
        s_dec = (
            f"The customer company ({p.effective_name}) agrees in name with hit {hit.primary_name}, "
            f"and both entities are linked to {shared}."
        )
        inc_note = (
            f" Incorporation date ({p.dob.iso_str}) is not contradicted by the hit's listed dates"
            if hit.dobs and p.dob
            else " No incorporation date on the hit contradicts the company"
        )
        s_end = f"{inc_note.strip()}, so the hit is confirmed as the same entity{_fmt_excluded(p)}."
        return f"{s_cust} {s_dec} {s_end}"

    # REVIEW: INSUFFICIENT_DATA
    details: list[str] = []
    if f.get("countries_disjoint") and f.get("name_agrees") and f.get("dob_status") == "AGREE":
        c_countries = _fmt_countries(set(p.country_sources.keys()), geo)
        h_countries = _fmt_countries(hit.countries, geo)
        details.append(
            f"The customer ({p.effective_name}) and hit {hit.primary_name} agree on name and date of birth "
            f"({p.dob.iso_str if p.dob else 'unknown'}), but their countries differ (customer: {c_countries}; hit: {h_countries})."
        )
        details.append(
            f"Under policy, a country difference cannot clear a hit whose name and date of birth agree, so human review is required{_fmt_excluded(p)}."
        )
    else:
        if f.get("shared_countries"):
            shared = _fmt_countries(f["shared_countries"], geo)
            details.append(f"Both the customer and hit {hit.primary_name} are linked to {shared}.")
        elif p.country_sources and hit.countries:
            details.append(
                f"Customer countries ({_fmt_countries(set(p.country_sources.keys()), geo)}) "
                f"and hit countries ({_fmt_countries(hit.countries, geo)}) show no settled match."
            )
        elif not hit.countries:
            details.append(f"Hit {hit.hit_id} ({hit.primary_name}) provides no country metadata.")
        elif not p.country_sources:
            details.append("The customer side provides no country metadata.")

        if f.get("short_individual_name"):
            details.append(
                f"The customer's name ({p.effective_name}) has fewer than two real name parts, so it cannot be confirmed or cleared on name."
            )
        elif p.dob is None:
            details.append("The customer has no settled date of birth to compare against the hit.")
        elif not hit.dobs:
            details.append(
                "The hit provides no date of birth in structured fields, first-paragraph profile notes, or anchored search results."
            )
        elif f.get("dob_status") == "INCONCLUSIVE":
            details.append(
                f"The customer's date of birth ({p.dob.iso_str}) falls within the 1-year tolerance of the hit's ({_fmt_hit_dobs(hit)}), leaving it inconclusive."
            )

        details.append(
            f"Without a decisive contradiction or sufficient confirmation, the hit is sent to review{_fmt_excluded(p)}."
        )

    return f"{s_cust} {' '.join(details[:3])}"


def _sentence_customer_resolution(resolved_alert: ResolvedAlert, person: PersonProfile) -> str:
    if resolved_alert.resolution_status == "UNRESOLVED" or person.is_unresolved:
        return (
            f"Alert {resolved_alert.alert_id} could not be matched to a CRM record, so evaluation relies on "
            f"the screened details ({person.effective_name or 'unnamed'})."
        )
    if len(person.record_ids) > 1:
        ids_str = ", ".join(person.record_ids)
        return f"Alert {resolved_alert.alert_id} resolved to customer person cluster {ids_str} (canonical {person.canonical_record_id})."
    return f"Alert {resolved_alert.alert_id} resolved to customer record {person.canonical_record_id}."


def _describe_missing_country(person: PersonProfile, hit: HitProfile) -> str:
    if not hit.countries and not person.country_sources:
        return "Neither side lists a country."
    if not hit.countries:
        return "The hit profile lists no country."
    if not person.country_sources:
        return "The customer records list no country."
    return ""


def _narrate_ambiguous(
    resolved_alert: ResolvedAlert,
    hit: HitProfile,
    decision: str,
    reason_code: str,
    eval_results: list[EvaluationResult],
    geo: GeoNormalizer,
) -> str:
    cand_ids = ", ".join(r.person.canonical_record_id or "?" for r in eval_results)
    if reason_code == "AMBIGUOUS_CUSTOMER_RECORD":
        summaries = "; ".join(
            f"{r.person.canonical_record_id}: {r.decision} ({r.reason_code})"
            for r in eval_results[:4]
        )
        return (
            f"Alert {resolved_alert.alert_id} fits {len(eval_results)} distinct customer people ({cand_ids}), "
            f"resulting in an ambiguous customer match. "
            f"Evaluating hit {hit.hit_id} ({hit.primary_name}) per candidate yields conflicting outcomes ({summaries}). "
            f"Per policy, this ambiguity requires human review."
        )

    # All candidates agreed on the decision
    if decision == "DIFFERENT_ENTITY":
        per_cand = "; ".join(
            f"{r.person.canonical_record_id}: {r.reason_code}" for r in eval_results[:4]
        )
        return (
            f"Alert {resolved_alert.alert_id} fits {len(eval_results)} distinct customer people ({cand_ids}), "
            f"so customer_record_id is AMBIGUOUS. "
            f"Every fitting customer person independently clears hit {hit.hit_id} ({hit.primary_name}) "
            f"({per_cand}). "
            f"Per policy, unanimous clearance applies across all candidates."
        )
    if decision == "SAME_ENTITY":
        return (
            f"Alert {resolved_alert.alert_id} fits {len(eval_results)} distinct customer people ({cand_ids}), "
            f"so customer_record_id is AMBIGUOUS. "
            f"Every fitting customer person independently matches hit {hit.hit_id} ({hit.primary_name}) "
            f"under {reason_code}. Per policy, the hit is confirmed as the same entity."
        )
    return (
        f"Alert {resolved_alert.alert_id} fits {len(eval_results)} distinct customer people ({cand_ids}), "
        f"so customer_record_id is AMBIGUOUS. "
        f"Evaluating hit {hit.hit_id} ({hit.primary_name}) across each candidate yields REVIEW ({reason_code}) "
        f"due to insufficient distinguishing data (e.g., missing DOB on one or both sides). "
        f"The hit is routed to review."
    )
