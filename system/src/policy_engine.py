"""Precedence-ordered policy engine translating POLICY.md principles into decisions."""

from __future__ import annotations

from datetime import date
from typing import Any

from .models import (
    DecisionOutput,
    EvaluationResult,
    HitProfile,
    PersonProfile,
    ResolvedAlert,
)
from .normalize import (
    GeoNormalizer,
    NameMatcher,
    compare_dob_against_hit_dobs,
    is_incorporated_after,
    is_strictly_before_date,
    is_younger_than,
    max_possible_age_at,
)
from .policy_spec import (
    CLEARANCE_PRECEDENCE,
    DECISION_DIFFERENT,
    DECISION_REVIEW,
    DECISION_SAME,
    MIN_AGE_AT_REPORT_YEARS,
)
from .proof_narrator import narrate_decision


class PolicyEngine:
    """Evaluates resolved customer candidate(s) against a resolved watchlist hit."""

    def __init__(self, name_matcher: NameMatcher, geo: GeoNormalizer) -> None:
        self.name_matcher = name_matcher
        self.geo = geo

    def evaluate_hit(
        self,
        resolved_alert: ResolvedAlert,
        hit: HitProfile,
        screened_at: date,
    ) -> DecisionOutput:
        eval_results = [
            self.evaluate_candidate(cand, hit, screened_at)
            for cand in resolved_alert.candidates
        ]

        if len(eval_results) == 1:
            r = eval_results[0]
            evidence = sorted(set(r.evidence_ids))
            reason = narrate_decision(
                resolved_alert, hit, r.decision, r.reason_code, eval_results, self.geo
            )
            return DecisionOutput(
                hit_id=hit.hit_id,
                alert_id=hit.alert_id,
                decision=r.decision,
                reason_code=r.reason_code,
                customer_record_id=resolved_alert.customer_record_id,
                reason=reason,
                evidence=evidence,
            )

        # Principle 3 multi-candidate aggregation when customer_record_id == "AMBIGUOUS"
        decisions = {r.decision for r in eval_results}
        evidence_union: set[str] = set()
        for r in eval_results:
            evidence_union.update(r.evidence_ids)
            if r.person.canonical_record_id:
                evidence_union.add(r.person.canonical_record_id)

        if len(decisions) == 1:
            only_dec = next(iter(decisions))
            if only_dec == DECISION_DIFFERENT:
                # Prefer a clearance reason code that genuinely holds for EVERY candidate;
                # fall back to highest-precedence union only when candidates clear on disjoint grounds.
                common_codes = set.intersection(
                    *(set(r.all_applicable_codes) for r in eval_results)
                )
                if common_codes:
                    chosen_code = next(
                        c for c in CLEARANCE_PRECEDENCE if c in common_codes
                    )
                    # Recompute candidate evidence specifically for the shared reason code
                    evidence_union = set()
                    for r in eval_results:
                        r.reason_code = chosen_code
                        ev_for_code = self._collect_clearance_evidence(chosen_code, r.person, hit)
                        evidence_union.update(ev_for_code)
                        if r.person.canonical_record_id:
                            evidence_union.add(r.person.canonical_record_id)
                else:
                    all_codes = {r.reason_code for r in eval_results}
                    chosen_code = next(
                        c for c in CLEARANCE_PRECEDENCE if c in all_codes
                    )
            elif only_dec == DECISION_SAME:
                all_codes = {r.reason_code for r in eval_results}
                if "NAME_DOB_LOCATION_MATCH" in all_codes and len(all_codes) == 1:
                    chosen_code = "NAME_DOB_LOCATION_MATCH"
                elif "NAME_LOCATION_MATCH" in all_codes:
                    chosen_code = "NAME_LOCATION_MATCH"
                else:
                    chosen_code = "NAME_DOB_MATCH"
            else:
                chosen_code = "INSUFFICIENT_DATA"

            reason = narrate_decision(
                resolved_alert, hit, only_dec, chosen_code, eval_results, self.geo
            )
            return DecisionOutput(
                hit_id=hit.hit_id,
                alert_id=hit.alert_id,
                decision=only_dec,
                reason_code=chosen_code,
                customer_record_id="AMBIGUOUS",
                reason=reason,
                evidence=sorted(evidence_union),
            )

        # Conflicting decisions across candidate people -> REVIEW + AMBIGUOUS_CUSTOMER_RECORD
        reason = narrate_decision(
            resolved_alert,
            hit,
            DECISION_REVIEW,
            "AMBIGUOUS_CUSTOMER_RECORD",
            eval_results,
            self.geo,
        )
        return DecisionOutput(
            hit_id=hit.hit_id,
            alert_id=hit.alert_id,
            decision=DECISION_REVIEW,
            reason_code="AMBIGUOUS_CUSTOMER_RECORD",
            customer_record_id="AMBIGUOUS",
            reason=reason,
            evidence=sorted(evidence_union),
        )

    def evaluate_candidate(
        self,
        person: PersonProfile,
        hit: HitProfile,
        screened_at: date,
    ) -> EvaluationResult:
        nm = self.name_matcher
        is_comp = person.entity_type == "Company" or hit.entity_type == "Company"

        # 1. Entity Type Mismatch (Principle 6)
        if person.entity_type != hit.entity_type:
            ev = []
            if person.canonical_record_id:
                ev.append(person.canonical_record_id)
            ev.append(hit.hit_id)
            return EvaluationResult(
                decision=DECISION_DIFFERENT,
                reason_code="ENTITY_TYPE_MISMATCH",
                all_applicable_codes=["ENTITY_TYPE_MISMATCH"],
                person=person,
                evidence_ids=ev,
            )

        # Precompute Name, DOB, Country, Temporal, Deceased predicates
        # Name comparison (Principles 4, 6, 8)
        short_individual_name = False
        name_contradicts_verified = False
        name_agrees_for_confirm = False

        if person.verified_name:
            v_parts = nm.real_part_count(person.verified_name, is_company=is_comp)
            if not is_comp and v_parts < 2:
                short_individual_name = True
            else:
                v_agrees = nm.name_list_agrees(
                    [person.verified_name], hit.all_names, is_company=is_comp
                )
                if v_agrees:
                    name_agrees_for_confirm = True
                elif not person.is_unresolved:
                    name_contradicts_verified = True
        else:
            # Declared names only: cannot clear on NAME_MISMATCH_VERIFIED.
            valid_declared: list[str] = []
            for d_name, _ in person.declared_names:
                cnt = nm.real_part_count(d_name, is_company=is_comp)
                if is_comp and cnt >= 1:
                    valid_declared.append(d_name)
                elif not is_comp and cnt >= 2:
                    valid_declared.append(d_name)
            if not is_comp and person.declared_names and not valid_declared:
                short_individual_name = True
            elif valid_declared:
                name_agrees_for_confirm = nm.name_list_agrees(
                    valid_declared, hit.all_names, is_company=is_comp
                )

        # DOB comparison (Principles 5, 6, 7)
        dob_status = compare_dob_against_hit_dobs(person.dob, hit.dobs)

        # Country comparison (Principles 5, 6, 7)
        cust_countries = set(person.country_sources.keys())
        hit_countries = set(hit.countries)
        shared_countries = cust_countries & hit_countries
        countries_disjoint = bool(cust_countries and hit_countries and not shared_countries)

        # Deceased check (Principle 6)
        deceased_clear = (
            hit.deceased_date is not None
            and is_strictly_before_date(hit.deceased_date, screened_at)
        )

        # Temporal impossibility check (Principle 6)
        temporal_clear = False
        max_age_at_report: int | None = None
        if person.dob is not None and hit.earliest_report_date is not None:
            if person.entity_type == "Company":
                temporal_clear = is_incorporated_after(person.dob, hit.earliest_report_date)
            else:
                temporal_clear = is_younger_than(
                    person.dob, hit.earliest_report_date, min_age=MIN_AGE_AT_REPORT_YEARS
                )
                max_age_at_report = max_possible_age_at(person.dob, hit.earliest_report_date)

        # Location mismatch check (Principle 6)
        # Cannot clear if customer is unresolved (CRM/notes unknown) OR if name & DOB agree!
        location_clear = (
            (not person.is_unresolved)
            and countries_disjoint
            and not (name_agrees_for_confirm and dob_status == "AGREE")
        )

        # Collect all applicable DIFFERENT_ENTITY grounds in precedence order
        applicable: list[str] = []
        if deceased_clear:
            applicable.append("DECEASED")
        if temporal_clear:
            applicable.append("TEMPORAL_IMPOSSIBILITY")
        if dob_status == "CONTRADICT":
            applicable.append("DOB_MISMATCH")
        if name_contradicts_verified:
            applicable.append("NAME_MISMATCH_VERIFIED")
        if location_clear:
            applicable.append("LOCATION_MISMATCH")

        facts: dict[str, Any] = {
            "screened_at": screened_at.isoformat(),
            "name_agrees": name_agrees_for_confirm,
            "short_individual_name": short_individual_name,
            "dob_status": dob_status,
            "shared_countries": shared_countries,
            "countries_disjoint": countries_disjoint,
            "max_age_at_report": max_age_at_report,
        }

        if applicable:
            primary_code = applicable[0]
            ev_ids = self._collect_clearance_evidence(primary_code, person, hit)
            return EvaluationResult(
                decision=DECISION_DIFFERENT,
                reason_code=primary_code,
                all_applicable_codes=applicable,
                person=person,
                evidence_ids=ev_ids,
                facts=facts,
            )

        # Positive confirmation checks (Principle 7)
        # Never confirm an unresolved alert without CRM backing, or when countries contradict.
        if not person.is_unresolved and not countries_disjoint:
            if is_comp:
                if name_agrees_for_confirm and shared_countries and dob_status != "CONTRADICT":
                    ev_ids = self._collect_confirmation_evidence(
                        "NAME_LOCATION_MATCH", person, hit, shared_countries
                    )
                    return EvaluationResult(
                        decision=DECISION_SAME,
                        reason_code="NAME_LOCATION_MATCH",
                        all_applicable_codes=["NAME_LOCATION_MATCH"],
                        person=person,
                        evidence_ids=ev_ids,
                        facts=facts,
                    )
            else:
                if name_agrees_for_confirm and dob_status == "AGREE":
                    code = (
                        "NAME_DOB_LOCATION_MATCH"
                        if shared_countries
                        else "NAME_DOB_MATCH"
                    )
                    ev_ids = self._collect_confirmation_evidence(
                        code, person, hit, shared_countries
                    )
                    return EvaluationResult(
                        decision=DECISION_SAME,
                        reason_code=code,
                        all_applicable_codes=[code],
                        person=person,
                        evidence_ids=ev_ids,
                        facts=facts,
                    )

        # Otherwise REVIEW (INSUFFICIENT_DATA)
        ev_ids = self._collect_review_evidence(
            person,
            hit,
            shared_countries,
            dob_status=dob_status,
            name_agrees=name_agrees_for_confirm,
            countries_disjoint=countries_disjoint,
        )
        return EvaluationResult(
            decision=DECISION_REVIEW,
            reason_code="INSUFFICIENT_DATA",
            all_applicable_codes=["INSUFFICIENT_DATA"],
            person=person,
            evidence_ids=ev_ids,
            facts=facts,
        )

    def _collect_clearance_evidence(
        self, code: str, person: PersonProfile, hit: HitProfile
    ) -> list[str]:
        ev: list[str] = []
        if code == "DECEASED":
            ev.append(hit.hit_id)
            if person.canonical_record_id:
                ev.append(person.canonical_record_id)
        elif code == "TEMPORAL_IMPOSSIBILITY":
            if person.dob_source_id:
                ev.append(person.dob_source_id)
            elif person.canonical_record_id:
                ev.append(person.canonical_record_id)
            else:
                ev.append(hit.hit_id)
        elif code == "DOB_MISMATCH":
            if person.dob_source_id:
                ev.append(person.dob_source_id)
            if hit.dob_source == "search_results":
                ev.extend(hit.dob_evidence_ids)
            if not ev:
                ev.append(hit.hit_id)
        elif code == "NAME_MISMATCH_VERIFIED":
            if person.verified_name_source_id:
                ev.append(person.verified_name_source_id)
            elif person.canonical_record_id:
                ev.append(person.canonical_record_id)
        elif code == "LOCATION_MISMATCH":
            for srcs in person.country_sources.values():
                ev.extend(srcs)
            if not ev and person.canonical_record_id:
                ev.append(person.canonical_record_id)
        return sorted(set(ev))

    def _collect_confirmation_evidence(
        self,
        code: str,
        person: PersonProfile,
        hit: HitProfile,
        shared_countries: set[str],
    ) -> list[str]:
        ev: list[str] = []
        if person.dob_source_id and code in ("NAME_DOB_LOCATION_MATCH", "NAME_DOB_MATCH"):
            ev.append(person.dob_source_id)
        if person.verified_name_source_id:
            ev.append(person.verified_name_source_id)
        if shared_countries:
            for iso in shared_countries:
                ev.extend(person.country_sources.get(iso, ()))
        if hit.dob_source == "search_results" and code in (
            "NAME_DOB_LOCATION_MATCH",
            "NAME_DOB_MATCH",
        ):
            ev.extend(hit.dob_evidence_ids)
        # Ensure at least canonical_record_id is included if CRM data contributed to name/location
        if person.canonical_record_id and (
            not person.verified_name_source_id
            or any(person.canonical_record_id in person.country_sources.get(iso, ()) for iso in shared_countries)
        ):
            ev.append(person.canonical_record_id)
        if not ev and person.canonical_record_id:
            ev.append(person.canonical_record_id)
        return sorted(set(ev))

    def _collect_review_evidence(
        self,
        person: PersonProfile,
        hit: HitProfile,
        shared_countries: set[str],
        dob_status: str = "INCONCLUSIVE",
        name_agrees: bool = False,
        countries_disjoint: bool = False,
    ) -> list[str]:
        ev: list[str] = []
        if shared_countries:
            for iso in shared_countries:
                ev.extend(person.country_sources.get(iso, ()))
        else:
            for srcs in person.country_sources.values():
                ev.extend(srcs)

        # Include DOB / verified name sources when they actually participated in a comparison
        dob_compared = person.dob is not None and bool(hit.dobs)
        if dob_compared or (name_agrees and dob_status == "AGREE" and countries_disjoint):
            if person.dob_source_id:
                ev.append(person.dob_source_id)
            if person.verified_name_source_id:
                ev.append(person.verified_name_source_id)
            if hit.dob_source == "search_results":
                ev.extend(hit.dob_evidence_ids)

        if not ev:
            if person.verified_name_source_id:
                ev.append(person.verified_name_source_id)
            if person.dob_source_id:
                ev.append(person.dob_source_id)
            if person.canonical_record_id:
                ev.append(person.canonical_record_id)
            if not ev:
                ev.append(hit.hit_id)
        return sorted(set(ev))
