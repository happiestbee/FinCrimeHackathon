"""Programmatic anti-hallucination verifier gating LLM outputs before Claim Ledger entry."""

from __future__ import annotations

import re
from typing import Any

from .models import KycRecord, PartialDate
from .normalize import GeoNormalizer, NameMatcher, fold_text, parse_partial_date


class ExtractionVerifier:
    """Verifies verbatim quotes, subject attribution, and numeric/date entailment."""

    def __init__(self, name_matcher: NameMatcher, geo: GeoNormalizer) -> None:
        self.name_matcher = name_matcher
        self.geo = geo

    def verify_quote_in_text(self, quote: str | None, source_text: str | None) -> bool:
        if not quote or not source_text:
            return False
        q_norm = " ".join(fold_text(quote).split())
        s_norm = " ".join(fold_text(source_text).split())
        return len(q_norm) >= 6 and q_norm in s_norm

    def verify_numbers_in_quote(self, numbers: list[int], quote: str) -> bool:
        for num in numbers:
            if not re.search(rf"\b{int(num)}\b", quote):
                return False
        return True

    def verify_subject_binding(
        self,
        about_subject: Any,
        subject_name_in_quote: str | None,
        quote: str,
        expected_names: list[str],
    ) -> bool:
        if about_subject is not True:
            return False
        if not subject_name_in_quote or not subject_name_in_quote.strip():
            return False
        q_norm = " ".join(fold_text(quote).split())
        subj_norm = " ".join(fold_text(subject_name_in_quote).split())
        if subj_norm not in q_norm:
            return False
        return self.name_matcher.name_list_agrees(
            [subject_name_in_quote], expected_names, is_company=False
        )

    def verify_kyc_note_extraction(
        self,
        payload: dict[str, Any],
        source_text: str,
        source_id: str,
        checked_at: str,
    ) -> KycRecord | None:
        quote = payload.get("verbatim_quote") or ""
        if not self.verify_quote_in_text(quote, source_text):
            return None

        status = payload.get("status")
        if status in ("rejected", "expired"):
            return KycRecord(
                source_id=source_id,
                source_kind="kyc_note",
                checked_at=checked_at,
                result=status,
            )
        if status != "approved":
            return None

        verified_name = payload.get("verified_name")
        if verified_name:
            vn_norm = " ".join(fold_text(str(verified_name)).split())
            s_norm = " ".join(fold_text(source_text).split())
            if len(vn_norm) < 2 or vn_norm not in s_norm:
                verified_name = None

        verified_dob: PartialDate | None = None
        raw_dob = payload.get("verified_dob")
        if raw_dob:
            pd = parse_partial_date(str(raw_dob), day_first=True)
            if pd is not None and self.verify_numbers_in_quote([pd.year], quote):
                verified_dob = pd

        doc_country: str | None = None
        raw_country = payload.get("document_country")
        if raw_country:
            iso = self.geo.normalize_structured_country(str(raw_country))
            if iso and iso in self.geo.extract_countries_from_prose(source_text):
                doc_country = iso

        return KycRecord(
            source_id=source_id,
            source_kind="kyc_note",
            checked_at=checked_at,
            result="approved",
            verified_name=verified_name,
            verified_dob=verified_dob,
            document_country=doc_country,
        )

    def verify_residence_note_extraction(
        self, payload: dict[str, Any], source_text: str
    ) -> list[str]:
        out: list[str] = []
        for item in payload.get("lived_countries") or []:
            if not isinstance(item, dict):
                continue
            quote = item.get("verbatim_quote") or ""
            raw_c = item.get("country") or ""
            if not self.verify_quote_in_text(quote, source_text):
                continue
            iso = self.geo.normalize_structured_country(raw_c)
            if iso and iso in self.geo.extract_countries_from_prose(quote):
                if iso not in out:
                    out.append(iso)
        return out

    def verify_hit_dob_extraction(
        self,
        payload: dict[str, Any],
        first_para: str,
        hit_names: list[str],
    ) -> list[PartialDate]:
        quote = payload.get("verbatim_quote") or ""
        if not self.verify_quote_in_text(quote, first_para):
            return []
        if not self.verify_subject_binding(
            payload.get("about_subject"),
            payload.get("subject_name_in_quote"),
            quote,
            hit_names,
        ):
            return []
        return self._derive_dobs_from_verified_payload(payload, quote)

    def verify_search_dob_extraction(
        self,
        payload: dict[str, Any],
        snippet: str,
        hit_names: list[str],
        hit_countries: set[str],
        listing_phrase: str,
    ) -> list[PartialDate]:
        if payload.get("listing_matches") is not True:
            return []
        if listing_phrase.lower() not in snippet.lower():
            return []
        if not (set(self.geo.extract_countries_from_prose(snippet)) & hit_countries):
            return []
        quote = payload.get("verbatim_quote") or ""
        if not self.verify_quote_in_text(quote, snippet):
            return []
        if not self.verify_subject_binding(
            payload.get("about_subject"),
            payload.get("subject_name_in_quote"),
            quote,
            hit_names,
        ):
            return []
        return self._derive_dobs_from_verified_payload(payload, quote)

    def _derive_dobs_from_verified_payload(
        self, payload: dict[str, Any], quote: str
    ) -> list[PartialDate]:
        raw_birth = payload.get("birth_date")
        if raw_birth:
            pd = parse_partial_date(str(raw_birth), day_first=False)
            if pd is not None and self.verify_numbers_in_quote([pd.year], quote):
                return [pd]

        age = payload.get("age")
        ev_year = payload.get("event_year")
        if isinstance(age, int) and isinstance(ev_year, int):
            if 12 <= age <= 105 and 1900 <= ev_year <= 2030:
                if self.verify_numbers_in_quote([age, ev_year], quote):
                    y1, y2 = ev_year - age - 1, ev_year - age
                    return [
                        PartialDate(y1, raw=f"{age} in {ev_year}"),
                        PartialDate(y2, raw=f"{age} in {ev_year}"),
                    ]
        return []
