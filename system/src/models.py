"""Shared data contracts for the watchlist entity resolution pipeline."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date
from typing import Any


TIER_1_VERIFIED = "TIER_1_VERIFIED"
TIER_2_DECLARED = "TIER_2_DECLARED"
TIER_3_CONTEXTUAL = "TIER_3_CONTEXTUAL"
TIER_0_EXCLUDED = "TIER_0_EXCLUDED"


@dataclass(frozen=True, order=True)
class PartialDate:
    """Represents a full or partial calendar date without inventing missing parts."""

    year: int
    month: int | None = None
    day: int | None = None
    raw: str = ""

    @property
    def earliest(self) -> date:
        m = self.month if self.month is not None else 1
        d = self.day if self.day is not None else 1
        return date(self.year, m, d)

    @property
    def latest(self) -> date:
        m = self.month if self.month is not None else 12
        if self.day is not None:
            d = self.day
        else:
            d = calendar.monthrange(self.year, m)[1]
        return date(self.year, m, d)

    @property
    def is_full(self) -> bool:
        return self.month is not None and self.day is not None

    @property
    def iso_str(self) -> str:
        if self.month is None:
            return f"{self.year:04d}"
        if self.day is None:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    def __str__(self) -> str:
        return self.iso_str


@dataclass(frozen=True)
class Claim:
    """Typed, provenance-backed attribute extracted from a bank or watchlist record."""

    side: str  # "CUSTOMER" | "WATCHLIST"
    attribute: str
    raw_value: str
    normalized_value: Any
    trust_tier: str
    source_id: str
    verbatim_quote: str = ""
    subject_attributed: bool = True


@dataclass(frozen=True)
class KycRecord:
    """Normalized identity check from kyc_checks.jsonl or notes.jsonl (channel=kyc)."""

    source_id: str
    source_kind: str  # "kyc_check" | "kyc_note"
    checked_at: str
    result: str  # "approved" | "rejected" | "expired" | "unknown"
    verified_name: str | None = None
    verified_dob: PartialDate | None = None
    document_country: str | None = None
    document_number: str | None = None
    customer_id: str | None = None


@dataclass
class PersonProfile:
    """Resolved customer person cluster with prioritized verified and declared facts."""

    cluster_id: str
    record_ids: list[str]
    canonical_record_id: str | None
    entity_type: str
    verified_name: str | None = None
    verified_name_source_id: str | None = None
    declared_names: list[tuple[str, str]] = field(default_factory=list)
    effective_name: str | None = None
    dob: PartialDate | None = None
    dob_source_id: str | None = None
    dob_is_verified: bool = False
    country_sources: dict[str, list[str]] = field(default_factory=dict)
    emails: list[str] = field(default_factory=list)
    excluded_kyc_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    is_unresolved: bool = False
    resolution_notes: list[str] = field(default_factory=list)


@dataclass
class ResolvedAlert:
    """Alert-level customer resolution outcome shared across all hits in the alert."""

    alert_id: str
    resolution_status: str  # "LINKED" | "INFERRED_UNIQUE" | "AMBIGUOUS" | "UNRESOLVED"
    customer_record_id: str | None
    candidates: list[PersonProfile] = field(default_factory=list)


@dataclass
class HitProfile:
    """Resolved watchlist hit with waterfall DOB, countries, dated reports, and provenance."""

    hit_id: str
    alert_id: str
    entity_type: str
    primary_name: str
    all_names: list[str]
    dobs: list[PartialDate] = field(default_factory=list)
    dob_source: str | None = None  # "dobs_field" | "profile_notes" | "search_results" | None
    dob_evidence_ids: list[str] = field(default_factory=list)
    dob_detail: str = ""
    countries: set[str] = field(default_factory=set)
    deceased_date: PartialDate | None = None
    life_status: str | None = None
    earliest_report_date: PartialDate | None = None
    listing_phrase: str = "named on a watchlist"
    claims: list[Claim] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Per-candidate policy evaluation outcome prior to multi-candidate aggregation."""

    decision: str
    reason_code: str
    all_applicable_codes: list[str]
    person: PersonProfile
    evidence_ids: list[str]
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionOutput:
    """Final serialized output schema line per hit."""

    hit_id: str
    alert_id: str
    decision: str
    reason_code: str
    customer_record_id: str | None
    reason: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_id": self.hit_id,
            "alert_id": self.alert_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "customer_record_id": self.customer_record_id,
            "reason": self.reason,
            "evidence": self.evidence,
        }
