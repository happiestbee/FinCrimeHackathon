"""Bidirectional two-stage Anti-Namesake Search Shield over search_results.jsonl."""

from __future__ import annotations

import re
import threading
from typing import Any

from .ingest import DataStore
from .models import PartialDate

_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_DATE_MASK_RE = re.compile(
    rf"\b(?:{_MONTH_PAT}\.?\s+\d{{1,2}},?\s+(?:19|20)\d\d"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?,?\s+(?:19|20)\d\d"
    rf"|{_MONTH_PAT}\.?,?\s+(?:19|20)\d\d"
    rf"|{_MONTH_PAT}\.?\s+\d{{1,2}}"
    rf"|\d{{1,2}}\s+{_MONTH_PAT}\.?"
    rf"|(?:19|20)\d\d-\d{{1,2}}-\d{{1,2}}"
    rf"|(?:19|20)\d\d)\b",
    re.I,
)

_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_PAT})\.?,?\s+((?:19|20)\d\d)\b", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d\d)\b")

_BORN_DATE_RE = re.compile(
    rf"\bborn\s+(?:on\s+|in\s+)?(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\s+{_MONTH_PAT}\s+\d{{4}}|{_MONTH_PAT}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}})\b",
    re.I,
)

_NON_AGE_NOUNS = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|%|percent|million|billion|thousand|years?|months?|days?|hours?"
    r"|members?|people|persons?|suspects?|individuals?|defendants?|counts?|charges?"
    r"|companies|entities|countries|states|officers?|officials?|dollars?|euros?|pounds?|cases?|accounts?"
)

_AGE_PATTERNS_STRICT = [
    re.compile(r"\b(\d{1,2})\s*-?\s*years?\s*-?\s*old\b", re.I),
    re.compile(r"\baged\s+(\d{1,2})\b", re.I),
    re.compile(r"\breached\s+(?:the\s+)?age\s+of\s+(\d{1,2})\b", re.I),
    re.compile(r"\breached\s+(\d{1,2})\s+years?\s+of\s+age\b", re.I),
    re.compile(r"\bat\s+(\d{1,2})\s+years?\s+of\s+age\b", re.I),
    re.compile(r"\bage\s+(?:of\s+)?(\d{1,2})\b", re.I),
    re.compile(
        rf"\b(?:at\s+age|then)\s+(\d{{1,2}})\b(?!\s*(?:{_NON_AGE_NOUNS})\b)",
        re.I,
    ),
    re.compile(
        rf"\b(?:he|she|subject|they|[A-Z][a-z]+)\s+(?:was|were)\s+(\d{{1,2}})\b"
        rf"(?!\s*(?:{_NON_AGE_NOUNS})\b)",
    ),
    re.compile(
        rf"\b(?:was|were)\s+(\d{{1,2}})\s*(?:,|\.|when\b|while\b|at\s+the\s+time\b|$)",
        re.I,
    ),
    re.compile(
        rf",\s*at\s+(\d{{1,2}})\s*(?:,|\.|when\b|while\b|$)",
        re.I,
    ),
]

_AGE_PATTERNS_RAW = [
    re.compile(r"\b(\d{1,2})\s*-?\s*years?\s*-?\s*old\b", re.I),
    re.compile(r"\baged?\s+(\d{1,2})\b", re.I),
    re.compile(r"\breached\s+(?:the\s+)?age\s+of\s+(\d{1,2})\b", re.I),
    re.compile(r"\breached\s+(\d{1,2})\s+years?\s+of\s+age\b", re.I),
    re.compile(r"\bat\s+(\d{1,2})\s+years?\s+of\s+age\b", re.I),
    re.compile(r"\bage\s+(?:of\s+)?(\d{1,2})\b", re.I),
    re.compile(
        r"(?<!\bSection )(?<!\bArticle )(?<!\bParagraph )(?<!\bClause )(?<!\bRule )(?<!\bOrder )(?<!\bItem )(?<!\bPart )"
        r"\b[A-ZÀ-ÖØ-ÞŁŚŹŻĆŃĘĄÓ][a-zà-öø-ÿłśźżćńęąó']+\s+\((\d{1,2})\)"
    ),
    re.compile(
        rf"\b(?:at|then|was|were|while|being)\s+(?:age\s+)?(\d{{1,2}})\b"
        rf"(?!\s*(?:{_NON_AGE_NOUNS})\b)",
        re.I,
    ),
]

_AGE_PATTERNS_MASKED = [
    re.compile(r",\s*(\d{1,2})\s*,"),
    re.compile(r",\s*(\d{1,2})\s*\."),
    re.compile(r",\s*(\d{1,2})\s+(?:and|who|a|an|the)\b", re.I),
    re.compile(r"\b(\d{1,2}),\s+(?:a|an|the|[A-Z])"),
    re.compile(
        rf"\b(?:a|the)\s+(\d{{1,2}})\b(?!\s*(?:{_NON_AGE_NOUNS})\b)",
        re.I,
    ),
    re.compile(r",\s*(\d{1,2})\s*$"),
    re.compile(
        r"\b(?:listed|on|with|to|for|featured|highlighted|profiled|identified|included|focused\s+on|named)\s+(\d{1,2})\s*,",
        re.I,
    ),
]


def _extract_event_year(text: str) -> int | None:
    m_my = _MONTH_YEAR_RE.search(text)
    if m_my:
        return int(m_my.group(2))
    years = sorted({int(y) for y in _YEAR_RE.findall(text)})
    if len(years) == 1:
        return years[0]
    return None


def extract_age_and_year_from_profile_note(text: str) -> tuple[int | None, int | None]:
    """Strict age+year extractor for watchlist profile_notes (guards against legal clause/count numbers)."""
    if not text:
        return None, None
    event_year = _extract_event_year(text)
    age: int | None = None
    for pat in _AGE_PATTERNS_STRICT:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 12 <= val <= 105:
                age = val
                break
    return age, event_year


def extract_age_and_year_from_prose(text: str) -> tuple[int | None, int | None]:
    """Deterministically extract (age, event_year) from a short biographical search snippet."""
    if not text:
        return None, None

    event_year = _extract_event_year(text)

    # Age extraction: raw explicit patterns first
    age: int | None = None
    for pat in _AGE_PATTERNS_RAW:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 12 <= val <= 105:
                age = val
                break

    if age is None:
        masked = _DATE_MASK_RE.sub(" <DATE> ", text)
        for pat in _AGE_PATTERNS_MASKED:
            m = pat.search(masked)
            if m:
                val = int(m.group(1))
                if 12 <= val <= 105:
                    age = val
                    break

    return age, event_year


class SearchShield:
    """Anti-namesake shield enforcing listing-category and country anchors."""

    def __init__(self, store: DataStore, llm_client: Any = None, verifier: Any = None) -> None:
        self.store = store
        self.llm_client = llm_client
        self.verifier = verifier
        self._stats_lock = threading.Lock()
        self.stats = {
            "hits_searched": 0,
            "snippets_matched": 0,
            "snippets_llm": 0,
        }

    def lookup_hit_dobs(
        self,
        hit_names: list[str],
        hit_countries: set[str],
        listing_phrase: str,
    ) -> tuple[list[PartialDate], list[str], str]:
        """Search search_results.jsonl for an Individual hit lacking DOB in structured/notes fields."""
        if not hit_countries or not hit_names:
            return [], [], ""

        with self._stats_lock:
            self.stats["hits_searched"] += 1
        nm = self.store.name_matcher
        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Stage A1: exact canonical query token-set match across hit names/aliases
        for name in hit_names:
            parts = nm.extract_name_parts(name, is_company=False)
            if not parts:
                continue
            for row in self.store.sr_by_query_tokens.get(frozenset(parts), ()):
                rid = row["result_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    candidates.append(row)

        # Stage A2: token-based fallback if exact token-set match returned nothing.
        # Guard: never allow a 1-part alias (e.g. "Ali") to match multi-part queries of other people.
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
                    candidates.append(row)

        # Stage B: Strict Policy Principle 5 filter (listing phrase + country anchor)
        phrase_lc = listing_phrase.lower()
        matched_rows: list[dict[str, Any]] = []
        for row in candidates:
            snip = row.get("snippet") or ""
            if phrase_lc not in snip.lower():
                continue
            title_name = (row.get("title") or "").split(" - ")[0].strip()
            mask_names = list(hit_names)
            if row.get("query"):
                mask_names.append(row["query"])
            if title_name:
                mask_names.append(title_name)
            snip_countries = set(
                self.store.geo.extract_countries_from_prose(snip, mask_names=mask_names)
            )
            if not (snip_countries & hit_countries):
                continue
            matched_rows.append(row)

        if not matched_rows:
            return [], [], ""

        matched_rows.sort(key=lambda r: r["result_id"])
        possible_dobs: list[PartialDate] = []
        used_sr_ids: list[str] = []
        detail_parts: list[str] = []

        for row in matched_rows:
            snip = row["snippet"]
            rid = row["result_id"]
            extracted_for_row: list[PartialDate] = []

            # Check explicit born date first
            from .normalize import parse_partial_date

            m_born = _BORN_DATE_RE.search(snip)
            if m_born:
                pd = parse_partial_date(m_born.group(1), day_first=False)
                if pd is not None:
                    extracted_for_row.append(pd)
                    detail_parts.append(f"born {pd.iso_str} ({rid})")

            if not extracted_for_row:
                age, ev_year = extract_age_and_year_from_prose(snip)
                if age is not None and ev_year is not None:
                    y1, y2 = ev_year - age - 1, ev_year - age
                    extracted_for_row.extend(
                        [PartialDate(y1, raw=f"{age} in {ev_year}"), PartialDate(y2, raw=f"{age} in {ev_year}")]
                    )
                    detail_parts.append(f"aged {age} in {ev_year} -> {y1} or {y2} ({rid})")

            # Trigger-gated LLM fallback for verified anchored snippet that failed regex
            if not extracted_for_row and self.llm_client is not None and self.llm_client.enabled:
                llm_res = self.llm_client.extract_search_snippet_dob(
                    snip, hit_names, listing_phrase, sorted(hit_countries)
                )
                if llm_res and self.verifier:
                    verified = self.verifier.verify_search_dob_extraction(
                        llm_res, snip, hit_names, hit_countries, listing_phrase
                    )
                    if verified:
                        with self._stats_lock:
                            self.stats["snippets_llm"] += 1
                        extracted_for_row.extend(verified)
                        detail_parts.append(
                            f"{' or '.join(d.iso_str for d in verified)} ({rid})"
                        )

            if extracted_for_row:
                with self._stats_lock:
                    self.stats["snippets_matched"] += 1
                used_sr_ids.append(rid)
                for pd in extracted_for_row:
                    if pd not in possible_dobs:
                        possible_dobs.append(pd)

        return possible_dobs, used_sr_ids, "; ".join(detail_parts)
