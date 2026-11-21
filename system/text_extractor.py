"""Extract structured facts from free-text profile notes, customer notes, and search snippets."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_client import LLMClient
from normalizer import (
    COUNTRY_ALIASES,
    MONTH_MAP,
    NON_COUNTRIES,
    PartialDate,
    normalize_country,
    parse_date,
    strip_accents,
)


NEGATIVE_KYC_RE = re.compile(
    r"\b(?:reject(?:ed)?|expir(?:ed)?|fail(?:ed)?|declin(?:ed)?|unverified|invalid|not\s+approved|unable\s+to\s+verify|could\s+not\s+verify)\b",
    re.I,
)
POSITIVE_KYC_RE = re.compile(
    r"\b(?:verif(?:y|ied|ication)|validat(?:ed|ion)|confirm(?:ed|ing)?|complet(?:ed|e)?|approv(?:ed)?|clear(?:ed)?|pass(?:ed)?|match(?:es|ed)?|process(?:ed)?|check(?:ed)?|accept(?:ed)?|vett(?:ed|ing)?|authenticat(?:ed|e)?|successful(?:ly)?)\b",
    re.I,
)

MONTH_NAMES_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

# Precompile country/demonym scanning regex (excluding ambiguous 2-letter codes in prose)
_COUNTRY_TERMS: List[str] = sorted(
    {k for k in COUNTRY_ALIASES.keys() if len(k) > 2} | set(COUNTRY_ALIASES.values()),
    key=len,
    reverse=True,
)
COUNTRY_SCAN_RE = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(t) for t in _COUNTRY_TERMS) + r")(?![a-z])",
    re.I,
)


def extract_countries_in_text(text: str) -> Set[str]:
    """Find canonical countries/demonyms mentioned by full name or demonym in English prose."""
    norm = strip_accents(text).lower()
    found: Set[str] = set()
    for m in COUNTRY_SCAN_RE.finditer(norm):
        c = normalize_country(m.group(1))
        if c and c not in NON_COUNTRIES:
            found.add(c)
    return found


def extract_countries_in_snippet(
    snippet: str,
    title: Optional[str] = None,
    hit_names: Optional[List[str]] = None,
) -> Set[str]:
    """Scan countries in a search snippet after masking out the subject's person name(s)."""
    masked = strip_accents(snippet)
    candidates: List[str] = []
    if title:
        head = re.split(r"\s+-\s+", strip_accents(title), maxsplit=1)[0].strip()
        if len(head.split()) >= 2:
            candidates.append(head)
    for nm in hit_names or []:
        clean = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", strip_accents(nm)).strip()
        if "," in clean:
            parts = [p.strip() for p in clean.split(",", 1)]
            if parts[0] and parts[1]:
                candidates.append(f"{parts[1]} {parts[0]}")
                candidates.append(f"{parts[0]} {parts[1]}")
        elif len(clean.split()) >= 2:
            candidates.append(clean)
    for cand in sorted(set(candidates), key=len, reverse=True):
        pat = re.compile(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", re.I)
        masked = pat.sub(" ", masked)
    return extract_countries_in_text(masked)


def extract_dated_reports(profile_notes: Optional[str]) -> List[PartialDate]:
    """Extract dated reports strictly matching 'According to ...; Month DD, YYYY:' (Principle 5)."""
    if not profile_notes:
        return []
    dates: List[PartialDate] = []
    pat = re.compile(rf"^According to [^\n]+?;\s*({MONTH_NAMES_RE}\s+\d{{1,2}},\s+\d{{4}}):")
    for line in profile_notes.splitlines():
        m = pat.match(line.strip())
        if m:
            dt = parse_date(m.group(1))
            if dt is not None:
                dates.append(dt)
    return dates


def _extract_age_and_ref_year_regex(text: str) -> Optional[Tuple[int, int]]:
    """Extract (age, reference_year) from prose like 'aged 30, in November 1998' or '33 year old ... May 2019'."""
    age: Optional[int] = None
    age_patterns = [
        r"\baged\s+(\d{1,3})\b",
        r"\bage\s+(\d{1,2})\b",
        r"\b(\d{1,3})[\s-]year[\s-]old\b",
        r"\bat\s+(?:the\s+)?age\s+(?:of\s+)?(\d{1,3})\b",
        r"\breached\s+(?:the\s+)?age\s+(?:of\s+)?(\d{1,3})\b",
        r"\breached\s+(\d{1,3})\s+years?\s+of\s+age\b",
        r"\bwas\s+(\d{1,3})\s+years?\s+old\b",
        r"\bwas\s+(\d{1,3})\b(?=\s+(?:and|when|in|on|at)\b|\s*[,.;])",
        r"\bage\s+of\s+[^.\n]{1,60}?\s+was\s+(\d{1,3})\b",
        r",\s*(\d{1,2}),\s+(?:[A-Z]|as\b|a\b|an\b|the\b|who\b|was\b|is\b|identified\b|listed\b|named\b|profiled\b|previously\b|working\b|from\b|in\b|took\b|worked\b|served\b|participated\b|had\b)",
        r",\s*(\d{1,2})\s+and\s+(?:named|a|the|worked|listed|identified)\b",
        r"\b(?:listed|identified|profiled|documented|featured|named|profile\s+on)\s+(\d{1,2}),\s+[A-Z]",
        r"\b(?:a|at|then)\s+(\d{1,2})\b(?=\s*[,.;]|\s+(?:was|is|who)\b)",
        r",\s*(?:at|then)\s+(\d{1,2})\s*,",
        r"\b(?!(?:Section|Article|Rule|Clause|Item|Part|Para|Paragraph|Chapter|Title|No|Vol)\b)[A-Z][a-z]+\s+\(\s*(\d{1,2})\s*\)",
    ]
    for pat in age_patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = int(m.group(1))
            if 5 <= val <= 115:
                age = val
                break
    if age is None:
        return None

    ref_year: Optional[int] = None
    ym_patterns = [
        rf"\b{MONTH_NAMES_RE}\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?(19\d{{2}}|20\d{{2}})\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_NAMES_RE},?\s+(19\d{{2}}|20\d{{2}})\b",
        r"\b(19\d{2}|20\d{2})-\d{2}-\d{2}\b",
        r"\b(?:A|In|From)\s+(19\d{2}|20\d{2})\b",
        r"\b(19\d{2}|20\d{2})\s+(?:news|public|regulator|media|official|report|file|files|notice|profile)\b",
    ]
    for pat in ym_patterns:
        m = re.search(pat, text, re.I)
        if m:
            ref_year = int(m.group(1))
            break
    if ref_year is None:
        return None
    return (age, ref_year)


def extract_profile_p1_dobs(
    profile_notes: Optional[str],
    is_company: bool,
    llm_client: Optional[LLMClient] = None,
) -> List[PartialDate]:
    """Extract DOBs or possible birth years from paragraph 1 of hit profile_notes (Principle 5)."""
    if is_company or not profile_notes:
        return []
    p1 = profile_notes.split("\n\n")[0].strip()
    if not p1 or p1.startswith("According to "):
        return []

    # 1. Explicit birth date / birth year
    m_date = re.search(
        rf"\b(?:born(?:\s+on)?|DOB\s*[:=]?|date\s+of\s+birth\s*[:=]?)\s+(\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}\s+{MONTH_NAMES_RE}\s+\d{{4}}|{MONTH_NAMES_RE}\s+\d{{1,2}},?\s+\d{{4}})\b",
        p1,
        re.I,
    )
    if m_date:
        dt = parse_date(m_date.group(1))
        if dt:
            return [dt]

    m_yr = re.search(r"\bborn\s+(?:in\s+)?(19\d{2}|20\d{2})\b", p1, re.I)
    if m_yr:
        return [PartialDate(int(m_yr.group(1)))]

    # 2. Age at date -> two possible birth years {ref_year - age - 1, ref_year - age}
    ar = _extract_age_and_ref_year_regex(p1)
    if ar is not None:
        age, ref_year = ar
        return [PartialDate(ref_year - age - 1), PartialDate(ref_year - age)]

    # 3. LLM fallback/extraction when enabled (handles novel unseen phrasing)
    if (
        llm_client
        and llm_client.enabled
        and not p1.startswith(("Career:", "Political Party:", "Position:"))
        and re.search(r"\b(?:born|birth|dob|d\.o\.b|yob|age|aged|old|year-old|years\s+old|then\s+\d{1,2})\b", p1, re.I)
    ):
        res = llm_client.extract_json(
            "hit_p1_dob",
            (
                "Extract the subject's birth date OR age at a dated event from this watchlist profile paragraph. "
                "NEVER treat a sanction, designation, arrest, election, or imposition date as a birth date. "
                "Return JSON only: "
                '{"birth_date": "YYYY-MM-DD or YYYY-MM or YYYY or null", "age": int_or_null, "event_year": int_or_null}. '
                "Do not guess missing parts. Only return facts explicitly stated in the text."
            ),
            p1,
        )
        if res:
            bd = parse_date(res.get("birth_date"))
            if bd and str(bd.year) in p1:
                return [bd]
            age_v = res.get("age")
            yr_v = res.get("event_year")
            if isinstance(age_v, int) and isinstance(yr_v, int) and str(age_v) in p1 and str(yr_v) in p1:
                return [PartialDate(yr_v - age_v - 1), PartialDate(yr_v - age_v)]

    return []


def extract_search_snippet_dobs(
    snippet: str,
    llm_client: Optional[LLMClient] = None,
) -> List[PartialDate]:
    """Extract possible birth years from a validated search_results snippet."""
    ar = _extract_age_and_ref_year_regex(snippet)
    if ar is not None:
        age, ref_year = ar
        return [PartialDate(ref_year - age - 1), PartialDate(ref_year - age)]

    m_yr = re.search(r"\bborn\s+(?:in\s+)?(19\d{2}|20\d{2})\b", snippet, re.I)
    if m_yr:
        return [PartialDate(int(m_yr.group(1)))]

    if llm_client and llm_client.enabled:
        res = llm_client.extract_json(
            "search_snippet_dob",
            (
                "Extract the person's age and the report year from this news/watchlist snippet. "
                'Return JSON only: {"age": int_or_null, "report_year": int_or_null, "birth_year": int_or_null}. '
                "Only extract numbers explicitly stated in the text."
            ),
            snippet,
        )
        if res:
            by = res.get("birth_year")
            if isinstance(by, int) and str(by) in snippet:
                return [PartialDate(by)]
            age_v = res.get("age")
            yr_v = res.get("report_year")
            if isinstance(age_v, int) and isinstance(yr_v, int) and str(age_v) in snippet and str(yr_v) in snippet:
                return [PartialDate(yr_v - age_v - 1), PartialDate(yr_v - age_v)]
    return []


@dataclass
class NoteKYC:
    """Identity verification extracted from a support/KYC note."""

    note_id: str
    checked_at: Optional[PartialDate]
    verified_name: Optional[str]
    verified_dob: Optional[PartialDate]
    document_country: Optional[str]
    approved: bool


@dataclass
class NoteFacts:
    """Structured facts extracted from a single notes.jsonl row."""

    note_id: str
    kyc: Optional[NoteKYC] = None
    lived_countries: Set[str] = field(default_factory=set)


_DATE_TOKEN_RE = re.compile(
    rf"(\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}\s+{MONTH_NAMES_RE}\s+\d{{4}}|{MONTH_NAMES_RE}\s+\d{{1,2}},?\s+\d{{4}})"
)


def _extract_kyc_note_regex(note: Dict[str, Any]) -> Optional[NoteKYC]:
    """Deterministically parse KYC verification details from channel='kyc' or KYC text."""
    text = note.get("text") or ""
    if not text:
        return None
    if NEGATIVE_KYC_RE.search(text):
        return NoteKYC(
            note_id=note["note_id"],
            checked_at=parse_date(note.get("created_at")),
            verified_name=None,
            verified_dob=None,
            document_country=None,
            approved=False,
        )

    is_kyc_channel = note.get("channel") == "kyc"
    has_kyc_words = bool(
        re.search(r"\b(passport|national\s+id|driving\s+licen[cs]e|identity\s+(?:check|verification|confirmed)|Sumsub|Onfido|Persona|Veriff|DOB)\b", text, re.I)
    )
    if not is_kyc_channel and not has_kyc_words:
        return None
    if not POSITIVE_KYC_RE.search(text):
        return None

    # Extract DOB explicitly tagged by DOB / born / parentheses
    verified_dob: Optional[PartialDate] = None
    dob_patterns = [
        rf"\b(?:DOB|date\s+of\s+birth|birth\s+date)(?:\s+on\s+document|\s+recorded|\s+verified|\s+confirmed|\s+noted|\s+provided)?(?:\s+(?:is|was))?(?:\s+(?:recorded|verified|confirmed|noted|provided))?(?:\s+(?:as|is|of))?\s*[:=]?\s*{_DATE_TOKEN_RE.pattern}",
        rf"\b(?:confirming|noting|lists|listing|recorded)\s+(?:a|his|her|their)?\s*(?:DOB|date\s+of\s+birth|birth\s+date)\s+(?:of|as)\s+{_DATE_TOKEN_RE.pattern}",
        rf"\b(?:lists|listing|shows)\s+{_DATE_TOKEN_RE.pattern}\s+as\s+(?:the\s+)?(?:DOB|date\s+of\s+birth|birth\s+date)\b",
        rf"{_DATE_TOKEN_RE.pattern}\s+(?:DOB|date\s+of\s+birth|birth\s+date)\b",
        rf"\bborn(?:\s+on)?\s+{_DATE_TOKEN_RE.pattern}",
        rf"\(\s*(?:DOB\s+)?{_DATE_TOKEN_RE.pattern}\s*\)",
    ]
    dob_span: Optional[Tuple[int, int]] = None
    for pat in dob_patterns:
        m = re.search(pat, text, re.I)
        if m:
            dt = parse_date(m.group(1))
            if dt:
                verified_dob = dt
                dob_span = m.span(1)
                break

    # Extract checked_at: fallback to note.created_at or remaining non-DOB date
    checked_at = parse_date(note.get("created_at"))
    for m in _DATE_TOKEN_RE.finditer(text):
        if dob_span and m.span(1) == dob_span:
            continue
        dt = parse_date(m.group(1))
        if dt:
            checked_at = dt
            break

    # Extract verified name: uppercase spans of >=2 words (or before "'s identity")
    verified_name: Optional[str] = None
    m_poss = re.search(r"^([^\n]{2,80}?)'s\s+(?:identity|passport|national|driving)\b", text)
    if m_poss and m_poss.group(1).strip().isupper():
        verified_name = m_poss.group(1).strip()
    if not verified_name:
        # Collect contiguous ALL-CAPS Unicode token spans (including N/A or digits inside names)
        tokens = text.split()
        spans: List[str] = []
        cur: List[str] = []
        for raw_tok in tokens:
            clean_tok = raw_tok.strip(".,;:()[]{}\"")
            clean_tok = re.sub(r"['’]s$", "", clean_tok)
            has_alpha = any(ch.isalpha() for ch in clean_tok)
            is_caps = (has_alpha and clean_tok == clean_tok.upper()) or (cur and clean_tok.isdigit())
            # Exclude standalone single-word markers like DOB / ID / KYC
            if is_caps and clean_tok not in {"DOB", "ID", "KYC"}:
                cur.append(clean_tok)
                # Break span if raw_tok ended with punctuation (, ; . :)
                if raw_tok.endswith((",", ";", ".", ":", ")")):
                    if len(cur) >= 2:
                        spans.append(" ".join(cur))
                    cur = []
            else:
                if len(cur) >= 2:
                    spans.append(" ".join(cur))
                cur = []
        if len(cur) >= 2:
            spans.append(" ".join(cur))
        filtered = [s for s in spans if any(any(c.isalpha() for c in w) and w not in {"DOB", "ID", "KYC", "N/A", "NA"} for w in s.split())]
        if filtered:
            verified_name = max(filtered, key=len)

    # Extract document country from text
    countries = extract_countries_in_text(text)
    doc_country = next(iter(countries), None) if len(countries) == 1 else None
    if len(countries) > 1:
        # Prefer explicit "<Country>-issued", "from <Country>", or "issued in/by <Country>"
        m_c = re.search(
            r"\b([A-Za-z\s'.-]+?)-issued\b|\b(?:from|issued\s+(?:in|by))\s+(?:the\s+)?([A-Z][A-Za-z\s'.-]+?)(?:\s+on\b|[.,;(]|$)",
            text,
        )
        if m_c:
            raw_c = m_c.group(1) or m_c.group(2)
            nc = normalize_country(raw_c)
            if nc:
                doc_country = nc

    return NoteKYC(
        note_id=note["note_id"],
        checked_at=checked_at,
        verified_name=verified_name,
        verified_dob=verified_dob,
        document_country=doc_country,
        approved=True,
    )


def extract_note_facts(note: Dict[str, Any], llm_client: Optional[LLMClient] = None) -> NoteFacts:
    """Extract KYC checks and residence countries from a single note using regex + optional LLM."""
    text = (note.get("text") or "").strip()
    nid = note["note_id"]
    facts = NoteFacts(note_id=nid)
    if not text:
        return facts

    channel = note.get("channel") or ""

    # 1. Try deterministic KYC note parsing
    kyc_res = _extract_kyc_note_regex(note)
    if kyc_res is not None:
        # Verify completeness; if regex missed name/dob/country and LLM is available, refine with LLM
        if (
            llm_client
            and llm_client.enabled
            and kyc_res.approved
            and (not kyc_res.verified_name or not kyc_res.verified_dob or not kyc_res.document_country)
        ):
            llm_kyc = _extract_kyc_note_llm(note, llm_client)
            if llm_kyc is not None:
                if not llm_kyc.approved:
                    facts.kyc = llm_kyc
                else:
                    facts.kyc = NoteKYC(
                        note_id=nid,
                        checked_at=kyc_res.checked_at or llm_kyc.checked_at,
                        verified_name=kyc_res.verified_name or llm_kyc.verified_name,
                        verified_dob=kyc_res.verified_dob or llm_kyc.verified_dob,
                        document_country=kyc_res.document_country or llm_kyc.document_country,
                        approved=True,
                    )
                return facts
        facts.kyc = kyc_res
        return facts

    # 2. If channel == 'kyc' (or mentions KYC words) and regex didn't recognize it, try LLM!
    if channel == "kyc" or re.search(r"\b(verif|passport|national id|driving licen|kyc|sumsub|onfido|persona|veriff|born|dob)\b", text, re.I):
        if llm_client and llm_client.enabled:
            llm_kyc = _extract_kyc_note_llm(note, llm_client)
            if llm_kyc is not None:
                facts.kyc = llm_kyc
                return facts

    # 3. Residence / lived-in notes (exclude KYC notes and pure non-country boilerplate)
    countries_mentioned = extract_countries_in_text(text)
    if countries_mentioned:
        # When LLM is enabled, verify whether the note actually states the customer lived/worked/resided there!
        if llm_client and llm_client.enabled:
            res = llm_client.extract_json(
                "note_residence",
                (
                    "Extract any countries where this customer lived, resided, was based, or worked according to this support note. "
                    'Return JSON only: {"lived_countries": ["CountryName", ...]}. '
                    "Return an empty list if the country is only about a card payment, transfer destination, or unrelated topic."
                ),
                text,
            )
            if res and isinstance(res.get("lived_countries"), list):
                grounded = set()
                for c in res["lived_countries"]:
                    nc = normalize_country(str(c))
                    if nc and nc in countries_mentioned:
                        grounded.add(nc)
                facts.lived_countries = grounded
                return facts

        # Pure deterministic fallback: accept countries in non-KYC notes when residence/work cues appear
        # or note isn't about a pure transaction destination
        if not re.search(r"\b(sent\s+(?:money|funds|a\s+transfer)\s+to|payment\s+(?:in|to)|merchant\s+in)\b", text, re.I):
            facts.lived_countries = countries_mentioned

    return facts


def _extract_kyc_note_llm(note: Dict[str, Any], llm_client: LLMClient) -> Optional[NoteKYC]:
    """Use Grok LLM to extract structured KYC note fields with literal grounding."""
    text = note.get("text") or ""
    res = llm_client.extract_json(
        "note_kyc",
        (
            "Extract identity verification facts from this bank KYC note. "
            "Return JSON only: "
            '{"is_kyc": bool, "approved": bool, "verified_name": string_or_null, '
            '"verified_dob": "YYYY-MM-DD or YYYY-MM or YYYY or null", '
            '"document_country": string_or_null, "checked_at": "YYYY-MM-DD or null"}. '
            "Set approved=false if the check was rejected, failed, declined, or expired. "
            "NEVER return the verification/check date as verified_dob; set verified_dob=null if no birth date, DOB, or incorporation date is explicitly stated. "
            "Do not invent facts not explicitly in the note."
        ),
        text,
    )
    if not res or not res.get("is_kyc"):
        return None
    approved = bool(res.get("approved")) and not bool(NEGATIVE_KYC_RE.search(text))
    vname = res.get("verified_name")
    if isinstance(vname, str):
        vname = vname.strip()
        if not vname or strip_accents(vname).lower() not in strip_accents(text).lower():
            # Allow simple case/punctuation differences; drop hallucinated names
            words = [w for w in re.split(r"\s+", strip_accents(vname).lower()) if len(w) > 1]
            tlow = strip_accents(text).lower()
            if not words or not all(w in tlow for w in words):
                vname = None
    else:
        vname = None

    vdob = parse_date(res.get("verified_dob"))
    if vdob and str(vdob.year) not in text:
        vdob = None

    doc_country = normalize_country(res.get("document_country"))
    if doc_country and doc_country not in extract_countries_in_text(text):
        doc_country = None

    created_dt = parse_date(note.get("created_at"))
    checked_at = parse_date(res.get("checked_at")) or created_dt
    if vdob and ((checked_at and vdob == checked_at) or (created_dt and vdob == created_dt)):
        vdob = None
    return NoteKYC(
        note_id=note["note_id"],
        checked_at=checked_at,
        verified_name=vname,
        verified_dob=vdob,
        document_country=doc_country,
        approved=approved,
    )
