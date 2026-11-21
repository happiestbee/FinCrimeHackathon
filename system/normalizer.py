"""Deterministic normalization and matching helpers for dates, countries, and names."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

PLACEHOLDERS: Set[str] = {
    "na", "n/a", "test", "unknown", "xxx", "none", "null", "nil", "n.a", "tbd",
}

NON_COUNTRIES: Set[str] = {
    "international", "europe", "global", "worldwide", "unknown", "n/a", "na", "none",
}

# Canonical lowercase country/territory mapping covering ISO-2 codes, demonyms, and variants.
COUNTRY_ALIASES: Dict[str, str] = {
    # ISO-2 codes observed and standard
    "af": "afghanistan", "al": "albania", "dz": "algeria", "ar": "argentina",
    "am": "armenia", "au": "australia", "at": "austria", "az": "azerbaijan",
    "bh": "bahrain", "bd": "bangladesh", "by": "belarus", "be": "belgium",
    "ba": "bosnia and herzegovina", "br": "brazil", "vg": "british virgin islands",
    "cm": "cameroon", "ca": "canada", "cn": "china", "co": "colombia",
    "cr": "costa rica", "ci": "cote d'ivoire", "hr": "croatia",
    "cz": "czech republic", "dk": "denmark", "do": "dominican republic",
    "ec": "ecuador", "eg": "egypt", "er": "eritrea", "ee": "estonia",
    "et": "ethiopia", "fi": "finland", "fr": "france", "ge": "georgia",
    "de": "germany", "gh": "ghana", "gt": "guatemala", "ht": "haiti",
    "hn": "honduras", "hk": "hong kong", "hu": "hungary", "is": "iceland",
    "in": "india", "id": "indonesia", "ir": "iran", "iq": "iraq",
    "ie": "ireland", "il": "israel", "it": "italy", "jm": "jamaica",
    "jp": "japan", "je": "jersey", "jo": "jordan", "ke": "kenya",
    "xk": "kosovo", "kw": "kuwait", "lv": "latvia", "lb": "lebanon",
    "li": "liechtenstein", "lu": "luxembourg", "mo": "macao",
    "my": "malaysia", "mx": "mexico", "md": "moldova", "me": "montenegro",
    "ma": "morocco", "np": "nepal", "nl": "netherlands", "nz": "new zealand",
    "ni": "nicaragua", "ng": "nigeria", "kp": "north korea",
    "mp": "northern mariana islands", "no": "norway", "om": "oman",
    "pk": "pakistan", "ps": "palestine", "pa": "panama", "py": "paraguay",
    "pe": "peru", "ph": "philippines", "pl": "poland", "pt": "portugal",
    "pr": "puerto rico", "qa": "qatar", "ro": "romania", "ru": "russia",
    "sa": "saudi arabia", "rs": "serbia", "sl": "sierra leone",
    "sg": "singapore", "sk": "slovakia", "si": "slovenia", "so": "somalia",
    "za": "south africa", "kr": "south korea", "es": "spain", "sd": "sudan",
    "se": "sweden", "ch": "switzerland", "sy": "syria", "tw": "taiwan",
    "th": "thailand", "tt": "trinidad and tobago", "tr": "turkey",
    "ug": "uganda", "ua": "ukraine", "ae": "united arab emirates",
    "gb": "united kingdom", "uk": "united kingdom", "us": "united states",
    "usa": "united states", "uy": "uruguay", "ve": "venezuela", "vn": "vietnam",
    # Demonyms & common variants
    "afghan": "afghanistan", "albanian": "albania", "algerian": "algeria",
    "american": "united states", "united states of america": "united states",
    "u.s.": "united states", "u.s": "united states",
    "argentine": "argentina", "argentinian": "argentina",
    "armenian": "armenia", "australian": "australia", "austrian": "austria",
    "azerbaijani": "azerbaijan", "bahraini": "bahrain",
    "bangladeshi": "bangladesh", "belarusian": "belarus", "belgian": "belgium",
    "bosnian": "bosnia and herzegovina", "brazilian": "brazil",
    "british": "united kingdom", "great britain": "united kingdom",
    "england": "united kingdom", "scotland": "united kingdom", "wales": "united kingdom",
    "cameroonian": "cameroon", "canadian": "canada",
    "chinese": "china", "colombian": "colombia", "costa rican": "costa rica",
    "ivorian": "cote d'ivoire", "ivory coast": "cote d'ivoire",
    "croatian": "croatia", "czech": "czech republic", "czechia": "czech republic",
    "danish": "denmark", "dominican": "dominican republic",
    "dutch": "netherlands", "the netherlands": "netherlands", "holland": "netherlands",
    "ecuadorian": "ecuador", "egyptian": "egypt", "eritrean": "eritrea",
    "estonian": "estonia", "ethiopian": "ethiopia", "finnish": "finland",
    "french": "france", "georgian": "georgia", "german": "germany",
    "ghanaian": "ghana", "guatemalan": "guatemala", "haitian": "haiti",
    "honduran": "honduras", "hungarian": "hungary", "icelandic": "iceland",
    "indian": "india", "indonesian": "indonesia", "iranian": "iran",
    "iraqi": "iraq", "irish": "ireland", "israeli": "israel",
    "italian": "italy", "jamaican": "jamaica", "japanese": "japan",
    "jordanian": "jordan", "kenyan": "kenya", "kosovar": "kosovo",
    "kuwaiti": "kuwait", "latvian": "latvia", "lebanese": "lebanon",
    "liechtensteiner": "liechtenstein", "luxembourgish": "luxembourg",
    "macau": "macao",
    "macao special administrative region of china": "macao",
    "hong kong special administrative region of china": "hong kong",
    "malaysian": "malaysia", "mexican": "mexico", "moldovan": "moldova",
    "montenegrin": "montenegro", "moroccan": "morocco", "nepalese": "nepal",
    "new zealander": "new zealand", "nicaraguan": "nicaragua",
    "nigerian": "nigeria", "north korean": "north korea",
    "democratic people's republic of korea": "north korea",
    "norwegian": "norway", "omani": "oman", "pakistani": "pakistan",
    "palestina": "palestine", "palestinian": "palestine",
    "panamanian": "panama", "paraguayan": "paraguay", "peruvian": "peru",
    "filipino": "philippines", "philippine": "philippines",
    "polish": "poland", "portuguese": "portugal", "qatari": "qatar",
    "romanian": "romania", "russian": "russia", "russian federation": "russia",
    "saudi": "saudi arabia", "saudi arabian": "saudi arabia",
    "serbian": "serbia", "sierra leonean": "sierra leone",
    "singaporean": "singapore", "slovak": "slovakia", "slovenian": "slovenia",
    "somali": "somalia", "south african": "south africa",
    "south korean": "south korea", "republic of korea": "south korea",
    "spanish": "spain", "sudanese": "sudan", "swedish": "sweden",
    "swiss": "switzerland", "syrian": "syria", "syrian arab republic": "syria",
    "taiwanese": "taiwan", "thai": "thailand", "turkish": "turkey",
    "türkiye": "turkey", "turkiye": "turkey",
    "ugandan": "uganda", "ukrainian": "ukraine", "emirati": "united arab emirates",
    "uae": "united arab emirates", "uruguayan": "uruguay",
    "venezuelan": "venezuela", "vietnamese": "vietnam", "viet nam": "vietnam",
}

DISPLAY_NAMES: Dict[str, str] = {
    "cote d'ivoire": "Côte d'Ivoire",
    "united kingdom": "United Kingdom",
    "united states": "United States",
    "united arab emirates": "United Arab Emirates",
    "czech republic": "Czech Republic",
    "dominican republic": "Dominican Republic",
    "new zealand": "New Zealand",
    "north korea": "North Korea",
    "south africa": "South Africa",
    "south korea": "South Korea",
    "saudi arabia": "Saudi Arabia",
    "sierra leone": "Sierra Leone",
    "costa rica": "Costa Rica",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "british virgin islands": "British Virgin Islands",
    "northern mariana islands": "Northern Mariana Islands",
    "puerto rico": "Puerto Rico",
    "trinidad and tobago": "Trinidad and Tobago",
    "hong kong": "Hong Kong",
}


def strip_accents(text: str) -> str:
    """Fold Unicode accents/diacritics and curly quotes to plain ASCII equivalents."""
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    repls = {
        "ß": "ss", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L",
        "ı": "i", "İ": "I", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "TH", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    }
    for src, dst in repls.items():
        if src in text:
            text = text.replace(src, dst)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalize_country(raw: Optional[str]) -> Optional[str]:
    """Normalize a single country/nationality/code string into canonical lowercase form."""
    if not raw:
        return None
    cleaned = strip_accents(str(raw)).strip()
    cleaned = re.sub(r"\s*\[[^\]]*\]\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).lower()
    if not cleaned or cleaned in NON_COUNTRIES:
        return None
    if cleaned in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[cleaned]
    if "," in cleaned:
        tail = cleaned.split(",")[-1].strip()
        if tail in NON_COUNTRIES:
            return None
        return COUNTRY_ALIASES.get(tail, tail or None)
    return cleaned


def display_country(canon: str) -> str:
    """Format canonical country key into readable English name."""
    if canon in DISPLAY_NAMES:
        return DISPLAY_NAMES[canon]
    return " ".join(w.capitalize() if w not in {"and", "of"} else w for w in canon.split())


def extract_country_from_location(raw: Optional[str]) -> Optional[str]:
    """Extract canonical country from an address, place_of_birth, or nationality string."""
    if not raw:
        return None
    cleaned = re.sub(r"\s*\[[^\]]*\]\s*$", "", str(raw)).strip()
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    for part in reversed(parts):
        norm = normalize_country(part)
        if norm:
            return norm
    return None


@dataclass(frozen=True)
class PartialDate:
    """Calendar date supporting year-only, year-month, or full YYYY-MM-DD."""

    year: int
    month: Optional[int] = None
    day: Optional[int] = None

    def format(self) -> str:
        if self.month is not None and self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.month is not None:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}"

    def is_full(self) -> bool:
        return self.month is not None and self.day is not None

    def tuple_key(self) -> Tuple[int, int, int]:
        return (self.year, self.month or 0, self.day or 0)


def parse_date(raw: Optional[str]) -> Optional[PartialDate]:
    """Parse common ISO, DD/MM/YYYY, Month DD YYYY, DD Month YYYY, and year formats."""
    if raw is None:
        return None
    s = strip_accents(str(raw)).strip()
    if not s:
        return None

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return PartialDate(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        return PartialDate(int(m.group(1)), int(m.group(2)), None)

    m = re.match(r"^(\d{4})$", s)
    if m:
        return PartialDate(int(m.group(1)), None, None)

    # DD/MM/YYYY (verified across dev set: day comes first)
    m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return PartialDate(y, mo, d)
        if 1 <= d <= 12 and 1 <= mo <= 31:
            return PartialDate(y, d, mo)

    # Month DD, YYYY or Month DD YYYY
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$", s)
    if m and m.group(1).lower() in MONTH_MAP:
        return PartialDate(int(m.group(3)), MONTH_MAP[m.group(1).lower()], int(m.group(2)))

    # DD Month YYYY
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})$", s)
    if m and m.group(2).lower() in MONTH_MAP:
        return PartialDate(int(m.group(3)), MONTH_MAP[m.group(2).lower()], int(m.group(1)))

    # Month YYYY
    m = re.match(r"^([A-Za-z]+),?\s+(\d{4})$", s)
    if m and m.group(1).lower() in MONTH_MAP:
        return PartialDate(int(m.group(2)), MONTH_MAP[m.group(1).lower()], None)

    return None


def dates_agree(a: PartialDate, b: PartialDate) -> bool:
    """True when years match and every part present in both agrees (Principle 7)."""
    if a.year != b.year:
        return False
    if a.month is not None and b.month is not None and a.month != b.month:
        return False
    if a.day is not None and b.day is not None and a.day != b.day:
        return False
    return True


def dob_contradicts(cust_dob: PartialDate, hit_dob: PartialDate) -> bool:
    """True if customer DOB contradicts a single hit DOB (Principle 6)."""
    if abs(cust_dob.year - hit_dob.year) > 1:
        return True
    if cust_dob.year == hit_dob.year:
        if cust_dob.month is not None and hit_dob.month is not None and cust_dob.month != hit_dob.month:
            return True
        if cust_dob.day is not None and hit_dob.day is not None and cust_dob.day != hit_dob.day:
            return True
    return False


def max_age_at(dob: PartialDate, ref: PartialDate) -> Optional[int]:
    """Conservative upper bound on age at `ref` (returns smallest possible certainty or exact max age)."""
    if ref.year < dob.year:
        return -1
    if dob.month is None or ref.month is None:
        # Unknown month: max possible age is ref.year - dob.year
        return ref.year - dob.year
    if ref.month < dob.month:
        return ref.year - dob.year - 1
    if ref.month > dob.month:
        return ref.year - dob.year
    # Same month
    if dob.day is None or ref.day is None:
        return ref.year - dob.year
    if ref.day < dob.day:
        return ref.year - dob.year - 1
    return ref.year - dob.year


def date_strictly_before(a: PartialDate, b: PartialDate) -> bool:
    """True only when every possible calendar day represented by `a` is strictly before `b`."""
    if a.year < b.year:
        return True
    if a.year > b.year:
        return False
    if a.month is None or b.month is None:
        return False
    if a.month < b.month:
        return True
    if a.month > b.month:
        return False
    if a.day is None or b.day is None:
        return False
    return a.day < b.day


def date_strictly_after(a: PartialDate, b: PartialDate) -> bool:
    """True only when every possible calendar day represented by `a` is strictly after `b`."""
    return date_strictly_before(b, a)


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


class NameMatcher:
    """Implements POLICY.md Principle 8 and name_variants.json rules."""

    def __init__(self, variants_path: str):
        with open(variants_path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.variant_groups: Dict[str, Set[int]] = {}
        for idx, group in enumerate(data.get("groups", [])):
            for token in group:
                norm = strip_accents(token).lower().strip()
                self.variant_groups.setdefault(norm, set()).add(idx)
        self.legal_forms: Set[str] = {
            strip_accents(x).lower().strip() for x in data.get("company_legal_forms", [])
        }

    def extract_parts(self, name: Optional[str], is_company: bool = False) -> List[str]:
        """Normalize a name into distinct real name parts per Principle 8."""
        if not name:
            return []
        s = strip_accents(str(name))
        # Remove anything inside () or []
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"\[[^\]]*\]", " ", s)
        s = s.lower()
        # Strip placeholders like n/a before punctuation splitting
        s = re.sub(r"\bn/a\b", " ", s)
        # Replace non-alphanumeric with spaces
        s = re.sub(r"[^a-z0-9]+", " ", s)

        parts: List[str] = []
        seen: Set[str] = set()
        for tok in s.split():
            if len(tok) <= 1:
                # Single letters are initials, not parts
                continue
            if tok in PLACEHOLDERS or tok.isdigit():
                continue
            if is_company and tok in self.legal_forms:
                continue
            if tok not in seen:
                seen.add(tok)
                parts.append(tok)
        return parts

    def search_query_key(self, name: Optional[str]) -> str:
        """Build search_results.jsonl query key: lowercase words sorted alphabetically."""
        if not name:
            return ""
        s = strip_accents(str(name))
        s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s).lower()
        words = [w for w in re.sub(r"[^a-z0-9]+", " ", s).split() if w]
        return " ".join(sorted(words))

    def parts_match(self, p1: str, p2: str) -> bool:
        """True if two individual name parts match under Principle 8."""
        if p1 == p2:
            return True
        g1 = self.variant_groups.get(p1)
        g2 = self.variant_groups.get(p2)
        if g1 and g2 and (g1 & g2):
            return True
        short_len = min(len(p1), len(p2))
        if short_len < 4:
            return False
        max_edits = 1 if short_len <= 6 else 2
        return levenshtein(p1, p2) <= max_edits

    def _parts_agree(self, parts_a: Sequence[str], parts_b: Sequence[str]) -> bool:
        """True when >=2 parts match or every part of either name is covered by matched parts."""
        if not parts_a or not parts_b:
            return False

        edges: List[Tuple[int, int]] = []
        for i, pa in enumerate(parts_a):
            for j, pb in enumerate(parts_b):
                if self.parts_match(pa, pb):
                    edges.append((1 << i, 1 << j))
            for j in range(len(parts_b) - 1):
                joined_b = parts_b[j] + parts_b[j + 1]
                if self.parts_match(pa, joined_b) and levenshtein(pa, joined_b) < min(
                    levenshtein(pa, parts_b[j]), levenshtein(pa, parts_b[j + 1])
                ):
                    edges.append((1 << i, (1 << j) | (1 << (j + 1))))
        for i in range(len(parts_a) - 1):
            joined_a = parts_a[i] + parts_a[i + 1]
            for j, pb in enumerate(parts_b):
                if self.parts_match(joined_a, pb) and levenshtein(joined_a, pb) < min(
                    levenshtein(parts_a[i], pb), levenshtein(parts_a[i + 1], pb)
                ):
                    edges.append(((1 << i) | (1 << (i + 1)), 1 << j))

        if not edges:
            return False

        full_a = (1 << len(parts_a)) - 1
        full_b = (1 << len(parts_b)) - 1

        if len(parts_a) <= 10 and len(parts_b) <= 10:
            best_at: Dict[Tuple[int, int], int] = {(0, 0): 0}
            stack: List[Tuple[int, int, int]] = [(0, 0, 0)]
            while stack:
                ma_cur, mb_cur, cnt = stack.pop()
                if cnt >= 2 or (cnt >= 1 and (ma_cur == full_a or mb_cur == full_b)):
                    return True
                for ma, mb in edges:
                    if not (ma_cur & ma) and not (mb_cur & mb):
                        na, nb, nc = ma_cur | ma, mb_cur | mb, cnt + 1
                        if nc >= 2 or (na == full_a or nb == full_b):
                            return True
                        if best_at.get((na, nb), -1) < nc:
                            best_at[(na, nb)] = nc
                            stack.append((na, nb, nc))
            return False

        # Fallback greedy matching for abnormally long company names
        used_a, used_b, count = 0, 0, 0
        for ma, mb in edges:
            if not (used_a & ma) and not (used_b & mb):
                used_a |= ma
                used_b |= mb
                count += 1
                if count >= 2 or used_a == full_a or used_b == full_b:
                    return True
        return False

    def names_agree(
        self,
        name_a: Optional[str],
        name_b: Optional[str],
        is_company: bool = False,
        require_min_two_a: bool = False,
    ) -> bool:
        """Check whether two name strings agree per Principle 8.

        Note: if `require_min_two_a` is True (e.g. an individual customer whose names
        must have >= 2 real parts to confirm or clear on name), fewer than 2 real
        parts in `name_a` returns False.
        """
        pa = self.extract_parts(name_a, is_company=is_company)
        pb = self.extract_parts(name_b, is_company=is_company)
        if not pa or not pb:
            return False
        if require_min_two_a and not is_company and len(pa) < 2:
            return False
        return self._parts_agree(pa, pb)
