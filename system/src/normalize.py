"""Normalization and comparison primitives for names, dates, geographies, and IDs."""

from __future__ import annotations

import calendar
from datetime import date
import json
import os
import re
import unicodedata
from typing import Iterable

from .models import PartialDate


_TRANSLIT_MAP = str.maketrans(
    {
        "ł": "l",
        "Ł": "l",
        "ø": "o",
        "Ø": "o",
        "đ": "d",
        "Đ": "d",
        "ð": "d",
        "Ð": "d",
        "þ": "th",
        "Þ": "th",
        "ß": "ss",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "ı": "i",
        "İ": "i",
        "ħ": "h",
        "Ħ": "h",
        "ŧ": "t",
        "Ŧ": "t",
    }
)

_BRACKETS_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NA_SLASH_RE = re.compile(r"(?i)\bn\s*/\s*a\b")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

PLACEHOLDER_TOKENS = frozenset(
    {"na", "test", "unknown", "xxx", "none", "null", "nil", "tbd", "n/a"}
)

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def fold_text(text: str | None) -> str:
    """Fold Unicode characters, transliterating non-NFKD letters and stripping diacritics."""
    if not text:
        return ""
    text = text.translate(_TRANSLIT_MAP)
    norm = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in norm if not unicodedata.combining(ch))
    return stripped.lower()


def normalize_phone(phone: str | None) -> str:
    """Normalize a phone number to digits only."""
    if not phone:
        return ""
    return re.sub(r"\D", "", phone)


def normalize_doc_number(doc: str | None) -> str:
    """Normalize an identity document number."""
    if not doc:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", doc).upper()
    return cleaned


def levenshtein(a: str, b: str) -> int:
    """Pure Levenshtein distance (insert/delete/substitute = 1; transposition = 2)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        bj = b[j - 1]
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(
                prev[i] + 1,      # insertion
                curr[i - 1] + 1,  # deletion
                prev[i - 1] + cost,  # substitution
            )
        prev = curr
    return prev[la]


class NameMatcher:
    """Principle 8 compliant name normalizer and comparator."""

    def __init__(self, variants_path: str | None = None, variants_data: dict | None = None):
        if variants_data is None and variants_path is not None:
            with open(variants_path, encoding="utf-8") as fh:
                variants_data = json.load(fh)
        variants_data = variants_data or {}

        self.variant_groups: dict[str, set[int]] = {}
        for idx, grp in enumerate(variants_data.get("groups", [])):
            for raw_tok in grp:
                tok = fold_text(raw_tok).strip()
                if tok:
                    self.variant_groups.setdefault(tok, set()).add(idx)

        self.company_legal_forms: set[str] = {
            fold_text(x).strip()
            for x in variants_data.get("company_legal_forms", [])
            if fold_text(x).strip()
        }

    def extract_name_parts(self, name: str | None, is_company: bool = False) -> list[str]:
        """Extract ordered unique non-trivial name parts per Principle 8."""
        if not name:
            return []
        cleaned = _BRACKETS_RE.sub(" ", name)
        cleaned = _NA_SLASH_RE.sub(" ", cleaned)
        folded = fold_text(cleaned)
        raw_tokens = _TOKEN_RE.findall(folded)

        parts: list[str] = []
        seen: set[str] = set()
        for tok in raw_tokens:
            if len(tok) < 2:
                continue
            if tok in PLACEHOLDER_TOKENS:
                continue
            if tok.isdigit():
                continue
            if re.fullmatch(r"x+", tok):
                continue
            if is_company and tok in self.company_legal_forms:
                continue
            if tok not in seen:
                seen.add(tok)
                parts.append(tok)
        return parts

    def real_part_count(self, name: str | None, is_company: bool = False) -> int:
        return len(self.extract_name_parts(name, is_company=is_company))

    def index_tokens(self, name: str | None, is_company: bool = False) -> list[str]:
        """Return canonical tokens (mapping variant group tokens to a group key) for indexing."""
        parts = self.extract_name_parts(name, is_company=is_company)
        keys: list[str] = []
        for p in parts:
            keys.append(p)
            for gid in sorted(self.variant_groups.get(p, ())):
                keys.append(f"__grp_{gid}")
        return list(dict.fromkeys(keys))

    def parts_match(self, a: str, b: str) -> bool:
        """Return True iff two normalized parts match under Principle 8."""
        if not a or not b:
            return False
        if a == b:
            return True
        ga = self.variant_groups.get(a)
        gb = self.variant_groups.get(b)
        if ga and gb and (ga & gb):
            return True
        min_len = min(len(a), len(b))
        if min_len < 4:
            return False
        max_edits = 1 if min_len <= 6 else 2
        if abs(len(a) - len(b)) > max_edits:
            return False
        return levenshtein(a, b) <= max_edits

    def names_agree(
        self,
        name_a: str | None,
        name_b: str | None,
        is_company: bool = False,
        extra_parts_a: Iterable[str] = (),
    ) -> bool:
        """Check whether two names agree per Principle 8 (including split/merged parts)."""
        parts_a = self.extract_name_parts(name_a, is_company=is_company)
        for ep in extra_parts_a:
            for tok in self.extract_name_parts(ep, is_company=is_company):
                if tok not in parts_a:
                    parts_a.append(tok)
        parts_b = self.extract_name_parts(name_b, is_company=is_company)
        return self._parts_lists_agree(parts_a, parts_b)

    def _split_pair_matches(self, single: str, p1: str, p2: str) -> bool:
        """Return True when a single token legitimately matches adjacent split parts p1 + p2.

        Any non-exact fuzzy split match must strictly reduce edit distance vs. matching either
        fragment alone so a long token (e.g. 'logistics') cannot absorb an unrelated short tag ('st').
        """
        joined = p1 + p2
        if not self.parts_match(single, joined):
            return False
        if single == joined:
            return True
        ga = self.variant_groups.get(single)
        gb = self.variant_groups.get(joined)
        if ga and gb and (ga & gb):
            return True
        d_joined = levenshtein(single, joined)
        return d_joined < levenshtein(single, p1) and d_joined < levenshtein(single, p2)

    def _parts_lists_agree(self, parts_a: list[str], parts_b: list[str]) -> bool:
        m, n = len(parts_a), len(parts_b)
        if m == 0 or n == 0:
            return False

        # Candidate match edges: (matched_a_tuple, matched_b_tuple)
        # 1-to-1, 2-to-1 (adjacent in A), and 1-to-2 (adjacent in B).
        edges_for_a: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = [[] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if self.parts_match(parts_a[i], parts_b[j]):
                    edges_for_a[i].append(((i,), (j,)))
                if j + 1 < n and self._split_pair_matches(parts_a[i], parts_b[j], parts_b[j + 1]):
                    edges_for_a[i].append(((i,), (j, j + 1)))
            if i + 1 < m:
                for j in range(n):
                    if self._split_pair_matches(parts_b[j], parts_a[i], parts_a[i + 1]):
                        edges_for_a[i].append(((i, i + 1), (j,)))

        # Quick upper bound check: if total potential single/pair start indices < 1, fail
        if not any(edges_for_a):
            return False

        # Exact backtracking search over A indices to see if any disjoint matching satisfies:
        # matched_count >= 2 OR covered_a == m OR covered_b == n
        used_a = [False] * m
        used_b = [False] * n

        def backtrack(idx: int, matched_count: int, cov_a: int, cov_b: int) -> bool:
            if matched_count >= 2 or cov_a == m or cov_b == n:
                return True
            if idx >= m:
                return False
            # Prune: remaining indices in A plus already matched_count cannot even reach 2 if neither cov_a==m nor cov_b==n is reachable
            rem_a = m - idx
            rem_b = n - cov_b
            if matched_count + min(rem_a, rem_b) < 2 and cov_a + rem_a < m and cov_b + rem_b < n:
                return False

            if not used_a[idx]:
                for a_idxs, b_idxs in edges_for_a[idx]:
                    if any(used_a[x] for x in a_idxs) or any(used_b[y] for y in b_idxs):
                        continue
                    for x in a_idxs:
                        used_a[x] = True
                    for y in b_idxs:
                        used_b[y] = True
                    if backtrack(
                        idx + 1,
                        matched_count + 1,
                        cov_a + len(a_idxs),
                        cov_b + len(b_idxs),
                    ):
                        return True
                    for x in a_idxs:
                        used_a[x] = False
                    for y in b_idxs:
                        used_b[y] = False

            return backtrack(idx + 1, matched_count, cov_a, cov_b)

        return backtrack(0, 0, 0, 0)

    def name_list_agrees(
        self,
        customer_names: Iterable[str | None],
        hit_names: Iterable[str | None],
        is_company: bool = False,
    ) -> bool:
        """Return True if any customer name agrees with any hit name/alias."""
        c_list = [x for x in customer_names if x and self.extract_name_parts(x, is_company)]
        h_list = [x for x in hit_names if x and self.extract_name_parts(x, is_company)]
        for cn in c_list:
            for hn in h_list:
                if self.names_agree(cn, hn, is_company=is_company):
                    return True
        return False


# --- Geographic / Country Normalization ---

NON_COUNTRY_REGIONS = frozenset(
    {
        "international",
        "europe",
        "european union",
        "eu",
        "asia",
        "africa",
        "north america",
        "south america",
        "latin america",
        "middle east",
        "global",
        "worldwide",
        "unknown",
        "stateless",
        "none",
        "n/a",
        "na",
        "other",
    }
)

# Canonical ISO-3166-1 alpha-2 -> Primary English display name.
# Territories carry their own ISO codes and MUST NOT be collapsed into parent sovereigns.
COUNTRIES_ISO_TO_NAME: dict[str, str] = {
    "AF": "Afghanistan",
    "AX": "Åland Islands",
    "AL": "Albania",
    "DZ": "Algeria",
    "AS": "American Samoa",
    "AD": "Andorra",
    "AO": "Angola",
    "AI": "Anguilla",
    "AQ": "Antarctica",
    "AG": "Antigua and Barbuda",
    "AR": "Argentina",
    "AM": "Armenia",
    "AW": "Aruba",
    "AU": "Australia",
    "AT": "Austria",
    "AZ": "Azerbaijan",
    "BS": "Bahamas",
    "BH": "Bahrain",
    "BD": "Bangladesh",
    "BB": "Barbados",
    "BY": "Belarus",
    "BE": "Belgium",
    "BZ": "Belize",
    "BJ": "Benin",
    "BM": "Bermuda",
    "BT": "Bhutan",
    "BO": "Bolivia",
    "BQ": "Bonaire, Sint Eustatius and Saba",
    "BA": "Bosnia and Herzegovina",
    "BW": "Botswana",
    "BV": "Bouvet Island",
    "BR": "Brazil",
    "IO": "British Indian Ocean Territory",
    "VG": "British Virgin Islands",
    "BN": "Brunei",
    "BG": "Bulgaria",
    "BF": "Burkina Faso",
    "BI": "Burundi",
    "CV": "Cabo Verde",
    "KH": "Cambodia",
    "CM": "Cameroon",
    "CA": "Canada",
    "KY": "Cayman Islands",
    "CF": "Central African Republic",
    "TD": "Chad",
    "CL": "Chile",
    "CN": "China",
    "CX": "Christmas Island",
    "CC": "Cocos (Keeling) Islands",
    "CO": "Colombia",
    "KM": "Comoros",
    "CG": "Republic of the Congo",
    "CD": "Democratic Republic of the Congo",
    "CK": "Cook Islands",
    "CR": "Costa Rica",
    "CI": "Côte d'Ivoire",
    "HR": "Croatia",
    "CU": "Cuba",
    "CW": "Curaçao",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DK": "Denmark",
    "DJ": "Djibouti",
    "DM": "Dominica",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "SV": "El Salvador",
    "GQ": "Equatorial Guinea",
    "ER": "Eritrea",
    "EE": "Estonia",
    "SZ": "Eswatini",
    "ET": "Ethiopia",
    "FK": "Falkland Islands",
    "FO": "Faroe Islands",
    "FJ": "Fiji",
    "FI": "Finland",
    "FR": "France",
    "GF": "French Guiana",
    "PF": "French Polynesia",
    "GA": "Gabon",
    "GM": "Gambia",
    "GE": "Georgia",
    "DE": "Germany",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GR": "Greece",
    "GL": "Greenland",
    "GD": "Grenada",
    "GP": "Guadeloupe",
    "GU": "Guam",
    "GT": "Guatemala",
    "GG": "Guernsey",
    "GN": "Guinea",
    "GW": "Guinea-Bissau",
    "GY": "Guyana",
    "HT": "Haiti",
    "VA": "Vatican City",
    "HN": "Honduras",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IS": "Iceland",
    "IN": "India",
    "ID": "Indonesia",
    "IR": "Iran",
    "IQ": "Iraq",
    "IE": "Ireland",
    "IM": "Isle of Man",
    "IL": "Israel",
    "IT": "Italy",
    "JM": "Jamaica",
    "JP": "Japan",
    "JE": "Jersey",
    "JO": "Jordan",
    "KZ": "Kazakhstan",
    "KE": "Kenya",
    "KI": "Kiribati",
    "KP": "North Korea",
    "KR": "South Korea",
    "XK": "Kosovo",
    "KW": "Kuwait",
    "KG": "Kyrgyzstan",
    "LA": "Laos",
    "LV": "Latvia",
    "LB": "Lebanon",
    "LS": "Lesotho",
    "LR": "Liberia",
    "LY": "Libya",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MO": "Macao",
    "MG": "Madagascar",
    "MW": "Malawi",
    "MY": "Malaysia",
    "MV": "Maldives",
    "ML": "Mali",
    "MT": "Malta",
    "MH": "Marshall Islands",
    "MQ": "Martinique",
    "MR": "Mauritania",
    "MU": "Mauritius",
    "YT": "Mayotte",
    "MX": "Mexico",
    "FM": "Micronesia",
    "MD": "Moldova",
    "MC": "Monaco",
    "MN": "Mongolia",
    "ME": "Montenegro",
    "MS": "Montserrat",
    "MA": "Morocco",
    "MZ": "Mozambique",
    "MM": "Myanmar",
    "NA": "Namibia",
    "NR": "Nauru",
    "NP": "Nepal",
    "NL": "Netherlands",
    "NC": "New Caledonia",
    "NZ": "New Zealand",
    "NI": "Nicaragua",
    "NE": "Niger",
    "NG": "Nigeria",
    "NU": "Niue",
    "NF": "Norfolk Island",
    "MK": "North Macedonia",
    "MP": "Northern Mariana Islands",
    "NO": "Norway",
    "OM": "Oman",
    "PK": "Pakistan",
    "PW": "Palau",
    "PS": "Palestine",
    "PA": "Panama",
    "PG": "Papua New Guinea",
    "PY": "Paraguay",
    "PE": "Peru",
    "PH": "Philippines",
    "PN": "Pitcairn Islands",
    "PL": "Poland",
    "PT": "Portugal",
    "PR": "Puerto Rico",
    "QA": "Qatar",
    "RE": "Réunion",
    "RO": "Romania",
    "RU": "Russia",
    "RW": "Rwanda",
    "BL": "Saint Barthélemy",
    "SH": "Saint Helena",
    "KN": "Saint Kitts and Nevis",
    "LC": "Saint Lucia",
    "MF": "Saint Martin",
    "PM": "Saint Pierre and Miquelon",
    "VC": "Saint Vincent and the Grenadines",
    "WS": "Samoa",
    "SM": "San Marino",
    "ST": "São Tomé and Príncipe",
    "SA": "Saudi Arabia",
    "SN": "Senegal",
    "RS": "Serbia",
    "SC": "Seychelles",
    "SL": "Sierra Leone",
    "SG": "Singapore",
    "SX": "Sint Maarten",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "SB": "Solomon Islands",
    "SO": "Somalia",
    "ZA": "South Africa",
    "SS": "South Sudan",
    "ES": "Spain",
    "LK": "Sri Lanka",
    "SD": "Sudan",
    "SR": "Suriname",
    "SJ": "Svalbard and Jan Mayen",
    "SE": "Sweden",
    "CH": "Switzerland",
    "SY": "Syria",
    "TW": "Taiwan",
    "TJ": "Tajikistan",
    "TZ": "Tanzania",
    "TH": "Thailand",
    "TL": "Timor-Leste",
    "TG": "Togo",
    "TK": "Tokelau",
    "TO": "Tonga",
    "TT": "Trinidad and Tobago",
    "TN": "Tunisia",
    "TR": "Turkey",
    "TM": "Turkmenistan",
    "TC": "Turks and Caicos Islands",
    "TV": "Tuvalu",
    "UG": "Uganda",
    "UA": "Ukraine",
    "AE": "United Arab Emirates",
    "GB": "United Kingdom",
    "US": "United States",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VU": "Vanuatu",
    "VE": "Venezuela",
    "VN": "Vietnam",
    "VI": "U.S. Virgin Islands",
    "WF": "Wallis and Futuna",
    "EH": "Western Sahara",
    "YE": "Yemen",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

_ISO3_TO_ISO2: dict[str, str] = {
    "AFG": "AF", "ALB": "AL", "DZA": "DZ", "ASM": "AS", "AND": "AD", "AGO": "AO", "AIA": "AI",
    "ARG": "AR", "ARM": "AM", "ABW": "AW", "AUS": "AU", "AUT": "AT", "AZE": "AZ", "BHS": "BS",
    "BHR": "BH", "BGD": "BD", "BRB": "BB", "BLR": "BY", "BEL": "BE", "BLZ": "BZ", "BEN": "BJ",
    "BMU": "BM", "BTN": "BT", "BOL": "BO", "BIH": "BA", "BWA": "BW", "BRA": "BR", "VGB": "VG",
    "BRN": "BN", "BGR": "BG", "BFA": "BF", "BDI": "BI", "KHM": "KH", "CMR": "CM", "CAN": "CA",
    "CPV": "CV", "CYM": "KY", "CAF": "CF", "TCD": "TD", "CHL": "CL", "CHN": "CN", "COL": "CO",
    "COM": "KM", "COG": "CG", "COD": "CD", "CRI": "CR", "CIV": "CI", "HRV": "HR", "CUB": "CU",
    "CUW": "CW", "CYP": "CY", "CZE": "CZ", "DNK": "DK", "DJI": "DJ", "DMA": "DM", "DOM": "DO",
    "ECU": "EC", "EGY": "EG", "SLV": "SV", "GNQ": "GQ", "ERI": "ER", "EST": "EE", "SWZ": "SZ",
    "ETH": "ET", "FRO": "FO", "FJI": "FJ", "FIN": "FI", "FRA": "FR", "GAB": "GA", "GMB": "GM",
    "GEO": "GE", "DEU": "DE", "GHA": "GH", "GIB": "GI", "GRC": "GR", "GRL": "GL", "GRD": "GD",
    "GUM": "GU", "GTM": "GT", "GGY": "GG", "GIN": "GN", "GNB": "GW", "GUY": "GY", "HTI": "HT",
    "HND": "HN", "HKG": "HK", "HUN": "HU", "ISL": "IS", "IND": "IN", "IDN": "ID", "IRN": "IR",
    "IRQ": "IQ", "IRL": "IE", "IMN": "IM", "ISR": "IL", "ITA": "IT", "JAM": "JM", "JPN": "JP",
    "JEY": "JE", "JOR": "JO", "KAZ": "KZ", "KEN": "KE", "PRK": "KP", "KOR": "KR", "KWT": "KW",
    "KGZ": "KG", "LAO": "LA", "LVA": "LV", "LBN": "LB", "LSO": "LS", "LBR": "LR", "LBY": "LY",
    "LIE": "LI", "LTU": "LT", "LUX": "LU", "MAC": "MO", "MDG": "MG", "MWI": "MW", "MYS": "MY",
    "MDV": "MV", "MLI": "ML", "MLT": "MT", "MRT": "MR", "MUS": "MU", "MEX": "MX", "MDA": "MD",
    "MCO": "MC", "MNG": "MN", "MNE": "ME", "MAR": "MA", "MOZ": "MZ", "MMR": "MM", "NAM": "NA",
    "NPL": "NP", "NLD": "NL", "NZL": "NZ", "NIC": "NI", "NER": "NE", "NGA": "NG", "MKD": "MK",
    "MNP": "MP", "NOR": "NO", "OMN": "OM", "PAK": "PK", "PSE": "PS", "PAN": "PA", "PNG": "PG",
    "PRY": "PY", "PER": "PE", "PHL": "PH", "POL": "PL", "PRT": "PT", "PRI": "PR", "QAT": "QA",
    "ROU": "RO", "RUS": "RU", "RWA": "RW", "SAU": "SA", "SEN": "SN", "SRB": "RS", "SYC": "SC",
    "SLE": "SL", "SGP": "SG", "SVK": "SK", "SVN": "SI", "SOM": "SO", "ZAF": "ZA", "SSD": "SS",
    "ESP": "ES", "LKA": "LK", "SDN": "SD", "SUR": "SR", "SWE": "SE", "CHE": "CH", "SYR": "SY",
    "TWN": "TW", "TJK": "TJ", "TZA": "TZ", "THA": "TH", "TLS": "TL", "TGO": "TG", "TTO": "TT",
    "TUN": "TN", "TUR": "TR", "TKM": "TM", "UGA": "UG", "UKR": "UA", "ARE": "AE", "GBR": "GB",
    "USA": "US", "URY": "UY", "UZB": "UZ", "VEN": "VE", "VNM": "VN", "VIR": "VI", "YEM": "YE",
    "ZMB": "ZM", "ZWE": "ZW", "XKX": "XK",
}

_COUNTRY_SYNONYMS_AND_DEMONYMS: dict[str, str] = {
    # Multi-word & official names / abbreviations
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "american": "US",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "british": "GB",
    "english": "GB",
    "scottish": "GB",
    "welsh": "GB",
    "uk": "GB",
    "u.k.": "GB",
    "united arab emirates": "AE",
    "uae": "AE",
    "u.a.e.": "AE",
    "emirati": "AE",
    "cote d'ivoire": "CI",
    "cote divoire": "CI",
    "ivory coast": "CI",
    "ivorian": "CI",
    "czechia": "CZ",
    "czech republic": "CZ",
    "czech": "CZ",
    "russian federation": "RU",
    "russia": "RU",
    "russian": "RU",
    "south korea": "KR",
    "republic of korea": "KR",
    "korea, republic of": "KR",
    "south korean": "KR",
    "north korea": "KP",
    "democratic people's republic of korea": "KP",
    "korea, democratic people's republic of": "KP",
    "dprk": "KP",
    "north korean": "KP",
    "palestina": "PS",
    "palestine": "PS",
    "state of palestine": "PS",
    "palestinian": "PS",
    "palestinian territory": "PS",
    "palestinian territories": "PS",
    "macao": "MO",
    "macau": "MO",
    "macao special administrative region of china": "MO",
    "macao sar": "MO",
    "hong kong": "HK",
    "hong kong sar": "HK",
    "hong kong special administrative region of china": "HK",
    "taiwan": "TW",
    "taiwan, province of china": "TW",
    "chinese taipei": "TW",
    "taiwanese": "TW",
    "republic of china": "TW",
    "people's republic of china": "CN",
    "prc": "CN",
    "china": "CN",
    "chinese": "CN",
    "viet nam": "VN",
    "vietnam": "VN",
    "vietnamese": "VN",
    "turkiye": "TR",
    "türkiye": "TR",
    "turkey": "TR",
    "turkish": "TR",
    "syrian arab republic": "SY",
    "syria": "SY",
    "syrian": "SY",
    "iran, islamic republic of": "IR",
    "islamic republic of iran": "IR",
    "iran": "IR",
    "iranian": "IR",
    "republic of moldova": "MD",
    "moldova, republic of": "MD",
    "moldova": "MD",
    "moldovan": "MD",
    "united republic of tanzania": "TZ",
    "tanzania, united republic of": "TZ",
    "bolivarian republic of venezuela": "VE",
    "venezuela, bolivarian republic of": "VE",
    "plurinational state of bolivia": "BO",
    "bolivia, plurinational state of": "BO",
    "holland": "NL",
    "the netherlands": "NL",
    "netherlands": "NL",
    "dutch": "NL",
    "myanmar": "MM",
    "burma": "MM",
    "burmese": "MM",
    "lao people's democratic republic": "LA",
    "lao pdr": "LA",
    "laos": "LA",
    "laotian": "LA",
    "brunei darussalam": "BN",
    "brunei": "BN",
    "cape verde": "CV",
    "cabo verde": "CV",
    "swaziland": "SZ",
    "eswatini": "SZ",
    "macedonia": "MK",
    "north macedonia": "MK",
    "bosnia": "BA",
    "bosnia and herzegovina": "BA",
    "bosnian": "BA",
    "trinidad": "TT",
    "trinidad and tobago": "TT",
    "congo": "CG",
    "republic of the congo": "CG",
    "congo, republic of the": "CG",
    "democratic republic of the congo": "CD",
    "congo, democratic republic of the": "CD",
    "dr congo": "CD",
    "drc": "CD",
    "holy see": "VA",
    "vatican": "VA",
    "vatican city": "VA",
    "u.s. virgin islands": "VI",
    "us virgin islands": "VI",
    "virgin islands, u.s.": "VI",
    "british virgin islands": "VG",
    "virgin islands, british": "VG",
    "northern mariana islands": "MP",
    # Single-word country demonyms
    "afghan": "AF",
    "albanian": "AL",
    "algerian": "DZ",
    "argentine": "AR",
    "argentinian": "AR",
    "armenian": "AM",
    "australian": "AU",
    "austrian": "AT",
    "azerbaijani": "AZ",
    "bahraini": "BH",
    "bangladeshi": "BD",
    "belarusian": "BY",
    "belgian": "BE",
    "bolivian": "BO",
    "brazilian": "BR",
    "bulgarian": "BG",
    "cambodian": "KH",
    "cameroonian": "CM",
    "canadian": "CA",
    "chilean": "CL",
    "colombian": "CO",
    "costa rican": "CR",
    "croatian": "HR",
    "cuban": "CU",
    "cypriot": "CY",
    "danish": "DK",
    "dominican": "DO",
    "ecuadorian": "EC",
    "egyptian": "EG",
    "salvadoran": "SV",
    "eritrean": "ER",
    "estonian": "EE",
    "ethiopian": "ET",
    "finnish": "FI",
    "french": "FR",
    "georgian": "GE",
    "german": "DE",
    "ghanaian": "GH",
    "greek": "GR",
    "guatemalan": "GT",
    "haitian": "HT",
    "honduran": "HN",
    "hungarian": "HU",
    "icelandic": "IS",
    "indian": "IN",
    "indonesian": "ID",
    "iraqi": "IQ",
    "irish": "IE",
    "israeli": "IL",
    "italian": "IT",
    "jamaican": "JM",
    "japanese": "JP",
    "jordanian": "JO",
    "kazakh": "KZ",
    "kazakhstani": "KZ",
    "kenyan": "KE",
    "kosovar": "XK",
    "kuwaiti": "KW",
    "kyrgyz": "KG",
    "latvian": "LV",
    "lebanese": "LB",
    "liberian": "LR",
    "libyan": "LY",
    "liechtensteiner": "LI",
    "lithuanian": "LT",
    "luxembourgish": "LU",
    "malaysian": "MY",
    "malian": "ML",
    "maltese": "MT",
    "mauritanian": "MR",
    "mexican": "MX",
    "mongolian": "MN",
    "montenegrin": "ME",
    "moroccan": "MA",
    "mozambican": "MZ",
    "namibian": "NA",
    "nepalese": "NP",
    "nepali": "NP",
    "new zealander": "NZ",
    "nicaraguan": "NI",
    "nigerian": "NG",
    "nigerien": "NE",
    "norwegian": "NO",
    "omani": "OM",
    "pakistani": "PK",
    "panamanian": "PA",
    "paraguayan": "PY",
    "peruvian": "PE",
    "filipino": "PH",
    "philippine": "PH",
    "polish": "PL",
    "portuguese": "PT",
    "puerto rican": "PR",
    "qatari": "QA",
    "romanian": "RO",
    "rwandan": "RW",
    "saudi": "SA",
    "saudi arabian": "SA",
    "senegalese": "SN",
    "serbian": "RS",
    "sierra leonean": "SL",
    "singaporean": "SG",
    "slovak": "SK",
    "slovenian": "SI",
    "somali": "SO",
    "south african": "ZA",
    "spanish": "ES",
    "sri lankan": "LK",
    "sudanese": "SD",
    "swedish": "SE",
    "swiss": "CH",
    "tajik": "TJ",
    "tanzanian": "TZ",
    "thai": "TH",
    "tunisian": "TN",
    "ugandan": "UG",
    "ukrainian": "UA",
    "uruguayan": "UY",
    "uzbek": "UZ",
    "venezuelan": "VE",
    "yemeni": "YE",
    "zambian": "ZM",
    "zimbabwean": "ZW",
}

_US_STATE_FALSE_POSITIVES = (
    "new jersey",
    "jersey city",
    "new mexico",
    "new england",
    "new south wales",
    "british columbia",
    "benin city",
    "bank of america",
    "american express",
    "american airlines",
    "latin america",
    "north america",
    "south america",
    "central america",
)

CITY_SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("munich", "munchen"),
    ("vienna", "wien"),
    ("cologne", "koln"),
    ("milan", "milano"),
    ("rome", "roma"),
    ("lisbon", "lisboa"),
    ("warsaw", "warszawa"),
    ("prague", "praha"),
    ("moscow", "moskva"),
    ("belgrade", "beograd"),
    ("florence", "firenze"),
    ("venice", "venezia"),
    ("naples", "napoli"),
    ("seville", "sevilla"),
    ("copenhagen", "kobenhavn"),
    ("gothenburg", "goteborg"),
    ("brussels", "bruxelles"),
    ("geneva", "geneve"),
    ("zurich", "zuerich"),
    ("frankfurt", "frankfurt am main"),
]


class GeoNormalizer:
    """Country, territory, and city normalizer following POLICY.md Principles 5 & 6."""

    def __init__(self) -> None:
        self.lookup: dict[str, str] = {}
        for iso, name in COUNTRIES_ISO_TO_NAME.items():
            self.lookup[iso.lower()] = iso
            self.lookup[fold_text(name)] = iso
            # Also without punctuation like apostrophes
            no_apos = fold_text(name).replace("'", "")
            self.lookup[no_apos] = iso
        for iso3, iso2 in _ISO3_TO_ISO2.items():
            self.lookup[iso3.lower()] = iso2
        for raw, iso in _COUNTRY_SYNONYMS_AND_DEMONYMS.items():
            f = fold_text(raw)
            self.lookup[f] = iso
            self.lookup[f.replace("'", "")] = iso

        # Build prose matching pattern from full country names + demonyms (NO bare 2-letter ISO codes!)
        prose_terms: dict[str, str] = {}
        for iso, name in COUNTRIES_ISO_TO_NAME.items():
            f = fold_text(name)
            if len(f) >= 4:
                prose_terms[f] = iso
                prose_terms[f.replace("'", "")] = iso
        for raw, iso in _COUNTRY_SYNONYMS_AND_DEMONYMS.items():
            f = fold_text(raw)
            if f in (
                "uk", "us", "usa", "u.s.", "u.s.a.", "u.k.", "uae", "u.a.e.",
                "dprk", "prc", "drc", "english",
            ):
                continue  # abbreviations handled case-sensitively; 'english' is usually the language
            if len(f) >= 4:
                prose_terms[f] = iso
                prose_terms[f.replace("'", "")] = iso
        self._prose_terms = prose_terms
        sorted_terms = sorted(prose_terms.keys(), key=len, reverse=True)
        self._prose_regex = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted_terms) + r")(?!\w)"
        )
        self._abbr_regex = re.compile(
            r"\b(U\.S\.A\.|U\.S\.|USA|U\.K\.|UK|U\.A\.E\.|UAE)\b"
        )
        self._abbr_map = {
            "U.S.A.": "US",
            "U.S.": "US",
            "USA": "US",
            "U.K.": "GB",
            "UK": "GB",
            "U.A.E.": "AE",
            "UAE": "AE",
        }
        self._false_pos_re = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(x) for x in _US_STATE_FALSE_POSITIVES) + r")(?!\w)"
        )
        self._ga_context_re = re.compile(
            r"\b(?:atlanta|savannah|augusta|macon|columbus|marietta|athens|fulton|cobb|dekalb),\s*georgia\b"
            r"|\bgeorgia\s*,\s*(?:united states|usa|u\.s\.|ga)\b"
        )

        self._city_map: dict[str, set[str]] = {}
        for grp in CITY_SYNONYM_GROUPS:
            gset = set(grp)
            for item in grp:
                self._city_map[item] = gset

    def display_name(self, iso: str) -> str:
        return COUNTRIES_ISO_TO_NAME.get(iso, iso)

    def normalize_structured_country(self, raw: str | None) -> str | None:
        """Normalize a structured country/nationality string to ISO-2, or None if unknown/region."""
        if not raw:
            return None
        cleaned = _BRACKETS_RE.sub("", raw).strip()
        if not cleaned:
            return None
        folded = fold_text(cleaned).strip(" .,:;")
        if folded in NON_COUNTRY_REGIONS:
            return None
        if folded in self.lookup:
            return self.lookup[folded]
        if folded.replace("'", "") in self.lookup:
            return self.lookup[folded.replace("'", "")]
        # Structured fields like nationality="Belgrade, Serbia" or "Birnin, Kebbi State, Nigeria"
        if "," in cleaned:
            last_part = cleaned.split(",")[-1].strip()
            return self.normalize_structured_country(last_part)
        return None

    def extract_countries_from_address(self, addr: str | None) -> list[str]:
        """Extract country from the last comma-separated part of an address (after stripping [tag])."""
        if not addr:
            return []
        cleaned = _BRACKETS_RE.sub("", addr).strip()
        if not cleaned:
            return []
        tail = cleaned.split(",")[-1].strip()
        iso = self.normalize_structured_country(tail)
        return [iso] if iso else []

    def extract_countries_from_pob(self, pob: str | None) -> list[str]:
        """Extract countries from place_of_birth (semicolon-separated places)."""
        if not pob:
            return []
        out: list[str] = []
        for segment in pob.split(";"):
            cleaned = _BRACKETS_RE.sub("", segment).strip()
            if not cleaned:
                continue
            tail = cleaned.split(",")[-1].strip()
            iso = self.normalize_structured_country(tail)
            if iso and iso not in out:
                out.append(iso)
        return out

    def extract_countries_from_prose(
        self, text: str | None, mask_names: Iterable[str] = ()
    ) -> list[str]:
        """Extract country ISO-2 codes mentioned in English prose (no 2-letter or name collisions)."""
        if not text:
            return []
        raw_masked = text
        # Mask exact occurrences of subject names before scanning for countries so names like
        # "Jordan Andrew Chartrand" or "Chad Miller" never self-anchor to Jordan (JO) or Chad (TD).
        name_spans: list[str] = []
        for nm in mask_names:
            if not nm or not nm.strip():
                continue
            cleaned_nm = _BRACKETS_RE.sub("", nm).strip()
            if len(cleaned_nm) >= 4:
                name_spans.append(cleaned_nm)
                if "," in cleaned_nm:
                    parts = [p.strip() for p in cleaned_nm.split(",") if p.strip()]
                    if len(parts) == 2:
                        name_spans.append(f"{parts[1]} {parts[0]}")
        name_spans = sorted(set(name_spans), key=len, reverse=True)

        folded = fold_text(raw_masked)
        for sp in name_spans:
            sp_folded = fold_text(sp)
            if len(sp_folded) >= 4:
                pat = r"(?<!\w)" + re.escape(sp_folded) + r"(?!\w)"
                folded = re.sub(pat, lambda m: " " * len(m.group(0)), folded)
                raw_masked = re.sub(
                    r"(?i)(?<!\w)" + re.escape(sp) + r"(?!\w)",
                    lambda m: " " * len(m.group(0)),
                    raw_masked,
                )

        found: list[str] = []
        # Check uppercase abbreviations on original text first
        for m in self._abbr_regex.finditer(raw_masked):
            iso = self._abbr_map.get(m.group(1))
            if iso and iso not in found:
                found.append(iso)

        # Mask false-positive US state / phrase occurrences before matching countries
        masked = self._false_pos_re.sub(lambda m: " " * len(m.group(0)), folded)
        masked = self._ga_context_re.sub(lambda m: " " * len(m.group(0)), masked)

        for m in self._prose_regex.finditer(masked):
            term = m.group(1)
            iso = self._prose_terms.get(term)
            if iso and iso not in found:
                found.append(iso)
        return found

    def city_matches(self, screened_city: str | None, record_city: str | None, record_address: str | None) -> bool:
        """Check whether screened city appears in CRM city or address (handling state suffixes & exonyms)."""
        if not screened_city:
            return True
        sc_main = fold_text(screened_city).split(",")[0].strip()
        if not sc_main:
            return True
        candidates = self._city_map.get(sc_main, {sc_main})
        haystack_city = fold_text(record_city)
        haystack_addr = fold_text(record_address)
        for cand in candidates:
            pat = r"(?<!\w)" + re.escape(cand) + r"(?!\w)"
            if re.search(pat, haystack_city) or re.search(pat, haystack_addr):
                return True
        return False


# --- Date & Interval Functions ---

_MONTH_NAMES_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)


def parse_partial_date(raw: str | None, day_first: bool = True) -> PartialDate | None:
    """Parse structured or semi-structured dates into PartialDate without inventing missing parts."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    # ISO YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return PartialDate(y, mo, d, raw=raw)
        return None

    # ISO YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1000 <= y <= 2100:
            return PartialDate(y, mo, None, raw=raw)
        return None

    # Year only YYYY
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        if 1000 <= y <= 2100:
            return PartialDate(y, None, None, raw=raw)
        return None

    # YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return PartialDate(y, mo, d, raw=raw)
        return None

    # Slash or dot numeric DD/MM/YYYY or MM/DD/YYYY or DD.MM.YYYY
    m = re.fullmatch(r"(\d{1,2})[/\.](\d{1,2})[/\.](\d{4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sep_is_dot = "." in s
        if sep_is_dot:
            day_first_eff = True
        else:
            day_first_eff = day_first
        if day_first_eff:
            d, mo = a, b
            if mo > 12 and d <= 12:
                mo, d = d, mo
        else:
            mo, d = a, b
            if mo > 12 and d <= 12:
                mo, d = d, mo
        if _valid_ymd(y, mo, d):
            return PartialDate(y, mo, d, raw=raw)
        return None

    # D Month YYYY or D Month, YYYY
    m = re.fullmatch(rf"(\d{{1,2}})\s+({_MONTH_NAMES_PAT})\.?,?\s+(\d{{4}})", s, re.I)
    if m:
        d = int(m.group(1))
        mo = MONTH_MAP.get(m.group(2).lower())
        y = int(m.group(3))
        if mo and _valid_ymd(y, mo, d):
            return PartialDate(y, mo, d, raw=raw)
        return None

    # Month D, YYYY or Month D YYYY
    m = re.fullmatch(rf"({_MONTH_NAMES_PAT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", s, re.I)
    if m:
        mo = MONTH_MAP.get(m.group(1).lower())
        d = int(m.group(2))
        y = int(m.group(3))
        if mo and _valid_ymd(y, mo, d):
            return PartialDate(y, mo, d, raw=raw)
        return None

    # Month YYYY or Month, YYYY
    m = re.fullmatch(rf"({_MONTH_NAMES_PAT})\.?,?\s+(\d{{4}})", s, re.I)
    if m:
        mo = MONTH_MAP.get(m.group(1).lower())
        y = int(m.group(2))
        if mo and 1000 <= y <= 2100:
            return PartialDate(y, mo, None, raw=raw)
        return None

    return None


def _valid_ymd(y: int, m: int, d: int) -> bool:
    if not (1000 <= y <= 2100 and 1 <= m <= 12):
        return False
    max_d = calendar.monthrange(y, m)[1]
    return 1 <= d <= max_d


def dob_agrees_single(cust: PartialDate, hit: PartialDate) -> bool:
    """Principle 7: birth year matches AND every part both give (month, day) is equal."""
    if cust.year != hit.year:
        return False
    if cust.month is not None and hit.month is not None and cust.month != hit.month:
        return False
    if cust.day is not None and hit.day is not None and cust.day != hit.day:
        return False
    return True


def dob_contradicts_single(cust: PartialDate, hit: PartialDate) -> bool:
    """Principle 6: birth year > 1 year outside hit's, OR same year & conflicting month/day."""
    if abs(cust.year - hit.year) > 1:
        return True
    if cust.year == hit.year:
        if cust.month is not None and hit.month is not None and cust.month != hit.month:
            return True
        if (
            cust.month is not None
            and hit.month is not None
            and cust.month == hit.month
            and cust.day is not None
            and hit.day is not None
            and cust.day != hit.day
        ):
            return True
    return False


def compare_dob_against_hit_dobs(
    cust_dob: PartialDate | None, hit_dobs: list[PartialDate]
) -> str:
    """Return 'AGREE', 'CONTRADICT', or 'INCONCLUSIVE' across a hit's list of possible DOBs."""
    if cust_dob is None or not hit_dobs:
        return "INCONCLUSIVE"
    if any(dob_agrees_single(cust_dob, hd) for hd in hit_dobs):
        return "AGREE"
    if all(dob_contradicts_single(cust_dob, hd) for hd in hit_dobs):
        return "CONTRADICT"
    return "INCONCLUSIVE"


def _add_years_date(d: date, years: int) -> date:
    try:
        return date(d.year + years, d.month, d.day)
    except ValueError:
        # Feb 29 -> Feb 28 in non-leap year
        return date(d.year + years, 2, 28)


def is_younger_than(cust_dob: PartialDate, event_date: PartialDate, min_age: int = 16) -> bool:
    """Return True only when customer is GUARANTEED younger than min_age at event_date."""
    # Maximum possible age occurs if customer was born on cust_dob.earliest
    # and event occurred on event_date.latest.
    cutoff = _add_years_date(cust_dob.earliest, min_age)
    return event_date.latest < cutoff


def max_possible_age_at(cust_dob: PartialDate, event_date: PartialDate) -> int:
    """Maximum integer age the customer could possibly have attained by event_date.latest."""
    b = cust_dob.earliest
    e = event_date.latest
    age = e.year - b.year
    if (e.month, e.day) < (b.month, b.day):
        age -= 1
    return age


def is_incorporated_after(incorp_date: PartialDate, event_date: PartialDate) -> bool:
    """Return True only when company incorporation is GUARANTEED after event_date."""
    return incorp_date.earliest > event_date.latest


def is_strictly_before_date(event_date: PartialDate, cutoff_date: date) -> bool:
    """Return True only when event_date is GUARANTEED strictly before cutoff_date."""
    return event_date.latest < cutoff_date
