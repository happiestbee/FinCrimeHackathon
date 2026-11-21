# Screening Entity Resolution System (`system/`)

A deterministic, Policy-driven entity resolution pipeline for Track 2 (**Sanctions Screening — Same Person or Not?**), built purely on Python 3 standard libraries with optional `temperature=0` OpenRouter / xAI Grok LLM extraction for unstructured free text.

## 1. Architecture Overview

```
<data_folder>/*.jsonl + name_variants.json
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 1. DataRepository & Indexer (resolver.py)              │
│    • O(1) indexes: account_id, login_handle, email     │
│    • Search inverted buckets: (country, dob.year)      │
│    • Union-Find person clustering:                     │
│      (phone + full DOB) ∪ (approved document_number)   │
└───────────────┬────────────────────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────────────────────────────┐
│ 2. Customer  │  │ 3. Watchlist Hit & Search Enrichment │
│ Resolution   │  │    (engine.py + text_extractor.py)   │
│ • Direct link│  │ • Structured dobs / deceased_date    │
│ • Joint acct │  │ • "According to ...; Date:" reports  │
│ • NO_LINK    │  │ • P1 age/DOB + search_results filter │
└──────┬───────┘  └──────────────┬───────────────────────┘
       │                         │
       │    ┌────────────────────┘
       ▼    ▼
┌────────────────────────────────────────────────────────┐
│ 4. Free-Text Fact Extractor (text_extractor.py)        │
│    • High-precision regex for standard KYC/age/dates   │
│    • Concurrent single-flight LLM fallback             │
│      (llm_client.py: OpenRouter / xAI, temp=0, seed=42)│
│    • Strict literal grounding against source text      │
└───────────────┬────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│ 5. Policy Rule Engine & Evidence Builder (engine.py)   │
│    • Normalizer & NameMatcher (normalizer.py)          │
│    • Hard contradictions -> DIFFERENT_ENTITY           │
│    • Strict agreement    -> SAME_ENTITY                │
│    • Multi-person voting -> AMBIGUOUS handling         │
│    • Minimal deciding evidence IDs + deterministic     │
│      2-4 sentence reason -> output JSONL               │
└────────────────────────────────────────────────────────┘
```

## 2. How We Find the Customer & the Listed Person

- **Customer side (`resolver.py`)**:
  1. Follow `account_ref` → `accounts.jsonl`, `login_handle` → `accounts.jsonl`, or `email` → `customers.jsonl`. If a link maps to multiple people (e.g., joint account holders), disambiguate using the screened attributes (`entity_type`, name parts, partial/full DOB, country, city).
  2. For unlinked (`NO_LINK`) alerts, narrow `customers.jsonl` via `(country, dob_year)` inverted indexes and evaluate Principle 2 matching across `full_name`, `preferred_name`, and `login_name`.
  3. Merge duplicate customer records into one person via Union-Find when they share `(normalized_phone, full_dob)` or an `approved` KYC check's `document_number` (via `onboarding.jsonl` + `kyc_checks.jsonl`). Sharing only a phone or address does not merge records.
  4. Rank all `approved` identity checks from both `kyc_checks.jsonl` and `notes.jsonl` (`channel="kyc"` or KYC text) by `checked_at` to establish the authoritative `verified_name`, `verified_dob`, and `document_country` (*Verified beats declared*). Rejected or expired checks are discarded. Support/email/phone notes are parsed for lived-in countries.

- **Watchlist hit side (`engine.py` + `text_extractor.py`)**:
  1. Extract `name`, `aliases`, `entity_type`, `deceased_date`, and all countries tied to `countries`, `nationality`, `citizenship`, `addresses` (last comma-separated segment excluding `International`/`Europe`), and `place_of_birth`.
  2. Parse dated reports strictly matching `According to ...; Month DD, YYYY:` to identify the earliest report date.
  3. Gather hit DOBs in Policy Principle 5 priority order: `dobs` field → first paragraph of `profile_notes` (including `aged X in Month YYYY` producing `{Y-X-1, Y-X}`) → `search_results.jsonl` queried by sorted lowercase name words, keeping only snippets containing both the hit's category listing phrase and at least one hit country.

## 3. How We Represent Evidence & Keep Outputs Minimal

- Every customer country and DOB carries provenance (which `CUS-`, `KYC-`, `NOTE-`, or `SR-` record supplied it).
- Instead of dumping every visited record, `engine.py` emits only the minimal deciding IDs for the chosen `reason_code`:
  - `NAME_MISMATCH_VERIFIED` cites the authoritative `KYC-` or `NOTE-` verification record.
  - `DOB_MISMATCH` cites the decisive customer DOB source plus any `SR-` snippet that established the hit's birth years.
  - `LOCATION_MISMATCH` cites the customer record IDs plus any `KYC-` / `NOTE-` records contributing customer countries.
  - `INSUFFICIENT_DATA` cites the overlapping-country sources (or customer + verification ID if no overlap exists).
  - `DECEASED` cites `<hit_id>#deceased_date`, `alert_id`, and `customer_record_id`.
- Reasons are generated from deterministic templates populated with each case's specific names, dates, countries, and IDs.

## 4. Reproducibility & LLM Usage (Models, Prompts, Grounding)

- **Zero-dependency HTTP client (`llm_client.py`)**: Uses Python standard library `urllib.request` only. Reads `OPENROUTER_API_KEY` (official endpoint `https://openrouter.ai/api/v1/chat/completions`, default model `x-ai/grok-4.20`) or `XAI_API_KEY` (`https://api.x.ai/api/v1/chat/completions`, default model `grok-4.20-0309-non-reasoning`), configurable via `LLM_MODEL` or disabled with `LLM_DISABLED=1`.
- **Where the LLM runs**: Only on unresolved free-text passages belonging to matched customer clusters or candidate hits:
  1. `note_residence`: Distinguishing whether a support/email/phone note mentioning a country actually describes where the customer lived, resided, or worked (vs. a transfer/card payment).
  2. `note_kyc`: Fallback extraction for novel KYC verification phrasing when regex misses name, DOB, or country.
  3. `hit_p1_dob`: Fallback extraction of birth date or age-at-event in paragraph 1 of `profile_notes` (prompt explicitly forbids treating sanction, designation, arrest, election, or imposition dates as birth dates).
  4. `search_snippet_dob`: Fallback extraction of `(age, report_year)` from category- and country-validated search snippets.
- **Prompts & Determinism**:
  - All LLM calls enforce `temperature=0`, `seed=42`, JSON-only output, single-flight thread deduplication by `(model, task_tag, text)` SHA-256 hash, and literal grounding checks (every extracted year/country/name must appear in the source text).
  - All name matching (`normalizer.py` Principle 8), date bounds, entity merging, and decision rules execute 100% deterministically in Python.

## 5. Runtime, Cost & Observed Results (`dev/` — 819 Hits)

- **Speed & Cost**: On the 819-hit development set using `grok-4.20-0309-non-reasoning`, a full run takes **15.9s – 17.6s** across 5 runs, making **88 API calls** (~23,577 prompt tokens + ~840 completion tokens per run, well under **$0.04 USD** per run).
- **Practice Labels (`validate.py --score dev/practice_labels.jsonl`)**:
  - **50 / 50 right decisions**, **0 unsafe clearances**
  - **Decision score: 100.0 / 100**
  - **Reason code match: 100%**, **Customer record match: 100%**
- **5-Run Reproducibility (all 819 hits)**: **0 differences** in `decision`, `reason_code`, `customer_record_id`, or `evidence` across 5 consecutive runs.
- **Unit & Rare-Case Tests (`python system/test_policy.py`)**: 12 / 12 passing, covering `DECEASED`, `ENTITY_TYPE_MISMATCH`, `AMBIGUOUS_CUSTOMER_RECORD`, placeholders, `MARK` vs `MARKUS` edit-distance boundary, and rejected KYC notes.

## 6. Three Case Explanations (from `output/dev.jsonl`)

1. **Clear Mismatch (`DIFFERENT_ENTITY` — `H-2C6B2E` / `A-D6F373`)**:
   - *Customer (`CUS-615500`)*: `NOTE-BCDE4B` records an Onfido-approved check for `ROBERT JAMES KEVIN KOWALCZYK`, DOB `1955-11-19`, US driving licence.
   - *Watchlist Hit (`H-2C6B2E`)*: `Richard James` has empty `dobs`, but `profile_notes` paragraph 1 states `DOB: 6/20/1955` (`1955-06-20`).
   - *Result*: Although both share birth year `1955` and US ties, the verified month/day contradicts (`1955-11-19` vs `1955-06-20`), yielding `DIFFERENT_ENTITY` (`DOB_MISMATCH`), citing `["NOTE-BCDE4B"]`.

2. **Confirmed Match (`SAME_ENTITY` — `H-0181BD` / `A-957E63`)**:
   - *Customer (`CUS-775419`, merged with `CUS-138588`)*: Linked via joint account `ACC-FB5DD0` (excluding family joint holder `CUS-717798` who has a different DOB `1960`). `CUS-775419` and `CUS-138588` share phone `+880 1777482865` and DOB `1985-09-12`. `NOTE-CB2CB8` confirms an Onfido-approved Brazil ID (`1985-09-12`).
   - *Watchlist Hit (`H-0181BD`)*: `COSTA, MARCOS ROBERTO` (`PEP`, `Brazil`) lacks `dobs`, so `search_results` is queried (`"costa marcos roberto"`). `SR-590704` matches both `politically exposed person` and `Brazil` ("May 2019 ... 33 year old" → `1985` or `1986`).
   - *Result*: Name, DOB (`1985-09-12` matches `1985`), and country (`Brazil`) all agree with no contradictions → `SAME_ENTITY` (`NAME_DOB_LOCATION_MATCH`), citing `["CUS-775419", "CUS-138588", "NOTE-CB2CB8", "SR-590704"]`.

3. **Held for Review (`REVIEW` — `H-FF3096` / `A-1EF738`)**:
   - *Customer (`CUS-459031`)*: Linked via email to `CUS-459031` (Australia, DOB `1965-05-19`), verified in `NOTE-588201`, and `NOTE-10AEAB` states the customer lived and worked in the `United States` from 2013 to 2023.
   - *Watchlist Hit (`H-FF3096`)*: `Abdul Rahman Yasin` (`Sanctions`, `United States`) has no DOB on file or matching search result (the Hebrew notice date `22.11.2011` is a designation date, not a birth date).
   - *Result*: Name and country (`United States`) agree with no contradiction, but DOB cannot be compared (`INSUFFICIENT_DATA`) → `REVIEW`, citing `["NOTE-10AEAB"]`.

## 7. Usage

```bash
# Official OpenRouter mode (judges)
export OPENROUTER_API_KEY="sk-or-..."
./system/run dev output/dev.jsonl

# Validate & score practice labels
python3 validate.py --data dev --output output/dev.jsonl --score dev/practice_labels.jsonl
```
