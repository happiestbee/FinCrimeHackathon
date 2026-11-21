# Watchlist Entity Resolution System ("Same Person or Not?")

A zero-dependency Python 3 neuro-symbolic entity resolution system implementing `POLICY.md`. It reconstructs customer identity across fragmented bank tables and notes, disambiguates watchlist profiles and web search snippets against namesakes, and evaluates every hit through a deterministic, precedence-ordered policy engine.

---

## 1. Architecture

```text
                    [ <data_folder> (JSONL + name_variants.json) ]
                                          │
          ┌───────────────────────────────┴───────────────────────────────┐
          ▼                                                               ▼
[ 1. Alert Customer Resolver ]                              [ 2. Watchlist & Search Resolver ]
• Direct links (account/login/email)                        • 3-step DOB waterfall (Company: dobs only)
• Joint-account screened filter                             • Step 1: structured hits.dobs
• Unlinked inverted CRM index                               • Step 2: first-paragraph profile_notes
• Principle 3 Union-Find (phone+fullDOB                     • Step 3: Anti-Namesake Search Shield
  & approved doc_number) + quarantine                         (name/alias + listing phrase + country)
• Verified KYC + KYC-note precedence                        • Dated reports ("According to …; Date:")
• Past-residence note mining (all emails)                   • Territories distinct; non-countries dropped
          │                                                               │
          └────────────────► [ 3. Typed Claim Ledger ] ◄──────────────────┘
                 Claim(side, attribute, value, trust_tier, source_id)
                 (TIER_1_VERIFIED .. TIER_3_CONTEXTUAL; TIER_0_EXCLUDED dropped)
                                          │
                                          ▼
          [ 4. Deterministic Policy Engine & Minimal Proof Narrator ]
          • Candidate-by-candidate evaluation + AMBIGUOUS aggregation
          • Precedence: ENTITY_TYPE > DECEASED > TEMPORAL > DOB > NAME_VERIFIED > LOCATION
          • Confirmation: NAME_DOB_LOCATION / NAME_DOB / NAME_LOCATION (companies)
          • Minimal leaf evidence pruning + 2–4 sentence deterministic reason
                                          │
                                          ▼
                       [ Atomic JSONL Output (<output_file>) ]

  * Trigger-Gated OpenRouter Fallback (src/llm_client.py + src/verifier.py):
    Invoked ONLY when OPENROUTER_API_KEY is set AND a triggered prose record
    cannot be parsed deterministically. All outputs must pass verbatim-quote,
    subject-binding, and Python numeric/date entailment checks before use.
```

### Package Layout

```text
system/
├── run                       # Executable pure Python runner: system/run <data> <output>
├── README.md                 # System documentation and case walkthroughs
├── src/
│   ├── main.py               # CLI entry point, concurrent alert/hit orchestration, atomic write
│   ├── models.py             # Shared contracts: PartialDate, Claim, PersonProfile, HitProfile
│   ├── normalize.py          # Unicode folding, Principle 8 names, ISO geo/territories, dates
│   ├── ingest.py             # JSONL ingestion, hash maps, inverted indexes, Union-Find clusters
│   ├── customer_resolver.py  # Direct/unlinked lookup, note mining, verified-over-declared merge
│   ├── watchlist_resolver.py # 3-step DOB waterfall, dated reports, territories, company rules
│   ├── search_shield.py      # Two-stage namesake filter (listing phrase + country anchor)
│   ├── llm_client.py         # Provider-pinned OpenRouter stdlib client, single-flight cache, retries
│   ├── verifier.py           # Verbatim-quote, subject-attribution, and math entailment gate
│   ├── policy_spec.py        # Declarative thresholds, reason codes, and precedence order
│   ├── policy_engine.py      # Deterministic evaluator, AMBIGUOUS aggregator, evidence pruner
│   └── proof_narrator.py     # Factual 2–4 sentence template compiler
└── tests/                    # Zero-dependency unittest suite (26 tests)
```

---

## 2. How Both Sides Are Resolved

### Customer Side (`ingest.py`, `customer_resolver.py`)
1. **Person Clustering & Family Quarantine (Principle 3):** At startup, a Union-Find graph over all ~72k `customers.jsonl` records creates hard merges strictly on:
   - Identical `(entity_type, normalized_phone, full_dob)` across CRM DOBs, approved `kyc_checks.jsonl` DOBs, and approved `kyc`-channel note DOBs.
   - Identical `(entity_type, normalized_document_number)` across `kyc_checks.jsonl` rows with `result == "approved"`.
   Shared addresses, shared phones with differing/missing DOBs, shared joint accounts, and rejected/expired checks are never merged, keeping family members strictly quarantined.
2. **Alert Resolution (Principle 2):**
   - **Linked alerts:** Follow `account_ref`, `login_handle`, and `email` (intersecting when multiple links are present). When a link leads to multiple person clusters (e.g., joint accounts), screened details (`entity_type`, name, shared DOB parts, country, city) disambiguate; unresolved multi-person fits preserve all candidates as `AMBIGUOUS`.
   - **Unlinked alerts:** Query inverted indexes (`cust_by_type_token`, `cust_by_type_year`, `cust_by_type_country`) and match Principle 8 names (combining single-word `preferred_name` with `full_name` so common given names never false-match unrelated surnames), shared DOB parts, residence/nationality, and city/exonym (`München` ↔ `Munich`). Yields `null` (`0` fits), a canonical `CUS-*` (`1` fit), or `AMBIGUOUS` (`>1` person clusters).
3. **Verified Beats Declared & Defensive Note Mining (Principle 4):** Collects all approved checks across `kyc_checks.jsonl` and `kyc`-channel entries in `notes.jsonl`, excluding rejected/expired checks (`TIER_0_EXCLUDED`). The most recent approved check supplies `verified_name`, `verified_dob`, and `document_country`. Non-KYC support notes across all cluster emails are mined for past countries of residence before any location clearance can trigger.

### Watchlist & Search Side (`watchlist_resolver.py`, `search_shield.py`)
1. **3-Step DOB Waterfall (Principle 5):**
   - **Companies:** Only `hits.jsonl` `dobs` is treated as an incorporation date; establishment dates inside `profile_notes` or `search_results.jsonl` are ignored.
   - **Individuals:** (1) `hits.dobs` if non-empty; else (2) explicit birth date or `(age, event_year)` in the **first paragraph** of `profile_notes` (producing `[Y - A - 1, Y - A]`); else (3) `search_results.jsonl`.
2. **Two-Stage Anti-Namesake Search Shield:**
   - **Stage A:** Matches search queries against the hit's canonicalized name and aliases (`NameMatcher`).
   - **Stage B:** Accepts a snippet **only** when it contains the exact Principle 5 listing phrase for the hit's primary/first category (e.g., `"a politically exposed person"`, `"a sanctioned individual"`, `"named in adverse media"`) **and** mentions a country overlapping `hit.countries` (extracted via word-boundary country/demonym matching that avoids 2-letter ISO false positives and US state collisions).
3. **Dated Reports & Geography (Principles 5 & 6):** Dated reports are parsed exclusively from lines matching `"According to …; <Date>: …"`. Territories (`PR`, `MP`, `VG`, `JE`, `MO`, `HK`, `TW`, `XK`, `PS`, etc.) are kept distinct from parent sovereigns, `[tag]` suffixes on addresses are stripped before extracting trailing countries, and non-country labels (`International`, `Europe`) are dropped.

---

## 3. Evidence Representation & Minimal Proof Pruning

Facts are normalized into typed `Claim` objects tagged with trust tiers (`TIER_1_VERIFIED`, `TIER_2_DECLARED`, `TIER_3_CONTEXTUAL`, `TIER_0_EXCLUDED`) and provenance `source_id`s. During evaluation, `PolicyEngine` selects the single highest-precedence winning proof and prunes `evidence` to the minimal leaf deciding record IDs (`KYC-*`, `NOTE-*`, `CUS-*`, `SR-*`, and `H-*` when a hit-level property such as `deceased_date` or `entity_type` decides), omitting intermediate join tables (`ACC-*`, `APP-*`) and excluded checks (`TIER_0_EXCLUDED`).

For `AMBIGUOUS` alerts (Principle 3), each fitting person cluster is evaluated independently:
- If all candidates produce the same `decision`, that decision and its primary code are emitted with `customer_record_id = "AMBIGUOUS"`.
- If candidates yield different decisions, `REVIEW` / `AMBIGUOUS_CUSTOMER_RECORD` is emitted with `customer_record_id = "AMBIGUOUS"`.

---

## 4. Determinism, Model Configuration & Runtime

### Reproducibility Engineering
- **Deterministic Symbolic Core:** All indexing, Union-Find clustering, Principle 8 name comparisons (pure Levenshtein with Unicode transliteration and multi-group variants from `name_variants.json`), closed `DateInterval` math, precedence routing, evidence pruning, and 2–4 sentence reason compilation execute deterministically in pure Python.
- **Trigger-Gated, Verified LLM Fallback (`llm_client.py`, `verifier.py`):** OpenRouter is called only when `OPENROUTER_API_KEY` is set and a triggered free-text record fails deterministic parsing.
  - **Model & Pinning:** `google/gemini-2.5-flash` (overridable via `OPENROUTER_MODEL`), pinned with `provider: {"order": ["Google"], "require_parameters": true, "allow_fallbacks": false}`, `temperature: 0`, `top_p: 1`, `seed: 42`, and `response_format: {"type": "json_schema", "strict": true}`.
  - **Prompts & Schemas:** Four narrow extraction-only JSON schemas (`kyc_note_extraction`, `residence_note_extraction`, `hit_dob_extraction`, `search_dob_extraction`) requiring an exact `verbatim_quote` and (for watchlist prose) `about_subject` + `subject_name_in_quote`.
  - **Verification Gate & Concurrency:** `ExtractionVerifier` rejects any claim whose quote is not a substring of the source, whose subject does not match the hit, or whose stated integers do not appear in the quote; birth-year intervals `[Y - A - 1, Y - A]` are computed strictly in Python. When an API key is active, `run_pipeline` resolves alerts and hits concurrently via `ThreadPoolExecutor` (`LLM_WORKERS=12`) with single-flight SHA-256 deduplication (`threading.Event`), token-bucket rate limiting, OS CA certificate fallback, 60s socket timeout, and exponential-backoff retries across HTTP `429`/`5xx`, timeouts, malformed JSON, and HTTP 200 `content: null` responses, while evaluating and emitting predictions in deterministic input order.

### Performance & Cost
- **Dev Set (`819` hits, `72,265` customers, `31,895` notes, `16,882` search results):**
  - Wall-clock runtime: **~6.8–7.5 seconds** per full run (`6.91s`–`7.01s` across 5 consecutive runs; **0** OpenRouter fallback triggers on `dev/`, **\$0.00** API cost on `dev/`).
  - Practice labels (`50` hits): **50 / 50** decisions (**100.0** score, **0** unsafe clearances), **100%** reason codes, **100%** `customer_record_id`.
  - 5-Run Repeatability SHA-256: **100% byte-for-byte identical** across all 5 consecutive runs.
- **Estimated Unseen API Cost (with `OPENROUTER_API_KEY` set):** Even if ~50–100 novel prose records on the unseen set trigger fallback extraction (~250 input + ~80 output tokens per call using `google/gemini-2.5-flash`), estimated OpenRouter cost is **< \$0.02 per run** (well under \$0.10 across 5 unseen runs) and runtime stays well under **1 minute**.

---

## 5. Three Case Walkthroughs (`dev/`)

### Case 1 — Clear Mismatch (`DIFFERENT_ENTITY` / `DOB_MISMATCH`): `H-622EDD`
- **Alert & Customer Resolution:** `A-AE9222` screened `Aleksandar Raskovic` with `login_handle = "aleksandr1262"` (`dob = null`, `country = null`). Traversing `accounts.jsonl` links to `CUS-270502`, whose email leads to KYC support note `NOTE-CA9BBC`: *"Veriff validated the Serbia national ID for ALEKSANDAR RASKOVIC, born 17 April 1979, on 2023-11-26."* Verified DOB is `1979-04-17`.
- **Watchlist & Search Resolution:** Hit `H-622EDD` (`Raković, Aleksandar`, category `PEP`, nationality `Belgrade, Serbia` → `RS`) has empty `dobs` and empty `profile_notes`. Querying `search_results.jsonl` through the Anti-Namesake Search Shield yields `SR-A88751` (*"A regulator's notice from May 2002 listed Aleksandar Raković, 29, a politically exposed person in Serbia."*), which matches both the exact `PEP` listing phrase and `Serbia` (`RS`), deriving birth years `1972` or `1973`.
- **Decision & Evidence:** Customer DOB `1979-04-17` differs by >1 year from `1972`/`1973` → `DIFFERENT_ENTITY` (`DOB_MISMATCH`), `customer_record_id = "CUS-270502"`, `evidence = ["NOTE-CA9BBC", "SR-A88751"]`.

### Case 2 — Confirmed Match (`SAME_ENTITY` / `NAME_DOB_LOCATION_MATCH`): `H-80B4B5`
- **Alert & Customer Resolution:** `A-A79F99` (`login_handle = "kervielj1155"`) links to `CUS-626609` (`KERVIEL Jerome`, France), whose declared CRM DOB is `1969-05-05`. Note mining recovers approved KYC note `NOTE-23FC8E` (*"The Veriff process successfully validated KERVIEL JEROME on 2025-02-02, referencing a France passport and a birth date of January 11, 1977."*). Per Principle 4 (*Verified beats declared*), `1977-01-11` supersedes `1969-05-05`.
- **Watchlist & Search Resolution:** Hit `H-80B4B5` (`Jérôme Kerviel`, `Adverse Media`, `France`) has empty `dobs` and `"According to..."` dated reports starting in 2008. Search Shield verifies `SR-01BFC3` (*"In February 2010, at the age of 33, Jérôme Kerviel was the subject of a news outlet profile after being named in adverse media in France."*) → `1976` or `1977`.
- **Decision & Evidence:** Normalized names agree (`KERVIEL JEROME` ↔ `Jérôme Kerviel`), verified DOB `1977-01-11` agrees with `1976 or 1977`, both share `France`, and no contradiction exists → `SAME_ENTITY` (`NAME_DOB_LOCATION_MATCH`), `customer_record_id = "CUS-626609"`, `evidence = ["CUS-626609", "NOTE-23FC8E", "SR-01BFC3"]`.

### Case 3 — Honest Review (`REVIEW` / `INSUFFICIENT_DATA`): `H-427D39`
- **Alert & Customer Resolution:** `A-9ACC6E` (`account_ref = "ACC-7FB32A"`, `Rajesh KUMAR`) links to `CUS-762804` (`Netherlands` / `Milan`, `dob = 1965-01-03`). Principle 3 Union-Find merges `CUS-762804` with `CUS-510447` (shared phone `+44 752166400` and full DOB `1965-01-03`), adding residence country India (`IN`), and support note `NOTE-7F183D` confirms historical work/residence in India (`2013`–`2014`).
- **Watchlist & Search Resolution:** Hit `H-427D39` (`Rajesh Kumar`, `PEP`, `India`) has empty `dobs`, only `Career:` in `profile_notes`, and no anchored search snippet.
- **Decision & Evidence:** Because both sides share `India` (preventing `LOCATION_MISMATCH`) and the hit lacks a date of birth (preventing confirmation or DOB clearance), nothing settles identity → `REVIEW` (`INSUFFICIENT_DATA`), `customer_record_id = "CUS-762804"`, `evidence = ["CUS-510447", "NOTE-7F183D"]`.
