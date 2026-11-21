"""Zero-dependency web visualizer server for the watchlist entity resolution system."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_DIR = REPO_ROOT / "system"
VISUALIZER_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = VISUALIZER_DIR / "index.html"

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from src.customer_resolver import CustomerResolver  # noqa: E402
from src.ingest import DataStore, read_jsonl  # noqa: E402
from src.llm_client import OpenRouterClient  # noqa: E402
from src.normalize import normalize_phone, parse_partial_date  # noqa: E402
from src.policy_engine import PolicyEngine  # noqa: E402
from src.search_shield import SearchShield  # noqa: E402
from src.verifier import ExtractionVerifier  # noqa: E402
from src.watchlist_resolver import WatchlistResolver  # noqa: E402


COST_MATRIX = {
    "DIFFERENT_ENTITY": {"DIFFERENT_ENTITY": 0, "SAME_ENTITY": 3, "REVIEW": 3},
    "SAME_ENTITY": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 0, "REVIEW": 2},
    "REVIEW": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 2, "REVIEW": 0},
}


def load_dotenv_if_present() -> None:
    """Load key=value pairs from .env at repo root without overriding set env vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and not os.environ.get(k) and v:
                os.environ[k] = v
    except Exception:
        pass


def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if (path / "hits.jsonl").exists():
            return path
    return None


def discover_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    workspace_parent = REPO_ROOT.parent

    sample_dir = _first_existing_dir(
        [
            SYSTEM_DIR / "tests" / "sample_data",
            workspace_parent / "sample_data" / "sample_data",
            workspace_parent / "sample_data",
        ]
    )
    if sample_dir is not None:
        datasets.append(
            {
                "id": "sample",
                "name": "Synthetic Sample Suite (20 curated cases)",
                "data_dir": str(sample_dir),
                "labels_path": str(sample_dir / "expected_labels.jsonl")
                if (sample_dir / "expected_labels.jsonl").exists()
                else None,
                "hit_count_hint": 20,
            }
        )

    adv_dir = _first_existing_dir(
        [
            SYSTEM_DIR / "tests" / "adversarial",
            workspace_parent / "screening-track" / "screening" / "adversarial",
        ]
    )
    if adv_dir is not None:
        datasets.append(
            {
                "id": "adversarial",
                "name": "Adversarial Stress Suite (180 hits)",
                "data_dir": str(adv_dir),
                "labels_path": str(adv_dir / "practice_labels.jsonl")
                if (adv_dir / "practice_labels.jsonl").exists()
                else None,
                "hit_count_hint": 180,
            }
        )

    dev_dir = _first_existing_dir(
        [
            REPO_ROOT / "data" / "screening" / "dev",
            workspace_parent / "screening-track" / "screening" / "dev",
        ]
    )
    if dev_dir is not None:
        datasets.append(
            {
                "id": "dev",
                "name": "Development Screening Set (819 hits)",
                "data_dir": str(dev_dir),
                "labels_path": str(dev_dir / "practice_labels.jsonl")
                if (dev_dir / "practice_labels.jsonl").exists()
                else None,
                "hit_count_hint": 819,
            }
        )

    return datasets


def _clip_excerpt(text: str, quote: str = "", max_len: int = 230) -> tuple[str, str]:
    """Return a clean single-paragraph excerpt containing `quote` when present."""
    clean = " ".join((text or "").split())
    if not clean:
        return "", ""
    q = " ".join((quote or "").split())
    if len(clean) <= max_len:
        if q and q.lower() in clean.lower():
            idx = clean.lower().find(q.lower())
            return clean, clean[idx : idx + len(q)]
        return clean, ""

    if q:
        idx = clean.lower().find(q.lower())
        if idx != -1:
            exact_q = clean[idx : idx + len(q)]
            margin = max(24, (max_len - len(exact_q)) // 2)
            start = max(0, idx - margin)
            end = min(len(clean), idx + len(exact_q) + margin)
            if start > 0:
                sp = clean.find(" ", start)
                if sp != -1 and sp < idx:
                    start = sp + 1
            if end < len(clean):
                sp = clean.rfind(" ", idx + len(exact_q), end)
                if sp != -1 and sp > idx + len(exact_q):
                    end = sp
            prefix = "… " if start > 0 else ""
            suffix = " …" if end < len(clean) else ""
            return f"{prefix}{clean[start:end]}{suffix}", exact_q

    cut = clean.rfind(" ", 0, max_len)
    if cut < max_len // 2:
        cut = max_len
    return f"{clean[:cut]} …", ""


class TracingLLMClient(OpenRouterClient):
    """OpenRouter/xAI client subclass that records real prompt/extraction telemetry for UI playback."""

    def __init__(self, *args: Any, on_trace: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trace_lock = threading.Lock()
        self._seen_trace_keys: set[str] = set()
        self.trace_events: list[dict[str, Any]] = []
        self._on_trace = on_trace

    def _record_trace(
        self,
        channel: str,
        agent_role: str,
        raw_text: str,
        matched: bool,
        quote: str,
        badge: str | None,
        record_id: str | None = None,
    ) -> None:
        excerpt, resolved_quote = _clip_excerpt(raw_text, quote if matched else "")
        if not excerpt:
            return
        if matched and not resolved_quote:
            matched = False
            badge = None
        dedup_key = f"{channel}|{excerpt[:120]}"
        with self._trace_lock:
            if dedup_key in self._seen_trace_keys:
                return
            self._seen_trace_keys.add(dedup_key)
            evt = {
                "id": f"llm-{len(self.trace_events) + 1}",
                "channel": channel,
                "agent_role": agent_role,
                "source_mode": "LLM",
                "record_id": record_id or channel,
                "text": excerpt,
                "matched": bool(matched and resolved_quote),
                "highlight_quote": resolved_quote if matched else "",
                "badge": badge if matched else None,
            }
            self.trace_events.append(evt)
        if self._on_trace is not None:
            self._on_trace(evt)

    def extract_kyc_note(self, note_text: str) -> dict[str, Any] | None:
        res = super().extract_kyc_note(note_text)
        matched = False
        quote = ""
        badge = None
        if isinstance(res, dict):
            quote = str(res.get("verbatim_quote") or "")
            status = str(res.get("status") or "unknown").lower()
            parts: list[str] = []
            if status in ("approved", "rejected", "expired"):
                parts.append(status.upper())
            if res.get("verified_dob"):
                parts.append(f"DOB {res['verified_dob']}")
            elif res.get("document_country"):
                parts.append(str(res["document_country"]))
            elif res.get("verified_name"):
                parts.append("NAME VERIFIED")
            if quote and (parts or status != "unknown"):
                matched = True
                badge = " · ".join(parts[:2]) or "KYC EXTRACTED"
        self._record_trace("KYC_NOTE", "Grok · KYC Reader", note_text, matched, quote, badge, "KYC NOTE")
        return res

    def extract_residence_note(self, note_text: str) -> dict[str, Any] | None:
        res = super().extract_residence_note(note_text)
        matched = False
        quote = ""
        badge = None
        if isinstance(res, dict):
            lived = res.get("lived_countries") or []
            if isinstance(lived, list) and lived:
                first = lived[0] if isinstance(lived[0], dict) else {}
                quote = str(first.get("verbatim_quote") or "")
                countries = [str(item.get("country")) for item in lived if isinstance(item, dict) and item.get("country")]
                if quote and countries:
                    matched = True
                    badge = "LIVED · " + ", ".join(countries[:2])
        self._record_trace("RESIDENCE_NOTE", "Grok · Geo Miner", note_text, matched, quote, badge, "NOTE PROSE")
        return res

    def extract_hit_first_para_dob(
        self, paragraph: str, hit_names: list[str]
    ) -> dict[str, Any] | None:
        res = super().extract_hit_first_para_dob(paragraph, hit_names)
        matched = False
        quote = ""
        badge = None
        if isinstance(res, dict) and res.get("about_subject") is True:
            quote = str(res.get("verbatim_quote") or "")
            if quote and res.get("birth_date"):
                matched = True
                badge = f"DOB · {res['birth_date']}"
            elif quote and res.get("age") is not None and res.get("event_year") is not None:
                matched = True
                badge = f"AGE {res['age']} @ {res['event_year']}"
        self._record_trace("WATCHLIST_BIO", "Grok · Bio Reader", paragraph, matched, quote, badge, "HIT BIO")
        return res

    def extract_search_snippet_dob(
        self,
        snippet: str,
        hit_names: list[str],
        listing_phrase: str,
        hit_countries: list[str],
    ) -> dict[str, Any] | None:
        res = super().extract_search_snippet_dob(snippet, hit_names, listing_phrase, hit_countries)
        matched = False
        quote = ""
        badge = None
        if isinstance(res, dict) and res.get("about_subject") is True:
            quote = str(res.get("verbatim_quote") or "")
            if quote and res.get("birth_date"):
                matched = True
                badge = f"WEB DOB · {res['birth_date']}"
            elif quote and res.get("age") is not None and res.get("event_year") is not None:
                matched = True
                badge = f"WEB AGE {res['age']} @ {res['event_year']}"
        self._record_trace("SEARCH_SNIPPET", "Grok · Web Shield", snippet, matched, quote, badge, "SEARCH")
        return res


class DatasetSession:
    """In-memory resolved pipeline session for one screening dataset directory."""

    def __init__(
        self,
        spec: dict[str, Any],
        offline: bool = False,
        progress_cb: Any = None,
    ) -> None:
        self.spec = spec
        self.id = spec["id"]
        self.name = spec["name"]
        t0 = time.monotonic()

        self.store = DataStore(spec["data_dir"])
        if progress_cb:
            progress_cb(0, len(self.store.hits), 0, None)

        def _on_llm_trace(evt: dict[str, Any]) -> None:
            if progress_cb:
                progress_cb(None, len(self.store.hits), None, evt)

        if offline:
            llm_client = TracingLLMClient(api_key="", on_trace=_on_llm_trace)
            llm_client.api_key = ""
            llm_client.enabled = False
        else:
            llm_client = TracingLLMClient(on_trace=_on_llm_trace)
        verifier = ExtractionVerifier(self.store.name_matcher, self.store.geo)
        self.customer_resolver = CustomerResolver(
            self.store, llm_client=llm_client, verifier=verifier
        )
        self.search_shield = SearchShield(
            self.store, llm_client=llm_client, verifier=verifier
        )
        self.watchlist_resolver = WatchlistResolver(
            self.store,
            search_shield=self.search_shield,
            llm_client=llm_client,
            verifier=verifier,
        )
        self.engine = PolicyEngine(self.store.name_matcher, self.store.geo)

        # Auxiliary O(1) indexes for UI inspection
        self.sr_by_id: dict[str, dict[str, Any]] = {
            r["result_id"]: r for r in self.store.search_results
        }
        self.kyc_by_id: dict[str, dict[str, Any]] = {
            k["check_id"]: k for k in self.store.kyc_checks
        }
        self.notes_by_id: dict[str, dict[str, Any]] = {
            n["note_id"]: n for n in self.store.notes
        }
        self.onboarding_by_id: dict[str, dict[str, Any]] = {
            o["kyc_applicant_id"]: o for o in self.store.onboarding
        }
        self.phone_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        for c in self.store.customers:
            p = normalize_phone(c.get("phone"))
            if p:
                self.phone_index[(c["entity_type"], p)].append(c["record_id"])

        # Optional ground-truth / practice labels
        self.labels_by_hit: dict[str, dict[str, Any]] = {}
        labels_path = spec.get("labels_path")
        if labels_path and os.path.exists(labels_path):
            for row in read_jsonl(labels_path):
                self.labels_by_hit[row["hit_id"]] = row

        self.hit_cache: dict[str, dict[str, Any]] = {}
        self.summary_hits: list[dict[str, Any]] = []
        decision_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        resolution_counts: Counter[str] = Counter()

        if llm_client.enabled and len(self.store.hits) > 1:
            unique_alerts = {
                h["alert_id"]: self.store.alerts_by_id[h["alert_id"]] for h in self.store.hits
            }
            with ThreadPoolExecutor(max_workers=12) as executor:
                alert_futures = {
                    aid: executor.submit(self.customer_resolver.resolve_alert, alert)
                    for aid, alert in unique_alerts.items()
                }
                hit_futures = [
                    executor.submit(self.watchlist_resolver.resolve_hit, hit)
                    for hit in self.store.hits
                ]
                resolved_pairs = []
                for idx, (hit, hf) in enumerate(zip(self.store.hits, hit_futures), 1):
                    res_alert = alert_futures[hit["alert_id"]].result()
                    hit_prof = hf.result()
                    resolved_pairs.append((res_alert, hit_prof))
                    if progress_cb and (idx % 20 == 0 or idx == len(self.store.hits)):
                        live_evt = self._scan_event_from_hit(hit, res_alert, hit_prof)
                        progress_cb(
                            idx,
                            len(self.store.hits),
                            llm_client.stats.get("requests_total", 0),
                            live_evt,
                        )
        else:
            resolved_pairs = []
            for idx, hit in enumerate(self.store.hits, 1):
                res_alert = self.customer_resolver.resolve_alert(self.store.alerts_by_id[hit["alert_id"]])
                hit_prof = self.watchlist_resolver.resolve_hit(hit)
                resolved_pairs.append((res_alert, hit_prof))
                if progress_cb and (idx % 20 == 0 or idx == len(self.store.hits)):
                    live_evt = self._scan_event_from_hit(hit, res_alert, hit_prof)
                    progress_cb(
                        idx,
                        len(self.store.hits),
                        llm_client.stats.get("requests_total", 0),
                        live_evt,
                    )

        for hit, (resolved_alert, hit_profile) in zip(self.store.hits, resolved_pairs):
            hid = hit["hit_id"]
            alert = self.store.alerts_by_id[hit["alert_id"]]
            screened_pd = parse_partial_date(alert.get("screened_at"), day_first=True)
            screened_dt = screened_pd.latest if screened_pd is not None else date.today()

            eval_results = [
                self.engine.evaluate_candidate(cand, hit_profile, screened_dt)
                for cand in resolved_alert.candidates
            ]
            output = self.engine.evaluate_hit(resolved_alert, hit_profile, screened_dt)
            out_dict = output.to_dict()

            decision_counts[output.decision] += 1
            reason_counts[output.reason_code] += 1
            resolution_counts[resolved_alert.resolution_status] += 1

            label = self.labels_by_hit.get(hid)
            label_match: str | None = None
            if label is not None:
                dec_ok = output.decision == label.get("decision")
                code_ok = output.reason_code in (label.get("reason_codes") or [])
                cid_expected = label.get("customer_record_ids") or []
                cid_ok = (output.customer_record_id in cid_expected) or (
                    cid_expected == ["AMBIGUOUS"]
                    and output.customer_record_id in (None, "", "AMBIGUOUS")
                )
                label_match = "MATCH" if (dec_ok and code_ok and cid_ok) else (
                    "DECISION_ONLY" if dec_ok else "MISMATCH"
                )

            primary_cand = resolved_alert.candidates[0] if resolved_alert.candidates else None
            cust_name = (
                (primary_cand.effective_name if primary_cand else None)
                or (alert.get("customer") or {}).get("name")
                or "Unknown"
            )

            self.hit_cache[hid] = {
                "alert": alert,
                "hit": hit,
                "resolved_alert": resolved_alert,
                "hit_profile": hit_profile,
                "screened_at": screened_dt.isoformat(),
                "eval_results": eval_results,
                "output": out_dict,
                "label": label,
                "label_match": label_match,
            }

            self.summary_hits.append(
                {
                    "hit_id": hid,
                    "alert_id": hit["alert_id"],
                    "decision": output.decision,
                    "reason_code": output.reason_code,
                    "customer_record_id": output.customer_record_id,
                    "resolution_status": resolved_alert.resolution_status,
                    "entity_type": hit["entity_type"],
                    "customer_name": cust_name,
                    "hit_name": hit_profile.primary_name,
                    "dob_source": hit_profile.dob_source,
                    "reason": output.reason,
                    "evidence_count": len(output.evidence),
                    "has_label": label is not None,
                    "label_match": label_match,
                }
            )

        self.load_time_ms = round((time.monotonic() - t0) * 1000.0, 1)
        self.label_stats = self._compute_label_stats()
        self.llm_info = {
            "enabled": llm_client.enabled,
            "backend": llm_client.backend if llm_client.enabled else "offline",
            "model": llm_client.model if llm_client.enabled else None,
            "stats": dict(llm_client.stats),
        }
        self.swarm_traces = self._curate_swarm_traces(llm_client)
        self.summary_payload = {
            "dataset": {
                "id": self.id,
                "name": self.name,
                "data_dir": self.spec["data_dir"],
            },
            "load_time_ms": self.load_time_ms,
            "llm_info": self.llm_info,
            "totals": {
                "hits": len(self.summary_hits),
                "alerts": len(self.store.alerts),
                "customers": len(self.store.customers),
                "search_results": len(self.store.search_results),
            },
            "decision_counts": dict(decision_counts),
            "reason_counts": dict(reason_counts.most_common()),
            "resolution_counts": dict(resolution_counts),
            "label_stats": self.label_stats,
            "swarm_traces": self.swarm_traces,
            "hits": self.summary_hits,
        }

    def _find_dob_quote_in_prose(self, text: str) -> str:
        import re as _re

        for pat in (
            r"\bborn\s+(?:on\s+|in\s+)?[^\.;,\n]{4,32}",
            r"(?i)\b(?:DOB|date\s+of\s+birth)\s*[:=]?\s*[^\n;]{4,28}",
            r"(?i)\baged?\s+\d{2}\b[^.\n]{0,45}\b(?:19|20)\d{2}\b",
            r"(?i)\b(?:at\s+(?:the\s+age\s+of\s+)?)?\d{2}\s*-?\s*years?\s*-?\s*old\b[^.\n]{0,45}\b(?:19|20)\d{2}\b",
            r"(?-i:[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,3}\s*(?:,\s*\d{2}\s*,|\(\d{2}\)))",
            r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b",
        ):
            m = _re.search(pat, text)
            if m:
                return m.group(0).strip()
        return ""

    def _scan_event_from_hit(
        self,
        hit: dict[str, Any],
        resolved_alert: Any,
        hit_profile: Any,
    ) -> dict[str, Any] | None:
        hid = hit["hit_id"]
        pn = (hit.get("profile_notes") or "").strip()
        if hit_profile.dob_source == "profile_notes" and pn:
            first_para = pn.split("\n\n")[0].strip()
            quote = self._find_dob_quote_in_prose(first_para)
            excerpt, resolved_q = _clip_excerpt(first_para, quote)
            if excerpt:
                return {
                    "id": f"scan-{hid}",
                    "channel": "WATCHLIST_BIO",
                    "agent_role": "Rule · Bio Scanner",
                    "source_mode": "SCAN",
                    "record_id": hid,
                    "text": excerpt,
                    "matched": bool(resolved_q),
                    "highlight_quote": resolved_q,
                    "badge": f"DOB · {hit_profile.dobs[0].iso_str}" if (resolved_q and hit_profile.dobs) else None,
                }
        if hit_profile.dob_source == "search_results" and hit_profile.dob_evidence_ids:
            sr_id = hit_profile.dob_evidence_ids[0]
            sr_row = self.sr_by_id.get(sr_id)
            if sr_row and sr_row.get("snippet"):
                quote = self._find_dob_quote_in_prose(sr_row["snippet"])
                excerpt, resolved_q = _clip_excerpt(sr_row["snippet"], quote)
                if excerpt:
                    return {
                        "id": f"scan-{sr_id}",
                        "channel": "SEARCH_SNIPPET",
                        "agent_role": "Shield · Web Scanner",
                        "source_mode": "SCAN",
                        "record_id": sr_id,
                        "text": excerpt,
                        "matched": bool(resolved_q),
                        "highlight_quote": resolved_q,
                        "badge": f"WEB DOB · {hit_profile.dobs[0].iso_str}" if (resolved_q and hit_profile.dobs) else None,
                    }
        for cand in resolved_alert.candidates:
            for cl in cand.claims:
                if cl.attribute == "RESIDENCE_HISTORICAL" and cl.raw_value:
                    cname = self.store.geo.display_name(cl.normalized_value)
                    excerpt, resolved_q = _clip_excerpt(cl.raw_value, cname)
                    if excerpt:
                        return {
                            "id": f"scan-{cl.source_id}",
                            "channel": "RESIDENCE_NOTE",
                            "agent_role": "Rule · Geo Scanner",
                            "source_mode": "SCAN",
                            "record_id": cl.source_id,
                            "text": excerpt,
                            "matched": bool(resolved_q),
                            "highlight_quote": resolved_q,
                            "badge": f"RESIDENCE · {cl.normalized_value}" if resolved_q else None,
                        }
        if pn:
            first_para = pn.split("\n\n")[0].strip()
            excerpt, _ = _clip_excerpt(first_para, "")
            if excerpt:
                return {
                    "id": f"scan-empty-{hid}",
                    "channel": "WATCHLIST_BIO",
                    "agent_role": "Rule · Bio Scanner",
                    "source_mode": "SCAN",
                    "record_id": hid,
                    "text": excerpt,
                    "matched": False,
                    "highlight_quote": "",
                    "badge": None,
                }
        return None

    def _curate_swarm_traces(self, llm_client: TracingLLMClient) -> list[dict[str, Any]]:
        """Build up to 6 genuine reading traces mixing highlights and no-signal scans."""
        llm_matched = [e for e in llm_client.trace_events if e["matched"]]
        llm_empty = [e for e in llm_client.trace_events if not e["matched"]]

        scan_matched: list[dict[str, Any]] = []
        scan_empty: list[dict[str, Any]] = []
        seen_texts: set[str] = {e["text"] for e in llm_client.trace_events}

        def _push(target: list[dict[str, Any]], evt: dict[str, Any] | None) -> None:
            if evt is None or not evt.get("text") or evt["text"] in seen_texts:
                return
            seen_texts.add(evt["text"])
            target.append(evt)

        for hid, cached in self.hit_cache.items():
            if len(scan_matched) >= 6:
                break
            evt = self._scan_event_from_hit(cached["hit"], cached["resolved_alert"], cached["hit_profile"])
            if evt and evt["matched"]:
                _push(scan_matched, evt)

        for note in self.store.notes:
            if len(scan_matched) >= 6 and len(scan_empty) >= 4:
                break
            txt = (note.get("text") or "").strip()
            if not txt:
                continue
            nid = note["note_id"]
            if note.get("channel") == "kyc" and len(scan_matched) < 6:
                kr = self.customer_resolver.parse_kyc_note(note)
                quote = ""
                if kr.verified_dob:
                    quote = self._find_dob_quote_in_prose(txt) or (kr.verified_dob.raw or kr.verified_dob.iso_str)
                if not quote:
                    import re as _re
                    m = _re.search(r"(?i)\b(?:approved|verified|rejected|expired|declined|failed)\b", txt)
                    if m:
                        quote = m.group(0)
                excerpt, resolved_q = _clip_excerpt(txt, quote)
                if resolved_q:
                    _push(
                        scan_matched,
                        {
                            "id": f"scan-{nid}",
                            "channel": "KYC_NOTE",
                            "agent_role": "Rule · KYC Scanner",
                            "source_mode": "SCAN",
                            "record_id": nid,
                            "text": excerpt,
                            "matched": True,
                            "highlight_quote": resolved_q,
                            "badge": f"KYC · {kr.result.upper()}",
                        },
                    )
            elif note.get("channel") != "kyc" and len(scan_empty) < 4:
                countries = self.customer_resolver.parse_residence_note(note)
                if not countries:
                    excerpt, _ = _clip_excerpt(txt, "")
                    _push(
                        scan_empty,
                        {
                            "id": f"scan-{nid}",
                            "channel": "RESIDENCE_NOTE",
                            "agent_role": "Rule · Geo Scanner",
                            "source_mode": "SCAN",
                            "record_id": nid,
                            "text": excerpt,
                            "matched": False,
                            "highlight_quote": "",
                            "badge": None,
                        },
                    )

        used_sr_ids: set[str] = {
            eid
            for cached in self.hit_cache.values()
            for eid in (cached["hit_profile"].dob_evidence_ids or [])
            if str(eid).startswith("SR-")
        }
        for sr in self.store.search_results:
            if len(scan_empty) >= 4:
                break
            if sr["result_id"] in used_sr_ids:
                continue
            snip = (sr.get("snippet") or "").strip()
            if not snip:
                continue
            excerpt, _ = _clip_excerpt(snip, "")
            _push(
                scan_empty,
                {
                    "id": f"scan-{sr['result_id']}",
                    "channel": "SEARCH_SNIPPET",
                    "agent_role": "Shield · Web Scanner",
                    "source_mode": "SCAN",
                    "record_id": sr["result_id"],
                    "text": excerpt,
                    "matched": False,
                    "highlight_quote": "",
                    "badge": None,
                },
            )

        matched_pool = llm_matched + scan_matched
        empty_pool = llm_empty + scan_empty

        # Interleave up to 4 matched + 2 empty (or fill up to 6 from whichever is available)
        curated: list[dict[str, Any]] = []
        take_matched = matched_pool[:4]
        take_empty = empty_pool[:2]
        while len(take_matched) + len(take_empty) < 6 and len(matched_pool) > len(take_matched):
            take_matched.append(matched_pool[len(take_matched)])
        while len(take_matched) + len(take_empty) < 6 and len(empty_pool) > len(take_empty):
            take_empty.append(empty_pool[len(take_empty)])

        m_idx = 0
        e_idx = 0
        pattern = ("M", "E", "M", "M", "E", "M")
        for kind in pattern:
            if kind == "M" and m_idx < len(take_matched):
                curated.append(take_matched[m_idx])
                m_idx += 1
            elif kind == "E" and e_idx < len(take_empty):
                curated.append(take_empty[e_idx])
                e_idx += 1
        curated.extend(take_matched[m_idx:])
        curated.extend(take_empty[e_idx:])
        return curated[:6]

    def _compute_label_stats(self) -> dict[str, Any] | None:
        if not self.labels_by_hit:
            return None
        labels = list(self.labels_by_hit.values())
        cost = 0
        baseline = 0
        right = []
        unsafe = 0
        for row in labels:
            hid = row["hit_id"]
            got_dec = self.hit_cache[hid]["output"]["decision"] if hid in self.hit_cache else "REVIEW"
            exp_dec = row["decision"]
            cost += COST_MATRIX[exp_dec][got_dec]
            baseline += COST_MATRIX[exp_dec]["REVIEW"]
            if got_dec == exp_dec:
                right.append(row)
            elif got_dec == "DIFFERENT_ENTITY":
                unsafe += 1

        codes_ok = sum(
            1
            for row in right
            if self.hit_cache[row["hit_id"]]["output"]["reason_code"] in row["reason_codes"]
        )
        cids_ok = sum(
            1
            for row in right
            if (
                self.hit_cache[row["hit_id"]]["output"]["customer_record_id"]
                in row["customer_record_ids"]
            )
            or (
                row["customer_record_ids"] == ["AMBIGUOUS"]
                and self.hit_cache[row["hit_id"]]["output"]["customer_record_id"]
                in (None, "", "AMBIGUOUS")
            )
        )
        decision_score = round(100.0 * max(0.0, 1.0 - cost / baseline), 1) if baseline else 100.0
        return {
            "total": len(labels),
            "right_decisions": len(right),
            "unsafe_clearances": unsafe,
            "decision_score": decision_score,
            "reason_code_pct": round(100.0 * codes_ok / len(right), 1) if right else 0.0,
            "customer_id_pct": round(100.0 * cids_ok / len(right), 1) if right else 0.0,
        }

    def get_hit_detail(self, hit_id: str) -> dict[str, Any] | None:
        cached = self.hit_cache.get(hit_id)
        if cached is None:
            return None

        alert = cached["alert"]
        hit = cached["hit"]
        resolved_alert = cached["resolved_alert"]
        hit_profile = cached["hit_profile"]
        eval_results = cached["eval_results"]
        out_dict = cached["output"]
        geo = self.store.geo

        records_by_id: dict[str, dict[str, Any]] = {
            alert["alert_id"]: {"kind": "Alert", "id": alert["alert_id"], "data": alert},
            hit["hit_id"]: {"kind": "Watchlist Hit", "id": hit["hit_id"], "data": hit},
        }

        # Direct account links from alert
        direct_accounts: list[dict[str, Any]] = []
        seen_acc: set[str] = set()
        if alert.get("account_ref"):
            for acc in self.store.accounts_by_id.get(alert["account_ref"], ()):
                if acc["account_id"] not in seen_acc:
                    seen_acc.add(acc["account_id"])
                    direct_accounts.append(acc)
                    records_by_id[acc["account_id"]] = {
                        "kind": "Account",
                        "id": acc["account_id"],
                        "data": acc,
                    }
        if alert.get("login_handle"):
            for acc in self.store.accounts_by_login.get(
                alert["login_handle"].strip().lower(), ()
            ):
                if acc["account_id"] not in seen_acc:
                    seen_acc.add(acc["account_id"])
                    direct_accounts.append(acc)
                    records_by_id[acc["account_id"]] = {
                        "kind": "Account",
                        "id": acc["account_id"],
                        "data": acc,
                    }

        eval_by_cluster = {er.person.cluster_id: er for er in eval_results}

        candidates_payload: list[dict[str, Any]] = []
        for idx, cand in enumerate(resolved_alert.candidates):
            er = eval_by_cluster.get(cand.cluster_id)
            crm_rows: list[dict[str, Any]] = []
            cand_accounts: list[dict[str, Any]] = []
            cand_onboarding: list[dict[str, Any]] = []
            cand_kyc: list[dict[str, Any]] = []
            cand_notes: list[dict[str, Any]] = []
            quarantined_records: list[dict[str, Any]] = []

            seen_q: set[str] = set()
            seen_cand_acc: set[str] = set()
            seen_ob: set[str] = set()
            seen_kyc: set[str] = set()
            seen_notes: set[str] = set()
            excluded_set = set(cand.excluded_kyc_ids)

            for cid in cand.record_ids:
                c_row = self.store.customers_by_id.get(cid)
                if c_row:
                    crm_rows.append(c_row)
                    records_by_id[cid] = {"kind": "Customer CRM", "id": cid, "data": c_row}
                    norm_p = normalize_phone(c_row.get("phone"))
                    if norm_p:
                        for qcid in self.phone_index.get((c_row["entity_type"], norm_p), ()):
                            if qcid not in cand.record_ids and qcid not in seen_q:
                                seen_q.add(qcid)
                                q_row = self.store.customers_by_id.get(qcid)
                                if q_row:
                                    quarantined_records.append(q_row)
                                    records_by_id[qcid] = {
                                        "kind": "Quarantined CRM (Family/Shared Phone)",
                                        "id": qcid,
                                        "data": q_row,
                                    }

                for acc in self.store.accounts_by_customer.get(cid, ()):
                    aid = acc["account_id"]
                    if aid not in seen_cand_acc:
                        seen_cand_acc.add(aid)
                        cand_accounts.append(acc)
                        records_by_id[aid] = {"kind": "Account", "id": aid, "data": acc}

                for ob in self.store.onboarding_by_customer.get(cid, ()):
                    oid = ob["kyc_applicant_id"]
                    if oid not in seen_ob:
                        seen_ob.add(oid)
                        cand_onboarding.append(ob)
                        records_by_id[oid] = {"kind": "Onboarding", "id": oid, "data": ob}

                for kyc in self.store.kyc_checks_by_customer.get(cid, ()):
                    kid = kyc["check_id"]
                    if kid not in seen_kyc:
                        seen_kyc.add(kid)
                        kyc_copy = dict(kyc)
                        kyc_copy["_tier_status"] = (
                            "TIER_0_EXCLUDED"
                            if kid in excluded_set
                            or (kyc.get("result") or "").lower() != "approved"
                            else "TIER_1_APPROVED"
                        )
                        cand_kyc.append(kyc_copy)
                        records_by_id[kid] = {"kind": "KYC Check", "id": kid, "data": kyc_copy}

            for em in cand.emails:
                for note in self.store.notes_by_email.get(em.lower(), ()):
                    nid = note["note_id"]
                    if nid not in seen_notes:
                        seen_notes.add(nid)
                        n_copy = dict(note)
                        n_copy["_tier_status"] = (
                            "TIER_0_EXCLUDED" if nid in excluded_set else "ACTIVE_NOTE"
                        )
                        cand_notes.append(n_copy)
                        records_by_id[nid] = {
                            "kind": f"Note ({note.get('channel') or 'note'})",
                            "id": nid,
                            "data": n_copy,
                        }

            # Fallback resolve any KYC/Note IDs referenced directly on profile if not yet indexed
            for extra_id in list(cand.excluded_kyc_ids) + (
                [cand.verified_name_source_id] if cand.verified_name_source_id else []
            ) + ([cand.dob_source_id] if cand.dob_source_id else []):
                if extra_id and extra_id not in records_by_id:
                    if extra_id in self.kyc_by_id:
                        records_by_id[extra_id] = {
                            "kind": "KYC Check",
                            "id": extra_id,
                            "data": self.kyc_by_id[extra_id],
                        }
                    elif extra_id in self.notes_by_id:
                        records_by_id[extra_id] = {
                            "kind": "Note",
                            "id": extra_id,
                            "data": self.notes_by_id[extra_id],
                        }

            country_map = {
                iso: {
                    "iso": iso,
                    "name": geo.display_name(iso),
                    "sources": srcs,
                }
                for iso, srcs in sorted(cand.country_sources.items())
            }

            candidates_payload.append(
                {
                    "index": idx,
                    "cluster_id": cand.cluster_id,
                    "record_ids": cand.record_ids,
                    "canonical_record_id": cand.canonical_record_id,
                    "entity_type": cand.entity_type,
                    "verified_name": cand.verified_name,
                    "verified_name_source_id": cand.verified_name_source_id,
                    "declared_names": [
                        {"name": nm, "source_id": src} for nm, src in cand.declared_names
                    ],
                    "effective_name": cand.effective_name,
                    "dob": cand.dob.iso_str if cand.dob else None,
                    "dob_source_id": cand.dob_source_id,
                    "dob_is_verified": cand.dob_is_verified,
                    "countries": list(country_map.values()),
                    "emails": cand.emails,
                    "excluded_kyc_ids": cand.excluded_kyc_ids,
                    "is_unresolved": cand.is_unresolved,
                    "resolution_notes": cand.resolution_notes,
                    "evaluation": {
                        "decision": er.decision,
                        "reason_code": er.reason_code,
                        "all_applicable_codes": er.all_applicable_codes,
                        "evidence_ids": er.evidence_ids,
                        "facts": {
                            "screened_at": er.facts.get("screened_at"),
                            "name_agrees": er.facts.get("name_agrees"),
                            "short_individual_name": er.facts.get("short_individual_name"),
                            "dob_status": er.facts.get("dob_status"),
                            "shared_countries": sorted(er.facts.get("shared_countries") or ()),
                            "countries_disjoint": er.facts.get("countries_disjoint"),
                            "max_age_at_report": er.facts.get("max_age_at_report"),
                        },
                    }
                    if er
                    else None,
                    "raw_trail": {
                        "customers": crm_rows,
                        "quarantined_customers": quarantined_records,
                        "accounts": cand_accounts,
                        "onboarding": cand_onboarding,
                        "kyc_checks": cand_kyc,
                        "notes": cand_notes,
                    },
                }
            )

        # Trace Anti-Namesake Search Shield candidates
        search_trace = self._trace_search_shield(hit_profile)
        for sr_item in search_trace["candidates"]:
            rid = sr_item["result_id"]
            if rid in self.sr_by_id:
                records_by_id[rid] = {
                    "kind": f"Search Result ({sr_item['status']})",
                    "id": rid,
                    "data": self.sr_by_id[rid],
                }

        # Guarantee every ID listed in output["evidence"] exists in records_by_id
        for evid in out_dict.get("evidence") or []:
            if evid not in records_by_id:
                if evid in self.sr_by_id:
                    records_by_id[evid] = {"kind": "Search Result", "id": evid, "data": self.sr_by_id[evid]}
                elif evid in self.kyc_by_id:
                    records_by_id[evid] = {"kind": "KYC Check", "id": evid, "data": self.kyc_by_id[evid]}
                elif evid in self.notes_by_id:
                    records_by_id[evid] = {"kind": "Note", "id": evid, "data": self.notes_by_id[evid]}
                elif evid in self.store.customers_by_id:
                    records_by_id[evid] = {"kind": "Customer CRM", "id": evid, "data": self.store.customers_by_id[evid]}
                elif evid in self.store.accounts_by_id:
                    records_by_id[evid] = {"kind": "Account", "id": evid, "data": self.store.accounts_by_id[evid][0]}

        hit_countries = [
            {"iso": iso, "name": geo.display_name(iso)}
            for iso in sorted(hit_profile.countries)
        ]

        return {
            "output": out_dict,
            "label": cached["label"],
            "label_match": cached["label_match"],
            "screened_at": cached["screened_at"],
            "alert": alert,
            "hit_raw": hit,
            "direct_accounts": direct_accounts,
            "resolved_alert": {
                "alert_id": resolved_alert.alert_id,
                "resolution_status": resolved_alert.resolution_status,
                "customer_record_id": resolved_alert.customer_record_id,
                "candidates": candidates_payload,
            },
            "hit_profile": {
                "hit_id": hit_profile.hit_id,
                "entity_type": hit_profile.entity_type,
                "primary_name": hit_profile.primary_name,
                "all_names": hit_profile.all_names,
                "dobs": [d.iso_str for d in hit_profile.dobs],
                "dob_source": hit_profile.dob_source,
                "dob_evidence_ids": hit_profile.dob_evidence_ids,
                "dob_detail": hit_profile.dob_detail,
                "countries": hit_countries,
                "deceased_date": hit_profile.deceased_date.iso_str
                if hit_profile.deceased_date
                else None,
                "life_status": hit_profile.life_status,
                "earliest_report_date": hit_profile.earliest_report_date.iso_str
                if hit_profile.earliest_report_date
                else None,
                "listing_phrase": hit_profile.listing_phrase,
            },
            "search_shield_trace": search_trace,
            "records_by_id": records_by_id,
        }

    def _trace_search_shield(self, hit_profile: Any) -> dict[str, Any]:
        is_comp = hit_profile.entity_type == "Company"
        if is_comp:
            return {
                "executed": False,
                "skip_reason": "Skipped: Policy Principle 5 restricts Companies to structured hit.dobs only.",
                "listing_phrase": hit_profile.listing_phrase,
                "candidates": [],
            }

        nm = self.store.name_matcher
        geo = self.store.geo
        hit_names = hit_profile.all_names
        hit_countries = set(hit_profile.countries)

        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for name in hit_names:
            parts = nm.extract_name_parts(name, is_company=False)
            if not parts:
                continue
            for row in self.store.sr_by_query_tokens.get(frozenset(parts), ()):
                rid = row["result_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    candidates.append(row)

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
                    if row["result_id"] not in seen_ids:
                        seen_ids.add(row["result_id"])
                        candidates.append(row)

        phrase = hit_profile.listing_phrase
        phrase_lc = phrase.lower()
        used_ids = set(hit_profile.dob_evidence_ids) if hit_profile.dob_source == "search_results" else set()
        executed = hit_profile.dob_source not in ("dobs_field", "profile_notes")

        traced_rows: list[dict[str, Any]] = []
        for row in sorted(candidates, key=lambda r: r["result_id"]):
            snip = row.get("snippet") or ""
            phrase_matched = phrase_lc in snip.lower()
            title_name = (row.get("title") or "").split(" - ")[0].strip()
            mask_names = list(hit_names)
            if row.get("query"):
                mask_names.append(row["query"])
            if title_name:
                mask_names.append(title_name)

            snip_isos = set(geo.extract_countries_from_prose(snip, mask_names=mask_names))
            matched_isos = sorted(snip_isos & hit_countries)
            country_matched = bool(matched_isos)
            anchored = bool(hit_countries) and phrase_matched and country_matched
            used_for_dob = row["result_id"] in used_ids

            block_reasons: list[str] = []
            if not hit_countries:
                block_reasons.append("Hit has no country anchor")
            if not phrase_matched:
                block_reasons.append(f'Missing listing phrase "{phrase}"')
            if hit_countries and not country_matched:
                found_names = ", ".join(geo.display_name(i) for i in sorted(snip_isos)) or "none"
                expected_names = ", ".join(geo.display_name(i) for i in sorted(hit_countries))
                block_reasons.append(
                    f"Country anchor mismatch (snippet: {found_names}; expected: {expected_names})"
                )

            if not executed and hit_profile.dob_source in ("dobs_field", "profile_notes"):
                status = "NOT_NEEDED"
            elif used_for_dob:
                status = "USED_FOR_DOB"
            elif anchored:
                status = "ANCHORED_NO_DOB"
            else:
                status = "BLOCKED"

            traced_rows.append(
                {
                    "result_id": row["result_id"],
                    "query": row.get("query"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "snippet": snip,
                    "phrase_matched": phrase_matched,
                    "country_matched": country_matched,
                    "found_countries": [
                        {"iso": i, "name": geo.display_name(i)} for i in sorted(snip_isos)
                    ],
                    "matched_countries": [
                        {"iso": i, "name": geo.display_name(i)} for i in matched_isos
                    ],
                    "anchored": anchored,
                    "used_for_dob": used_for_dob,
                    "status": status,
                    "block_reason": "; ".join(block_reasons) if block_reasons else None,
                }
            )

        skip_reason = None
        if not executed:
            skip_reason = f"Waterfall stopped earlier at {hit_profile.dob_source}; Stage A candidates shown for inspection."

        return {
            "executed": executed,
            "skip_reason": skip_reason,
            "listing_phrase": phrase,
            "candidates": traced_rows,
        }


class VisualizerAppState:
    """Thread-safe registry managing lazy loading, progress telemetry, and caching."""

    def __init__(self, preload_sample: bool = True, offline: bool = False) -> None:
        self.offline = offline
        self.specs = {d["id"]: d for d in discover_datasets()}
        self.sessions: dict[str, DatasetSession] = {}
        self._lock = threading.Lock()
        self._ds_locks: dict[str, threading.Lock] = {did: threading.Lock() for did in self.specs}
        self._progress_lock = threading.Lock()
        self._progress: dict[str, dict[str, Any]] = {}
        if preload_sample and "sample" in self.specs:
            self.get_session("sample")

    def list_datasets(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for did, spec in self.specs.items():
            out.append(
                {
                    "id": did,
                    "name": spec["name"],
                    "loaded": did in self.sessions,
                    "hit_count_hint": spec.get("hit_count_hint"),
                }
            )
        return out

    def get_progress(self, dataset_id: str) -> dict[str, Any] | None:
        if dataset_id not in self.specs:
            return None
        if dataset_id in self.sessions:
            sess = self.sessions[dataset_id]
            return {
                "status": "ready",
                "processed": len(sess.summary_hits),
                "total": len(sess.summary_hits),
                "llm_calls": sess.llm_info["stats"].get("requests_total", 0),
                "events": list(sess.swarm_traces),
            }
        with self._progress_lock:
            state = self._progress.get(dataset_id)
            if state is None:
                return {
                    "status": "idle",
                    "processed": 0,
                    "total": self.specs[dataset_id].get("hit_count_hint") or 0,
                    "llm_calls": 0,
                    "events": [],
                }
            return {
                "status": state["status"],
                "processed": state["processed"],
                "total": state["total"],
                "llm_calls": state["llm_calls"],
                "events": list(state["events"]),
            }

    def get_session(self, dataset_id: str) -> DatasetSession | None:
        if dataset_id not in self.specs:
            return None
        sess = self.sessions.get(dataset_id)
        if sess is not None:
            return sess
        ds_lock = self._ds_locks[dataset_id]
        with ds_lock:
            sess = self.sessions.get(dataset_id)
            if sess is not None:
                return sess

            with self._progress_lock:
                self._progress[dataset_id] = {
                    "status": "loading",
                    "processed": 0,
                    "total": self.specs[dataset_id].get("hit_count_hint") or 0,
                    "llm_calls": 0,
                    "events": [],
                    "seen": set(),
                }

            def _progress_cb(
                processed: int | None,
                total: int | None,
                llm_calls: int | None,
                evt: dict[str, Any] | None,
            ) -> None:
                with self._progress_lock:
                    st = self._progress.get(dataset_id)
                    if st is None:
                        return
                    if processed is not None:
                        st["processed"] = processed
                    if total is not None:
                        st["total"] = total
                    if llm_calls is not None:
                        st["llm_calls"] = llm_calls
                    if evt and evt.get("text"):
                        key = f"{evt.get('source_mode')}|{evt['text'][:80]}"
                        if key not in st["seen"] and len(st["events"]) < 24:
                            st["seen"].add(key)
                            st["events"].append(evt)

            sess = DatasetSession(
                self.specs[dataset_id],
                offline=self.offline,
                progress_cb=_progress_cb,
            )
            with self._lock:
                self.sessions[dataset_id] = sess
            with self._progress_lock:
                st = self._progress.get(dataset_id)
                if st is not None:
                    st["status"] = "ready"
                    st["processed"] = len(sess.summary_hits)
                    st["total"] = len(sess.summary_hits)
                    st["llm_calls"] = sess.llm_info["stats"].get("requests_total", 0)
            return sess


def make_handler(app_state: VisualizerAppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_bytes: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html_bytes)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                if not INDEX_HTML_PATH.exists():
                    self._send_json({"error": "index.html not found"}, status=500)
                    return
                self._send_html(INDEX_HTML_PATH.read_bytes())
                return

            if path == "/api/datasets":
                self._send_json({"datasets": app_state.list_datasets()})
                return

            parts = [unquote(p) for p in path.strip("/").split("/") if p]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "datasets":
                dataset_id = parts[2]
                if len(parts) == 4 and parts[3] == "progress":
                    prog = app_state.get_progress(dataset_id)
                    if prog is None:
                        self._send_json({"error": f"Unknown dataset: {dataset_id}"}, status=404)
                        return
                    self._send_json(prog)
                    return

                session = app_state.get_session(dataset_id)
                if session is None:
                    self._send_json({"error": f"Unknown dataset: {dataset_id}"}, status=404)
                    return

                if len(parts) == 4 and parts[3] == "summary":
                    self._send_json(session.summary_payload)
                    return

                if len(parts) == 5 and parts[3] == "hit":
                    hit_id = parts[4]
                    detail = session.get_hit_detail(hit_id)
                    if detail is None:
                        self._send_json({"error": f"Unknown hit: {hit_id}"}, status=404)
                        return
                    self._send_json(detail)
                    return

            self._send_json({"error": "Not found"}, status=404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(description="Start the Screening Entity Resolution Visualizer.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Do not preload sample dataset on startup",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable LLM extraction fallback even if an API key is configured",
    )
    args = parser.parse_args(argv)

    app_state = VisualizerAppState(preload_sample=not args.no_preload, offline=args.offline)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app_state))
    print(f"[visualizer] listening on http://{args.host}:{args.port} (datasets={list(app_state.specs)}, offline={args.offline})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
