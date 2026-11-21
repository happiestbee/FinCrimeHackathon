"""CLI entry point and end-to-end screening pipeline orchestrator."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import os
import sys
import tempfile
import time

from .customer_resolver import CustomerResolver
from .ingest import DataStore
from .llm_client import OpenRouterClient
from .normalize import parse_partial_date
from .policy_engine import PolicyEngine
from .search_shield import SearchShield
from .verifier import ExtractionVerifier
from .watchlist_resolver import WatchlistResolver


def run_pipeline(
    data_dir: str,
    output_file: str,
    llm_client=None,
    verifier=None,
) -> dict:
    """Execute full screening resolution over data_dir and atomically write JSONL to output_file."""
    t0 = time.monotonic()
    store = DataStore(data_dir)
    if llm_client is None:
        llm_client = OpenRouterClient()
    if verifier is None:
        verifier = ExtractionVerifier(store.name_matcher, store.geo)
    customer_resolver = CustomerResolver(store, llm_client=llm_client, verifier=verifier)
    search_shield = SearchShield(store, llm_client=llm_client, verifier=verifier)
    watchlist_resolver = WatchlistResolver(
        store, search_shield=search_shield, llm_client=llm_client, verifier=verifier
    )
    engine = PolicyEngine(store.name_matcher, store.geo)

    decisions_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    results_lines: list[str] = []

    for hit in store.hits:
        alert = store.alerts_by_id[hit["alert_id"]]
        resolved_alert = customer_resolver.resolve_alert(alert)
        hit_profile = watchlist_resolver.resolve_hit(hit)
        screened_pd = parse_partial_date(alert.get("screened_at"), day_first=True)
        screened_dt = screened_pd.latest if screened_pd is not None else date.today()

        decision_out = engine.evaluate_hit(resolved_alert, hit_profile, screened_dt)
        decisions_counter[decision_out.decision] += 1
        reason_counter[decision_out.reason_code] += 1
        results_lines.append(json.dumps(decision_out.to_dict(), ensure_ascii=False))

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_out_", suffix=".jsonl", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for line in results_lines:
                fh.write(line + "\n")
        os.replace(tmp_path, output_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    elapsed = time.monotonic() - t0
    return {
        "hits": len(results_lines),
        "elapsed_sec": round(elapsed, 3),
        "decisions": dict(decisions_counter),
        "reason_codes": dict(reason_counter),
        "customer_stats": dict(customer_resolver.stats),
        "watchlist_stats": dict(watchlist_resolver.stats),
        "search_stats": dict(search_shield.stats),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run watchlist entity resolution system on a screening data folder."
    )
    parser.add_argument("data_folder", help="Path to data folder containing JSONL files")
    parser.add_argument("output_file", help="Path to write JSONL predictions")
    args = parser.parse_args(argv)

    summary = run_pipeline(args.data_folder, args.output_file)
    print(
        f"[screening] processed {summary['hits']} hits in {summary['elapsed_sec']}s | "
        f"decisions={summary['decisions']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
