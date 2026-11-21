"""Export static GitHub Pages bundle into `docs/` so EntityLens runs serverlessly on mobile/web."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
VISUALIZER_DIR = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
STATIC_DATA_DIR = DOCS_DIR / "static_data"

if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))

from server import VisualizerAppState, load_dotenv_if_present  # noqa: E402


def dump_min_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def export_static(chunk_size: int = 100, include_dev: bool = False) -> None:
    load_dotenv_if_present()
    app_state = VisualizerAppState(preload_sample=False, offline=False)

    if STATIC_DATA_DIR.exists():
        shutil.rmtree(STATIC_DATA_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy frontend HTML + disable Jekyll processing on GitHub Pages
    shutil.copy2(VISUALIZER_DIR / "index.html", DOCS_DIR / "index.html")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    datasets_meta = []
    for did, spec in app_state.specs.items():
        if did == "dev" and not include_dev:
            continue
        session = app_state.get_session(did)
        if session is None:
            continue
        datasets_meta.append(
            {
                "id": did,
                "name": spec["name"],
                "loaded": True,
                "hit_count_hint": len(session.summary_hits),
            }
        )

        ds_dir = STATIC_DATA_DIR / did
        summary_copy = json.loads(json.dumps(session.summary_payload))

        hits_list = summary_copy.get("hits", [])
        for start in range(0, len(hits_list), chunk_size):
            chunk_idx = start // chunk_size
            chunk_rel = f"chunks/hits_{chunk_idx}.json"
            chunk_map = {}
            for row in hits_list[start : start + chunk_size]:
                hid = row["hit_id"]
                row["chunk"] = chunk_rel
                detail = session.get_hit_detail(hid)
                if detail is not None:
                    chunk_map[hid] = detail
            dump_min_json(ds_dir / chunk_rel, chunk_map)

        dump_min_json(ds_dir / "summary.json", summary_copy)
        print(
            f"[export] {did}: {len(hits_list)} hits -> {(len(hits_list) + chunk_size - 1) // chunk_size} chunks",
            flush=True,
        )

    dump_min_json(STATIC_DATA_DIR / "datasets.json", {"datasets": datasets_meta})
    print(f"[export] static site written to {DOCS_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export static EntityLens site for GitHub Pages.")
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Include dev dataset locally (blocked from git via .gitignore).",
    )
    args = parser.parse_args()
    export_static(include_dev=args.include_dev)
