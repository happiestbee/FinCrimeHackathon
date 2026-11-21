# FinCrimeHackathon — Watchlist Entity Resolution

Zero-dependency Python 3 neuro-symbolic entity resolution system built for the **"Same Person or Not?"** watchlist screening track.

## Repository Layout

- `system/` — Pure Python 3 standard-library entity resolution pipeline (`system/run <data_folder> <output_file>`), architecture walkthrough ([`system/README.md`](system/README.md)), and unit/end-to-end tests (`system/tests/`).
- `output/dev.jsonl` — Predictions generated on the development screening set.
- `visualizer/` — Local zero-dependency web dashboard (`EntityLens`) for inspecting resolution graphs, Search Shield filtering, and policy evidence.
- `.env.example` — Environment template for `OPENROUTER_API_KEY` (and optional local `XAI_API_KEY`).

## Quick Start

```bash
# Run predictions on a screening dataset folder
export OPENROUTER_API_KEY="sk-or-..."
./system/run <data_folder> <output_file>

# Run tests
python3 -m unittest discover -s system/tests -v

# Launch local visualizer
python3 visualizer/server.py --port 8765
```

> **Note:** Raw datasets (`data/`) and API secrets (`.env`) are intentionally excluded via `.gitignore`.
