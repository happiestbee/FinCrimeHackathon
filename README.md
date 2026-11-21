# FinCrimeHackathon — Watchlist Entity Resolution

Zero-dependency Python 3 neuro-symbolic entity resolution system built for the **"Same Person or Not?"** watchlist screening track.

## Repository Layout

- `system/` — Pure Python 3 standard-library entity resolution pipeline (`system/run <data_folder> <output_file>`), architecture walkthrough ([`system/README.md`](system/README.md)), and unit/end-to-end tests (`system/tests/`).
- `output/dev.jsonl` — Predictions generated on the development screening set.
- `visualizer/` — Local zero-dependency web dashboard (`EntityLens`) for inspecting resolution graphs, Search Shield filtering, and policy evidence.
- `.env.example` — Environment template for `OPENROUTER_API_KEY` (and optional local `XAI_API_KEY`).

Make sure to add the OPENROUTER_API_KEY to your environment variables before running the system in .env file or directly in your shell.
Note that there is sample data being used for the site hosting, but none of the actual data has been published in accordance with the competition rules. The data used for the site is purely synthetic. 
Partial inspiration for the swarm UI from [swarm-site](https://califa.github.io/swarm-site/)

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
