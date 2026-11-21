# FinCrimeHackathon — Watchlist Entity Resolution

Zero-dependency Python 3 neuro-symbolic entity resolution system built for the **"Same Person or Not?"** watchlist screening track.

> **Active Development Branch:** All implementation code, CLI runner (`system/run`), dev predictions (`output/dev.jsonl`), tests, and the local visualizer live on branch [`steven`](https://github.com/happiestbee/FinCrimeHackathon/tree/steven).

## Getting the Code

```bash
git checkout steven
```

## Branch Overview (`steven`)

- `system/` — Pure Python 3 standard-library entity resolution pipeline (`system/run <data_folder> <output_file>`), architecture README, and unit/end-to-end test suite.
- `output/dev.jsonl` — Predictions generated on the development screening set.
- `visualizer/` — Local zero-dependency web dashboard (`EntityLens`) for inspecting resolution graphs, Search Shield filtering, and policy evidence.
- `.env.example` — Environment template for `OPENROUTER_API_KEY` (and optional local `XAI_API_KEY`).

> **Note:** Raw datasets (`data/`) and API secrets (`.env`) are intentionally excluded via `.gitignore`.
