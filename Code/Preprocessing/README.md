# Stage 1 — Preprocessing

Turns the raw SUNT Origin–Destination (OD) Parquet files into a tidy, chronologically ordered
**labelled instance stream**, `data/processed/features.csv` (one row per bus arrival).

This implements Algorithm 1 of the report ("Constructing the streaming dataset from SUNT OD"):
load and scope the OD events, drop trips with non-monotonic timestamps, compute the forward headway
and a rolling local-median "normal" headway, label each arrival by looking a few stops ahead
(ok / warning / bunching), engineer the causal features, drop unlabelable rows, and sort by time.

## Run (from `Code/`)

```bash
python Preprocessing/prepare_data.py [--config config.yaml]
```

## Requires

- The raw SUNT OD data (see the root `README.md` → **Data**). Point `config.yaml: paths.od_dir` at it.
- If you only want the downstream stages, skip this: `data/processed/features.csv` is already
  committed.

## Reads / writes

- **Reads:** `od-YYYY-MM-DD.parquet` from `paths.od_dir`, scoped by `scope.*` in `config.yaml`.
- **Writes:** `data/processed/features.csv` (~109k rows for the default 8-route, 1–3 March 2024 scope).

## Code

- Entry: `prepare_data.py`
- Logic: `src/headway.py` (OD load, forward headway, local-median normal) and `src/features.py`
  (labels + causal features), driven by `src/config.py`.
