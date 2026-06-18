# Stage 3: Simulation

Estimates whether acting on the model's predictions would have reduced bunching, with a
**counterfactual simulation**: replay the recorded arrival stream and, whenever the model fires a
nudge on a bus that is genuinely closing on the one ahead, add a small bounded ease-off, then
recompute the downstream headways and compare against the untouched baseline. This is a simulation
under stated assumptions, not a measured field outcome. Includes a true-positive-only ablation and an
ease-off sensitivity sweep.

## Run (from `Code/`)

```bash
python Simulation/run_simulation.py  [--config config.yaml]   # counterfactual + sensitivity
python Simulation/make_beforeafter.py [--config config.yaml]  # before/after panel + figures
python Simulation/make_figures.py     [--config config.yaml]  # string diagram, rolling acc, lead time
```

## Requires

- `outputs/predictions.csv` from Stage 2 (committed, or rebuild with `run_experiment.py`).
- For `make_beforeafter.py`, also `data/processed/features.csv`.
- The raw SUNT OD data is **optional**: `make_beforeafter.py` uses it only for the peak-vs-off-peak
  travel-time context table. If it is absent the script skips just that table (printing a clear note),
  keeps the committed `outputs/beforeafter_travel_time.csv`, and still writes everything else. All the
  before/after numbers and figures reproduce without it.

## Reads / writes

- **Reads:** `outputs/predictions.csv` (and `data/processed/features.csv` for the before/after panel;
  the raw OD parquet only for the optional travel-time table).
- **Writes:** `outputs/simulation_results.json`, `outputs/beforeafter_panel.csv`,
  `outputs/beforeafter_summary.json`, figures under `figures/`, and
  `outputs/beforeafter_travel_time.csv` (only when the raw SUNT data is present).

## Code

- Entry: `run_simulation.py`, `make_beforeafter.py`, `make_figures.py`
- Logic: `src/simulate.py` (ease-off replay, counterfactual metrics, sensitivity), reusing
  `src/headway.py` and `src/plots.py`.
