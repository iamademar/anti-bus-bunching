# Stage 2 — Experiment

Runs the streaming learners prequentially (test-then-train, in chronological order) on
`data/processed/features.csv` and produces the predictions, metrics, fired nudges, and drift log.
Also produces the descriptive "before" evidence (headway irregularity and avoidable wait) with no ML.

## Run (from `Code/`)

```bash
python Experiment/run_experiment.py [--config config.yaml]   # CapyMOA HAT + ARF
python Experiment/make_evidence.py  [--config config.yaml]   # descriptive evidence + figures
```

## Requires

- `data/processed/features.csv` (committed, or rebuild it with Stage 1).
- **Java 11+** on the `PATH` / `JAVA_HOME` — CapyMOA runs on a JVM.

## Reads / writes

- **Reads:** `data/processed/features.csv`.
- **Writes (run_experiment):** `outputs/predictions.csv`, `outputs/metrics.json`,
  `outputs/nudges.json`, `outputs/drift_events.json`.
- **Writes (make_evidence):** `outputs/evidence_*.csv`, `outputs/evidence_summary.json`, and figures
  under `figures/`.

## Code

- Entry: `run_experiment.py`, `make_evidence.py`
- Logic: `src/experiment.py` (prequential run, metrics, nudges, drift), `src/baselines.py`
  (non-ML baselines), `src/evidence.py` (descriptive analysis), `src/plots.py` (figures).
