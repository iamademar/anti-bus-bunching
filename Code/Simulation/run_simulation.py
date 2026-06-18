#!/usr/bin/env python
"""Deliverable 2: counterfactual simulation — would the driver nudge reduce bunching?

Reads `outputs/predictions.csv` (the recorded arrival stream + the model's fired nudges), replays
it under a small bounded ease-off, recomputes headways, and compares bunching / CV / excess-wait
against the untouched baseline. Includes a true-positive-only ablation and an ease-off sensitivity
sweep. This is a SIMULATION under stated assumptions, not a measured field outcome.

Usage:
    python run_simulation.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Simulation/run_simulation.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json

import pandas as pd

from src.config import load_config
from src import simulate as sim
from src.plots import fig_counterfactual_bars, fig_sensitivity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.outputs_dir
    figs = cfg.figures_dir
    figs.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(out / "predictions.csv", parse_dates=["stop_time"])
    pred["route_short_name"] = pred["route_short_name"].astype(str)
    pred["direction_id"] = pred["direction_id"].astype(str)
    print(f"[sim] loaded {len(pred):,} predicted arrivals from {out/'predictions.csv'}")

    results = sim.run_all(pred, cfg)
    with open(out / "simulation_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    sweep_df = pd.DataFrame(results["sensitivity"])
    p1 = fig_counterfactual_bars(results, figs); print(f"  [fig] {p1}")
    p2 = fig_sensitivity(sweep_df, figs); print(f"  [fig] {p2}")

    base = results["baseline"]
    allf = results["all_fired"]
    tp = results["true_positive_only"]
    print("\n[sim] counterfactual results (simulation; transcribe into the report):")
    print(f"  baseline       : bunching={base['bunch_count']:,}  CV={base['mean_cv']:.3f}  "
          f"excess_wait={base['total_excess_wait_min']:,.0f} min")
    print(f"  nudge (all)    : bunching={allf['bunch_count']:,}  CV={allf['mean_cv']:.3f}  "
          f"excess_wait={allf['total_excess_wait_min']:,.0f} min  "
          f"| -{allf['d_bunch_count_pct']:.0f}% bunching, -{allf['d_excess_wait_pct']:.0f}% wait  "
          f"({allf['n_easeoffs']:,} ease-offs)")
    print(f"  nudge (TP only): bunching={tp['bunch_count']:,}  CV={tp['mean_cv']:.3f}  "
          f"excess_wait={tp['total_excess_wait_min']:,.0f} min  "
          f"| -{tp['d_bunch_count_pct']:.0f}% bunching, -{tp['d_excess_wait_pct']:.0f}% wait")
    print("\n  sensitivity sweep (ease-off s -> %bunching reduction / %wait reduction):")
    for r in results["sensitivity"]:
        print(f"    {r['ease_off_seconds']:>4.0f}s : "
              f"-{r['d_bunch_count_pct']:.0f}% / -{r['d_excess_wait_pct']:.0f}%")


if __name__ == "__main__":
    main()
