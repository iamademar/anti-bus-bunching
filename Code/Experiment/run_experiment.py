#!/usr/bin/env python
"""Stage 2: processed features CSV -> predictions / metrics / nudges / drift (CapyMOA).

Runs the full model set (HAT, ARF, Hoeffding Tree, EFDT, KNN, Naive Bayes) prequentially
(test-then-train, chronological), computes imbalance-aware metrics + nudge lead time for
each, and saves the artifacts the demo app consumes.

Usage:
    python run_experiment.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Experiment/run_experiment.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json

import pandas as pd

from src.config import load_config
from src.experiment import run_prequential, build_metrics_table, save_outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg.processed_path, parse_dates=["stop_time"])
    print(f"[run] loaded {len(df):,} instances from {cfg.processed_path}")

    res = run_prequential(df, cfg)
    pred_frame = res["predictions"]
    metrics = build_metrics_table(pred_frame, cfg)
    save_outputs(pred_frame, res, metrics, cfg)

    print("\n[run] metrics (bunching class):")
    for name, m in metrics.items():
        lead = m.get("nudge_lead", {}).get("median_lead_min")
        print(f"  {name:18s}  recall={m['recall_bunching']:.3f}  "
              f"F1={m['f1_bunching']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
              f"lead={lead}")


if __name__ == "__main__":
    main()
