#!/usr/bin/env python
"""Stage 3: saved outputs -> figures/. Same artifacts the report and demo use.

Usage:
    python make_figures.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Simulation/make_figures.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.plots import fig_string_diagram, fig_rolling_accuracy, fig_lead_time_hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.outputs_dir
    figs = cfg.figures_dir
    figs.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(out / "predictions.csv", parse_dates=["stop_time"])
    with open(out / "metrics.json") as fh:
        metrics = json.load(fh)
    # rolling-accuracy summary is recomputed cheaply from predictions for plotting
    # (run_experiment already stored drift points in drift_events.json)
    with open(out / "drift_events.json") as fh:
        drift = json.load(fh)

    # reconstruct a minimal summary structure for the rolling plot
    summary = {"summary": {"models": {}}}
    for name in ("HAT", "ARF"):
        col = f"pred_{name}"
        if col in pred:
            correct = (pred[col] == pred["label"]).astype(int).tolist()
            from src.experiment import _rolling
            summary["summary"]["models"][name] = {
                "rolling_accuracy": _rolling(correct, int(cfg.model["windowed_eval_size"])),
                "drift_points": drift.get(name, []),
            }

    p1 = fig_rolling_accuracy(summary, figs)
    print(f"  [fig] {p1}")
    r, d = cfg.scope["routes"][0], cfg.scope["directions"][0]
    p2 = fig_string_diagram(pred, r, d, figs)
    print(f"  [fig] {p2}")
    p3 = fig_lead_time_hist(metrics, figs, model="ARF")
    if p3:
        print(f"  [fig] {p3}")


if __name__ == "__main__":
    main()
