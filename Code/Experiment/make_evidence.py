#!/usr/bin/env python
"""Deliverable 1: descriptive evidence that bunching is a real, costly problem (no ML).

Reads the processed `features.csv`, measures headway irregularity and the avoidable passenger
wait it causes, writes the evidence tables + figures, and prints the headline numbers to drop
straight into the report.

Usage:
    python make_evidence.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Experiment/make_evidence.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import date

import pandas as pd

from src.config import load_config
from src import evidence as ev
from src.plots import (fig_headway_ratio_hist, fig_cv_by_route, fig_excess_wait,
                       fig_severity_timeseries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.outputs_dir
    figs = cfg.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.processed_path, parse_dates=["stop_time"])
    print(f"[evidence] loaded {len(df):,} arrivals from {cfg.processed_path}")

    e = cfg.evidence
    min_arr = int(e.get("min_arrivals_per_stop", 5))
    cv_clip = e.get("cv_clip")
    lab = cfg.label
    bunch_frac, warn_frac = float(lab["bunch_frac"]), float(lab["warn_frac"])

    # --- tables ---------------------------------------------------------------------------
    irr = ev.headway_irregularity(df, min_arrivals=min_arr, cv_clip=cv_clip)
    exc = ev.excess_wait_minutes(df, min_arrivals=min_arr, cv_clip=cv_clip)
    peak = ev.peak_offpeak_irregularity(df, min_arrivals=min_arr, cv_clip=cv_clip)
    bf = ev.bunched_fraction(df, bunch_frac, warn_frac)
    # scoped service-day count from config (data can spill one row past midnight -> +1)
    sc = cfg.scope
    n_days = (date.fromisoformat(str(sc["end_date"])) - date.fromisoformat(str(sc["start_date"]))).days + 1
    summ = ev.excess_wait_summary(df, min_arrivals=min_arr, cv_clip=cv_clip, n_days=n_days)
    sev = ev.severity_index(df)

    irr.to_csv(out / "evidence_irregularity.csv", index=False)
    exc.drop(columns=["mean_wait_min"], errors="ignore").to_csv(
        out / "evidence_excess_wait.csv", index=False)
    peak.to_csv(out / "evidence_peak.csv", index=False)
    with open(out / "evidence_summary.json", "w") as fh:
        json.dump({"summary": summ,
                   "frac_bunched_now": bf["frac_bunched_now"],
                   "frac_warning_now": bf["frac_warning_now"],
                   "label_frac": bf.get("label_frac", {})}, fh, indent=2)

    # --- figures --------------------------------------------------------------------------
    p1 = fig_headway_ratio_hist(bf, figs, bunch_frac, warn_frac); print(f"  [fig] {p1}")
    p2 = fig_cv_by_route(peak, figs); print(f"  [fig] {p2}")
    p3 = fig_excess_wait(exc, figs); print(f"  [fig] {p3}")
    p4 = fig_severity_timeseries(sev, figs); print(f"  [fig] {p4}")

    # --- headline numbers for the report --------------------------------------------------
    print("\n[evidence] headline numbers (transcribe into the report):")
    print(f"  stops / stop-hour windows     : {summ['n_stops']:,} / {summ['n_windows']:,}  over {summ['n_days']} day(s)")
    print(f"  median / mean headway CV      : {summ['median_cv']:.2f} / {summ['mean_cv']:.2f}")
    print(f"  bunched-now fraction          : {bf['frac_bunched_now']:.1%}")
    print(f"  warning-now fraction          : {bf['frac_warning_now']:.1%}")
    print(f"  forward-label balance         : {bf.get('label_frac', {})}")
    print(f"  mean excess wait / passenger  : {summ['mean_excess_wait_min']:.2f} min")
    print(f"  total avoidable pax-min (capped)   : {summ['total_excess_pax_min_capped']:,.0f}")
    print(f"  total avoidable pax-min (uncapped) : {summ['total_excess_pax_min_uncapped']:,.0f}")
    print(f"  avoidable pax-min PER DAY (capped) : {summ['total_excess_pax_min_per_day_capped']:,.0f}")


if __name__ == "__main__":
    main()
