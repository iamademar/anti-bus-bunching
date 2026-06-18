#!/usr/bin/env python
"""Assemble the BEFORE / AFTER service-quality panel for the standalone /Explanations writeup.

This does NOT re-run the ML. It reuses the already-saved experiment outputs:
  * outputs/simulation_results.json  -> the counterfactual nudge run (bunching, CV, excess wait)
  * outputs/predictions.csv          -> per-arrival headways/labels/predictions (for extra rows)
  * the raw OD parquet               -> per-trip travel time A->B (peak vs off-peak context only)

The panel, summary, and figures need only the committed outputs; they reproduce the report without
the raw SUNT download. The travel-time context is the one exception: it reads the OD parquet, so if
that data is absent this script skips it (printing a clear note), keeps the committed
beforeafter_travel_time.csv, and still writes everything else.

before = actual historical SUNT behaviour (8 busy routes, 1-3 March 2024)
after  = a counterfactual simulation of the anti-bunching nudge on the SAME data

Honesty: occupancy and stop-congestion show no measurable relationship to headway in SUNT, so
their "after" is reported as "no measurable change", not an improvement. Travel time A->B is a
congestion CONTEXT metric (peak >> off-peak); the nudge targets wait/regularity, not ride time,
so its "after" cell is left as "--".

Usage:
    python make_beforeafter.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Simulation/make_beforeafter.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config
from src import simulate as sim
from src.headway import load_od

PEAK_HOURS = {6, 7, 8, 17, 18, 19}  # matches features._time_feats is_peak


def _pct(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before else float("nan")


def travel_time_context(cfg) -> pd.DataFrame:
    """Per-trip terminal-to-terminal travel time, median peak vs off-peak per route (BEFORE only).

    Congestion context: a real, intuitive 'traffic' signal, but NOT something the driver nudge
    changes (bunching barely affects ride time; easing off slightly adds to it)."""
    s = cfg.scope
    od = load_od(cfg.od_dir, s["routes"], s["directions"], s["start_date"], s["end_date"])
    od["start_trip"] = pd.to_datetime(od["start_trip"], errors="coerce")
    od["end_trip"] = pd.to_datetime(od["end_trip"], errors="coerce")
    trips = (od.dropna(subset=["start_trip", "end_trip"])
               .groupby(["route_short_name", "trip_id"], observed=True)
               .agg(start=("start_trip", "first"), end=("end_trip", "first")).reset_index())
    trips["travel_min"] = (trips["end"] - trips["start"]).dt.total_seconds() / 60.0
    trips = trips[(trips["travel_min"] > 0) & (trips["travel_min"] < 600)]
    trips["is_peak"] = trips["start"].dt.hour.isin(PEAK_HOURS)
    rows = []
    for route, g in trips.groupby("route_short_name", observed=True):
        peak = g[g["is_peak"]]["travel_min"]
        off = g[~g["is_peak"]]["travel_min"]
        if len(peak) and len(off):
            rows.append({"route": str(route),
                         "peak_min": float(peak.median()),
                         "offpeak_min": float(off.median()),
                         "delta_pct": 100.0 * (peak.median() - off.median()) / off.median(),
                         "n_peak": int(len(peak)), "n_off": int(len(off))})
    return pd.DataFrame(rows).sort_values("route").reset_index(drop=True)


def crowding_correlations(df: pd.DataFrame) -> dict:
    """Flat checks: does loading / boardings track headway? (They don't, in SUNT.)"""
    d = df.copy()
    d["ratio"] = (d["headway_min"] / d["local_median_headway"]).replace([np.inf, -np.inf], np.nan)
    out = {}
    if "loading" in d:
        v = d.dropna(subset=["loading", "ratio"])
        out["loading_vs_ratio_r"] = float(np.corrcoef(v["loading"], v["ratio"])[0, 1]) if len(v) > 2 else float("nan")
        out["loading_cv"] = float(v["loading"].std() / v["loading"].mean()) if len(v) else float("nan")
    if "n-boardings" in d:
        v = d.dropna(subset=["n-boardings", "headway_min"])
        out["boardings_vs_headway_r"] = float(np.corrcoef(v["n-boardings"], v["headway_min"])[0, 1]) if len(v) > 2 else float("nan")
    return out


def make_after_figures(cf: pd.DataFrame, cfg) -> None:
    """Emit the two 'after' figures the Explanations doc pairs with its before figures.

    `cf` is the counterfactual frame from simulate.apply_easeoff (has stop_time_cf, headway_cf),
    so both figures reflect the SAME ease-off as the panel numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plots import fig_string_diagram

    figs = cfg.figures_dir
    figs.mkdir(parents=True, exist_ok=True)

    # (a) severity-by-hour, before vs after, averaged over all routes ----------------------
    d = cf.copy()
    d["hour"] = d["stop_time"].dt.hour
    for col, name in (("headway_min", "before"), ("headway_cf", "after")):
        ratio = (d[col] / d["local_median_headway"]).replace([np.inf, -np.inf], np.nan)
        d[f"sev_{name}"] = (1.0 - ratio).clip(lower=0.0)
    by_hour = d.groupby("hour")[["sev_before", "sev_after"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_hour.index, by_hour["sev_before"], "-o", lw=1.3, color="indianred",
            label="before (recorded)")
    ax.plot(by_hour.index, by_hour["sev_after"], "-s", lw=1.3, color="seagreen",
            label="after (nudge, simulated)")
    for lo, hi in [(6, 8), (17, 19)]:
        ax.axvspan(lo, hi, color="grey", alpha=0.10)
    ax.set_xlabel("hour of day"); ax.set_ylabel("mean severity  (1 - headway ratio)")
    ax.set_title("Bunching severity by hour: before vs after the nudge (grey = peak)")
    ax.legend(fontsize=8)
    p_sev = figs / "severity_timeseries_after.png"
    fig.tight_layout(); fig.savefig(p_sev, dpi=130); plt.close(fig)
    print(f"  [fig] {p_sev}")

    # (b) string diagrams: BEFORE (recorded) and AFTER (counterfactual) on the same route ----
    # We regenerate the "before" diagram here too, because cf already has route/direction cast to
    # str; the stand-alone make_figures.py compares the int-typed predictions column to a string
    # and silently produces an empty plot. Emitting both from `cf` guarantees a matched pair.
    route, direction = str(cfg.scope["routes"][0]), str(cfg.scope["directions"][0])
    p_str_b = fig_string_diagram(cf, route, direction, figs, time_col="stop_time",
                                 suffix="_before", title_tag="  (before, recorded)")
    print(f"  [fig] {p_str_b}")
    p_str = fig_string_diagram(cf, route, direction, figs, time_col="stop_time_cf",
                               suffix="_after", title_tag="  (after nudge, simulated)")
    print(f"  [fig] {p_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg.outputs_dir

    # --- the counterfactual (already computed) ----------------------------------------------
    with open(out / "simulation_results.json") as fh:
        simr = json.load(fh)
    base, after = simr["baseline"], simr["all_fired"]
    tp = simr["true_positive_only"]

    # --- extra before/after rows from predictions.csv via the existing apply_easeoff --------
    pred = pd.read_csv(out / "predictions.csv", parse_dates=["stop_time"])
    pred["route_short_name"] = pred["route_short_name"].astype(str)
    pred["direction_id"] = pred["direction_id"].astype(str)
    s = cfg.simulate
    cf = sim.apply_easeoff(pred, ease_off_seconds=float(s["ease_off_seconds"]),
                           cap_min=float(s["max_total_easeoff_min"]),
                           close_threshold=float(s["close_threshold"]), mode="all_fired")

    # --- after-figures for the Explanations doc (severity-by-hour, string diagram) -----------
    # Both come from the SAME counterfactual frame `cf` used for the panel numbers, so the
    # picture matches the reported -13% effect (it is a modest, honest change, not a hand-drawn one).
    make_after_figures(cf, cfg)

    def dev(col):  # mean |headway - local normal|
        v = cf.dropna(subset=[col, "local_median_headway"])
        return float((v[col] - v["local_median_headway"]).abs().mean())

    def wait_std(col):  # std of per-stop wait (= headway/2): the (un)predictability of waiting
        v = cf.dropna(subset=[col])
        return float((v[col] / 2.0).std())

    head_dev_before, head_dev_after = dev("headway_min"), dev("headway_cf")
    wait_std_before, wait_std_after = wait_std("headway_min"), wait_std("headway_cf")
    wait_med_before = float((cf["headway_min"].dropna() / 2.0).median())
    wait_med_after = float((cf["headway_cf"].dropna() / 2.0).median())

    # --- crowding / congestion flat checks --------------------------------------------------
    feats = pd.read_csv(cfg.processed_path)
    crowd = crowding_correlations(feats)

    # --- travel-time context (peak vs off-peak) --------------------------------------------
    # This is the ONLY part of this script that needs the raw SUNT OD data: peak-vs-off-peak
    # travel time is computed from per-trip start/end times that live only in the OD parquet,
    # not in the committed predictions.csv. If the raw data is absent we skip it, leave the
    # committed beforeafter_travel_time.csv untouched, and still write the panel/summary/figures
    # (none of which depend on travel time). The before/after numbers reproduce the report
    # exactly; only the congestion-context travel-time numbers require the SUNT download.
    tt_path = out / "beforeafter_travel_time.csv"
    try:
        tt = travel_time_context(cfg)
        tt.to_csv(tt_path, index=False)
    except FileNotFoundError:
        tt = None
        print(f"  [skip] travel-time context needs the raw SUNT OD data (none found at "
              f"{cfg.od_dir}).")
        print(f"         Leaving the committed {tt_path.name} in place; everything else is "
              f"regenerated.")

    # --- assemble the panel, GROUPED by whether the nudge actually moves it ------------------
    # Group A: indicators the nudge improves.  Group B: checked but ~unchanged (honest).
    moved = [
        ("Bunching events (count)", f"{base['bunch_count']:,}", f"{after['bunch_count']:,}",
         after['d_bunch_count_pct']),
        ("Bunching rate (% of arrivals)", f"{100*base['frac_bunched']:.1f}%",
         f"{100*after['frac_bunched']:.1f}%", _pct(base['frac_bunched'], after['frac_bunched'])),
        ("Avoidable wait (passenger-min)", f"{base['total_excess_wait_min']:,.0f}",
         f"{after['total_excess_wait_min']:,.0f}", after['d_excess_wait_pct']),
        ("Headway irregularity (mean CV)", f"{base['mean_cv']:.3f}", f"{after['mean_cv']:.3f}",
         after['d_cv_pct']),
    ]
    unchanged = [
        ("Mean headway deviation (min)", f"{head_dev_before:.1f}", f"{head_dev_after:.1f}",
         _pct(head_dev_before, head_dev_after)),
        ("Median wait (min)", f"{wait_med_before:.1f}", f"{wait_med_after:.1f}",
         _pct(wait_med_before, wait_med_after)),
        ("Wait spread (std of wait, min)", f"{wait_std_before:.1f}",
         f"{wait_std_after:.1f}", _pct(wait_std_before, wait_std_after)),
    ]
    panel_df = pd.DataFrame(
        [(g, *row) for g, rows in [("improves", moved), ("~unchanged", unchanged)] for row in rows],
        columns=["group", "indicator", "before", "after", "improvement_pct"])
    panel_df.to_csv(out / "beforeafter_panel.csv", index=False)

    summary = {
        "moved": moved,
        "unchanged": unchanged,
        "ablation_tp_only": {"d_bunch_count_pct": tp["d_bunch_count_pct"],
                             "d_excess_wait_pct": tp["d_excess_wait_pct"]},
        "n_easeoffs": after["n_easeoffs"],
        "crowding": crowd,
    }
    if tt is not None:
        tt_peak_med = float(tt["peak_min"].median())
        tt_off_med = float(tt["offpeak_min"].median())
        summary["travel_time_peak_offpeak"] = tt.to_dict(orient="records")
        summary["travel_time_median"] = {
            "peak_min": tt_peak_med, "offpeak_min": tt_off_med,
            "delta_pct": 100.0 * (tt_peak_med - tt_off_med) / tt_off_med}
    else:
        # No raw SUNT data: mark the travel-time fields as unavailable rather than fabricating
        # them. The committed beforeafter_summary.json already holds the SUNT-derived values.
        summary["travel_time_peak_offpeak"] = None
        summary["travel_time_median"] = None
        summary["travel_time_note"] = ("requires the raw SUNT OD data; not regenerated. "
                                       "See the committed beforeafter_travel_time.csv.")
    with open(out / "beforeafter_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- print for transcription ------------------------------------------------------------
    def _row(name, b, a, p):
        ptxt = f"{p:+.1f}%" if p == p else "n/a"
        print(f"{name:40s} {b:>14s} {a:>14s} {ptxt:>12s}")

    print("BEFORE / AFTER PANEL  (before = actual March 2024; after = counterfactual nudge)\n")
    print(f"{'indicator':40s} {'before':>14s} {'after':>14s} {'improvement':>12s}")
    print("-- improves with the nudge --")
    for r in moved:
        _row(*r)
    print("-- checked but ~unchanged (the nudge is not a speed-up; it evens spacing) --")
    for r in unchanged:
        _row(*r)
    print(f"\nease-offs applied: {after['n_easeoffs']:,}")
    print(f"ablation (true-positive nudges only): "
          f"-{tp['d_bunch_count_pct']:.1f}% bunching / -{tp['d_excess_wait_pct']:.1f}% wait")
    print("\nNOT MOVED BY THE NUDGE (reported honestly):")
    print(f"  bus occupancy vs headway ratio  r = {crowd.get('loading_vs_ratio_r', float('nan')):+.3f}  (weak/no link)")
    print(f"  stop boardings vs headway       r = {crowd.get('boardings_vs_headway_r', float('nan')):+.3f}  (no link)")
    if tt is not None:
        print("\nTRAVEL TIME A->B  (congestion context; NOT a nudge target):")
        print(tt.to_string(index=False))
        print(f"  median across routes: peak {tt_peak_med:.1f} vs off-peak {tt_off_med:.1f} min "
              f"(+{100*(tt_peak_med-tt_off_med)/tt_off_med:.1f}%)")
        wrote = (f"{out/'beforeafter_panel.csv'}, {out/'beforeafter_travel_time.csv'}, "
                 f"{out/'beforeafter_summary.json'}")
    else:
        print("\nTRAVEL TIME A->B  (congestion context): skipped (needs the raw SUNT OD data).")
        wrote = f"{out/'beforeafter_panel.csv'}, {out/'beforeafter_summary.json'}"
    print(f"\n[beforeafter] wrote {wrote}")


if __name__ == "__main__":
    main()
