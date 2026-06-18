"""Figures for the report and demo — all generated from the SAME saved outputs.

  fig_string_diagram      - vehicles' arrival times along the route (bunching = lines converging)
  fig_rolling_accuracy    - prequential accuracy over time for HAT vs ARF + ADWIN drift markers
  fig_lead_time_hist      - distribution of nudge lead time (minutes before observed bunching)

  Evidence of bunching (Deliverable 1, src/evidence.py):
  fig_headway_ratio_hist  - distribution of current headway / local-normal, with the warn/bunch lines
  fig_cv_by_route         - headway coefficient of variation per route, peak vs off-peak
  fig_excess_wait         - ideal vs excess passenger wait minutes per route

  Counterfactual nudge (Deliverable 2, src/simulate.py):
  fig_counterfactual_bars - baseline vs nudged vs true-positive-only on bunching / CV / wait
  fig_sensitivity         - reduction in bunching & wait as the ease-off magnitude is swept
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / notebook-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_string_diagram(pred_frame: pd.DataFrame, route: str, direction: str,
                       out_dir: Path, time_col: str = "stop_time",
                       suffix: str = "", title_tag: str = "") -> Path:
    """Stop sequence (y) vs time (x), one line per vehicle. Converging lines = bunching.

    `time_col` selects the arrival-time column on the x-axis: "stop_time" for the recorded
    ("before") diagram, or "stop_time_cf" for the counterfactual ("after") diagram produced by
    simulate.apply_easeoff. `suffix`/`title_tag` distinguish the saved file and the title."""
    d = pred_frame[(pred_frame["route_short_name"] == str(route)) &
                   (pred_frame["direction_id"] == str(direction))].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for veh, g in d.groupby("trip_id", sort=False):
        g = g.sort_values("pt_sequence")
        ax.plot(pd.to_datetime(g[time_col]), g["pt_sequence"],
                lw=0.8, alpha=0.6)
    # mark predicted nudges (ARF) where they fired
    col = "pred_ARF" if "pred_ARF" in d else "pred_HAT"
    fired = d[d[col] >= 1]
    ax.scatter(pd.to_datetime(fired[time_col]), fired["pt_sequence"],
               s=8, c="orange", label="nudge fired", zorder=5)
    bunch = d[d["label"] == 2]
    ax.scatter(pd.to_datetime(bunch[time_col]), bunch["pt_sequence"],
               s=8, c="red", marker="x", label="observed bunching", zorder=6)
    ax.set_xlabel("time"); ax.set_ylabel("stop sequence")
    ax.set_title(f"String diagram — route {route} dir {direction}{title_tag}")
    ax.legend(loc="upper left", fontsize=8)
    p = Path(out_dir) / f"string_diagram_{route}_{direction}{suffix}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_rolling_accuracy(summary: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    models = summary["summary"]["models"]
    for name, info in models.items():
        roll = info["rolling_accuracy"]
        ax.plot(range(len(roll)), roll, lw=1.0, label=f"{name} (rolling acc)")
        for dp in info["drift_points"]:
            ax.axvline(dp, color="grey", alpha=0.15, lw=0.6)
    ax.set_xlabel("instance (chronological)"); ax.set_ylabel("rolling accuracy")
    ax.set_title("Prequential accuracy over time (grey = ADWIN drift)")
    ax.legend(fontsize=8)
    p = Path(out_dir) / "rolling_accuracy.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_lead_time_hist(metrics: dict, out_dir: Path, model: str = "ARF") -> Path | None:
    info = metrics.get(model, {}).get("nudge_lead", {})
    leads = info.get("lead_times_min")
    if not leads:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(leads, bins=20, color="steelblue", alpha=0.85)
    med = float(np.median(leads))
    ax.axvline(med, color="red", lw=1.2, label=f"median {med:.1f} min")
    ax.set_xlabel("nudge lead time (min before observed bunching)")
    ax.set_ylabel("count"); ax.set_title(f"Nudge lead time — {model}")
    ax.legend(fontsize=8)
    p = Path(out_dir) / f"lead_time_hist_{model}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ============================================================================================
# Deliverable 1 — descriptive evidence that bunching is a real, costly problem
# ============================================================================================
def fig_headway_ratio_hist(bf: dict, out_dir: Path, bunch_frac: float = 0.40,
                           warn_frac: float = 0.60) -> Path:
    """Histogram of current headway / local-normal. Mass left of the lines is bunched service."""
    vals = np.asarray(bf["ratio_values"], dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals <= 3.0]  # clip the long right tail (service gaps) for readability
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(bunch_frac, color="red", lw=1.4, ls="--",
               label=f"bunching < {bunch_frac:.2f}")
    ax.axvline(warn_frac, color="orange", lw=1.4, ls="--",
               label=f"warning < {warn_frac:.2f}")
    ax.axvline(1.0, color="grey", lw=1.0, ls=":", label="on-headway (=1.0)")
    ax.axvspan(0, bunch_frac, color="red", alpha=0.07)
    ax.set_xlabel("headway / local-normal headway (ratio)")
    ax.set_ylabel("arrivals"); ax.set_title("How irregular is the realised service?")
    ax.legend(fontsize=8)
    p = Path(out_dir) / "headway_ratio_hist.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_cv_by_route(peak_df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bars: per-route headway CV, peak vs off-peak (from evidence.peak_offpeak)."""
    piv = peak_df.pivot(index="route_short_name", columns="period", values="cv")
    routes = list(piv.index)
    x = np.arange(len(routes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, period in enumerate([c for c in ("off-peak", "peak") if c in piv.columns]):
        ax.bar(x + (i - 0.5) * w, piv[period].to_numpy(), width=w, label=period)
    ax.set_xticks(x); ax.set_xticklabels(routes, rotation=0)
    ax.set_xlabel("route"); ax.set_ylabel("headway CV (std / mean)")
    ax.set_title("Headway irregularity by route (higher = more bunching)")
    ax.legend(fontsize=8)
    p = Path(out_dir) / "headway_cv_by_route.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_excess_wait(exc: pd.DataFrame, out_dir: Path) -> Path:
    """Stacked bars per route: ideal wait + avoidable (excess) wait, in minutes per passenger."""
    per_route = (exc.groupby("route_short_name")
                    .agg(ideal=("ideal_wait_min", "mean"),
                         excess=("excess_wait_min", "mean"))
                    .reset_index())
    routes = per_route["route_short_name"].tolist()
    x = np.arange(len(routes))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, per_route["ideal"], label="ideal wait (even headways)", color="lightsteelblue")
    ax.bar(x, per_route["excess"], bottom=per_route["ideal"],
           label="avoidable excess wait (irregularity)", color="indianred")
    ax.set_xticks(x); ax.set_xticklabels(routes)
    ax.set_xlabel("route"); ax.set_ylabel("mean passenger wait (min)")
    ax.set_title("Excess passenger wait caused by headway irregularity")
    ax.legend(fontsize=8)
    p = Path(out_dir) / "excess_wait_by_route.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_severity_timeseries(sev: pd.DataFrame, out_dir: Path) -> Path:
    """Mean bunching severity by hour-of-day, one line per route; peak bands shaded."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for route, g in sev.groupby("route_short_name"):
        g = g.sort_values("hour")
        ax.plot(g["hour"], g["mean_severity"], lw=1.0, alpha=0.8, label=str(route))
    for lo, hi in [(6, 8), (17, 19)]:
        ax.axvspan(lo, hi, color="grey", alpha=0.10)
    ax.set_xlabel("hour of day"); ax.set_ylabel("mean severity  (1 - headway ratio)")
    ax.set_title("Bunching severity over the day (grey = peak)")
    ax.legend(fontsize=7, ncol=4)
    p = Path(out_dir) / "severity_timeseries.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ============================================================================================
# Deliverable 2 — counterfactual: would the nudge help?
# ============================================================================================
def fig_counterfactual_bars(results: dict, out_dir: Path) -> Path:
    """Baseline vs nudged (all-fired) vs true-positive-only on the three headline quantities."""
    base = results["baseline"]
    allf = results["all_fired"]
    tp = results["true_positive_only"]
    metrics = [
        ("bunching events", "bunch_count"),
        ("mean headway CV", "mean_cv"),
        ("excess wait (min)", "total_excess_wait_min"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    labels = ["baseline", "nudge\n(all fired)", "nudge\n(TP only)"]
    colors = ["grey", "seagreen", "darkseagreen"]
    for ax, (title, key) in zip(axes, metrics):
        vals = [base[key], allf[key], tp[key]]
        ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        # annotate the % reduction of the all-fired bar vs baseline
        if base[key]:
            red = 100.0 * (base[key] - allf[key]) / base[key]
            ax.text(1, allf[key], f"-{red:.0f}%", ha="center", va="bottom",
                    fontsize=8, color="seagreen")
    fig.suptitle("Counterfactual effect of the driver nudge (simulation)", fontsize=11)
    p = Path(out_dir) / "counterfactual_comparison.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_sensitivity(sweep_df: pd.DataFrame, out_dir: Path) -> Path:
    """Reduction in bunching and in excess wait as the per-stop ease-off magnitude is swept."""
    s = sweep_df.sort_values("ease_off_seconds")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s["ease_off_seconds"], s["d_bunch_count_pct"], "-o",
            label="bunching reduction", color="seagreen")
    ax.plot(s["ease_off_seconds"], s["d_excess_wait_pct"], "-s",
            label="excess-wait reduction", color="steelblue")
    ax.set_xlabel("ease-off per nudged stop (seconds)")
    ax.set_ylabel("reduction vs baseline (%)")
    ax.set_title("Sensitivity to the assumed ease-off magnitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    p = Path(out_dir) / "counterfactual_sensitivity.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p
