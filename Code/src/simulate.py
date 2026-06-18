"""Counterfactual: WOULD the driver nudge actually reduce bunching?

The real effect of drivers easing off cannot be observed from historical data, so this is a
clearly-labelled SIMULATION, not a measured outcome. It is, however, a far more defensible
estimate than the illustrative variance comparison it replaces: it actually re-runs the headway
geometry under a small, bounded behavioural change.

Idea
----
Replay the recorded arrival stream. Whenever the deployed model fires a nudge on a bus that is
genuinely closing on the bus ahead, apply a small, bounded "ease-off" delay to that bus's
subsequent arrivals (the driver lets the gap reopen). Then RECOMPUTE the forward headways on the
perturbed arrival times and re-measure bunching and excess wait. Compare against the untouched
baseline.

Grounding / assumptions (stated in the report)
* The ease-off magnitude (a few seconds per stop, capped per trip) is the decentralised analogue
  of the small adaptive holds in Daganzo (2009); Enayatollahi et al. (2019) show that shaving a
  little dwell / boarding time per stop materially reduces bunching formation.
* The nudge NEVER asks a driver to speed up — only ease off — matching the report's safe action.
* `local_median_headway` is held at its BASELINE value, so the comparison is against the same
  "normal" reference rather than a moving goalpost.
* Drivers are assumed to comply when nudged and closing; compliance and magnitude are exactly the
  assumptions the sensitivity sweep and the true-positive-only ablation stress-test.

Input: `outputs/predictions.csv` (stop_time, trip_id, route_short_name, direction_id, stop_id,
pt_sequence, headway_min, label, local_median_headway, pred_HAT, pred_ARF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .headway import recompute_forward_headway

WARNING = 1
BUNCHING = 2


# --------------------------------------------------------------------------------------------
# Metrics on a headway column (baseline or counterfactual) — reused for both sides of the compare
# --------------------------------------------------------------------------------------------
def _metrics_from_headway(df: pd.DataFrame, headway_col: str, bunch_frac: float,
                          min_arrivals: int, cv_clip: float | None) -> dict:
    """Bunching count, mean headway CV, and total excess passenger-minutes for one headway column.

    CV / excess-wait are computed within hourly windows (per route, dir, stop, hour), matching
    src/evidence.py, so the daily demand cycle is not mistaken for bunching."""
    cols = ["route_short_name", "direction_id", "stop_id", "stop_time",
            "local_median_headway", headway_col]
    d = df[[c for c in cols if c in df.columns]].copy()
    d = d.dropna(subset=[headway_col, "local_median_headway"])
    ratio = d[headway_col] / d["local_median_headway"]
    bunch_count = int((ratio < bunch_frac).sum())

    d["hour"] = d["stop_time"].dt.hour
    key = ["route_short_name", "direction_id", "stop_id", "hour"]
    g = d.groupby(key, sort=False)[headway_col]
    stat = pd.DataFrame({"n": g.size(), "mean_h": g.mean(), "std_h": g.std(ddof=0)})
    stat = stat[stat["n"] >= int(min_arrivals)]
    cv = (stat["std_h"] / stat["mean_h"])
    if cv_clip is not None:
        cv = cv.clip(lower=0.0, upper=float(cv_clip))
    excess_wait = (stat["mean_h"] / 2.0) * (cv ** 2)
    return {
        "bunch_count": bunch_count,
        "n_arrivals": int(len(d)),
        "frac_bunched": float((ratio < bunch_frac).mean()),
        "mean_cv": float(cv.mean()),
        "total_excess_wait_min": float(excess_wait.sum()),
    }


# --------------------------------------------------------------------------------------------
# The ease-off replay
# --------------------------------------------------------------------------------------------
def apply_easeoff(pred: pd.DataFrame, ease_off_seconds: float, cap_min: float,
                  close_threshold: float, mode: str = "all_fired",
                  pred_col: str = "pred_ARF") -> pd.DataFrame:
    """Return a copy with `stop_time_cf` (counterfactual arrival times) and `headway_cf`.

    For each bus (trip), accumulate a bounded ease-off delay every time the model fires AND the
    bus is closing (headway_ratio < close_threshold). The delay shifts that bus's arrival at this
    and all later stops. Forward headways are then recomputed on the shifted times.

    mode: 'all_fired'           -> every fired nudge eases off (the deployable system)
          'true_positive_only'  -> only nudges that coincide with a true warning/bunching label
                                   act (an oracle gate; the ablation that isolates whether the
                                   model's *predictions* — not perfect foresight — drive the gain)
    """
    d = pred.sort_values(["trip_id", "pt_sequence", "stop_time"]).copy()
    ratio = (d["headway_min"] / d["local_median_headway"]).to_numpy()
    fired = (d[pred_col].to_numpy() >= WARNING)
    if mode == "true_positive_only":
        fired = fired & (d["label"].to_numpy() >= WARNING)
    closing = ratio < float(close_threshold)
    act = fired & closing  # this arrival triggers an ease-off

    cap_s = float(cap_min) * 60.0
    delta_s = float(ease_off_seconds)
    trip_ids = d["trip_id"].to_numpy()

    # Walk arrivals in trip order, accumulate a capped per-trip delay, and apply it forward.
    delay_applied = np.zeros(len(d), dtype=float)  # seconds added to THIS arrival
    cur_trip = None
    cur_delay = 0.0
    for i in range(len(d)):
        if trip_ids[i] != cur_trip:
            cur_trip = trip_ids[i]
            cur_delay = 0.0
        # the ease-off decided at the PREVIOUS stop is already in cur_delay and applies here;
        # if this arrival also triggers, it raises the delay for subsequent stops.
        delay_applied[i] = cur_delay
        if act[i] and cur_delay < cap_s:
            cur_delay = min(cur_delay + delta_s, cap_s)

    d["stop_time_cf"] = d["stop_time"] + pd.to_timedelta(delay_applied, unit="s")
    d["headway_cf"] = recompute_forward_headway(d, time_col="stop_time_cf",
                                                min_headway_seconds=30)
    d["easeoff_applied_s"] = delay_applied
    return d


def baseline_metrics(pred: pd.DataFrame, cfg) -> dict:
    """Untouched (recorded) bunching / CV / excess-wait — the reference the nudge is compared to."""
    lab, ev = cfg.label, cfg.evidence
    return _metrics_from_headway(
        pred, "headway_min", float(lab["bunch_frac"]),
        int(ev.get("min_arrivals_per_stop", 5)), ev.get("cv_clip"))


def run_counterfactual(pred: pd.DataFrame, cfg, mode: str = "all_fired",
                       ease_off_seconds: float | None = None) -> dict:
    """Apply the ease-off and return counterfactual bunching / CV / excess-wait."""
    lab, ev, sim = cfg.label, cfg.evidence, cfg.simulate
    secs = float(sim["ease_off_seconds"]) if ease_off_seconds is None else float(ease_off_seconds)
    d = apply_easeoff(
        pred,
        ease_off_seconds=secs,
        cap_min=float(sim["max_total_easeoff_min"]),
        close_threshold=float(sim["close_threshold"]),
        mode=mode,
    )
    m = _metrics_from_headway(
        d, "headway_cf", float(lab["bunch_frac"]),
        int(ev.get("min_arrivals_per_stop", 5)), ev.get("cv_clip"))
    m["mode"] = mode
    m["ease_off_seconds"] = secs
    m["n_easeoffs"] = int((d["easeoff_applied_s"].diff().fillna(d["easeoff_applied_s"]) > 0).sum())
    return m


def _deltas(base: dict, nudged: dict) -> dict:
    """Relative reductions (nudged vs baseline) for the headline quantities."""
    def rel(b, n):
        return float((b - n) / b) if b else float("nan")
    return {
        "d_bunch_count": base["bunch_count"] - nudged["bunch_count"],
        "d_bunch_count_pct": 100.0 * rel(base["bunch_count"], nudged["bunch_count"]),
        "d_cv_pct": 100.0 * rel(base["mean_cv"], nudged["mean_cv"]),
        "d_excess_wait_pct": 100.0 * rel(base["total_excess_wait_min"], nudged["total_excess_wait_min"]),
    }


def sensitivity(pred: pd.DataFrame, cfg, seconds: list[float] | None = None,
                mode: str = "all_fired") -> pd.DataFrame:
    """Sweep the ease-off magnitude; report reduction in bunching and excess wait at each."""
    sim = cfg.simulate
    secs = seconds if seconds is not None else list(sim.get("sensitivity_seconds", [4, 8, 12, 16]))
    base = baseline_metrics(pred, cfg)
    rows = []
    for s in secs:
        nudged = run_counterfactual(pred, cfg, mode=mode, ease_off_seconds=s)
        d = _deltas(base, nudged)
        rows.append({"ease_off_seconds": float(s), **d,
                     "bunch_count": nudged["bunch_count"],
                     "total_excess_wait_min": nudged["total_excess_wait_min"]})
    return pd.DataFrame(rows)


def run_all(pred: pd.DataFrame, cfg) -> dict:
    """Full Deliverable-2 result: baseline, all-fired, true-positive-only, deltas, and a sweep."""
    base = baseline_metrics(pred, cfg)
    allf = run_counterfactual(pred, cfg, mode="all_fired")
    tp = run_counterfactual(pred, cfg, mode="true_positive_only")
    sweep = sensitivity(pred, cfg, mode="all_fired")
    return {
        "baseline": base,
        "all_fired": {**allf, **_deltas(base, allf)},
        "true_positive_only": {**tp, **_deltas(base, tp)},
        "sensitivity": sweep.to_dict(orient="records"),
    }
