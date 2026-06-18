"""Descriptive evidence that bunching is a real, costly problem in the SUNT data.

This module is deliberately MODEL-FREE: it does not train or predict anything. It only
measures, from the reconstructed forward headways, how irregular the service actually is and
converts that irregularity into a concrete cost — minutes of avoidable passenger waiting.

Key quantities
--------------
* Headway coefficient of variation (CV = std/mean) per (route, direction, stop): the standard
  "headway irregularity" measure. CV = 0 is perfectly even service; bunching inflates it.
* Bunched / warning fraction: share of arrivals whose CURRENT forward headway is below
  `bunch_frac` / `warn_frac` of the local-median ("normal") headway — the same thresholds the
  model's label uses, but applied to the realised present rather than the forward window.
* Excess passenger wait: for a stop with headways h_i, the mean wait of a passenger arriving at
  random (Poisson) is the renewal-theory result

      E[W] = E[h^2] / (2 E[h]) = (mean_h / 2) * (1 + CV^2).

  The even-headway ideal is mean_h / 2, so the AVOIDABLE wait caused purely by irregularity is

      W_excess = (mean_h / 2) * CV^2     [minutes per passenger].

  Weighting W_excess by boardings turns it into total avoidable passenger-minutes — the concrete
  "there is a real problem" number.

All functions take the processed `features.csv` frame (one row per arrival) and reuse the same
`headway_min` / `local_median_headway` columns the rest of the pipeline already produces.

Windowing
---------
Irregularity is measured WITHIN hourly windows (per route, direction, stop, hour-of-day), not
pooled across the whole day. This is deliberate: pooling a 5-minute peak gap with a 90-minute
late-evening gap would attribute the natural daily demand cycle to "bunching" and overstate the
cost. Comparing like-with-like demand isolates the genuine irregularity, and peak windows still
come out worse than off-peak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import static_threshold

STOP_KEY = ["route_short_name", "direction_id", "stop_id"]
WINDOW_KEY = ["route_short_name", "direction_id", "stop_id", "hour"]


def _ratio(df: pd.DataFrame) -> pd.Series:
    """Current forward headway as a fraction of the local 'normal' (same expr as static_threshold)."""
    return (df["headway_min"] / df["local_median_headway"]).replace([np.inf, -np.inf], np.nan)


def _with_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Attach an `hour` column (hour-of-day) used to window the irregularity computation."""
    if "hour" in df.columns:
        return df
    d = df.copy()
    d["hour"] = d["stop_time"].dt.hour
    return d


def headway_irregularity(df: pd.DataFrame, min_arrivals: int = 5,
                         cv_clip: float | None = None) -> pd.DataFrame:
    """Per (route, direction, stop, hour-of-day): count, mean/std/median headway, CV = std/mean.

    Windowing by hour compares like-with-like demand so the daily cycle is not mistaken for
    bunching. Windows with fewer than `min_arrivals` arrivals are dropped (CV is unreliable on a
    handful of points). If `cv_clip` is given, CV is clipped to [0, cv_clip] so a few artefact
    windows with a huge spread cannot dominate downstream aggregates."""
    d = _with_hour(df).dropna(subset=["headway_min"])
    g = d.groupby(WINDOW_KEY, sort=False)["headway_min"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_h": g.mean(),
        "std_h": g.std(ddof=0),
        "median_h": g.median(),
    }).reset_index()
    out = out[out["n"] >= int(min_arrivals)].copy()
    out["cv"] = out["std_h"] / out["mean_h"]
    if cv_clip is not None:
        out["cv"] = out["cv"].clip(lower=0.0, upper=float(cv_clip))
    return out


def bunched_fraction(df: pd.DataFrame, bunch_frac: float, warn_frac: float) -> dict:
    """Share of arrivals that are bunched / in warning RIGHT NOW (current headway ratio).

    Uses `static_threshold` (the existing baseline) so the definition matches the rest of the
    pipeline exactly. Also echoes the forward-looking `label` balance so the descriptive section
    and the ML section report consistent class shares."""
    valid = df.dropna(subset=["headway_min", "local_median_headway"])
    cls = static_threshold(valid, bunch_frac, warn_frac)
    n = len(cls)
    out = {
        "n_arrivals": int(n),
        "frac_bunched_now": float((cls == 2).mean()) if n else float("nan"),
        "frac_warning_now": float((cls == 1).mean()) if n else float("nan"),
        "ratio_values": _ratio(valid).dropna().to_numpy(),  # for the histogram
    }
    if "label" in df.columns:
        lab = df["label"].value_counts(normalize=True).to_dict()
        out["label_frac"] = {int(k): float(v) for k, v in lab.items()}
    return out


def excess_wait_minutes(df: pd.DataFrame, min_arrivals: int = 5,
                        cv_clip: float | None = None) -> pd.DataFrame:
    """Per-stop mean / ideal / excess passenger wait, and boarding-weighted avoidable minutes.

    mean_wait_min  = (mean_h / 2) * (1 + CV^2)     -- renewal-theory mean wait
    ideal_wait_min = mean_h / 2                     -- even-headway service
    excess_wait_min = (mean_h / 2) * CV^2           -- AVOIDABLE wait caused by irregularity

    The boarding-weighted `excess_pax_min` = excess_wait_min * boardings_at_stop estimates the
    total avoidable passenger-minutes per stop over the scoped period."""
    irr = headway_irregularity(df, min_arrivals=min_arrivals, cv_clip=cv_clip)
    irr["mean_wait_min"] = (irr["mean_h"] / 2.0) * (1.0 + irr["cv"] ** 2)
    irr["ideal_wait_min"] = irr["mean_h"] / 2.0
    irr["excess_wait_min"] = (irr["mean_h"] / 2.0) * (irr["cv"] ** 2)

    # boardings per window (passenger weight). `n-boardings` is the OD boarding count.
    if "n-boardings" in df.columns:
        board = (_with_hour(df).groupby(WINDOW_KEY, sort=False)["n-boardings"]
                   .sum().rename("boardings").reset_index())
        irr = irr.merge(board, on=WINDOW_KEY, how="left")
    else:
        irr["boardings"] = np.nan
    irr["boardings"] = irr["boardings"].fillna(0.0)
    irr["excess_pax_min"] = irr["excess_wait_min"] * irr["boardings"]
    return irr


def excess_wait_summary(df: pd.DataFrame, min_arrivals: int = 5,
                        cv_clip: float | None = 3.0, n_days: int | None = None) -> dict:
    """Scalar headline numbers: overall CV, mean excess wait, total avoidable passenger-minutes.

    Reports BOTH the cv-clipped total and the uncapped total so the effect of a few artefact
    windows is transparent (per the report's honesty stance). `n_days` is the number of scoped
    service days for the per-day rate; if None it is inferred from the date span (which can be
    off by one when the last trip spills past midnight, so callers should pass the scoped count)."""
    capped = excess_wait_minutes(df, min_arrivals=min_arrivals, cv_clip=cv_clip)
    uncapped = excess_wait_minutes(df, min_arrivals=min_arrivals, cv_clip=None)
    if n_days is None:
        n_days = max(df["stop_time"].dt.normalize().nunique(), 1) if "stop_time" in df else 1
    return {
        "n_windows": int(len(capped)),
        "n_stops": int(capped[STOP_KEY].drop_duplicates().shape[0]),
        "n_days": int(n_days),
        "median_cv": float(capped["cv"].median()),
        "mean_cv": float(capped["cv"].mean()),
        "mean_excess_wait_min": float(capped["excess_wait_min"].mean()),
        "total_excess_pax_min_capped": float(capped["excess_pax_min"].sum()),
        "total_excess_pax_min_uncapped": float(uncapped["excess_pax_min"].sum()),
        "total_excess_pax_min_per_day_capped": float(capped["excess_pax_min"].sum() / n_days),
    }


def peak_offpeak_irregularity(df: pd.DataFrame, min_arrivals: int = 5,
                              cv_clip: float | None = 3.0) -> pd.DataFrame:
    """Per route, CV and excess wait split by peak vs off-peak (uses the existing `is_peak`).

    Demonstrates that irregularity (and the wait cost) is worse in the peak, where the most
    passengers are affected."""
    if "is_peak" not in df.columns:
        raise KeyError("expected an `is_peak` column (built by features._time_feats)")
    rows = []
    for is_peak, sub in df.groupby("is_peak", sort=True):
        exc = excess_wait_minutes(sub, min_arrivals=min_arrivals, cv_clip=cv_clip)
        per_route = (exc.groupby("route_short_name")
                        .agg(cv=("cv", "mean"),
                             excess_wait_min=("excess_wait_min", "mean"),
                             excess_pax_min=("excess_pax_min", "sum"))
                        .reset_index())
        per_route["period"] = "peak" if int(is_peak) == 1 else "off-peak"
        rows.append(per_route)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["route_short_name", "period"]).reset_index(drop=True)


def severity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Mean bunching severity by hour-of-day, per route (Enayatollahi-2019-style index).

    Per arrival, severity = max(0, 1 - headway_ratio): 0 when on-headway, -> 1 as the gap to the
    bus ahead collapses to zero. Aggregated by route and hour-of-day for a time series."""
    d = df.copy()
    ratio = _ratio(d)
    d["severity"] = (1.0 - ratio).clip(lower=0.0)
    d["hour"] = d["stop_time"].dt.hour
    out = (d.dropna(subset=["severity"])
             .groupby(["route_short_name", "hour"])["severity"]
             .mean().rename("mean_severity").reset_index())
    return out
