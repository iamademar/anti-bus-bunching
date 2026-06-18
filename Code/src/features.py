"""Causal feature builder + forward-looking bunching label.

CAUSAL DISCIPLINE (no leakage):
  - FEATURES use only information available at or before the current stop (<= t).
  - The LABEL uses FUTURE headway (> t) and is the supervised target ONLY — never a feature.

Label (3-class) for a bus at a stop:
  Look K stops ahead on the SAME trip. Take the minimum forward headway over those
  upcoming arrivals (this bus vs the bus ahead). Compare to the local "normal" headway:
    min_future_headway < bunch_frac * local_median_headway   -> "bunching"   (2)
    min_future_headway < warn_frac  * local_median_headway   -> "warning"    (1)
    otherwise                                                 -> "ok"         (0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {0: "ok", 1: "warning", 2: "bunching"}

# Causal feature columns the models consume (all known at <= t).
FEATURE_COLS = [
    "headway_min",            # current forward headway
    "headway_ratio",          # current headway / local normal
    "headway_trend",          # change vs previous arrival's headway at this stop
    "since_prev_stop_min",    # running time from previous stop
    "loading",                # current onboard load
    "recent_boardings",       # boarding pressure (this + recent stops)
    "pt_sequence",            # position along the route
    "minute_of_day",
    "dow",                    # day of week
    "is_peak",
]


def _time_feats(df: pd.DataFrame) -> pd.DataFrame:
    t = df["stop_time"].dt
    df["minute_of_day"] = t.hour * 60 + t.minute
    df["dow"] = t.dayofweek
    df["is_peak"] = (
        ((t.hour >= 6) & (t.hour <= 8)) | ((t.hour >= 17) & (t.hour <= 19))
    ).astype(int)
    return df


def build_label(od: pd.DataFrame, bunch_frac: float, warn_frac: float,
                horizon_stops: int) -> pd.DataFrame:
    """Forward-looking 3-class bunching label per stop (uses FUTURE headway; target only)."""
    od = od.sort_values(["trip_id", "pt_sequence"]).copy()

    # minimum forward headway over the next K arrivals on this trip (the future risk)
    def _min_future(s: pd.Series) -> pd.Series:
        # reverse rolling-min over the next K (shift -1 so we look strictly AHEAD)
        rev = s[::-1]
        fut = rev.shift(1).rolling(horizon_stops, min_periods=1).min()[::-1]
        return fut

    od["min_future_headway"] = (
        od.groupby("trip_id", sort=False)["headway_min"].transform(_min_future)
    )
    ref = od["local_median_headway"]
    bunch = od["min_future_headway"] < (bunch_frac * ref)
    warn = od["min_future_headway"] < (warn_frac * ref)
    label = np.where(bunch, 2, np.where(warn, 1, 0))
    od["label"] = label.astype(int)
    return od


def build_features(od: pd.DataFrame) -> pd.DataFrame:
    """Add causal features. Assumes headway/segment columns already present."""
    od = od.sort_values(["route_short_name", "direction_id", "stop_id", "stop_time"]).copy()
    key = ["route_short_name", "direction_id", "stop_id"]

    od["headway_ratio"] = od["headway_min"] / od["local_median_headway"]
    od["headway_trend"] = (
        od.groupby(key, sort=False)["headway_min"].diff()  # vs previous arrival here
    )
    # boarding pressure: this stop + recent stops on the same trip
    od = od.sort_values(["trip_id", "pt_sequence"])
    od["recent_boardings"] = (
        od.groupby("trip_id", sort=False)["n-boardings"]
          .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    od = _time_feats(od)
    return od


def make_dataset(od: pd.DataFrame, cfg) -> pd.DataFrame:
    """Full pipeline tail: label + features, then a clean modelling frame in stream order."""
    lab = cfg.label
    od = build_label(od, lab["bunch_frac"], lab["warn_frac"], int(lab["horizon_stops"]))
    od = build_features(od)

    # require a defined current headway and a defined local normal to be a valid instance
    # (do this BEFORE subsetting columns so local_median_headway is still available)
    od = od.dropna(subset=["headway_min", "local_median_headway", "label"])

    meta = ["route_short_name", "direction_id", "vehicle", "trip_id",
            "stop_id", "stop_time",
            "local_median_headway",  # kept: the baselines need it (note: pt_sequence is in FEATURE_COLS)
            "n-boardings"]           # kept: the descriptive evidence weights excess wait by boardings
    keep = meta + FEATURE_COLS + ["label"]
    keep = list(dict.fromkeys(keep))  # dedupe, preserve order
    out = od[keep].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)

    # CHRONOLOGICAL stream order — this is what makes prequential evaluation honest
    out = out.sort_values("stop_time").reset_index(drop=True)
    return out
