"""Simple non-ML baselines — the bar the streaming models must beat.

If the Adaptive Random Forest cannot beat the static threshold here, the proposal's
premise (that you need streaming, adaptive ML rather than a fixed rule) collapses.
All baselines are evaluated in the SAME chronological order as the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def static_threshold(df: pd.DataFrame, bunch_frac: float, warn_frac: float) -> np.ndarray:
    """'Is the gap small RIGHT NOW?' — classify on the CURRENT headway ratio only."""
    ratio = (df["headway_min"] / df["local_median_headway"]).to_numpy()
    pred = np.where(ratio < bunch_frac, 2, np.where(ratio < warn_frac, 1, 0))
    return pred.astype(int)


def previous_headway_persistence(df: pd.DataFrame, bunch_frac: float,
                                 warn_frac: float) -> np.ndarray:
    """Carry the previous arrival's class forward (per route/dir/stop)."""
    ratio = (df["headway_min"] / df["local_median_headway"])
    cur = np.where(ratio < bunch_frac, 2, np.where(ratio < warn_frac, 1, 0))
    s = pd.Series(cur, index=df.index)
    prev = (
        s.groupby([df["route_short_name"], df["direction_id"], df["stop_id"]])
         .shift(1)
         .fillna(0)
    )
    return prev.astype(int).to_numpy()


def historical_average(df: pd.DataFrame) -> np.ndarray:
    """Most-frequent class by (route, direction, stop, hour-of-day), seen so far (expanding)."""
    hod = df["stop_time"].dt.hour
    grp_keys = [df["route_short_name"], df["direction_id"], df["stop_id"], hod]
    # expanding mode is expensive; approximate with expanding mean of label rounded.
    lab = df["label"]
    exp_mean = lab.groupby(grp_keys).apply(lambda s: s.expanding().mean()).reset_index(level=list(range(4)), drop=True)
    exp_mean = exp_mean.reindex(df.index).fillna(0.0)
    return np.clip(np.rint(exp_mean.to_numpy()), 0, 2).astype(int)
