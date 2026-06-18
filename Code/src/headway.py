"""Load OD parquet, reconstruct per-bus trajectories, and compute FORWARD HEADWAY.

Forward headway = at a given stop (for one route+direction), the time gap between a
vehicle's arrival (`stop_time`) and the *previous distinct vehicle's* arrival at that
same stop. This is the core signal bunching is defined on; the 5-/30-min aggregated
10-stop table cannot provide it, which is why we use the raw per-event OD data.

NOTE: OD boarding/alighting columns are HYPHENATED: `n-boardings`, `n-alighting`.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OD_COLUMNS = [
    "route_short_name", "direction_id", "pt_sequence", "stop_id", "vehicle",
    "trip_number", "trip_id", "start_trip", "end_trip", "stop_time",
    "n-boardings", "n-alighting", "lag_loading", "balance", "loading",
]


def _date_range(start: str, end: str) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out, d = [], d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def load_od(od_dir: Path, routes: list[str], directions: list[str],
            start_date: str, end_date: str) -> pd.DataFrame:
    """Read od-*.parquet for the requested dates, keep only scoped routes/directions."""
    frames = []
    for ds in _date_range(start_date, end_date):
        fp = Path(od_dir) / f"od-{ds}.parquet"
        if not fp.exists():
            print(f"  [skip] missing {fp.name}")
            continue
        df = pd.read_parquet(fp)
        df = df[df["route_short_name"].astype(str).isin([str(r) for r in routes])]
        df = df[df["direction_id"].astype(str).isin([str(x) for x in directions])]
        frames.append(df)
        print(f"  [load] {fp.name}: {len(df):,} scoped rows")
    if not frames:
        raise FileNotFoundError(
            f"No OD rows found for routes={routes} dirs={directions} in {od_dir}")
    od = pd.concat(frames, ignore_index=True)
    # stop_time is stored as object/string -> parse to datetime
    od["stop_time"] = pd.to_datetime(od["stop_time"], errors="coerce")
    od = od.dropna(subset=["stop_time"])
    # normalise key dtypes
    od["route_short_name"] = od["route_short_name"].astype(str)
    od["direction_id"] = od["direction_id"].astype(str)
    od["vehicle"] = od["vehicle"].astype(str)
    od["trip_id"] = od["trip_id"].astype(str)
    return od


def drop_nonmonotonic_trips(od: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Drop trips whose stop_time is not non-decreasing along pt_sequence (~2-3%)."""
    od = od.sort_values(["trip_id", "pt_sequence", "stop_time"])
    good_ids = []
    for tid, g in od.groupby("trip_id", sort=False):
        st = g.sort_values("pt_sequence")["stop_time"].values
        if len(st) >= 2 and np.all(np.diff(st).astype("timedelta64[s]").astype(float) >= 0):
            good_ids.append(tid)
    n_before = od["trip_id"].nunique()
    kept = od[od["trip_id"].isin(good_ids)].copy()
    dropped_frac = 1.0 - (len(good_ids) / max(n_before, 1))
    return kept, dropped_frac


def add_segment_and_dwell(od: pd.DataFrame) -> pd.DataFrame:
    """Per trip: segment travel time to the NEXT stop and an approximate dwell proxy."""
    od = od.sort_values(["trip_id", "pt_sequence"]).copy()
    grp = od.groupby("trip_id", sort=False)["stop_time"]
    # minutes to the next stop on this trip
    od["segment_time_min"] = (grp.shift(-1) - od["stop_time"]).dt.total_seconds() / 60.0
    # minutes since the previous stop on this trip (proxy for running time / dwell pressure)
    od["since_prev_stop_min"] = (od["stop_time"] - grp.shift(1)).dt.total_seconds() / 60.0
    return od


def recompute_forward_headway(od: pd.DataFrame, time_col: str = "stop_time",
                              min_headway_seconds: int = 30) -> pd.Series:
    """Forward headway (minutes) to the previous distinct vehicle at each (route, dir, stop).

    Pure helper: groups by (route, direction, stop), orders by `time_col`, and returns the gap
    to the previous arrival as a Series aligned to `od.index`. Implausibly tiny gaps
    (< `min_headway_seconds`) become NaN. The counterfactual simulation calls this on a SHIFTED
    arrival-time column to recompute headways after an ease-off, so `add_forward_headway` below
    delegates to it (DRY)."""
    key = ["route_short_name", "direction_id", "stop_id"]
    ordered = od.sort_values(key + [time_col])
    prev_time = ordered.groupby(key, sort=False)[time_col].shift(1)
    h = (ordered[time_col] - prev_time).dt.total_seconds() / 60.0
    # discard implausibly tiny gaps (same vehicle re-logged / artefacts)
    h = h.where(h * 60.0 >= min_headway_seconds)
    return h.reindex(od.index)


def add_forward_headway(od: pd.DataFrame, min_headway_seconds: int = 30,
                        headway_window: int = 8) -> pd.DataFrame:
    """For each (route, direction, stop), compute the gap to the previous vehicle's arrival.

    Adds:
      headway_min          - minutes since the previous distinct vehicle at this stop
      local_median_headway - rolling median of recent headways at this stop (the "normal")
    """
    od = od.sort_values(["route_short_name", "direction_id", "stop_id", "stop_time"]).copy()
    key = ["route_short_name", "direction_id", "stop_id"]
    od["headway_min"] = recompute_forward_headway(
        od, time_col="stop_time", min_headway_seconds=min_headway_seconds)
    # rolling local "normal" headway per stop (shifted so it uses only PAST arrivals)
    od["local_median_headway"] = (
        od.groupby(key, sort=False)["headway_min"]
          .transform(lambda s: s.shift(1).rolling(headway_window, min_periods=2).median())
    )
    return od


def build_headway_table(cfg) -> pd.DataFrame:
    """End-to-end: load -> (optional) drop non-monotonic -> segment/dwell -> forward headway."""
    s = cfg.scope
    od = load_od(cfg.od_dir, s["routes"], s["directions"], s["start_date"], s["end_date"])
    if cfg.preprocess.get("drop_nonmonotonic", True):
        od, frac = drop_nonmonotonic_trips(od)
        print(f"  [clean] dropped {frac:.1%} of trips for non-monotonic stop_time")
    od = add_segment_and_dwell(od)
    od = add_forward_headway(
        od,
        min_headway_seconds=int(cfg.preprocess.get("min_headway_seconds", 30)),
        headway_window=int(cfg.label.get("headway_window", 8)),
    )
    return od
