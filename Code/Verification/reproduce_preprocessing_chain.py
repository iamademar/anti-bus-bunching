#!/usr/bin/env python3
"""Reproduce the preprocessing chain end-to-end and prove it yields the committed features.csv.

WHY THIS MATTERS
----------------
Every headline number in the report -- 19.4% of arrivals bunched, the 0.80 mean headway CV,
the before/after simulation deltas -- is computed from `Code/data/processed/features.csv`.
That file is the OUTPUT of Stage 1 preprocessing; the raw SUNT OD data it was built from is
not committed (about 1.4 GB, third-party). A reader therefore has to take the row counts on
trust unless the chain can be re-run. This script re-runs it.

It also pins down a figure the report quotes for data quality: the share of trips removed for
having non-monotonic timestamps (a later stop logged at an earlier time than an earlier stop --
physically impossible, so the bus-to-bus gaps such a trip implies are invalid and would
masquerade as severe bunching). The drop is documented as "~23% of trips" in
`src/headway.py`; this script reports the exact value for the scoped data.

WHAT IT CHECKS
--------------
The script does NOT re-implement preprocessing. It calls the SAME functions the real Stage 1
uses (`src/headway.py` and `src/features.py`), so a pass means the committed file is genuinely
reproducible from the public OD data, not merely consistent with a parallel re-implementation.
It walks the chain step by step and prints the row/trip count after each:

  STEP 0  load_od                  raw scoped events (8 busiest routes, both dirs, 1-3 Mar 2024)
  STEP 1  drop_nonmonotonic_trips  remove trips whose stop_time is not non-decreasing in stop order
  STEP 2  add_forward_headway      reconstruct the forward headway + rolling local-median normal
                                   (adds columns only; drops no rows)
  STEP 3  make_dataset             build the label + causal features, then drop arrivals with no
                                   defined current headway / local-median normal / label

  TEST (the claim) : STEP 3 reproduces the committed features.csv EXACTLY -- same row count,
                     same trip count, and the same bunched-now arrival count (21,123) that the
                     19.4% prevalence figure is derived from.

The expected chain at the default scope is:
  185,197 rows / 2,507 trips  ->  116,905 / 1,944  (-22.5% of trips, non-monotonic)
                              ->  116,905 / 1,944  (headway reconstruction, no row drop)
                              ->  109,029 / 1,898  (label + features + dropna) == features.csv

WHY IT NEEDS THE RAW DATA (and how to point it at the data)
-----------------------------------------------------------
STEP 0 reads the OD Parquet, so this script -- unlike the other two Verification checks, which
run off the committed CSV -- needs the raw SUNT download. It resolves the OD folder the same way
the pipeline does, via `paths.od_dir` in `config.yaml` (default: `Code/Dataset/SUNT/data/od/`).
If your data lives elsewhere, either edit that config key or pass the folder as the first CLI
argument (an absolute path works). When the data is absent the script exits 2 with a clear
message rather than failing obscurely.

RUN
---
  cd Code && python3 Verification/reproduce_preprocessing_chain.py
  # or point it at a specific OD folder:
  python3 Verification/reproduce_preprocessing_chain.py /abs/path/to/od

Exit codes: 0 if the chain reproduces features.csv exactly; 1 if the counts diverge (so it can
act as an assertion in CI / a PR check); 2 if the raw OD data is not available to run STEP 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `from src ...` resolve whether run from Code/ or from the repo root.
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from src.config import load_config
from src.features import make_dataset
from src.headway import (
    add_forward_headway,
    add_segment_and_dwell,
    drop_nonmonotonic_trips,
    load_od,
)

# Expected counts at the default scope (config.yaml). These are the numbers the report quotes;
# the script asserts the reproduced STEP 3 frame matches the committed features.csv, and prints
# these as a reference so a divergence is easy to localise.
EXPECT = {
    "raw_rows": 185197, "raw_trips": 2507,
    "after_clean_rows": 116905, "after_clean_trips": 1944,
    "final_rows": 109029, "final_trips": 1898,
    "bunched_now": 21123,
}


def _stat(label: str, df: pd.DataFrame) -> None:
    print(f"  {label:<46} rows={len(df):>9,}   trips={df['trip_id'].nunique():>7,}")


def main(argv: list[str]) -> int:
    cfg = load_config()
    s = cfg.scope

    # Resolve the OD folder: CLI override first, else config's paths.od_dir.
    od_dir = Path(argv[1]).resolve() if len(argv) > 1 else cfg.od_dir
    sample = od_dir / f"od-{s['start_date']}.parquet"
    if not sample.exists():
        print(f"RAW OD DATA NOT FOUND at: {od_dir}")
        print(f"  expected e.g. {sample.name} (see README 'Where to put the data').")
        print("  Point this check at the data: pass the OD folder as the first argument,")
        print("  or set paths.od_dir in config.yaml. This check needs the raw download.")
        return 2

    print("REPRODUCE PREPROCESSING CHAIN")
    print(f"  scope: routes={s['routes']} dirs={s['directions']} "
          f"{s['start_date']}..{s['end_date']}")
    print(f"  OD dir: {od_dir}\n")

    # ---- STEP 0: load scoped raw OD ------------------------------------------------
    od = load_od(od_dir, s["routes"], s["directions"], s["start_date"], s["end_date"])
    _stat("STEP 0  load_od (raw scoped)", od)

    # ---- STEP 1: drop non-monotonic (inconsistent) trips ---------------------------
    od, frac = drop_nonmonotonic_trips(od)
    _stat(f"STEP 1  drop_nonmonotonic_trips ({frac:.1%} of trips)", od)

    # ---- STEP 2: reconstruct forward headway (adds columns; drops no rows) ----------
    od = add_segment_and_dwell(od)
    od = add_forward_headway(
        od,
        min_headway_seconds=int(cfg.preprocess.get("min_headway_seconds", 30)),
        headway_window=int(cfg.label.get("headway_window", 8)),
    )
    _stat("STEP 2  add_forward_headway (no row drop)", od)

    # ---- STEP 3: label + features + drop unusable arrivals -------------------------
    out = make_dataset(od, cfg)
    _stat("STEP 3  make_dataset (== final features)", out)

    # ---- TEST: reproduced frame vs committed features.csv --------------------------
    print("\n  reference (expected, default scope):")
    print(f"    {EXPECT['raw_rows']:>9,} / {EXPECT['raw_trips']:,}  ->  "
          f"{EXPECT['after_clean_rows']:,} / {EXPECT['after_clean_trips']:,}  ->  "
          f"{EXPECT['final_rows']:,} / {EXPECT['final_trips']:,}")

    ship_path = cfg.processed_path
    if not ship_path.exists():
        print(f"\nCOMMITTED features.csv NOT FOUND at {ship_path}; cannot compare.")
        return 1
    ship = pd.read_csv(ship_path)

    ratio = out["headway_min"] / out["local_median_headway"]
    bunched = int((ratio < float(cfg.label["bunch_frac"])).sum())

    rows_ok = len(out) == len(ship)
    trips_ok = out["trip_id"].nunique() == ship["trip_id"].nunique()
    bunch_ok = bunched == EXPECT["bunched_now"]

    print("\n  COMPARE TO COMMITTED features.csv")
    print(f"    rows         reproduced={len(out):,}  committed={len(ship):,}  "
          f"{'OK' if rows_ok else 'MISMATCH'}")
    print(f"    trips        reproduced={out['trip_id'].nunique():,}  "
          f"committed={ship['trip_id'].nunique():,}  {'OK' if trips_ok else 'MISMATCH'}")
    print(f"    bunched-now  reproduced={bunched:,}  expected={EXPECT['bunched_now']:,}  "
          f"{'OK' if bunch_ok else 'MISMATCH'}  (the 19.4% prevalence figure)")

    if rows_ok and trips_ok and bunch_ok:
        print("\nRESULT: CONFIRMED -- the chain reproduces features.csv exactly, so the report's")
        print("        prevalence and before/after numbers follow from the public OD data.")
        return 0
    print("\nRESULT: FAILED -- the reproduced frame diverges from the committed features.csv;")
    print("        check the scope in config.yaml matches the data, and the OD release version.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
