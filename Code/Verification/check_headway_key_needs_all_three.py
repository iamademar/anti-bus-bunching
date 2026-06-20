#!/usr/bin/env python3
"""Prove that headway MUST be keyed on all three: route + direction + stop.

WHY THIS MATTERS
----------------
`add_forward_headway` (in src/headway.py) computes `headway_min` by grouping arrivals on
`[route_short_name, direction_id, stop_id]`, ordering each group by `stop_time`, and taking the
gap to the previous arrival. A natural question (from Zaimei) is: why not key on the route alone?

Because headway is the gap between two buses that are genuinely one behind the other: same line,
same direction, AT THE SAME STOP. If you drop columns from the key, the "previous arrival" can be
a bus that is NOT a real predecessor, producing physically meaningless "headways":

  - Drop `stop_id`        -> you pair buses at DIFFERENT stops (e.g. one at stop 3, the next record
                             at stop 40). A gap between two different places is not a headway.
  - Drop `direction_id`   -> you pair an INBOUND bus with an OUTBOUND one at the same stop. Buses
                             going opposite ways are not following each other.

This script measures exactly how many such impossible pairs each wrong key creates, and shows the
correct key creates zero.

SUNT OD columns used (data dictionary, https://github.com/LabIA-UFBA/SUNT, docs/datasets.md):
  route_short_name - "Bus line identifier."
  direction_id     - "Direction of the trip: 'I' (one-way) or 'V' (return)."
  stop_id          - physical stop identifier.
  stop_time        - "Time at the stop."

WHAT IT CHECKS
--------------
For three grouping keys it counts, among consecutive (time-ordered) arrival pairs, how many are
at a DIFFERENT stop or in OPPOSITE directions (= impossible headways):
  WRONG KEY A : [route]                      -> expect almost all pairs are false
  WRONG KEY B : [route, direction]           -> still almost all at different stops
  CORRECT KEY : [route, direction, stop]     -> exactly 0 false pairs (the assertion)

Uses only the Python standard library (no pandas needed).

RUN
---
  cd Code && python3 Verification/check_headway_key_needs_all_three.py
  # or from the repo root:
  python3 Code/Verification/check_headway_key_needs_all_three.py
  # or point it at a specific file:
  python3 Verification/check_headway_key_needs_all_three.py path/to/features.csv

Exit code 0 if the correct key yields zero impossible pairs, 1 otherwise (so it can act as an
assertion in CI / a PR check).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

# Locations to try (relative to the current working directory) when no path is given,
# so the script works whether it's run from the repo root or from Code/.
CANDIDATE_PATHS = [
    "data/processed/features.csv",
    "Code/data/processed/features.csv",
    "../data/processed/features.csv",
]

REQUIRED_COLUMNS = ["route_short_name", "direction_id", "stop_id", "stop_time"]


def find_csv(explicit: str | None) -> str:
    """Return a readable path to features.csv, or exit with a helpful message."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        sys.exit(f"ERROR: file not found: {explicit}")

    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.getcwd(), here, os.path.join(here, "..")]
    for base in search_dirs:
        for rel in CANDIDATE_PATHS:
            cand = os.path.normpath(os.path.join(base, rel))
            if os.path.isfile(cand):
                return cand

    sys.exit(
        "ERROR: could not find features.csv automatically.\n"
        "  Tried: " + ", ".join(CANDIDATE_PATHS) + "\n"
        "  Run from the repo root or Code/, or pass the path explicitly:\n"
        "    python3 Verification/check_headway_key_needs_all_three.py path/to/features.csv"
    )


def consecutive_pairs(rows: list[dict], group_keys: list[str]):
    """Bucket rows by group_keys, sort each bucket by time, yield (prev, cur) consecutive pairs.

    This mimics how `add_forward_headway` finds "the previous arrival": group, sort by stop_time,
    look one back. Changing group_keys is exactly the experiment of using a different headway key.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in group_keys)].append(r)
    for g in buckets.values():
        g.sort(key=lambda r: r["t"])
        for prev, cur in zip(g, g[1:]):
            yield prev, cur


def pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.1f}%" if total else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to features.csv (auto-detected if omitted).")
    args = parser.parse_args()

    path = find_csv(args.csv_path)
    print(f"Reading: {path}\n")

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: {path} is missing required column(s): {missing}\n"
                     f"  Found columns: {reader.fieldnames}")
        for r in reader:
            rows.append({
                "route": r["route_short_name"],
                "dir":   r["direction_id"],
                "stop":  r["stop_id"],
                "t":     datetime.fromisoformat(r["stop_time"]),
            })

    print(f"rows: {len(rows):,}\n")

    def tally(group_keys):
        n = diff_stop = diff_dir = 0
        for prev, cur in consecutive_pairs(rows, group_keys):
            n += 1
            if prev["stop"] != cur["stop"]:
                diff_stop += 1
            if prev["dir"] != cur["dir"]:
                diff_dir += 1
        return n, diff_stop, diff_dir

    # ---- WRONG KEY A: route only ----------------------------------------------------
    nA, ds_A, dd_A = tally(["route"])
    print("WRONG KEY A:  group by [route] only")
    print(f"  consecutive pairs                   : {nA:,}")
    print(f"  pairs at a DIFFERENT stop (FALSE)    : {ds_A:,}  ({pct(ds_A, nA)})")
    print(f"  pairs in OPPOSITE directions (FALSE) : {dd_A:,}  ({pct(dd_A, nA)})")
    print()

    # ---- WRONG KEY B: route + direction (still no stop) -----------------------------
    nB, ds_B, dd_B = tally(["route", "dir"])
    print("WRONG KEY B:  group by [route, direction]  (still missing stop)")
    print(f"  consecutive pairs                   : {nB:,}")
    print(f"  pairs at a DIFFERENT stop (FALSE)    : {ds_B:,}  ({pct(ds_B, nB)})")
    print(f"  pairs in OPPOSITE directions         : {dd_B:,}  ({pct(dd_B, nB)})")
    print()

    # ---- CORRECT KEY: route + direction + stop --------------------------------------
    nC, ds_C, dd_C = tally(["route", "dir", "stop"])
    print("CORRECT KEY:  group by [route, direction, stop]")
    print(f"  consecutive pairs                   : {nC:,}")
    print(f"  pairs at a DIFFERENT stop            : {ds_C:,}  (must be 0)")
    print(f"  pairs in OPPOSITE directions         : {dd_C:,}  (must be 0)")
    print()

    # ---- Verdict --------------------------------------------------------------------
    ok = (ds_C == 0 and dd_C == 0)
    print(f"=> {'PASS' if ok else 'FAIL'}: the correct key produces "
          f"{ds_C} different-stop and {dd_C} opposite-direction pairs (both must be 0).")
    print()
    if ok:
        print("RESULT: CONFIRMED -- route alone is not enough. Keying on [route] would make")
        print(f"        {pct(ds_A, nA)} of 'headways' span different stops and {pct(dd_A, nA)} span")
        print("        opposite directions. Only [route, direction, stop] gives genuine headways.")
        return 0
    print("RESULT: FAILED -- the correct key produced impossible pairs; investigate the data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
