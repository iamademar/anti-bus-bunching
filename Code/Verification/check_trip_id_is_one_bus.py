#!/usr/bin/env python3
"""Prove that a SUNT `trip_id` is exactly ONE physical bus on ONE run.

WHY THIS MATTERS
----------------
Bus bunching is the gap between *two different* buses arriving at the same stop.
`add_forward_headway` (in src/headway.py) therefore keys the headway calculation on
`(route_short_name, direction_id, stop_id)` ordered by `stop_time` -- so the "previous
arrival" is the *other* bus just ahead in the queue at that stop.

A natural question is: why not key on `trip_id` instead? Because a `trip_id` is a single
bus's single run, so keying on it would compare a bus only to *itself* at its previous stop
(its travel time), and the gap to the following bus -- the bunching signal -- would never be
computed. This script proves the underlying fact: one `trip_id` == one bus.

In the SUNT Origin-Destination (OD) data the relevant columns are documented as:
  trip_id           - "Unique identifier for the trip."   (value = vehicle_route_run, e.g. 20194_1346_1)
  vehicle           - "Vehicle code."
  route_short_name  - "Bus line identifier."
  direction_id      - "Direction of the trip: 'I' (one-way) or 'V' (return)."
Source: SUNT data dictionary, https://github.com/LabIA-UFBA/SUNT (docs/datasets.md)

WHAT IT CHECKS
--------------
  TEST 1 (the claim) : every trip_id maps to exactly ONE vehicle (and one route, one direction).
  TEST 2 (the mirror): one vehicle appears under MANY trip_ids -> a bus does many runs a day,
                       which is why trip_id != vehicle.

Uses only the Python standard library (no pandas needed).

RUN
---
  cd Code && python3 Verification/check_trip_id_is_one_bus.py
  # or from the repo root:
  python3 Code/Verification/check_trip_id_is_one_bus.py
  # or point it at a specific file:
  python3 Verification/check_trip_id_is_one_bus.py path/to/features.csv

Exit code 0 if the claim holds, 1 otherwise (so it can act as an assertion in CI / a PR check).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

# Locations to try (relative to the current working directory) when no path is given,
# so the script works whether it's run from the repo root or from Code/.
CANDIDATE_PATHS = [
    "data/processed/features.csv",
    "Code/data/processed/features.csv",
    "../data/processed/features.csv",
]

REQUIRED_COLUMNS = ["trip_id", "vehicle", "route_short_name", "direction_id"]


def find_csv(explicit: str | None) -> str:
    """Return a readable path to features.csv, or exit with a helpful message."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        sys.exit(f"ERROR: file not found: {explicit}")

    # Search relative to CWD, and also relative to this script's own location.
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
        "    python3 Verification/check_trip_id_is_one_bus.py path/to/features.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to features.csv (auto-detected if omitted).")
    args = parser.parse_args()

    path = find_csv(args.csv_path)
    print(f"Reading: {path}\n")

    veh_per_trip = defaultdict(set)    # trip_id -> {vehicle}
    route_per_trip = defaultdict(set)  # trip_id -> {route_short_name}
    dir_per_trip = defaultdict(set)    # trip_id -> {direction_id}
    trips_per_veh = defaultdict(set)   # vehicle -> {trip_id}
    n_rows = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: {path} is missing required column(s): {missing}\n"
                     f"  Found columns: {reader.fieldnames}")
        for row in reader:
            n_rows += 1
            t, v = row["trip_id"], row["vehicle"]
            veh_per_trip[t].add(v)
            route_per_trip[t].add(row["route_short_name"])
            dir_per_trip[t].add(row["direction_id"])
            trips_per_veh[v].add(t)

    print(f"rows             : {n_rows:,}")
    print(f"distinct trip_id : {len(veh_per_trip):,}")
    print(f"distinct vehicle : {len(trips_per_veh):,}")
    print()

    # ---- TEST 1: one bus per trip ----------------------------------------------------
    max_veh = max(len(s) for s in veh_per_trip.values())
    bad_trips = [t for t, s in veh_per_trip.items() if len(s) > 1]
    max_route = max(len(s) for s in route_per_trip.values())
    max_dir = max(len(s) for s in dir_per_trip.values())
    test1_pass = (max_veh == 1 and max_route == 1 and max_dir == 1)

    print("TEST 1  one trip_id == one bus (one route, one direction)")
    print(f"  max distinct vehicles in any one trip_id : {max_veh}")
    print(f"  trip_ids with >1 vehicle                 : {len(bad_trips)}")
    print(f"  max distinct routes per trip_id          : {max_route}  (expect 1)")
    print(f"  max distinct directions per trip_id      : {max_dir}  (expect 1)")
    print(f"  => {'PASS' if test1_pass else 'FAIL'}: "
          f"every trip is exactly one bus = {test1_pass}")
    if bad_trips:
        print(f"  !! offending trip_ids (first 5): {bad_trips[:5]}")
    print()

    # ---- TEST 2: one bus -> many runs ------------------------------------------------
    counts = [len(s) for s in trips_per_veh.values()]
    mean_runs = sum(counts) / len(counts)
    print("TEST 2  one bus -> many runs over the day (so trip_id != vehicle)")
    print(f"  mean trip_ids per vehicle : {mean_runs:.1f}")
    print(f"  max  trip_ids per vehicle : {max(counts)}")
    print()

    # ---- Worked example: the busiest vehicle and its distinct runs -------------------
    busy = max(trips_per_veh, key=lambda v: len(trips_per_veh[v]))
    print(f"EXAMPLE  vehicle {busy} appears under {len(trips_per_veh[busy])} distinct "
          f"trip_ids (separate runs):")
    for t in sorted(trips_per_veh[busy])[:8]:
        route = next(iter(route_per_trip[t]))
        direction = next(iter(dir_per_trip[t]))
        print(f"    {t:<18} route={route} dir={direction}")
    print("  (note the value format: vehicle_route_run -- the naming encodes the proof)\n")

    # ---- Verdict ---------------------------------------------------------------------
    if test1_pass:
        print("RESULT: CONFIRMED -- a trip_id is one physical bus, so headway must be keyed on")
        print("        the stop (route, direction, stop_id), not on trip_id.")
        return 0
    print("RESULT: FAILED -- some trip_id spans more than one vehicle/route/direction; "
          "investigate the data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
