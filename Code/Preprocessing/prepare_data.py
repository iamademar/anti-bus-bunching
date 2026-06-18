#!/usr/bin/env python
"""Stage 1: OD parquet -> processed features CSV.

Loads scoped OD, reconstructs per-bus trajectories and forward headway, builds the
causal features and the forward-looking bunching label, and writes a chronologically
ordered CSV ready for streaming.

Usage:
    python prepare_data.py [--config config.yaml]
"""
from __future__ import annotations

# Make the repo root (which holds src/, config.yaml, data/, outputs/, figures/) importable
# regardless of the current working directory, so `from src ...` resolves when run as
# `python Preprocessing/prepare_data.py` from the repo root.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from src.config import load_config
from src.headway import build_headway_table
from src.features import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[prepare] OD dir: {cfg.od_dir}")
    print(f"[prepare] scope: routes={cfg.scope['routes']} dirs={cfg.scope['directions']} "
          f"{cfg.scope['start_date']}..{cfg.scope['end_date']}")

    od = build_headway_table(cfg)
    ds = make_dataset(od, cfg)

    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(cfg.processed_path, index=False)

    counts = ds["label"].value_counts().sort_index().to_dict()
    print(f"[prepare] wrote {len(ds):,} instances -> {cfg.processed_path}")
    print(f"[prepare] class balance (0=ok,1=warn,2=bunching): {counts}")


if __name__ == "__main__":
    main()
