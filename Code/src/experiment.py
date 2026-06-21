"""CapyMOA prequential (test-then-train) run for the Anti-Bunching Copilot.

For each instance, in strict chronological order:
    1. the model PREDICTS the bunching class,
    2. we record the prediction,
    3. the model LEARNS from the true label.
This mirrors deployment and avoids any future leakage (no random train/test split).

Models: Hoeffding Adaptive Tree (+ built-in ADWIN) and Adaptive Random Forest.
A standalone ADWIN over the error stream logs drift points for the demo/figures.
Metrics are imbalance-aware (we care about recall/F1 on the rare 'bunching' class)
plus 'nudge lead time' — how early a warning fires before observed bunching.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_java_home
from .features import FEATURE_COLS, FEATURE_TRAIN

BUNCHING = 2  # class id


def _make_stream(X: np.ndarray, y: np.ndarray):
    from capymoa.stream import NumpyStream
    # Ensure all 3 classes appear so the inferred schema is stable even on tiny scopes:
    # the caller guarantees this by appending one synthetic row per missing class if needed.
    return NumpyStream(
        X.astype(float), y.astype(int),
        dataset_name="anti_bunching",
        feature_names=list(FEATURE_TRAIN),
        target_type="categorical",
    )


def _build_models(schema, cfg):
    from capymoa.classifier import HoeffdingAdaptiveTree, AdaptiveRandomForestClassifier
    m = cfg.model
    seed = int(m["random_seed"])
    return {
        # HAT has a built-in ADWIN drift detector; grace_period controls split frequency.
        "HAT": HoeffdingAdaptiveTree(
            schema=schema, random_seed=seed, grace_period=int(m["grace_period"]),
        ),
        # ARF: online bagging (lambda=6), random feature subspaces, per-tree drift adaptation.
        # Note: ARF tunes its base trees internally; ensemble_size is the key knob here.
        "ARF": AdaptiveRandomForestClassifier(
            schema=schema, random_seed=seed,
            ensemble_size=int(m["arf_ensemble_size"]),
            lambda_param=6.0,
        ),
    }


def run_prequential(df: pd.DataFrame, cfg) -> dict:
    """Run HAT + ARF prequentially. Returns per-instance predictions and rolling metrics."""
    ensure_java_home()
    from capymoa.drift.detectors import ADWIN

    X = df[FEATURE_TRAIN].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    n = len(df)

    # Schema stability: CapyMOA infers the label set from y. If a tiny scope happens to
    # miss a class, prepend one synthetic priming row per missing class (excluded from
    # reported metrics via `prime`), so the 3-class schema {0,1,2} always exists.
    present = set(np.unique(y).tolist())
    missing = [c for c in (0, 1, 2) if c not in present]
    prime = len(missing)
    if prime:
        X = np.vstack([np.zeros((prime, X.shape[1])), X])
        y = np.concatenate([np.array(missing, dtype=int), y])
        n = len(y)

    results = {"n_instances": n, "models": {}}
    # carry the columns the baselines need (headway_min, local_median_headway) so all
    # methods are scored on the SAME frame. local_median_headway may be absent if the
    # processed CSV dropped it; fall back to deriving it from headway_min/headway_ratio.
    carry = ["stop_time", "trip_id", "route_short_name", "direction_id",
             "stop_id", "pt_sequence", "headway_min", "label"]
    preds_frame = df[[c for c in carry if c in df.columns]].copy().reset_index(drop=True)
    if "local_median_headway" in df.columns:
        preds_frame["local_median_headway"] = df["local_median_headway"].to_numpy()
    elif {"headway_min", "headway_ratio"} <= set(df.columns):
        # local normal = headway_min / headway_ratio (recover what features.py used)
        ratio = df["headway_ratio"].replace(0, np.nan).to_numpy()
        preds_frame["local_median_headway"] = df["headway_min"].to_numpy() / ratio

    for name in ("HAT", "ARF"):
        stream = _make_stream(X, y)
        schema = stream.get_schema()
        model = _build_models(schema, cfg)[name]
        drift = ADWIN(delta=float(cfg.model["adwin_delta"]))

        preds = np.empty(n, dtype=int)
        correct_roll = []
        drift_points = []

        i = 0
        while stream.has_more_instances():
            inst = stream.next_instance()
            yhat = model.predict(inst)          # 1. predict
            yhat = 0 if yhat is None else int(yhat)
            preds[i] = yhat
            ytrue = int(inst.y_index)
            model.train(inst)                   # 3. learn
            # drift over the 0/1 error stream
            err = 0.0 if yhat == ytrue else 1.0
            drift.add_element(err)
            if drift.detected_change():
                drift_points.append(int(i))
            correct_roll.append(1 if yhat == ytrue else 0)
            i += 1

        # strip the synthetic priming rows so predictions align with the original df
        preds_frame[f"pred_{name}"] = preds[prime:]
        results["models"][name] = {
            "drift_points": [max(0, dp - prime) for dp in drift_points],
            "rolling_accuracy": _rolling(correct_roll[prime:],
                                         int(cfg.model["windowed_eval_size"])),
        }

    return {"summary": results, "predictions": preds_frame}


def bake_off(df: pd.DataFrame, cfg, models: list[str] | None = None) -> pd.DataFrame:
    """Prequentially compare several CapyMOA classifiers on the SAME stream.

    Returns a DataFrame ranked by bunching-class F1, with imbalance-aware metrics and
    throughput. Lets model choice be EMPIRICAL rather than inferred. Each model runs
    test-then-train in chronological order on identical data, so the comparison is fair.

    Candidates (override via `models`): NaiveBayes, KNN, HoeffdingTree, EFDT,
    HoeffdingAdaptiveTree (HAT), AdaptiveRandomForest (ARF). MajorityClass is added as a
    floor. Includes the non-learning 'static_threshold' baseline for reference.
    """
    import time
    ensure_java_home()
    from capymoa.classifier import (
        NaiveBayes, KNN, HoeffdingTree, EFDT,
        HoeffdingAdaptiveTree, AdaptiveRandomForestClassifier, MajorityClass,
    )
    from . import baselines as B

    X = df[FEATURE_TRAIN].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    seed = int(cfg.model["random_seed"])
    gp = int(cfg.model["grace_period"])
    ens = int(cfg.model["arf_ensemble_size"])

    # ensure all 3 classes exist for a stable schema
    present = set(np.unique(y).tolist())
    missing = [c for c in (0, 1, 2) if c not in present]
    prime = len(missing)
    if prime:
        X = np.vstack([np.zeros((prime, X.shape[1])), X])
        y = np.concatenate([np.array(missing, dtype=int), y])

    def factories(schema):
        d = {
            "MajorityClass": lambda: MajorityClass(schema=schema),
            "NaiveBayes": lambda: NaiveBayes(schema=schema, random_seed=seed),
            "KNN": lambda: KNN(schema=schema, random_seed=seed, k=5, window_size=1000),
            "HoeffdingTree": lambda: HoeffdingTree(schema=schema, random_seed=seed, grace_period=gp),
            "EFDT": lambda: EFDT(schema=schema, random_seed=seed, grace_period=gp),
            "HAT": lambda: HoeffdingAdaptiveTree(schema=schema, random_seed=seed, grace_period=gp),
            "ARF": lambda: AdaptiveRandomForestClassifier(
                schema=schema, random_seed=seed, ensemble_size=ens, lambda_param=6.0),
        }
        return d

    want = models or ["MajorityClass", "NaiveBayes", "KNN", "HoeffdingTree", "EFDT", "HAT", "ARF"]
    rows = []

    for name in want:
        stream = _make_stream(X, y)
        schema = stream.get_schema()
        model = factories(schema)[name]()
        preds = np.empty(len(y), dtype=int)
        t0 = time.perf_counter()
        i = 0
        while stream.has_more_instances():
            inst = stream.next_instance()
            yhat = model.predict(inst)
            preds[i] = 0 if yhat is None else int(yhat)
            model.train(inst)
            i += 1
        dt = time.perf_counter() - t0
        m = classification_metrics(y[prime:], preds[prime:])
        m["model"] = name
        m["sec"] = round(dt, 1)
        m["instances_per_sec"] = int(len(y) / dt) if dt > 0 else None
        rows.append(m)

    # reference: the non-learning static threshold baseline
    bt = classification_metrics(
        df["label"].to_numpy(),
        B.static_threshold(_baseline_frame(df), cfg.label["bunch_frac"], cfg.label["warn_frac"]))
    bt["model"] = "static_threshold (baseline)"
    bt["sec"] = 0.0
    bt["instances_per_sec"] = None
    rows.append(bt)

    cols = ["model", "f1_bunching", "recall_bunching", "precision_bunching",
            "balanced_accuracy", "kappa", "sec", "instances_per_sec"]
    out = pd.DataFrame(rows)[cols].sort_values("f1_bunching", ascending=False).reset_index(drop=True)
    return out


def _baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the columns the static-threshold baseline needs exist (headway_min, local_median_headway)."""
    if "local_median_headway" in df.columns:
        return df
    out = df.copy()
    if {"headway_min", "headway_ratio"} <= set(df.columns):
        ratio = df["headway_ratio"].replace(0, np.nan)
        out["local_median_headway"] = df["headway_min"] / ratio
    return out


def _rolling(bits: list[int], w: int) -> list[float]:
    a = np.asarray(bits, dtype=float)
    if len(a) == 0:
        return []
    c = np.cumsum(a)
    out = []
    for i in range(len(a)):
        lo = max(0, i - w + 1)
        out.append(float((c[i] - (c[lo - 1] if lo > 0 else 0)) / (i - lo + 1)))
    return out


# --- metrics ---------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Imbalance-aware metrics, focused on the rare 'bunching' class."""
    from sklearn.metrics import (
        precision_recall_fscore_support, balanced_accuracy_score, cohen_kappa_score,
    )
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision_bunching": float(p[BUNCHING]),
        "recall_bunching": float(r[BUNCHING]),
        "f1_bunching": float(f[BUNCHING]),
        "support_bunching": int((y_true == BUNCHING).sum()),
    }


def nudge_lead_time(pred_frame: pd.DataFrame, pred_col: str) -> dict:
    """Median minutes between a fired nudge (pred>=warning) and the next true bunching, per trip."""
    leads = []
    for _tid, g in pred_frame.groupby("trip_id", sort=False):
        g = g.sort_values("stop_time")
        times = g["stop_time"].to_numpy()
        fired = (g[pred_col].to_numpy() >= 1)
        is_bunch = (g["label"].to_numpy() == BUNCHING)
        bunch_idx = np.where(is_bunch)[0]
        for bi in bunch_idx:
            # most recent nudge at or before this bunching event
            earlier = np.where(fired[: bi + 1])[0]
            if len(earlier):
                dt = (times[bi] - times[earlier[0]]) / np.timedelta64(1, "m")
                if dt >= 0:
                    leads.append(float(dt))
    if not leads:
        return {"median_lead_min": None, "n": 0}
    return {"median_lead_min": float(np.median(leads)), "n": len(leads),
            "lead_times_min": leads}


def build_metrics_table(pred_frame: pd.DataFrame, cfg) -> dict:
    """Metrics for baselines + models, in one comparable structure (mirrors §6 Table 1)."""
    from . import baselines as B
    lab = cfg.label
    y = pred_frame["label"].to_numpy()

    table = {}
    # baselines computed on the same frame
    table["static_threshold"] = classification_metrics(
        y, B.static_threshold(pred_frame, lab["bunch_frac"], lab["warn_frac"]))
    table["previous_headway"] = classification_metrics(
        y, B.previous_headway_persistence(pred_frame, lab["bunch_frac"], lab["warn_frac"]))
    # models
    for name in ("HAT", "ARF"):
        col = f"pred_{name}"
        if col in pred_frame:
            m = classification_metrics(y, pred_frame[col].to_numpy())
            m["nudge_lead"] = nudge_lead_time(pred_frame, col)
            table[name] = m
    return table


def save_outputs(pred_frame: pd.DataFrame, summary: dict, metrics: dict, cfg) -> None:
    """Write the artifacts the FastAPI demo expects."""
    out = Path(cfg.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred_frame.to_csv(out / "predictions.csv", index=False)

    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # nudges = rows where the chosen model (ARF) fired a warning/bunching prediction
    pred_col = "pred_ARF" if "pred_ARF" in pred_frame else "pred_HAT"
    nudges = pred_frame[pred_frame[pred_col] >= 1][
        ["stop_time", "trip_id", "route_short_name", "direction_id", "stop_id", pred_col]
    ].copy()
    nudges = nudges.rename(columns={pred_col: "risk_class"})
    nudges.to_json(out / "nudges.json", orient="records", date_format="iso")

    drift = {name: summary["summary"]["models"][name]["drift_points"]
             for name in summary["summary"]["models"]}
    with open(out / "drift_events.json", "w") as fh:
        json.dump(drift, fh, indent=2)

    print(f"  [save] outputs -> {out}")
