from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps (kept from your rich-feature script)
_HAS_SCIPY = False
_HAS_STATSMODELS = False
try:
    from scipy.optimize import curve_fit, OptimizeWarning  # type: ignore
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


def compute_label_thresholds(meta_df: pd.DataFrame) -> Dict[str, float]:
    # data-adaptive thresholds (do NOT depend on Is_Valid)
    df = meta_df.copy()
    if "too_sparse" in df.columns:
        sparse = df["too_sparse"]
        if sparse.dtype == bool:
            mask = ~sparse
        else:
            mask = pd.to_numeric(sparse, errors="coerce").fillna(0).astype(int) == 0
        df = df[mask].copy()
    if len(df) < 20:
        return {
            "max_slope_p99": 1.0,
            "lag_p90": 10.0,
            "noise_p95": 0.05,
            "min_growth": 0.02,
        }
    return {
        "max_slope_p99": float(df["max_slope"].quantile(0.99)),
        "lag_p90": float(df["lag_time"].quantile(0.90)),
        "noise_p95": float(df["noise_residual_std"].quantile(0.95)),
        "min_growth": 0.02,
    }


def assign_meta_label(row: pd.Series, thr: Dict[str, float]) -> str:
    # HARD invalid
    if bool(row.get("too_sparse", False)):
        return "Invalid"

    ms = row.get("max_slope", np.nan)
    if not np.isfinite(ms) or ms <= 0:
        return "Invalid"

    init = row.get("initial_OD", np.nan)
    mx = row.get("max_OD", np.nan)
    if np.isfinite(init) and np.isfinite(mx):
        if (mx - init) < thr["min_growth"]:
            return "Invalid"

    dip = row.get("dip_fraction", np.nan)
    if np.isfinite(dip) and dip > 0.5:
        return "Invalid"

    ldf = row.get("largest_drop_frac", np.nan)
    if np.isfinite(ldf) and ldf > 0.6:
        return "Invalid"

    if np.isfinite(ms) and ms > thr["max_slope_p99"]:
        return "Invalid"

    # SOFT flags
    soft = 0
    plateau = row.get("plateau_OD", np.nan)
    if not np.isfinite(plateau):
        soft += 1

    lag = row.get("lag_time", np.nan)
    if np.isfinite(lag) and lag > thr["lag_p90"]:
        soft += 1

    noise = row.get("noise_residual_std", np.nan)
    if np.isfinite(noise) and noise > thr["noise_p95"]:
        soft += 1

    if soft >= 2:
        return "Unsure"
    return "Valid"


def add_meta_label(meta: pd.DataFrame) -> pd.DataFrame:
    thr = compute_label_thresholds(meta)
    meta = meta.copy()
    meta["meta_label"] = meta.apply(lambda rr: assign_meta_label(rr, thr), axis=1)

    # place meta_label next to Is_Valid
    front = ["FileName", "Test Id", "Model Name", "Is_Valid", "meta_label", "too_sparse", "low_resolution", "had_outliers"]
    rest = [c for c in meta.columns if c not in front]
    return meta[front + rest]

