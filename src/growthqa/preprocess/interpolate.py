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

from growthqa.preprocess.timegrid import build_common_grid, choose_auto_tmax, get_time_columns, make_header_from_times, parse_time_from_header

REQUIRED_META_COLS = ["FileName", "Test Id", "Model Name", "Is_Valid"]


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    tcols = get_time_columns(df_wide)
    long = df_wide.melt(
        id_vars=REQUIRED_META_COLS,
        value_vars=tcols,
        var_name="time_col",
        value_name="OD",
    )
    long["time_h"] = long["time_col"].map(lambda c: parse_time_from_header(str(c)))
    long["OD"] = pd.to_numeric(long["OD"], errors="coerce")
    long["time_h"] = pd.to_numeric(long["time_h"], errors="coerce")
    long = long.drop(columns=["time_col"])
    long = long.dropna(subset=["time_h"])
    return long


def interpolate_linear_no_extrap(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_src = np.array(t_src, dtype=float)
    y_src = np.array(y_src, dtype=float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t = t_src[m]
    y = y_src[m]
    if t.size < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    # sort
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # de-duplicate times (mean duplicates)
    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size != t.size:
        y_acc = np.zeros_like(uniq_t, dtype=float)
        cnt = np.zeros_like(uniq_t, dtype=float)
        for i, u in enumerate(inv):
            y_acc[u] += y[i]
            cnt[u] += 1
        y = y_acc / np.maximum(cnt, 1.0)
        t = uniq_t

    if t.size < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    out = np.full_like(t_grid, np.nan, dtype=float)
    lo, hi = float(t[0]), float(t[-1])
    inside = (t_grid >= lo) & (t_grid <= hi)
    if np.any(inside):
        out[inside] = np.interp(t_grid[inside], t, y)
    return out


def build_raw_merged(df_all_wide: pd.DataFrame,
                     step_hours: float,
                     min_points: int,
                     tmax_hours: Optional[float],
                     auto_tmax: bool,
                     auto_tmax_coverage: float,
                     low_res_threshold: int) -> pd.DataFrame:
    long = wide_to_long(df_all_wide)
    all_times = long["time_h"].dropna().astype(float).to_numpy()

    eff_tmax = tmax_hours
    if auto_tmax:
        eff_tmax = choose_auto_tmax(long, coverage=auto_tmax_coverage, user_cap=tmax_hours)

    t_grid = build_common_grid(all_times, step_hours=step_hours, tmax_hours=eff_tmax)
    time_headers = make_header_from_times(t_grid)

    rows = []
    grouped = long.groupby(REQUIRED_META_COLS, sort=True, dropna=False)

    for (fname, tid, mname, is_valid), grp in grouped:
        t_src = grp["time_h"].to_numpy(dtype=float)
        y_src = grp["OD"].to_numpy(dtype=float)

        finite = np.isfinite(t_src) & np.isfinite(y_src)
        n_fin = int(np.sum(finite))

        too_sparse = n_fin < int(min_points)
        low_resolution = (n_fin >= int(min_points)) and (n_fin < int(low_res_threshold))

        y_grid = (
            interpolate_linear_no_extrap(t_src, y_src, t_grid)
            if not too_sparse else np.full_like(t_grid, np.nan, dtype=float)
        )

        row = {
            "FileName": fname,
            "Test Id": tid,
            "Model Name": mname,
            "Is_Valid": bool(is_valid),  # keep as-is
            "too_sparse": bool(too_sparse),
            "low_resolution": bool(low_resolution),
        }
        for h, v in zip(time_headers, y_grid):
            row[h] = float(v) if np.isfinite(v) else np.nan

        rows.append(row)

    cols = ["FileName", "Test Id", "Model Name", "Is_Valid", "too_sparse", "low_resolution"] + time_headers
    return pd.DataFrame(rows)[cols]

