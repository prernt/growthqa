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

from growthqa.preprocess.timegrid import parse_time_from_header


HAS_SCIPY = False
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


def baseline_correct_curve(
    t: np.ndarray,
    y: np.ndarray,
    blank_subtracted: bool,
    global_blank: Optional[float],
    clip_negatives: bool,
    early_hours: float = 0.5,
    min_early_points: int = 3,
    fallback_k: int = 5,
) -> np.ndarray:
    """
    Baseline (blank) subtraction.

    Priority:
      1) If blank_subtracted is False -> return y unchanged
      2) If global_blank is provided -> subtract that constant
      3) Else -> subtract robust baseline:
           - median of early-window points (t <= early_hours)
           - if too few points -> median of earliest fallback_k finite points

    Negative handling:
      - if clip_negatives True -> clamp corrected values at 0 (only for finite points)
    """
    if not blank_subtracted:
        return y

    tt = np.array(t, dtype=float)
    yy = np.array(y, dtype=float).copy()

    m = np.isfinite(tt) & np.isfinite(yy)
    if not np.any(m):
        return yy

    # 1) constant blank if given
    if global_blank is not None:
        yy[m] = yy[m] - float(global_blank)
    else:
        # 2) robust baseline from early window
        early_mask = m & (tt <= float(early_hours))

        if np.sum(early_mask) >= int(min_early_points):
            base = float(np.nanmedian(yy[early_mask]))
        else:
            # fallback: median of earliest K finite points
            idx = np.where(m)[0]
            idx_sorted = idx[np.argsort(tt[idx])]
            k = min(int(fallback_k), idx_sorted.size)
            base = float(np.nanmedian(yy[idx_sorted[:k]]))

        yy[m] = yy[m] - base

    if clip_negatives:
        yy[m] = np.maximum(yy[m], 0.0)

    return yy


def rolling_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    yy = y.copy()
    m = np.isfinite(yy)
    if np.sum(m) < window:
        return yy
    s = pd.Series(yy).ffill().bfill()
    filled = s.to_numpy(dtype=float)
    kernel = np.ones(int(window), dtype=float) / float(window)
    sm = np.convolve(filled, kernel, mode="same")
    sm[~m] = np.nan
    return sm


def lowess_smooth(t: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    # frac is fraction of data used (0..1)
    yy = y.copy()
    m = np.isfinite(yy) & np.isfinite(t)
    if np.sum(m) < 6:
        return yy
    if not _HAS_STATSMODELS:
        return yy

    tt = t[m]
    vv = yy[m]
    # return sorted, so we need to map back -> simplest: interpolate lowess result onto tt
    lw = lowess(vv, tt, frac=frac, return_sorted=True)
    tt2 = lw[:, 0]
    vv2 = lw[:, 1]
    yy_out = yy.copy()
    yy_out[m] = np.interp(tt, tt2, vv2)
    return yy_out


def savgol_smooth(y: np.ndarray, window: int) -> np.ndarray:
    yy = y.copy()
    if window < 3:
        return yy
    if window % 2 == 0:
        window += 1
    m = np.isfinite(yy)
    if np.sum(m) < window:
        return yy
    try:
        from scipy.signal import savgol_filter  # type: ignore
        s = pd.Series(yy).ffill().bfill()
        filled = s.to_numpy(dtype=float)
        sm = savgol_filter(filled, window_length=int(window), polyorder=2, mode="interp")
        sm[~m] = np.nan
        return sm
    except Exception:
        return rolling_smooth(yy, window=max(3, int(window)))


def smooth_curve(t: np.ndarray, y: np.ndarray, method: str, window: int) -> np.ndarray:
    method = (method or "NONE").upper()
    if method in {"NONE"}:
        return y
    if method in {"RAW"}:
        return rolling_smooth(y, window=int(window))
    if method in {"SGF"}:
        return savgol_smooth(y, window=int(window))
    if method in {"LWS"}:
        # map window to frac heuristically
        # window ~ number of points; frac = window/n
        m = np.isfinite(y)
        n = int(np.sum(m))
        if n <= 0:
            return y
        frac = min(1.0, max(0.2, float(window) / float(max(n, 1))))
        return lowess_smooth(t, y, frac=frac)
    return y


def normalize_curve(y: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "NONE").upper()
    yy = y.copy()
    m = np.isfinite(yy)
    if not np.any(m):
        return yy
    if mode == "NONE":
        return yy
    if mode == "MAX":
        mx = float(np.nanmax(yy[m]))
        if mx > 0:
            yy[m] = yy[m] / mx
        return yy
    if mode == "MINMAX":
        lo = float(np.nanmin(yy[m]))
        hi = float(np.nanmax(yy[m]))
        if hi > lo:
            yy[m] = (yy[m] - lo) / (hi - lo)
        return yy
    return yy


def preprocess_wide(raw_wide: pd.DataFrame,
                    blank_subtracted: bool,
                    global_blank: Optional[float],
                    smooth_method: str,
                    smooth_window: int,
                    clip_negatives: bool,
                    blank_status_map: Dict[str, Dict[str, object]],
                    blank_default: str,
                    normalize_mode: str) -> Tuple[pd.DataFrame, pd.Series]:
    # adjust blank_subtracted per-file if needed

    if blank_status_map is None:
        blank_status_map = {}

    base_meta_cols = [
        "FileName",
        "Test Id",
        "Model Name",
        "Is_Valid",
        "too_sparse",
        "low_resolution",
        "base_curve_id",
        "aug_id",
        "train_horizon",
        "tmax_original",
        "is_censored",
        "source_type",
        "is_synthetic",
    ]
    meta_cols = [c for c in base_meta_cols if c in raw_wide.columns]
    if "Concentration" in raw_wide.columns and "Concentration" not in meta_cols:
        meta_cols.append("Concentration")
    time_cols = [c for c in raw_wide.columns if parse_time_from_header(str(c)) is not None]
    t_grid = np.array([parse_time_from_header(c) for c in time_cols], dtype=float)

    out = raw_wide.copy()
    had_outliers = []

    # outlier flagging (light): count extreme z points after baseline (doesn't modify)
    for i in range(len(out)):
        y = pd.to_numeric(out.loc[i, time_cols], errors="coerce").to_numpy(dtype=float)

        # baseline correction first (for meaningful z-score)
        # y0 = baseline_correct_curve(
        #     y,
        #     blank_subtracted=blank_subtracted,
        #     global_blank=global_blank,
        #     clip_negatives=bool(clip_negatives),
        # )
        fname = str(out.loc[i, "FileName"]).strip()

        # Determine per-file rule
        entry = blank_status_map.get(fname)
        if entry is None:
            status = blank_default  # "RAW" or "ALREADY"
            per_file_blank = None
        else:
            status = str(entry["status"])
            per_file_blank = entry.get("blank_value")

        # Apply blank subtraction only if:
        #  - CLI requested blank subtraction (blank_subtracted=True)
        #  - AND this file is RAW (not already blank-subtracted)
        apply_blank = bool(blank_subtracted) and (status == "RAW")

        # Choose blank value priority:
        # per-file blank_value (if provided) > CLI global_blank > first-3-points baseline
        blank_value_to_use = per_file_blank if per_file_blank is not None else global_blank

        y0 = baseline_correct_curve(
        t_grid,
        y,
        blank_subtracted=apply_blank,
        global_blank=blank_value_to_use,
        clip_negatives=bool(clip_negatives),
        early_hours=0.5,        # recommended default
        min_early_points=3,
        fallback_k=5,
    )



        # outlier heuristic (flag only)
        m = np.isfinite(y0)
        flag = False
        if np.sum(m) >= 8:
            mu = float(np.nanmean(y0[m]))
            sd = float(np.nanstd(y0[m]))
            if sd > 1e-12:
                z = np.abs((y0[m] - mu) / sd)
                flag = bool(np.any(z > 6.0))
        had_outliers.append(flag)

        # now apply preprocessing
        # Apply transformations strictly on observed points to avoid borrowing
        # any information from NaN tails beyond the observed horizon.
        y3 = y0.copy()
        obs = np.isfinite(y0) & np.isfinite(t_grid)
        if np.any(obs):
            t_obs = t_grid[obs]
            y_obs = y0[obs]
            y_obs = smooth_curve(t_obs, y_obs, method=smooth_method, window=int(smooth_window))
            y_obs = normalize_curve(y_obs, mode=normalize_mode)
            y3[obs] = y_obs

        out.loc[i, time_cols] = y3

    out["had_outliers"] = pd.Series(had_outliers, index=out.index).astype(bool)

    # keep had_outliers adjacent to flags
    cols = meta_cols + ["had_outliers"] + time_cols
    return out[cols], out["had_outliers"]
