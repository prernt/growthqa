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

_TIME_COL_RE = re.compile(r"^\s*T\s*([0-9]+(?:\.[0-9]+)?)\s*\(h\)\s*$", re.I)


def parse_time_from_header(col: str) -> Optional[float]:
    if not isinstance(col, str):
        return None
    m = _TIME_COL_RE.match(col.strip())
    if not m:
        return None
    return float(m.group(1))


def make_header_from_times(t_grid: np.ndarray) -> List[str]:
    headers: List[str] = []
    for t in t_grid:
        ts = f"{float(t):.2f}".rstrip("0").rstrip(".")
        if "." not in ts:
            ts = f"{ts}.0"
        headers.append(f"T{ts} (h)")
    return headers


def get_time_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if parse_time_from_header(str(c)) is not None:
            cols.append(str(c))
    return cols

def build_common_grid(all_times: np.ndarray, step_hours: float, tmax_hours: Optional[float]) -> np.ndarray:
    all_times = np.array(all_times, dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    if tmax_hours is not None:
        # Canonical mode: force 0..tmax even when observed curves are shorter.
        tmin = 0.0
        tmax = float(tmax_hours)
        if tmax < tmin:
            tmax = tmin
        n = int(np.floor((tmax - tmin) / step_hours + 1e-9)) + 1
        grid = tmin + step_hours * np.arange(n, dtype=float)
        if grid.size == 0:
            grid = np.array([0.0], dtype=float)
        return grid

    if all_times.size == 0:
        return np.array([0.0], dtype=float)

    tmin = float(np.nanmin(all_times))
    if tmin > 0:
        tmin = 0.0

    tmax = float(np.nanmax(all_times))
    n = int(np.floor((tmax - tmin) / step_hours + 1e-9)) + 1
    grid = tmin + step_hours * np.arange(n, dtype=float)
    if grid.size == 0:
        grid = np.array([0.0], dtype=float)
    return grid


def choose_auto_tmax(long_df: pd.DataFrame, coverage: float, user_cap: Optional[float]) -> Optional[float]:
    """
    Pick tmax such that >=coverage fraction of curves have observations up to that time.
    Helps prevent the tail being dominated by NaNs.
    """
    curve_max = long_df.groupby("Test Id")["time_h"].max().dropna().to_numpy(dtype=float)
    if curve_max.size == 0:
        return user_cap

    candidate_max = float(np.max(curve_max))
    if user_cap is not None:
        candidate_max = min(candidate_max, float(user_cap))

    sorted_unique = np.sort(np.unique(curve_max))
    sorted_unique = sorted_unique[sorted_unique <= candidate_max + 1e-9]
    if sorted_unique.size == 0:
        return candidate_max

    chosen = float(sorted_unique[0])
    for t_candidate in sorted_unique:
        frac = float(np.mean(curve_max >= t_candidate))
        if frac >= float(coverage):
            chosen = float(t_candidate)
        else:
            break
    return chosen

