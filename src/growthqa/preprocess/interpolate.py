from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from growthqa.io.wide_loader import REQUIRED_META_COLS
from growthqa.preprocess.timegrid import (
    build_common_grid,
    get_time_columns,
    make_header_from_times,
    parse_time_from_header,
)


def _get_meta_cols(df_wide: pd.DataFrame) -> List[str]:
    cols = list(REQUIRED_META_COLS)
    if "Concentration" in df_wide.columns:
        cols.insert(3, "Concentration")
    for c in [
        "base_curve_id",
        "aug_id",
        "train_horizon",
        "tmax_original",
        "is_censored",
        "source_type",
        "is_synthetic",
        "gap_augmented",
        "gap_pattern",
    ]:
        if c in df_wide.columns and c not in cols:
            cols.append(c)
    return cols


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    tcols = get_time_columns(df_wide)
    meta_cols = _get_meta_cols(df_wide)
    long = df_wide.melt(
        id_vars=meta_cols,
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

    order = np.argsort(t)
    t = t[order]
    y = y[order]

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
                     tmax_hours: float) -> pd.DataFrame:
    long = wide_to_long(df_all_wide)

    t_grid = build_common_grid(step_hours=step_hours, tmax_hours=tmax_hours)
    time_headers = make_header_from_times(t_grid)

    rows = []
    meta_cols = _get_meta_cols(df_all_wide)
    grouped = long.groupby(meta_cols, sort=True, dropna=False)

    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(meta_cols, keys))
        t_src = grp["time_h"].to_numpy(dtype=float)
        y_src = grp["OD"].to_numpy(dtype=float)

        finite = np.isfinite(t_src) & np.isfinite(y_src)
        n_fin = int(np.sum(finite))

        too_sparse = n_fin < int(min_points)
        if n_fin >= 2:
            t_fin_sorted = np.sort(t_src[finite])
            raw_gaps = np.diff(t_fin_sorted)
            raw_max_gap_hours = float(np.max(raw_gaps)) if raw_gaps.size else 0.0
            raw_span_hours = float(t_fin_sorted[-1] - t_fin_sorted[0])
            expected_pts = int(round(raw_span_hours / float(step_hours))) + 1 if raw_span_hours > 0 else 1
            raw_missing_frac_on_grid = (
                float(max(0, expected_pts - n_fin) / expected_pts) if expected_pts > 0 else np.nan
            )
        elif n_fin == 1:
            raw_max_gap_hours = np.nan
            raw_missing_frac_on_grid = 0.0
        else:
            raw_max_gap_hours = np.nan
            raw_missing_frac_on_grid = np.nan

        y_grid = (
            interpolate_linear_no_extrap(t_src, y_src, t_grid)
            if not too_sparse else np.full_like(t_grid, np.nan, dtype=float)
        )

        row = {
            **meta,
            "too_sparse": bool(too_sparse),
            "n_points_observed_raw": int(n_fin),
            "max_gap_hours_raw": raw_max_gap_hours,
            "missing_frac_on_grid_raw": raw_missing_frac_on_grid,
        }
        for h, v in zip(time_headers, y_grid):
            row[h] = float(v) if np.isfinite(v) else np.nan

        rows.append(row)

    cols = meta_cols + [
        "too_sparse",
        "n_points_observed_raw", "max_gap_hours_raw", "missing_frac_on_grid_raw",
    ] + time_headers
    return pd.DataFrame(rows)[cols]