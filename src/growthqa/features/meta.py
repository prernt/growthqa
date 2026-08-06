from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.preprocess.transform import rolling_smooth


def _safe_label(v):
    if pd.isna(v):
        return pd.NA
    try:
        iv = int(v)
        return 1 if iv != 0 else 0
    except Exception:
        return pd.NA


def _time_cols_from_row(row: pd.Series) -> List[str]:
    cols = [c for c in row.index if parse_time_from_header(str(c)) is not None]
    return sorted(cols, key=lambda c: parse_time_from_header(str(c)) or 0.0)


def compute_features_from_row(row: pd.Series) -> Dict[str, object]:
    time_cols = _time_cols_from_row(row)
    t_all = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    y_all = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(t_all) & np.isfinite(y_all)
    t = t_all[finite]
    od = y_all[finite]

    observed_tmax = float(np.nanmax(t)) if t.size else np.nan
    n_points_observed_grid = int(t.size)
    if t.size >= 2:
        t_sorted = np.sort(t)
        dt = np.diff(t_sorted)
        max_gap_hours_grid = float(np.nanmax(dt)) if dt.size else np.nan
        median_dt_hours = float(np.nanmedian(dt)) if dt.size else np.nan
    else:
        max_gap_hours_grid = np.nan
        median_dt_hours = np.nan

    if np.isfinite(observed_tmax):
        in_range = (t_all >= 0) & (t_all <= observed_tmax)
        denom = int(np.sum(in_range))
        numer = int(np.sum(in_range & ~np.isfinite(y_all)))
        missing_frac_on_grid_calc = float(numer / denom) if denom > 0 else np.nan
    else:
        missing_frac_on_grid_calc = np.nan

    _raw_n = row.get("n_points_observed_raw", None)
    _raw_gap = row.get("max_gap_hours_raw", None)
    _raw_missing = row.get("missing_frac_on_grid_raw", None)

    n_points_observed = int(_raw_n) if _raw_n is not None and pd.notna(_raw_n) else n_points_observed_grid
    max_gap_hours = float(_raw_gap) if _raw_gap is not None and pd.notna(_raw_gap) else max_gap_hours_grid
    missing_frac_on_grid = (
        float(_raw_missing) if _raw_missing is not None and pd.notna(_raw_missing) else missing_frac_on_grid_calc
    )
    sparse_ctx = bool(row.get("too_sparse", False))
    grid_resolution_mismatch = (
        (np.isfinite(median_dt_hours) and median_dt_hours > 0.5)
        or (np.isfinite(max_gap_hours) and max_gap_hours > 2.0)
        or (np.isfinite(missing_frac_on_grid) and missing_frac_on_grid > 0.25)
    )
    train_horizon = pd.to_numeric(pd.Series([row.get("train_horizon", np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(train_horizon):
        train_horizon = 16.0
    if np.isfinite(observed_tmax):
        is_censored = int(float(observed_tmax) < 16.0 - 1e-9)
    else:
        is_censored = int(float(train_horizon) < 16.0 - 1e-9)

    if t.size == 0:
        return {
            "train_horizon": float(train_horizon),
            "observed_tmax": observed_tmax,
            "is_censored": is_censored,
            "n_points_observed": n_points_observed,
            "max_gap_hours": max_gap_hours,
            "missing_frac_on_grid": missing_frac_on_grid,
            "too_sparse": int(sparse_ctx),
            "grid_resolution_mismatch": int(grid_resolution_mismatch),
            "initial_OD": np.nan,
            "final_OD": np.nan,
            "auc": np.nan,
            "auc_per_hour": np.nan,
            "net_change_per_hour": np.nan,
            "max_slope": np.nan,
            "time_of_max_slope": np.nan,
            "time_of_max_OD": np.nan,
            "monotonicity_fraction": np.nan,
            "largest_drop_frac": np.nan,
            "roughness": np.nan,
            "noise_residual_std": np.nan,
            "noise_residual_std_is_fallback": np.nan,
            "lag_time_est": np.nan,
            "plateau_OD": np.nan,
            "growth_phase_duration": np.nan,
            "symmetry_factor": np.nan,
            "num_slope_sign_changes": np.nan,
            "multi_phase_flag": np.nan,
        }

    order = np.argsort(t)
    t = t[order]
    od = od[order]
    initial_od = float(od[0])
    final_od = float(od[-1])
    max_od = float(np.nanmax(od))
    min_od = float(np.nanmin(od))
    range_od = float(max_od - min_od)

    if t.size >= 2:
        auc = float(np.trapezoid(od, t)) if hasattr(np, "trapezoid") else float(np.trapz(od, t))
        dy = np.diff(od)
        dt = np.diff(t)
        slopes = np.full_like(dy, np.nan, dtype=float)
        good = dt > 1e-12
        slopes[good] = dy[good] / dt[good]
    else:
        auc = 0.0
        dy = np.array([], dtype=float)
        dt = np.array([], dtype=float)
        slopes = np.array([], dtype=float)
    auc_per_hour = float(auc / observed_tmax) if np.isfinite(observed_tmax) and observed_tmax > 0 else np.nan
    net_change_per_hour = (
        float((final_od - initial_od) / observed_tmax)
        if np.isfinite(observed_tmax) and observed_tmax > 0
        else np.nan
    )

    max_slope = np.nan
    time_of_max_slope = np.nan
    if slopes.size > 0 and np.any(np.isfinite(slopes)):
        i = int(np.nanargmax(slopes))
        max_slope = float(slopes[i])
        time_of_max_slope = float(t[i + 1]) if i + 1 < t.size else float(t[-1])

    i_max = int(np.nanargmax(od))
    time_of_max_od = float(t[i_max])
    mono = float(np.mean(dy > 0)) if dy.size > 0 else np.nan
    largest_drop_frac = np.nan
    if dy.size > 0:
        if range_od > 1e-9:
            largest_drop_frac = float(abs(np.nanmin(dy)) / range_od) if np.any(dy < 0) else 0.0
        else:
            largest_drop_frac = 0.0  # flat curve: no variation, no drop possible

    roughness = float(np.nanstd(dy)) if dy.size > 0 else np.nan
    noise_residual_std = np.nan
    noise_residual_std_is_fallback = False
    
    if n_points_observed >= 8:
        base = rolling_smooth(od, window=5)
        noise_residual_std = float(np.nanstd(od - base))
    elif n_points_observed >= 4:
        d = np.diff(od)
        if d.size > 0:
            noise_residual_std = float(np.nanstd(d) / np.sqrt(2.0))
            noise_residual_std_is_fallback = True

    lag_time = np.nan
    if range_od > 1e-9:
        thr = initial_od + 0.10 * range_od
        idx = np.where(od >= thr)[0]
        if idx.size:
            lag_time = float(t[idx[0]])

    plateau_od = float(np.nanmedian(od[-max(2, int(np.ceil(0.15 * od.size))):])) if od.size else np.nan
    growth_phase_duration = np.nan
    if range_od > 1e-9:
        thr10 = initial_od + 0.10 * range_od
        thr90 = initial_od + 0.90 * range_od
        i10 = np.where(od >= thr10)[0]
        i90 = np.where(od >= thr90)[0]
        if i10.size and i90.size:
            t10 = float(t[i10[0]])
            t90 = float(t[i90[0]])
            if t90 >= t10:
                growth_phase_duration = float(t90 - t10)

    total_dur = float(t[-1] - t[0]) if t.size >= 2 else np.nan
    symmetry_factor = (
        float((time_of_max_od - t[0]) / total_dur) if np.isfinite(total_dur) and total_dur > 1e-12 else np.nan
    )
    num_slope_sign_changes = 0
    if slopes.size and np.any(np.isfinite(slopes)):
        s = np.sign(slopes.copy())
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i - 1]
        nz = s[s != 0]
        if nz.size > 1:
            num_slope_sign_changes = int(np.sum(np.diff(nz) != 0))

    multi_phase_flag = np.nan
    
    if n_points_observed >= 7:
        multi_phase_flag = False
        if range_od > 1e-9:
            local_max_idxs = []
            for i in range(1, len(od) - 1):
                if od[i] >= od[i - 1] and od[i] >= od[i + 1]:
                    local_max_idxs.append(i)
            if len(local_max_idxs) >= 2:
                p1, p2 = local_max_idxs[0], local_max_idxs[-1]
                if (od[p1] - initial_od) > 0.2 * range_od and (od[p2] - initial_od) > 0.2 * range_od:
                    mid_min = float(np.min(od[p1:p2 + 1]))
                    if (max_od - mid_min) > 0.2 * range_od:
                        multi_phase_flag = True
    return {
        "train_horizon": float(train_horizon),
        "observed_tmax": observed_tmax,
        "is_censored": int(is_censored),
        "n_points_observed": n_points_observed,
        "max_gap_hours": max_gap_hours,
        "missing_frac_on_grid": missing_frac_on_grid,
        "too_sparse": int(sparse_ctx),
        "grid_resolution_mismatch": int(grid_resolution_mismatch),
        "initial_OD": initial_od,
        "final_OD": final_od,
        "auc": auc,
        "auc_per_hour": auc_per_hour,
        "net_change_per_hour": net_change_per_hour,
        "max_slope": max_slope,
        "time_of_max_slope": time_of_max_slope,
        "time_of_max_OD": time_of_max_od,
        "monotonicity_fraction": mono,
        "largest_drop_frac": largest_drop_frac,
        "roughness": roughness,
        "noise_residual_std": noise_residual_std,
        "noise_residual_std_is_fallback": int(noise_residual_std_is_fallback),
        "lag_time_est": lag_time,
        "plateau_OD": plateau_od,
        "growth_phase_duration": growth_phase_duration,
        "symmetry_factor": symmetry_factor,
        "num_slope_sign_changes": num_slope_sign_changes,
        "multi_phase_flag": (np.nan if pd.isna(multi_phase_flag) else int(bool(multi_phase_flag))),
    }


def build_metadata_from_wide(final_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in final_wide.iterrows():
        feats = compute_features_from_row(r)
        source_type = str(r.get("source_type", "")).strip().lower()
        if source_type not in {"synthetic", "lab"}:
            fname = str(r.get("FileName", "")).lower()
            source_type = "synthetic" if ("syn" in fname or "synthetic" in fname) else "lab"
        is_synth = r.get("is_synthetic", np.nan)
        is_synth = int(is_synth) if pd.notna(is_synth) else int(source_type == "synthetic")

        row = {
            "FileName": r.get("FileName"),
            "Test Id": r.get("Test Id"),
            "Model Name": r.get("Model Name"),
            "Concentration": r.get("Concentration", np.nan),
            "Is_Valid": _safe_label(r.get("Is_Valid", pd.NA)),
            "source_type": source_type,
            "is_synthetic": int(is_synth),
            "base_curve_id": r.get("base_curve_id"),
            "aug_id": r.get("aug_id"),
            "tmax_original": r.get("tmax_original", np.nan),
            "gap_augmented": int(pd.to_numeric(pd.Series([r.get("gap_augmented", 0)]), errors="coerce").fillna(0).iloc[0]),
            "gap_pattern": r.get("gap_pattern", None),
            **feats,
        }
        rows.append(row)
    meta = pd.DataFrame(rows)
    if "Concentration" in meta.columns and meta["Concentration"].isna().all():
        meta = meta.drop(columns=["Concentration"])
    return meta