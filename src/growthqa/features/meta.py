from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.preprocess.transform import rolling_smooth

_HAS_SCIPY = False
try:
    from scipy.optimize import curve_fit, OptimizeWarning  # type: ignore
    import warnings

    warnings.filterwarnings("ignore", category=OptimizeWarning)
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _logistic_model(t, A, mu, lam):
    return A / (1.0 + np.exp(-mu * (t - lam)))


def _gompertz_model(t, A, mu, lam):
    inner = (mu * np.e / max(A, 1e-9)) * (lam - t) + 1.0
    inner = np.clip(inner, -50, 50)
    return A * np.exp(-np.exp(inner))


def _richards_model(t, A, mu, lam, nu):
    nu = max(float(nu), 1e-3)
    expo = np.clip(-mu * (t - lam), -50.0, 50.0)
    term = nu * np.exp(expo)
    log_denom = np.clip((1.0 / nu) * np.log1p(term), -50.0, 50.0)
    denom = np.exp(log_denom)
    return A / denom


def _flat_model(t, c):
    return np.full_like(t, float(c), dtype=float)


def _aic_from_residuals(resid: np.ndarray, k_params: int) -> float:
    resid = np.array(resid, dtype=float)
    n = int(np.sum(np.isfinite(resid)))
    if n <= 0:
        return np.nan
    rss = float(np.nansum(resid ** 2))
    rss = max(rss, 1e-12)
    return float(n * np.log(rss / n) + 2 * int(k_params))


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


def compute_late_growth_features(times: np.ndarray, ods: np.ndarray, start: float = 16.0) -> Dict[str, object]:
    t = np.array(times, dtype=float)
    y = np.array(ods, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]

    out = {
        "has_late_data": 0,
        "late_window_start": float(start),
        "late_tmax": np.nan,
        "late_n_points": 0,
        "late_slope": np.nan,
        "late_delta": np.nan,
        "late_max_increase": np.nan,
        "late_growth_detected": 0,
    }
    if t.size == 0:
        return out

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    late = t > float(start)
    if not np.any(late):
        return out

    t_l = t[late]
    y_l = y[late]
    out["has_late_data"] = 1
    out["late_tmax"] = float(np.nanmax(t_l))
    out["late_n_points"] = int(t_l.size)

    if t_l.size >= 2:
        dt = np.diff(t_l)
        dy = np.diff(y_l)
        good = dt > 1e-12
        if np.any(good):
            slopes = dy[good] / dt[good]
            out["late_slope"] = float(np.nanmedian(slopes))

    ref = (t >= float(start)) & (t <= float(start + 2.0))
    if np.any(ref):
        ref_med = float(np.nanmedian(y[ref]))
        k = max(1, int(np.ceil(0.2 * y_l.size)))
        tail_med = float(np.nanmedian(y_l[-k:]))
        out["late_delta"] = float(tail_med - ref_med)

    od_at_start = np.nan
    if t.size >= 2 and np.nanmin(t) <= float(start) <= np.nanmax(t):
        od_at_start = float(np.interp(float(start), t, y))
    if np.isfinite(od_at_start):
        out["late_max_increase"] = float(np.nanmax(y_l) - od_at_start)

    slope_thr = 0.01
    delta_thr = 0.03
    inc_thr = 0.05
    grow = False
    if np.isfinite(out["late_slope"]) and out["late_slope"] > slope_thr:
        grow = True
    if np.isfinite(out["late_delta"]) and out["late_delta"] > delta_thr:
        grow = True
    if np.isfinite(out["late_max_increase"]) and out["late_max_increase"] > inc_thr:
        grow = True
    out["late_growth_detected"] = int(grow)
    return out


def compute_features_from_row(row: pd.Series, *, rich_meta: bool = False) -> Dict[str, object]:
    time_cols = _time_cols_from_row(row)
    t_all = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    y_all = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(t_all) & np.isfinite(y_all)
    t = t_all[finite]
    od = y_all[finite]

    observed_tmax = float(np.nanmax(t)) if t.size else np.nan
    n_points_observed = int(t.size)

    if t.size >= 2:
        t_sorted = np.sort(t)
        dt = np.diff(t_sorted)
        max_gap_hours = float(np.nanmax(dt)) if dt.size else np.nan
        median_dt_hours = float(np.nanmedian(dt)) if dt.size else np.nan
    else:
        max_gap_hours = np.nan
        median_dt_hours = np.nan

    if np.isfinite(observed_tmax):
        in_range = (t_all >= 0) & (t_all <= observed_tmax)
        denom = int(np.sum(in_range))
        numer = int(np.sum(in_range & ~np.isfinite(y_all)))
        missing_frac_on_grid = float(numer / denom) if denom > 0 else np.nan
    else:
        missing_frac_on_grid = np.nan

    low_res_ctx = (
        (np.isfinite(median_dt_hours) and median_dt_hours > 0.5)
        or (n_points_observed > 0 and n_points_observed < 10)
        or bool(row.get("low_resolution", False))
    )
    sparse_ctx = (
        (np.isfinite(max_gap_hours) and max_gap_hours > 2.0)
        or (np.isfinite(missing_frac_on_grid) and missing_frac_on_grid > 0.25)
        or bool(row.get("too_sparse", False))
    )

    train_horizon = pd.to_numeric(pd.Series([row.get("train_horizon", np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(train_horizon):
        train_horizon = 16.0
    if np.isfinite(observed_tmax):
        is_censored = int(float(observed_tmax) < 16.0 - 1e-9)
    else:
        is_censored = int(float(train_horizon) < 16.0 - 1e-9)

    if t.size == 0:
        late = compute_late_growth_features(t_all, y_all, start=16.0)
        return {
            "train_horizon": float(train_horizon),
            "observed_tmax": observed_tmax,
            "is_censored": is_censored,
            "n_points_observed": n_points_observed,
            "max_gap_hours": max_gap_hours,
            "median_dt_hours": median_dt_hours,
            "missing_frac_on_grid": missing_frac_on_grid,
            "low_resolution": int(low_res_ctx),
            "too_sparse": int(sparse_ctx),
            "initial_OD": np.nan,
            "final_OD": np.nan,
            "max_OD": np.nan,
            "min_OD": np.nan,
            "range_OD": np.nan,
            "auc": np.nan,
            "auc_per_hour": np.nan,
            "net_change_per_hour": np.nan,
            "max_slope": np.nan,
            "time_of_max_slope": np.nan,
            "time_of_max_OD": np.nan,
            "monotonicity_fraction": np.nan,
            "largest_drop_frac": np.nan,
            "dip_fraction": np.nan,
            "roughness": np.nan,
            "noise_residual_std": np.nan,
            "lag_time": np.nan,
            "lag_time_est": np.nan,
            "plateau_OD": np.nan,
            "growth_phase_duration": np.nan,
            "symmetry_factor": np.nan,
            "final_to_peak_ratio": np.nan,
            "time_to_peak": np.nan,
            "num_slope_sign_changes": np.nan,
            "multi_phase_flag": np.nan,
            "logistic_fit_mse": np.nan,
            "logistic_AIC": np.nan,
            "gompertz_AIC": np.nan,
            "richards_AIC": np.nan,
            "flat_AIC": np.nan,
            "best_model_AIC": np.nan,
            "best_model_name": np.nan,
            **late,
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
    if dy.size > 0 and range_od > 1e-9 and np.any(dy < 0):
        largest_drop_frac = float(abs(np.nanmin(dy)) / range_od)
    dip_fraction = np.nan
    if dy.size > 0 and range_od > 1e-9:
        dip_fraction = float(np.sum(np.clip(-dy, 0, None)) / range_od)

    roughness = float(np.nanstd(dy)) if dy.size > 0 else np.nan
    noise_residual_std = np.nan
    if od.size >= 8:
        base = rolling_smooth(od, window=5)
        noise_residual_std = float(np.nanstd(od - base))

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
    final_to_peak_ratio = float(final_od / max_od) if max_od > 1e-12 else np.nan
    num_slope_sign_changes = 0
    if slopes.size and np.any(np.isfinite(slopes)):
        s = np.sign(slopes.copy())
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i - 1]
        nz = s[s != 0]
        if nz.size > 1:
            num_slope_sign_changes = int(np.sum(np.diff(nz) != 0))

    multi_phase_flag = False
    if od.size >= 7 and range_od > 1e-9:
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

    logistic_fit_mse = np.nan
    logistic_AIC = np.nan
    gompertz_AIC = np.nan
    richards_AIC = np.nan
    flat_AIC = np.nan
    best_model_AIC = np.nan
    best_model_name = np.nan
    if rich_meta and _HAS_SCIPY and t.size >= 5:
        fit_residuals: Dict[str, Tuple[np.ndarray, int]] = {}
        try:
            popt_log, _ = curve_fit(_logistic_model, t, od, p0=[max_od, 1.0, float(np.median(t))], maxfev=2000)
            resid = od - _logistic_model(t, *popt_log)
            logistic_fit_mse = float(np.mean(resid ** 2))
            fit_residuals["Logistic"] = (resid, 3)
        except Exception:
            pass
        try:
            popt_gom, _ = curve_fit(_gompertz_model, t, od, p0=[max_od, 1.0, float(np.median(t))], maxfev=2000)
            fit_residuals["Gompertz"] = (od - _gompertz_model(t, *popt_gom), 3)
        except Exception:
            pass
        try:
            popt_r, _ = curve_fit(_richards_model, t, od, p0=[max_od, 1.0, float(np.median(t)), 1.0], maxfev=4000)
            fit_residuals["Richards"] = (od - _richards_model(t, *popt_r), 4)
        except Exception:
            pass
        fit_residuals["Flat"] = (od - _flat_model(t, float(np.nanmean(od))), 1)

        logistic_AIC = _aic_from_residuals(*fit_residuals["Logistic"]) if "Logistic" in fit_residuals else np.nan
        gompertz_AIC = _aic_from_residuals(*fit_residuals["Gompertz"]) if "Gompertz" in fit_residuals else np.nan
        richards_AIC = _aic_from_residuals(*fit_residuals["Richards"]) if "Richards" in fit_residuals else np.nan
        flat_AIC = _aic_from_residuals(*fit_residuals["Flat"]) if "Flat" in fit_residuals else np.nan
        aics = {
            "Logistic": logistic_AIC,
            "Gompertz": gompertz_AIC,
            "Richards": richards_AIC,
            "Flat": flat_AIC,
        }
        finite_aics = {k: v for k, v in aics.items() if np.isfinite(v)}
        if finite_aics:
            best_model_name = min(finite_aics, key=finite_aics.get)
            best_model_AIC = float(finite_aics[best_model_name])

    late = compute_late_growth_features(t_all, y_all, start=16.0)

    return {
        "train_horizon": float(train_horizon),
        "observed_tmax": observed_tmax,
        "is_censored": int(is_censored),
        "n_points_observed": n_points_observed,
        "max_gap_hours": max_gap_hours,
        "median_dt_hours": median_dt_hours,
        "missing_frac_on_grid": missing_frac_on_grid,
        "low_resolution": int(low_res_ctx),
        "too_sparse": int(sparse_ctx),
        "initial_OD": initial_od,
        "final_OD": final_od,
        "max_OD": max_od,
        "min_OD": min_od,
        "range_OD": range_od,
        "auc": auc,
        "auc_per_hour": auc_per_hour,
        "net_change_per_hour": net_change_per_hour,
        "max_slope": max_slope,
        "time_of_max_slope": time_of_max_slope,
        "time_of_max_OD": time_of_max_od,
        "monotonicity_fraction": mono,
        "largest_drop_frac": largest_drop_frac,
        "dip_fraction": dip_fraction,
        "roughness": roughness,
        "noise_residual_std": noise_residual_std,
        "lag_time": lag_time,
        "lag_time_est": lag_time,
        "plateau_OD": plateau_od,
        "growth_phase_duration": growth_phase_duration,
        "symmetry_factor": symmetry_factor,
        "final_to_peak_ratio": final_to_peak_ratio,
        "time_to_peak": time_of_max_od,
        "num_slope_sign_changes": num_slope_sign_changes,
        "multi_phase_flag": int(bool(multi_phase_flag)),
        "logistic_fit_mse": logistic_fit_mse,
        "logistic_AIC": logistic_AIC,
        "gompertz_AIC": gompertz_AIC,
        "richards_AIC": richards_AIC,
        "flat_AIC": flat_AIC,
        "best_model_AIC": best_model_AIC,
        "best_model_name": best_model_name,
        **late,
    }


def build_metadata_from_wide(final_wide: pd.DataFrame, *, rich_meta: bool = False) -> pd.DataFrame:
    rows = []
    for _, r in final_wide.iterrows():
        feats = compute_features_from_row(r, rich_meta=rich_meta)
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
            "had_outliers": int(bool(r.get("had_outliers", False))),
            "source_type": source_type,
            "is_synthetic": int(is_synth),
            "base_curve_id": r.get("base_curve_id"),
            "aug_id": r.get("aug_id"),
            "tmax_original": r.get("tmax_original", np.nan),
            **feats,
        }
        rows.append(row)
    meta = pd.DataFrame(rows)
    if "Concentration" in meta.columns and meta["Concentration"].isna().all():
        meta = meta.drop(columns=["Concentration"])
    return meta
