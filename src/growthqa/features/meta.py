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
from growthqa.preprocess.transform import rolling_smooth

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


def _logistic_model(t, A, mu, lam):
    return A / (1.0 + np.exp(-mu * (t - lam)))


def _gompertz_model(t, A, mu, lam):
    inner = (mu * np.e / A) * (lam - t) + 1.0
    inner = np.clip(inner, -50, 50)
    return A * np.exp(-np.exp(inner))


def _richards_model(t, A, mu, lam, nu):
    nu = max(nu, 1e-3)  # avoid zero
    expo = np.clip(-mu * (t - lam), -50.0, 50.0)  # prevent overflow in exp
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
    if rss <= 1e-12:
        rss = 1e-12
    # AIC = n*ln(rss/n) + 2k
    return float(n * np.log(rss / n) + 2 * int(k_params))


def compute_features_from_row(row: pd.Series) -> Dict[str, object]:
    meta_cols = {"FileName", "Test Id", "Model Name", "Is_Valid", "too_sparse", "low_resolution", "had_outliers"}
    time_cols = [c for c in row.index if c not in meta_cols and parse_time_from_header(str(c)) is not None]
    time_cols = sorted(time_cols, key=lambda c: parse_time_from_header(str(c)) or 0.0)

    t = np.array([parse_time_from_header(c) for c in time_cols], dtype=float)
    od = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t) & np.isfinite(od)
    t = t[m]
    od = od[m]

    if t.size == 0:
        return {k: np.nan for k in [
            "initial_OD", "plateau_OD", "max_slope", "dip_fraction", "lag_time", "auc",
            "growth_phase_duration", "symmetry_factor", "max_OD", "final_OD",
            "final_to_peak_ratio", "time_to_peak", "noise_residual_std",
            "num_slope_sign_changes", "largest_drop_frac",
            "logistic_fit_mse", "multi_phase_flag",
            "logistic_AIC", "gompertz_AIC", "richards_AIC", "flat_AIC",
            "best_model_AIC", "best_model_name"
        ]}

    # basic stats
    initial_od = float(od[0])
    final_od = float(od[-1])
    max_od = float(np.nanmax(od))
    delta = float(max_od - initial_od)

    # AUC
    if t.size >= 2:
        # Use trapezoidal rule; fall back to trapz for older NumPy without np.trapezoid
        auc = float(np.trapezoid(od, t)) if hasattr(np, "trapezoid") else float(np.trapz(od, t))
    else:
        auc = 0.0

    # slopes
    if t.size >= 3:
        dy = np.diff(od)
        dt = np.diff(t)
        slopes = np.full_like(dy, np.nan, dtype=float)
        good = dt > 1e-12
        slopes[good] = dy[good] / dt[good]
        max_slope = float(np.nanmax(slopes)) if np.any(np.isfinite(slopes)) else np.nan
    elif t.size == 2:
        dt = float(t[1] - t[0])
        max_slope = float((od[1] - od[0]) / dt) if dt > 1e-12 else np.nan
        slopes = np.array([max_slope], dtype=float)
        dy = np.diff(od)
    else:
        max_slope = np.nan
        slopes = np.array([], dtype=float)
        dy = np.array([], dtype=float)

    # plateau (median of last 15% points if possible)
    if t.size >= 10:
        k = max(2, int(np.ceil(0.15 * t.size)))
        plateau_od = float(np.nanmedian(od[-k:]))
    else:
        plateau_od = float(od[-1])

    # dip_fraction (total drop relative to delta)
    dip_fraction = np.nan
    if t.size >= 3 and delta > 1e-6:
        drops = np.clip(-(np.diff(od)), 0, None)
        dip_fraction = float(np.nansum(drops) / delta)

    # lag time (first reach 10% of delta)
    lag_time = np.nan
    if delta > 1e-6:
        thresh = initial_od + 0.10 * delta
        idx = np.where(od >= thresh)[0]
        if idx.size > 0:
            lag_time = float(t[idx[0]])

    # growth phase duration (rough: time from 10% to 90% of delta)
    growth_duration = np.nan
    if delta > 1e-6:
        t10 = np.nan
        t90 = np.nan
        thr10 = initial_od + 0.10 * delta
        thr90 = initial_od + 0.90 * delta
        i10 = np.where(od >= thr10)[0]
        i90 = np.where(od >= thr90)[0]
        if i10.size > 0:
            t10 = float(t[i10[0]])
        if i90.size > 0:
            t90 = float(t[i90[0]])
        if np.isfinite(t10) and np.isfinite(t90) and t90 >= t10:
            growth_duration = float(t90 - t10)

    # symmetry factor (rough: time-to-peak / total duration)
    symmetry_factor = np.nan
    time_to_peak = np.nan
    if t.size >= 2:
        peak_idx = int(np.nanargmax(od))
        time_to_peak = float(t[peak_idx])
        total_dur = float(t[-1] - t[0]) if t.size >= 2 else np.nan
        if np.isfinite(total_dur) and total_dur > 1e-12:
            symmetry_factor = float((time_to_peak - t[0]) / total_dur)

    final_to_peak_ratio = np.nan
    if max_od > 1e-12:
        final_to_peak_ratio = float(final_od / max_od)

    # noise residual std (against rolling smooth baseline)
    noise_residual_std = np.nan
    if od.size >= 8:
        base = rolling_smooth(od, window=5)
        resid = od - base
        noise_residual_std = float(np.nanstd(resid))

    # slope sign changes
    num_slope_sign_changes = 0
    if slopes.size > 0 and np.any(np.isfinite(slopes)):
        sign_slopes = np.sign(slopes)
        for i in range(1, len(sign_slopes)):
            if sign_slopes[i] == 0:
                sign_slopes[i] = sign_slopes[i - 1]
        nonzero = sign_slopes[sign_slopes != 0]
        if nonzero.size > 1:
            num_slope_sign_changes = int(np.sum(np.diff(nonzero) != 0))

    # largest drop frac
    largest_drop_frac = np.nan
    if delta > 1e-6 and dy.size > 0:
        if np.any(dy < 0):
            largest_drop = float(np.min(dy))  # negative
            largest_drop_frac = float(abs(largest_drop) / delta)

    # --- Parametric fits + AIC (kept from your older file) ---
    logistic_fit_mse = np.nan
    logistic_AIC = np.nan
    gompertz_AIC = np.nan
    richards_AIC = np.nan
    flat_AIC = np.nan
    best_model_AIC = np.nan
    best_model_name = None

    if _HAS_SCIPY and t.size >= 5:
        fit_residuals: Dict[str, Tuple[np.ndarray, int]] = {}

        # Logistic
        try:
            A0 = max_od
            mu0 = 1.0
            grad = np.gradient(od)
            lam0 = float(t[np.argmax(grad)]) if np.any(np.isfinite(grad)) else float(np.median(t))
            popt_log, _ = curve_fit(_logistic_model, t, od, p0=[A0, mu0, lam0], maxfev=2000)
            pred_log = _logistic_model(t, *popt_log)
            resid_log = od - pred_log
            logistic_fit_mse = float(np.mean(resid_log ** 2))
            fit_residuals["Logistic"] = (resid_log, 3)
        except Exception:
            pass

        # Gompertz
        try:
            A0 = max_od
            mu0 = 1.0
            lam0 = float(np.median(t))
            popt_gom, _ = curve_fit(_gompertz_model, t, od, p0=[A0, mu0, lam0], maxfev=2000)
            pred_gom = _gompertz_model(t, *popt_gom)
            resid_gom = od - pred_gom
            fit_residuals["Gompertz"] = (resid_gom, 3)
        except Exception:
            pass

        # Richards
        try:
            A0 = max_od
            mu0 = 1.0
            lam0 = float(np.median(t))
            nu0 = 1.0
            popt_rich, _ = curve_fit(_richards_model, t, od, p0=[A0, mu0, lam0, nu0], maxfev=4000)
            pred_rich = _richards_model(t, *popt_rich)
            resid_rich = od - pred_rich
            fit_residuals["Richards"] = (resid_rich, 4)
        except Exception:
            pass

        # Flat
        baseline_hat = float(np.nanmean(od)) if np.any(np.isfinite(od)) else 0.0
        pred_flat = _flat_model(t, baseline_hat)
        resid_flat = od - pred_flat
        fit_residuals["Flat"] = (resid_flat, 1)

        if "Logistic" in fit_residuals:
            logistic_AIC = _aic_from_residuals(*fit_residuals["Logistic"])
        if "Gompertz" in fit_residuals:
            gompertz_AIC = _aic_from_residuals(*fit_residuals["Gompertz"])
        if "Richards" in fit_residuals:
            richards_AIC = _aic_from_residuals(*fit_residuals["Richards"])
        if "Flat" in fit_residuals:
            flat_AIC = _aic_from_residuals(*fit_residuals["Flat"])

        aic_map = {
            "Logistic": logistic_AIC,
            "Gompertz": gompertz_AIC,
            "Richards": richards_AIC,
            "Flat": flat_AIC,
        }
        finite_aics = {k: v for k, v in aic_map.items() if np.isfinite(v)}
        if finite_aics:
            best_model_name = min(finite_aics, key=finite_aics.get)
            best_model_AIC = float(finite_aics[best_model_name])

    # --- heuristic multi-phase detection (kept) ---
    multi_phase_flag = False
    if len(od) >= 7 and delta > 1e-6:
        local_max_idxs = []
        for i in range(1, len(od) - 1):
            if od[i] >= od[i - 1] and od[i] >= od[i + 1]:
                local_max_idxs.append(i)
        if len(local_max_idxs) >= 2:
            p1, p2 = local_max_idxs[0], local_max_idxs[-1]
            od1, od2 = od[p1], od[p2]
            if (od1 - initial_od) > 0.2 * delta and (od2 - initial_od) > 0.2 * delta:
                mid_min = float(np.min(od[p1:p2 + 1]))
                if (max_od - mid_min) > 0.2 * delta:
                    multi_phase_flag = True

    return {
        "initial_OD": initial_od,
        "plateau_OD": plateau_od,
        "max_slope": max_slope,
        "dip_fraction": dip_fraction,
        "lag_time": lag_time,
        "auc": auc,
        "growth_phase_duration": growth_duration,
        "symmetry_factor": symmetry_factor,
        "max_OD": max_od,
        "final_OD": final_od,
        "final_to_peak_ratio": final_to_peak_ratio,
        "time_to_peak": time_to_peak,
        "noise_residual_std": noise_residual_std,
        "num_slope_sign_changes": num_slope_sign_changes,
        "largest_drop_frac": largest_drop_frac,
        "logistic_fit_mse": logistic_fit_mse,
        "multi_phase_flag": bool(multi_phase_flag),
        "logistic_AIC": logistic_AIC,
        "gompertz_AIC": gompertz_AIC,
        "richards_AIC": richards_AIC,
        "flat_AIC": flat_AIC,
        "best_model_AIC": best_model_AIC,
        "best_model_name": best_model_name,
    }


def build_metadata_from_wide(final_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in final_wide.iterrows():
        feats = compute_features_from_row(r)
        rows.append({
            "FileName": r["FileName"],
            "Test Id": r["Test Id"],
            "Model Name": r["Model Name"],
            "Is_Valid": bool(r["Is_Valid"]),
            "too_sparse": bool(r.get("too_sparse", False)),
            "low_resolution": bool(r.get("low_resolution", False)),
            "had_outliers": bool(r.get("had_outliers", False)),
            **feats
        })
    return pd.DataFrame(rows)
