# src/growthqa/grofit/dr_fit_spline.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

from .dr_fit_model import dr_fit_model
from .parametric_models import aic_from_rss


def _cv_mse_for_spline(
    x: np.ndarray,
    y: np.ndarray,
    s: float,
    *,
    k_folds: int,
    k_spline: int,
) -> float:
    n = len(x)
    k_folds = max(2, min(int(k_folds), n))
    idx = np.arange(n)
    folds = np.array_split(idx, k_folds)
    errs = []
    for val_idx in folds:
        train_idx = np.setdiff1d(idx, val_idx)
        if len(train_idx) <= k_spline:
            continue
        try:
            sp = UnivariateSpline(x[train_idx], y[train_idx], k=k_spline, s=float(s))
            y_hat = sp(x[val_idx])
            errs.append(float(np.mean((y[val_idx] - y_hat) ** 2)))
        except Exception:
            continue
    if not errs:
        return float("inf")
    return float(np.mean(errs))


def _auto_select_s(
    x: np.ndarray,
    y: np.ndarray,
    s_grid: np.ndarray,
    *,
    k_folds: int,
    k_spline: int,
) -> float:
    best_s = float(s_grid[0])
    best = float("inf")
    for s in s_grid:
        score = _cv_mse_for_spline(x, y, float(s), k_folds=k_folds, k_spline=k_spline)
        if score < best:
            best = score
            best_s = float(s)
    return best_s


def _is_monotonic(deriv: np.ndarray, eps: float) -> bool:
    d = np.asarray(deriv, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return False
    return bool(np.all(d >= -eps) or np.all(d <= eps))


def _pick_ec50_crossing(
    x_grid: np.ndarray,
    y_hat: np.ndarray,
    dy: np.ndarray,
    target: float,
) -> tuple[float, str]:
    diff = np.asarray(y_hat, float) - float(target)
    xg = np.asarray(x_grid, float)
    dyg = np.asarray(dy, float)

    roots: list[tuple[float, float]] = []
    n = len(xg)
    for i in range(max(0, n - 1)):
        d0, d1 = float(diff[i]), float(diff[i + 1])
        x0, x1 = float(xg[i]), float(xg[i + 1])
        if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(x0) and np.isfinite(x1)):
            continue
        if d0 == 0.0:
            slope = float(np.interp(x0, xg, np.abs(dyg)))
            roots.append((x0, slope))
        if d0 * d1 < 0.0:
            if abs(d1 - d0) < 1e-12:
                xr = x0
            else:
                xr = x0 + (0.0 - d0) * (x1 - x0) / (d1 - d0)
            slope = float(np.interp(xr, xg, np.abs(dyg)))
            roots.append((float(xr), slope))
    if n > 0 and float(diff[-1]) == 0.0:
        xr = float(xg[-1])
        roots.append((xr, float(np.interp(xr, xg, np.abs(dyg)))))

    if not roots:
        idx = int(np.argmin(np.abs(diff)))
        return float(xg[idx]), "NO_CROSS_NEAREST"

    if len(roots) == 1:
        return float(roots[0][0]), "OK"

    slopes = np.array([r[1] for r in roots], dtype=float)
    best = float(np.nanmax(slopes))
    tie_mask = np.isclose(slopes, best, rtol=0.0, atol=max(1e-6, 0.01 * max(1.0, abs(best))))
    tie_count = int(np.sum(tie_mask))
    if tie_count > 1:
        return float("nan"), "AMBIGUOUS"
    best_idx = int(np.nanargmax(slopes))
    return float(roots[best_idx][0]), "MULTI_STEEPEST"


def dr_fit_spline(
    conc: np.ndarray,
    resp: np.ndarray,
    x_transform: Optional[str] = "log1p",
    s: Optional[float] = None,
    auto_cv: bool = True,
    *,
    enforce_monotonic: bool = True,
    fallback_to_4pl: bool = True,
) -> Dict[str, Any]:
    x = np.asarray(conc, float)
    y = np.asarray(resp, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return {"success": False, "message": "Need >=4 points for dose-response", "n": len(x)}

    if x_transform == "log1p":
        xt = np.log1p(x)
    else:
        xt = x

    order = np.argsort(xt)
    xt = xt[order]
    x = x[order]
    y = y[order]

    k_spline = min(3, len(xt) - 1)

    if s is None and auto_cv:
        scale = float(np.nanvar(y) * len(y))
        s_grid = np.logspace(-3, 2, 20) * max(scale, 1e-6)
        k_folds = min(5, len(xt))
        s_fit = _auto_select_s(xt, y, s_grid, k_folds=k_folds, k_spline=k_spline)
    elif s is None:
        s_fit = float(np.nanvar(y) * len(y))
    else:
        s_fit = float(s)

    last_err: Optional[Exception] = None
    monotonic = False
    sp: Optional[UnivariateSpline] = None
    grid = np.array([], dtype=float)
    y_hat = np.array([], dtype=float)
    dy = np.array([], dtype=float)
    chosen_s = float(s_fit)

    for mult in [1.0, 3.0, 10.0, 30.0]:
        try:
            candidate_s = float(max(1e-12, s_fit * mult))
            sp_c = UnivariateSpline(xt, y, k=k_spline, s=candidate_s)
            g = np.linspace(float(np.min(xt)), float(np.max(xt)), 2000)
            yh = sp_c(g)
            d1 = sp_c.derivative(1)(g)
            eps = max(1e-8, 0.02 * float(np.nanstd(d1)))
            mono = _is_monotonic(d1, eps=eps)
            sp = sp_c
            grid = g
            y_hat = yh
            dy = d1
            monotonic = mono
            chosen_s = candidate_s
            if (not enforce_monotonic) or mono:
                break
        except Exception as e:
            last_err = e
            continue

    if sp is None or grid.size == 0:
        return {
            "success": False,
            "message": f"dr spline fit failed: {last_err}" if last_err is not None else "dr spline fit failed",
            "n": int(len(x)),
        }

    if enforce_monotonic and (not monotonic) and fallback_to_4pl:
        model_fit = dr_fit_model(x, y)
        if model_fit.get("success"):
            return {
                "success": True,
                "message": "ok",
                "n": int(len(x)),
                "x_transform": x_transform,
                "method": "4pl",
                "dr_monotonic": True,
                "ec50_status": "OK",
                "ec50": model_fit.get("ec50"),
                "ec50_x_transformed": model_fit.get("ec50"),
                "y_ec50": model_fit.get("y_ec50"),
                "endpoint_low": model_fit.get("top"),
                "endpoint_high": model_fit.get("bottom"),
                "aic": model_fit.get("aic"),
                "rss": model_fit.get("rss"),
                "s": np.nan,
                "x_grid": model_fit.get("x_grid"),
                "y_hat": model_fit.get("y_hat"),
                "model_fit": model_fit,
            }

    y0 = float(y_hat[0])
    y1 = float(y_hat[-1])
    target = 0.5 * (y0 + y1)
    ec50_xt, ec50_status = _pick_ec50_crossing(grid, y_hat, dy, target)

    if x_transform == "log1p":
        ec50 = float(np.expm1(ec50_xt)) if np.isfinite(ec50_xt) else float("nan")
    else:
        ec50 = float(ec50_xt) if np.isfinite(ec50_xt) else float("nan")

    y_ec50 = float(np.interp(ec50_xt, grid, y_hat)) if np.isfinite(ec50_xt) else float("nan")
    residual = y - sp(xt)
    rss = float(np.sum(residual**2))
    aic = float(aic_from_rss(rss, int(len(y)), int(k_spline + 1)))

    return {
        "success": True,
        "message": "ok",
        "n": int(len(x)),
        "x_transform": x_transform,
        "method": "spline",
        "s": float(chosen_s),
        "ec50": ec50,
        "ec50_x_transformed": float(ec50_xt) if np.isfinite(ec50_xt) else np.nan,
        "ec50_status": ec50_status,
        "y_ec50": y_ec50,
        "endpoint_low": y0,
        "endpoint_high": y1,
        "dr_monotonic": bool(monotonic),
        "aic": aic,
        "rss": rss,
        "x_grid": grid,
        "y_hat": y_hat,
    }
