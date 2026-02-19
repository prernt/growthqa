# src/growthqa/grofit/gc_fit_spline.py
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from .types import FitResult

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


def gc_fit_spline(
    t: np.ndarray,
    y: np.ndarray,
    s: Optional[float] = None,
    auto_cv: bool = True,
    s_grid: Optional[np.ndarray] = None,
) -> FitResult:
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 4:
        return FitResult(method="spline", model="spline", success=False,
                         message="Too few points for spline", n=len(t))

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    k_spline = min(3, len(t) - 1)

    # Choose smoothing via CV when s is None.
    # If auto_cv is False and s is None, use a light smoothing heuristic.
    if s is None and auto_cv:
        if s_grid is None:
            scale = float(np.nanvar(y) * len(y))
            s_grid = np.logspace(-3, 2, 20) * max(scale, 1e-6)
        k_folds = min(5, len(t))
        s = _auto_select_s(t, y, s_grid, k_folds=k_folds, k_spline=k_spline)
    elif s is None:
        s = float(np.nanvar(y) * len(y))

    try:
        sp = UnivariateSpline(t, y, k=k_spline, s=s)
        t_min, t_max = float(np.min(t)), float(np.max(t))
        t_grid = np.linspace(t_min, t_max, 400)
        y_grid = sp(t_grid)

        dt = float(np.nanmedian(np.diff(t_grid))) if t_grid.size > 1 else 1.0
        n_grid = int(t_grid.size)
        w = min(11, n_grid if (n_grid % 2 == 1) else (n_grid - 1))
        if w >= 5:
            poly = min(2, w - 1)
            dy = savgol_filter(y_grid, window_length=w, polyorder=poly, deriv=1, delta=max(dt, 1e-12), mode="interp")
        else:
            dy = np.gradient(y_grid, t_grid)
        idx = int(np.nanargmax(dy))
        mu = float(dy[idx])
        t_star = float(t_grid[idx])
        y_star = float(np.interp(t_star, t_grid, y_grid))

        y0 = float(np.nanpercentile(y_grid, 5))
        A = float(np.nanmax(y_grid) - y0)
        A = max(A, 0.0)

        if mu <= 1e-12:
            lag = float("nan")
        else:
            lag = float(t_star - (y_star - y0) / mu)

        integral = float(np.trapz(y_grid, t_grid))

        rss = float(np.sum((y - sp(t)) ** 2))
        return FitResult(
            method="spline",
            model="spline",
            success=True,
            message="ok",
            lag=lag,
            mu=mu,
            A=A,
            integral=integral,
            rss=rss,
            n=int(len(t)),
            k=None,
            extra={
                "s": float(s) if s is not None else None,
                "knots": sp.get_knots(),
                "mu_method": "savgol",
                "savgol_window": int(w) if w >= 5 else np.nan,
            },
        )

    except Exception as e:
        return FitResult(method="spline", model="spline", success=False,
                         message=f"Spline fit failed: {e}", n=int(len(t)))
