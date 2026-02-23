# src/growthqa/grofit/gc_fit_spline.py
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .types import FitResult


def _dedupe_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= 1:
        return x, y
    xu, inv = np.unique(x, return_inverse=True)
    if xu.size == x.size:
        return x, y
    y_sum = np.zeros_like(xu, dtype=float)
    cnt = np.zeros_like(xu, dtype=float)
    np.add.at(y_sum, inv, y)
    np.add.at(cnt, inv, 1.0)
    return xu, y_sum / np.maximum(cnt, 1.0)


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
        return FitResult(method="spline", model="spline", success=False, message="Too few points for spline", n=len(t))

    # Defensive: treat non-finite smoothing inputs as "not provided".
    if s is not None:
        try:
            s_num = float(s)
        except Exception:
            s_num = np.nan
        s = s_num if np.isfinite(s_num) else None

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    t, y = _dedupe_sorted_xy(t, y)

    if len(t) < 4:
        return FitResult(method="spline", model="spline", success=False, message="Too few unique points for spline", n=len(t))

    # NOTE: keep `s` argument for API compatibility; it maps to smoothing lambda (`lam`).
    lam_fit: Optional[float]
    if auto_cv:
        lam_fit = None if s is None else float(max(float(s), 0.0))
    else:
        if s is None:
            # Deterministic fallback when CV/GCV search is disabled but no lambda was provided.
            lam_fit = float(max(np.nanvar(y) * max(len(y), 1), 1e-12))
        else:
            lam_fit = float(max(float(s), 0.0))

    try:
        # make_smoothing_spline uses GCV natively when lam is None (R smooth.spline-like behavior).
        sp = make_smoothing_spline(t, y, lam=lam_fit)
        t_min, t_max = float(np.min(t)), float(np.max(t))
        t_grid = np.linspace(t_min, t_max, 400)
        y_grid = sp(t_grid)

        # Updated math: spline derivative is analytical; no Savitzky-Golay post-processing needed.
        dy = sp.derivative(1)(t_grid)
        
        # ROBUST MU EXTRACTION: 
        # Restrict the search for max slope to the active growth phase (5% to 95% of amplitude)
        # to prevent the spline from grabbing a noise spike in the lag or stationary phase.
        y_min, y_max = float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
        A_est = y_max - y_min
        
        if A_est > 1e-6:
            valid_mask = (y_grid >= y_min + 0.05 * A_est) & (y_grid <= y_min + 0.95 * A_est)
            if np.any(valid_mask):
                masked_dy = np.where(valid_mask, dy, -np.inf)
                idx = int(np.nanargmax(masked_dy))
            else:
                idx = int(np.nanargmax(dy))
        else:
            idx = int(np.nanargmax(dy))

        mu = float(dy[idx])
        t_star = float(t_grid[idx])
        y_star = float(np.interp(t_star, t_grid, y_grid))

        y0 = float(np.nanpercentile(y_grid, 5))
        # Fix: A must be defined as amplitude (max above baseline), not absolute maximum.
        A = float(np.nanmax(y_grid)) - y0

        if mu <= 1e-12:
            lag = float("nan")
        else:
            lag = float(t_star - (y_star - y0) / mu)

        integral = float(np.trapz(y_grid, t_grid))
        rss = float(np.sum((y - sp(t)) ** 2))

        lam_out = float(lam_fit) if lam_fit is not None and np.isfinite(lam_fit) else np.nan
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
                "lam": lam_out,
                "s": lam_out,  # backward-compatible field name
                "knots": np.asarray(getattr(sp, "t", []), float),
                "mu_method": "spline_derivative",
                "t_star": float(t_star),
                "y_star": float(y_star),
                "y0": float(y0),
            },
        )
    except Exception as e:
        return FitResult(method="spline", model="spline", success=False, message=f"Spline fit failed: {e}", n=int(len(t)))
