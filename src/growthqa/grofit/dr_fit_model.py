from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import curve_fit

from .parametric_models import aic_from_rss


def _hill_4pl(x: np.ndarray, bottom: float, top: float, ec50: float, hill: float) -> np.ndarray:
    xx = np.maximum(np.asarray(x, float), 0.0)
    ec50 = max(float(ec50), 1e-12)
    hill = max(float(hill), 1e-6)
    return float(bottom) + (float(top) - float(bottom)) / (1.0 + np.power(xx / ec50, hill))


def dr_fit_model(conc: np.ndarray, resp: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(conc, float)
    y = np.asarray(resp, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return {"success": False, "message": "Need >=4 points for 4PL", "n": int(len(x)), "converged": False}

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_pos = x[x > 0]
    x_span = float(np.nanmax(x) - np.nanmin(x)) if len(x) else 0.0
    y_pad = max(1e-3, 0.2 * float(np.nanmax(y) - np.nanmin(y)))
    xmin_pos = float(np.nanmin(x_pos)) if len(x_pos) else 1e-6
    xmax = float(np.nanmax(x)) if len(x) else 1.0
    ec50_guess = float(np.nanmedian(x_pos)) if len(x_pos) else max(1e-4, xmax * 0.5)
    if not np.isfinite(ec50_guess) or ec50_guess <= 0:
        ec50_guess = max(1e-4, xmax * 0.5)

    p0 = np.array([float(np.nanmin(y)), float(np.nanmax(y)), ec50_guess, 1.0], dtype=float)
    lower = np.array([float(np.nanmin(y) - y_pad), float(np.nanmin(y) - y_pad), max(1e-12, xmin_pos * 1e-3), 0.05], dtype=float)
    upper = np.array([float(np.nanmax(y) + y_pad), float(np.nanmax(y) + y_pad), max(1e-6, xmax * 1e3 + max(1e-3, x_span)), 8.0], dtype=float)

    try:
        params, cov = curve_fit(
            _hill_4pl,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        y_hat = _hill_4pl(x, *params)
        rss = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - rss / max(ss_tot, 1e-12))
        aic = float(aic_from_rss(rss, int(len(x)), 4))

        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
        y_grid = _hill_4pl(x_grid, *params)

        return {
            "success": True,
            "message": "ok",
            "n": int(len(x)),
            "model": "4pl",
            "bottom": float(params[0]),
            "top": float(params[1]),
            "ec50": float(params[2]),
            "hill": float(params[3]),
            "y_ec50": float(_hill_4pl(np.array([float(params[2])]), *params)[0]),
            "r2": r2,
            "aic": aic,
            "rss": rss,
            "converged": True,
            "params": params,
            "cov": cov,
            "x_grid": x_grid,
            "y_hat": y_grid,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"4PL fit failed: {e}",
            "n": int(len(x)),
            "model": "4pl",
            "converged": False,
            "ec50": np.nan,
            "y_ec50": np.nan,
            "aic": np.nan,
            "r2": np.nan,
            "rss": np.nan,
        }
