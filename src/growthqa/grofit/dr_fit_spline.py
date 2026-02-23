# src/growthqa/grofit/dr_fit_spline.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .dr_fit_model import dr_fit_model
from .parametric_models import aic_from_rss


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
    x_transform: Optional[str] = "log10",
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

    # Defensive: treat non-finite smoothing inputs as "not provided".
    if s is not None:
        try:
            s_num = float(s)
        except Exception:
            s_num = np.nan
        s = s_num if np.isfinite(s_num) else None

    x_transform_norm = (x_transform or "none").strip().lower()
    if x_transform_norm in {"log10", "log"}:
        pos = x[x > 0]
        if pos.size == 0:
            return {"success": False, "message": "Need at least one positive concentration for log10 transform", "n": len(x)}
        pseudo = max(float(np.nanmin(pos)) / 10.0, 1e-12)
        x_for_log = np.where(x > 0, x, pseudo)
        xt = np.log10(x_for_log)
    elif x_transform_norm == "log1p":
        xt = np.log1p(x)
    else:
        xt = x

    order = np.argsort(xt)
    xt = xt[order]
    x = x[order]
    y = y[order]
    # Keep x/y/xt aligned after deduplication on transformed x.
    xt_u, inv = np.unique(xt, return_inverse=True)
    if xt_u.size != xt.size:
        y_sum = np.zeros_like(xt_u, dtype=float)
        x_sum = np.zeros_like(xt_u, dtype=float)
        cnt = np.zeros_like(xt_u, dtype=float)
        np.add.at(y_sum, inv, y)
        np.add.at(x_sum, inv, x)
        np.add.at(cnt, inv, 1.0)
        xt = xt_u
        y = y_sum / np.maximum(cnt, 1.0)
        x = x_sum / np.maximum(cnt, 1.0)

    if len(xt) < 4:
        return {"success": False, "message": "Need >=4 unique points for dose-response", "n": len(xt)}

    # NOTE: keep `s` argument for API compatibility; it maps to smoothing lambda (`lam`).
    lam_fit: Optional[float]
    if auto_cv:
        # MATHEMATICAL FIX: GCV (lam=None) severely over-smooths sparse DR data into flat lines.
        # We force a light smoothing penalty (0.001) to preserve the biological S-curve shape.
        lam_fit = 0.001 if s is None else float(max(float(s), 0.0))
    else:
        if s is None:
            lam_fit = 0.001
        else:
            lam_fit = float(max(float(s), 0.0))

    try:
        # GCV happens internally when lam is None, matching smooth.spline-style behavior.
        sp = make_smoothing_spline(xt, y, lam=lam_fit)
        grid = np.linspace(float(np.min(xt)), float(np.max(xt)), 2000)
        y_hat = sp(grid)
        dy = sp.derivative(1)(grid)
        # A true sigmoidal curve has flat ends and a steep middle; a straight line has near-constant slope.
        dy_abs = np.abs(dy)
        is_linear = bool(np.max(dy_abs) < 1.5 * max(float(np.mean(dy_abs)), 1e-8))

        eps = max(1e-8, 0.02 * float(np.nanstd(dy)))
        monotonic = _is_monotonic(dy, eps=eps)
    except Exception as e:
        return {"success": False, "message": f"dr spline fit failed: {e}", "n": int(len(x))}

    if enforce_monotonic and ((not monotonic) or is_linear) and fallback_to_4pl:
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

    if x_transform_norm in {"log10", "log"}:
        ec50 = float(np.power(10.0, ec50_xt)) if np.isfinite(ec50_xt) else float("nan")
    elif x_transform_norm == "log1p":
        ec50 = float(np.expm1(ec50_xt)) if np.isfinite(ec50_xt) else float("nan")
    else:
        ec50 = float(ec50_xt) if np.isfinite(ec50_xt) else float("nan")

    y_ec50 = float(np.interp(ec50_xt, grid, y_hat)) if np.isfinite(ec50_xt) else float("nan")
    residual = y - sp(xt)
    rss = float(np.sum(residual**2))
    aic = float(aic_from_rss(rss, int(len(y)), 4))
    lam_out = float(lam_fit) if lam_fit is not None and np.isfinite(lam_fit) else np.nan

    return {
        "success": True,
        "message": "ok",
        "n": int(len(x)),
        "x_transform": x_transform,
        "method": "spline",
        "s": lam_out,
        "lam": lam_out,
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
