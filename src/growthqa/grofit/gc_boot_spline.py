# src/growthqa/grofit/gc_boot_spline.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Literal

from .gc_fit_spline import gc_fit_spline

BootstrapMethod = Literal["pairs", "residual"]


def gc_boot_spline(
    t: np.ndarray,
    y: np.ndarray,
    B: int = 200,
    ci: float = 0.95,
    random_state: Optional[int] = None,
    spline_s: Optional[float] = None,
    auto_cv: bool = True,
    bootstrap_method: BootstrapMethod = "pairs",
) -> Dict[str, Any]:
    """
    Bootstrap spline parameters A, mu, lag, integral.
    Supports:
      - pairs: resample (t, y) pairs
      - residual: fit once, resample residuals, refit
    Returns mean + sd + (lower, upper) CI.
    """
    rng = np.random.default_rng(random_state)
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    n = len(t)

    if n < 6:
        return {"success": False, "message": "Need >=6 points for bootstrap", "n": n}

    stats = {"A": [], "mu": [], "lag": [], "integral": []}

    # Pre-fit once and lock smoothing for all bootstrap resamples.
    locked_s = spline_s
    if locked_s is None:
        pre_fit = gc_fit_spline(t, y, s=None, auto_cv=auto_cv)
        if not pre_fit.success:
            return {"success": False, "message": "Base spline fit failed while estimating smoothing", "n": n}
        pre_extra = pre_fit.extra or {}
        lam_guess = pre_extra.get("lam", pre_extra.get("s", np.nan))
        lam_num = np.nan
        try:
            lam_num = float(lam_guess)
        except Exception:
            lam_num = np.nan
        locked_s = float(lam_num) if np.isfinite(lam_num) else None

    # Fit once for residual bootstrap
    base_fit = None
    tb_sorted = None
    y_fit = None
    resid = None
    n_resid = n
    if bootstrap_method == "residual":
        base_fit = gc_fit_spline(t, y, s=locked_s, auto_cv=False)
        if not base_fit.success:
            return {"success": False, "message": "Base spline fit failed for residual bootstrap", "n": n}
        try:
            from scipy.interpolate import make_smoothing_spline
            order = np.argsort(t)
            tb_sorted = t[order]
            y_sorted = y[order]
            xu, inv = np.unique(tb_sorted, return_inverse=True)
            if xu.size != tb_sorted.size:
                y_sum = np.zeros_like(xu, dtype=float)
                cnt = np.zeros_like(xu, dtype=float)
                np.add.at(y_sum, inv, y_sorted)
                np.add.at(cnt, inv, 1.0)
                tb_sorted = xu
                y_sorted = y_sum / np.maximum(cnt, 1.0)
            lam_used = float(max(float(locked_s), 0.0)) if locked_s is not None else None
            sp = make_smoothing_spline(tb_sorted, y_sorted, lam=lam_used)
            y_fit = sp(tb_sorted)
            resid = y_sorted - y_fit
            n_resid = int(len(tb_sorted))
        except Exception:
            return {"success": False, "message": "Base spline prediction failed for residual bootstrap", "n": n}

    for _ in range(B):
        if bootstrap_method == "pairs":
            idx = rng.integers(0, n, size=n)
            tb = t[idx]
            yb = y[idx]
            order = np.argsort(tb)
            tb = tb[order]
            yb = yb[order]
        else:
            # residual bootstrap
            if resid is None or y_fit is None or tb_sorted is None:
                continue
            resid_b = rng.choice(resid, size=n_resid, replace=True)
            yb = y_fit + resid_b
            tb = tb_sorted

        fit = gc_fit_spline(tb, yb, s=locked_s, auto_cv=False)
        if fit.success:
            stats["A"].append(fit.A)
            stats["mu"].append(fit.mu)
            stats["lag"].append(fit.lag)
            stats["integral"].append(fit.integral)

    def summarize(arr):
        arr = np.asarray([a for a in arr if a is not None and np.isfinite(a)], float)
        if len(arr) == 0:
            return {
                "mean": np.nan,
                "sd": np.nan,
                "std": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "lo90": np.nan,
                "hi90": np.nan,
                "lo95": np.nan,
                "hi95": np.nan,
                "n": 0,
            }
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1.0 - alpha))
        lo90 = float(np.quantile(arr, 0.05))
        hi90 = float(np.quantile(arr, 0.95))
        lo95 = float(np.quantile(arr, 0.025))
        hi95 = float(np.quantile(arr, 0.975))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return {
            "mean": float(np.mean(arr)),
            "sd": sd,
            "std": sd,
            "lo": lo,
            "hi": hi,
            "lo90": lo90,
            "hi90": hi90,
            "lo95": lo95,
            "hi95": hi95,
            "n": int(len(arr)),
        }

    out = {k: summarize(v) for k, v in stats.items()}
    out["success"] = True
    out["B"] = B
    out["ci"] = ci
    out["n"] = n
    out["bootstrap_method"] = bootstrap_method
    return out
