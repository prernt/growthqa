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

    # Fit once for residual bootstrap
    base_fit = None
    tb_sorted = None
    y_fit = None
    resid = None
    if bootstrap_method == "residual":
        base_fit = gc_fit_spline(t, y, s=spline_s, auto_cv=auto_cv)
        if not base_fit.success:
            return {"success": False, "message": "Base spline fit failed for residual bootstrap", "n": n}
        try:
            from scipy.interpolate import UnivariateSpline
            order = np.argsort(t)
            tb_sorted = t[order]
            y_sorted = y[order]
            k_spline = min(3, len(tb_sorted) - 1)
            s_used = (base_fit.extra or {}).get("s")
            if s_used is None:
                s_used = 0.0
            sp = UnivariateSpline(tb_sorted, y_sorted, k=k_spline, s=s_used)
            y_fit = sp(tb_sorted)
            resid = y_sorted - y_fit
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
            resid_b = rng.choice(resid, size=n, replace=True)
            yb = y_fit + resid_b
            tb = tb_sorted

        fit = gc_fit_spline(tb, yb, s=spline_s, auto_cv=auto_cv)
        if fit.success:
            stats["A"].append(fit.A)
            stats["mu"].append(fit.mu)
            stats["lag"].append(fit.lag)
            stats["integral"].append(fit.integral)

    def summarize(arr):
        arr = np.asarray([a for a in arr if a is not None and np.isfinite(a)], float)
        if len(arr) == 0:
            return {"mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1.0 - alpha))
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "lo": lo,
            "hi": hi,
            "n": int(len(arr)),
        }

    out = {k: summarize(v) for k, v in stats.items()}
    out["success"] = True
    out["B"] = B
    out["ci"] = ci
    out["n"] = n
    out["bootstrap_method"] = bootstrap_method
    return out
