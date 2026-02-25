# src/growthqa/grofit/dr_boot_spline.py
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .dr_fit_spline import dr_fit_spline


def dr_boot_spline(
    conc: np.ndarray,
    resp: np.ndarray,
    B: int = 300,
    ci: float = 0.95,
    refit_lambda: bool = False,  
    random_state: Optional[int] = None,
    x_transform: Optional[str] = "log1p",
    lam: Optional[float] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    x = np.asarray(conc, float)
    y = np.asarray(resp, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return {"success": False, "message": "Need >=6 points for DR bootstrap", "n": n}

    # Pre-fit once and lock smoothing to avoid repeated GCV inside bootstrap loop.
    locked_lam = lam
    if locked_lam is None:
        base_fit = dr_fit_spline(
            x,
            y,
            x_transform=x_transform,
            lam=None,
            auto_cv=True,
            enforce_monotonic=False,
            fallback_to_4pl=False,
        )
        if not base_fit.get("success"):
            return {"success": False, "message": "Base DR spline fit failed while estimating smoothing", "n": n}
        s_guess = base_fit.get("lam", base_fit.get("s", np.nan))
        try:
            s_num = float(s_guess)
        except Exception:
            s_num = np.nan
        locked_lam = float(s_num) if np.isfinite(s_num) else None

    ec50s = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        lam_to_use = None if refit_lambda else locked_lam
        if np.any(yb < 0):
            yb = np.clip(yb, 0.0, None) 

        fit = dr_fit_spline(
            xb,
            yb,
            x_transform=x_transform,
            lam=lam_to_use,
            auto_cv=False,
            enforce_monotonic=True,
            fallback_to_4pl=True,
        )
        ec50 = fit.get("ec50", np.nan) if fit.get("success") else np.nan
        try:
            ec50 = float(ec50)
        except Exception:
            ec50 = np.nan
        if np.isfinite(ec50):
            ec50s.append(ec50)

    ec50s = np.asarray(ec50s, float)
    if len(ec50s) == 0:
        return {"success": False, "message": "All boot fits failed", "n": n}

    alpha = (1.0 - ci) / 2.0
    return {
        "success": True,
        "message": "ok",
        "n": n,
        "B": B,
        "ci": ci,
        "ec50_mean": float(np.mean(ec50s)),
        "ec50_sd": float(np.std(ec50s, ddof=1)) if len(ec50s) > 1 else 0.0,
        "ec50_lo": float(np.quantile(ec50s, alpha)),
        "ec50_hi": float(np.quantile(ec50s, 1.0 - alpha)),
        "ec50_samples_n": int(len(ec50s)),
    }
