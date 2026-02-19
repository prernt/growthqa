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
    random_state: Optional[int] = None,
    x_transform: Optional[str] = "log1p",
    s: Optional[float] = None,
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

    ec50s = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        fit = dr_fit_spline(xb, yb, x_transform=x_transform, s=s, auto_cv=(s is None))
        if fit.get("success") and np.isfinite(fit.get("ec50", np.nan)):
            ec50s.append(float(fit["ec50"]))

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
