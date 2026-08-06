from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from growthqa.grofit.dr_fit_spline import dr_fit_spline
from growthqa.grofit.dr_fit_model import dr_fit_model


def dr_boot_spline(
    conc: np.ndarray,
    resp: np.ndarray,
    B: int = 300,
    ci: float = 0.95,
    refit_lambda: bool = False,
    random_state: Optional[int] = None,
    x_transform: Optional[str] = "log1p",
    lam: Optional[float] = None,
    auto_cv: bool = True,
    smooth: Optional[float] = None,
    y_transform: Optional[str] = None,
    fit_method: str = "spline",
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

    method = str(fit_method or "spline").strip().lower()
    if method in {"4pl", "4pl_fallback"}:
        return _boot_4pl(x, y, B=B, ci=ci, rng=rng, n=n)

    locked_lam = lam
    if locked_lam is None:
        base_fit = dr_fit_spline(
            x, y,
            x_transform=x_transform,
            lam=None,
            auto_cv=True,
            smooth=smooth,
            y_transform=y_transform,
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

    ec50s: list[float] = []
    n_total = 0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        lam_to_use = None if refit_lambda else locked_lam
        if np.any(yb < 0):
            yb = np.clip(yb, 0.0, None)

        fit = dr_fit_spline(
            xb, yb,
            x_transform=x_transform,
            lam=lam_to_use,
            auto_cv=False,
            smooth=smooth,
            y_transform=y_transform,
            enforce_monotonic=True,
            fallback_to_4pl=True,
        )
        ec50 = fit.get("ec50", np.nan) if fit.get("success") else np.nan
        ec50_status = str(fit.get("ec50_status", "")).strip().upper()
        try:
            ec50 = float(ec50)
        except Exception:
            ec50 = np.nan
        n_total += 1
        if np.isfinite(ec50) and ec50_status != "NO_CROSS_NEAREST":
            ec50s.append(ec50)

    return _summarize(ec50s, n_total, n=n, B=B, ci=ci, fit_method="spline")


def _boot_4pl(
    x: np.ndarray,
    y: np.ndarray,
    *,
    B: int,
    ci: float,
    rng: np.random.Generator,
    n: int,
) -> Dict[str, Any]:
    ec50s: list[float] = []
    n_total = 0

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.any(yb < 0):
            yb = np.clip(yb, 0.0, None)

        n_total += 1

        if len(np.unique(xb)) < 4:
            continue

        fit = dr_fit_model(xb, yb)
        if not fit.get("success"):
            continue

        try:
            ec50 = float(fit.get("ec50", np.nan))
        except Exception:
            ec50 = np.nan
        if not np.isfinite(ec50):
            continue

        lo_x = float(np.nanmin(xb))
        hi_x = float(np.nanmax(xb))
        if not (lo_x <= ec50 <= hi_x):
            continue

        ec50s.append(ec50)

    return _summarize(ec50s, n_total, n=n, B=B, ci=ci, fit_method="4pl")


def _summarize(
    ec50s: list[float],
    n_total: int,
    *,
    n: int,
    B: int,
    ci: float,
    fit_method: str,
) -> Dict[str, Any]:
    ec50s = np.asarray(ec50s, float)
    if len(ec50s) == 0:
        return {
            "success": False,
            "message": "All boot fits failed",
            "n": n,
            "fit_method": fit_method,
            "ec50_crossing_rate": 0.0 if n_total > 0 else float("nan"),
        }

    alpha = (1.0 - ci) / 2.0
    return {
        "success": True,
        "message": "ok",
        "n": n,
        "B": B,
        "ci": ci,
        "fit_method": fit_method,
        "ec50_mean": float(np.mean(ec50s)),
        "ec50_sd": float(np.std(ec50s, ddof=1)) if len(ec50s) > 1 else 0.0,
        "ec50_lo": float(np.quantile(ec50s, alpha)),
        "ec50_hi": float(np.quantile(ec50s, 1.0 - alpha)),
        "ec50_samples_n": int(len(ec50s)),
        "ec50_crossing_rate": float(len(ec50s) / n_total) if n_total > 0 else float("nan"),
    }