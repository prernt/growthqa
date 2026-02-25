# src/growthqa/grofit/gc_fit_spline.py
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .types import FitResult

# ── Shared smoothing constants (items 1, 2) ──────────────────────────────────
# spar ∈ (0,1] maps to SciPy λ via the same log-linear scale R uses internally.
# R's spar range [0,1] maps to λ ∈ [λ_min, λ_max] for a given dataset.
# We use a data-scale-invariant mapping: λ = 10^(-6 + 6*spar)
# so spar=0 → λ=1e-6 (very wiggly), spar=1 → λ=1 (very smooth).
# This is consistent between GC and DR (item 2).
SPAR_LAM_LOG_MIN = -6.0   # log10(λ) at spar=0
SPAR_LAM_LOG_MAX =  2.0   # log10(λ) at spar=1 (tighter than DR since GC data is denser)

# Minimum effective df for GC splines (sigmoid needs ≥4)
GC_MIN_DF = 4.0
DR_MIN_DF = 3.5


def spar_to_lam(spar: float, *, log_min: float = SPAR_LAM_LOG_MIN, log_max: float = SPAR_LAM_LOG_MAX) -> float:
    """Convert Grofit-like spar ∈ (0,1] to SciPy smoothing λ."""
    s = float(np.clip(spar, 1e-6, 1.0))
    return float(10.0 ** (log_min + (log_max - log_min) * s))


def lam_to_spar(lam: float, *, log_min: float = SPAR_LAM_LOG_MIN, log_max: float = SPAR_LAM_LOG_MAX) -> float:
    """Convert SciPy λ back to approximate spar ∈ (0,1]."""
    if lam <= 0:
        return 0.0
    s = (np.log10(max(lam, 1e-12)) - log_min) / (log_max - log_min)
    return float(np.clip(s, 0.0, 1.0))


def effective_df(sp, x: np.ndarray) -> float:
    """Estimate effective degrees of freedom of a smoothing spline.

    Uses curvature/slope energy ratio as a fast proxy for trace(H).
    df≈2 → straight line; df≈4+ → sigmoid with curvature.
    Shared by GC and DR (item 2).
    """
    try:
        dy = sp.derivative(1)(x)
        d2y = sp.derivative(2)(x)
        e1 = float(np.sum(dy ** 2))
        e2 = float(np.sum(d2y ** 2))
        if e1 < 1e-20:
            return 2.0
        return float(np.clip(2.0 + np.log1p(e2 / e1 * float(len(x))), 2.0, float(len(x))))
    except Exception:
        return 2.0


def _find_bounded_lambda(t: np.ndarray, y: np.ndarray, min_df: float, n_search: int = 30) -> float:
    """Binary-search for largest λ (smoothest spline) that achieves ≥ min_df effective df."""
    lam_lo, lam_hi = 1e-12, 1e6
    try:
        sp_test = make_smoothing_spline(t, y, lam=lam_lo)
        if effective_df(sp_test, t) < min_df:
            return lam_lo
    except Exception:
        return lam_lo
    for _ in range(n_search):
        lam_mid = np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15))))
        try:
            df_mid = effective_df(make_smoothing_spline(t, y, lam=lam_mid), t)
        except Exception:
            df_mid = 2.0
        if df_mid >= min_df:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    return float(np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15)))))


def _select_lam_and_fit(t: np.ndarray, y: np.ndarray, lam: Optional[float], auto_cv: bool, min_df: float):
    """Unified lambda selection + spline fitting. Returns (sp, lam_used, method_label).

    Shared between GC and DR for consistent smoothing semantics (item 2).
    Priority: explicit lam → GCV-with-df-floor → variance-fallback.
    """
    if lam is not None:
        sp = make_smoothing_spline(t, y, lam=float(max(lam, 0.0)))
        return sp, float(lam), "user"
    if auto_cv:
        try:
            sp_gcv = make_smoothing_spline(t, y, lam=None)
            if effective_df(sp_gcv, t) >= min_df:
                return sp_gcv, float("nan"), "gcv_ok"
            lam_b = _find_bounded_lambda(t, y, min_df=min_df)
            return make_smoothing_spline(t, y, lam=lam_b), lam_b, "gcv_bounded"
        except Exception:
            pass
    lam_fb = float(max(np.nanvar(y) * max(len(y), 1), 1e-12))
    return make_smoothing_spline(t, y, lam=lam_fb), lam_fb, "fallback"


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
    lam: Optional[float] = None,        # raw SciPy λ (backward compat)
    auto_cv: bool = True,
    s_grid: Optional[np.ndarray] = None,
    *,
    smooth: Optional[float] = None,     # NEW (item 1): Grofit-like spar ∈ (0,1]
    df: Optional[float] = None,         # NEW (item 1): target effective df (overrides smooth)
) -> FitResult:
    """Fit a smoothing spline to a growth curve.

    Smoothing control (item 1) — in priority order:
      df     : target effective degrees of freedom (most interpretable)
      smooth : spar ∈ (0,1] — Grofit-like knob, maps to λ via log-linear scale
      lam    : raw SciPy penalty λ (backward compat)
      auto_cv: True = GCV with df floor (default)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    if len(t) < 4:
        return FitResult(
            method="spline", model="spline", success=False,
            message="Too few points for spline", n=len(t),
            fit_status="failed", fail_reason="insufficient_points",
        )

    # ── Resolve smoothing parameter (item 1) ─────────────────────────────
    # df takes priority, then smooth (spar), then raw lam, then GCV
    resolved_lam: Optional[float] = lam
    if smooth is not None and resolved_lam is None:
        resolved_lam = spar_to_lam(smooth)
    # df is handled after fitting by iterating; for now we pass it as a
    # target to the bounded-lambda search.

    if resolved_lam is not None:
        try:
            lam_num = float(resolved_lam)
        except Exception:
            lam_num = float("nan")
        resolved_lam = lam_num if np.isfinite(lam_num) else None

    order = np.argsort(t)
    t, y = t[order], y[order]
    t, y = _dedupe_sorted_xy(t, y)

    if len(t) < 4:
        return FitResult(
            method="spline", model="spline", success=False,
            message="Too few unique points for spline", n=len(t),
            fit_status="failed", fail_reason="insufficient_unique_points",
        )

    # If df target given, override resolved_lam with bounded search result
    if df is not None:
        try:
            resolved_lam = _find_bounded_lambda(t, y, min_df=float(df))
            auto_cv = False
        except Exception:
            pass

    try:
        sp, lam_used, lam_method = _select_lam_and_fit(
            t, y, lam=resolved_lam, auto_cv=auto_cv, min_df=GC_MIN_DF,
        )

        t_min, t_max = float(np.min(t)), float(np.max(t))
        t_grid = np.linspace(t_min, t_max, 400)
        y_grid = sp(t_grid)
        dy = sp.derivative(1)(t_grid)

        # Amplitude-band μ extraction (robust, prevents noise spikes)
        y_min_g, y_max_g = float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
        A_est_raw = y_max_g - y_min_g
        if A_est_raw > 1e-6:
            valid_mask = (y_grid >= y_min_g + 0.05 * A_est_raw) & (y_grid <= y_min_g + 0.95 * A_est_raw)
            masked_dy = np.where(valid_mask if np.any(valid_mask) else np.ones_like(y_grid, bool), dy, -np.inf)
            idx = int(np.nanargmax(masked_dy))
        else:
            idx = int(np.nanargmax(dy))

        mu = float(dy[idx])
        t_star = float(t_grid[idx])
        y_star = float(np.interp(t_star, t_grid, y_grid))

        # ── Lag (item 3): standardized baseline = mean of first 3 grid points ──
        # This matches Grofit R's spline lag: uses fitted curve start as y0, not raw data.
        y0_baseline = float(np.nanmean(y_grid[:3])) if y_grid.size >= 3 else float(y_grid[0])
        A = float(np.nanmax(y_grid)) - y0_baseline
        lag_method_str = "tangent_spline"

        if mu <= 1e-12:
            lag = float("nan")
            lag_method_str = "tangent_spline_undefined"
        else:
            lag = float(max(0.0, t_star - (y_star - y0_baseline) / mu))

        integral = float(np.trapz(y_grid, t_grid))
        rss = float(np.sum((y - sp(t)) ** 2))

        # ── Smoothing diagnostics (item 1) ───────────────────────────────
        df_eff = effective_df(sp, t)
        smooth_out = lam_to_spar(lam_used) if np.isfinite(lam_used) else float("nan")

        lam_out = float(lam_used) if np.isfinite(lam_used) else float("nan")

        warn_list = []
        if lam_method == "gcv_bounded":
            warn_list.append("GCV over-smoothed; df floor enforced")
        if lam_method == "fallback":
            warn_list.append("GCV failed; variance-scaled fallback used")

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
            # new fields
            smooth_used=smooth_out,
            df_effective=df_eff,
            lam_raw=lam_out,
            lag_method=lag_method_str,
            y0_baseline=y0_baseline,
            fit_status="ok",
            fail_reason=None,
            warnings=warn_list if warn_list else None,
            extra={
                "lam": lam_out,
                "s": lam_out,
                "lam_method": lam_method,
                "knots": np.asarray(getattr(sp, "t", []), float),
                "mu_method": "spline_derivative",
                "t_star": float(t_star),
                "y_star": float(y_star),
                "y0": float(y0_baseline),
            },
        )
    except Exception as e:
        return FitResult(
            method="spline", model="spline", success=False,
            message=f"Spline fit failed: {e}", n=int(len(t)),
            fit_status="failed", fail_reason="fit_exception",
        )