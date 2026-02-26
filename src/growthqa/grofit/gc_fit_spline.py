# src/growthqa/grofit/gc_fit_spline.py
# ============================================================
# CORRECTED VERSION  – see analysis document for full rationale
# Key changes vs original:
#   1. Amplitude A: computed as max(y_grid) - y_grid[0] using the
#      SPLINE fitted curve, which already agrees with R's spline
#      approach.  No change needed here for the spline path because
#      the spline does not have a separate "A parameter" — it is
#      always derived geometrically from the fitted curve.
#      HOWEVER, the baseline y0 is now consistently defined as
#      y_grid[0] (the fitted spline at t_min), NOT mean(y_grid[:3]),
#      to match R's gcFitSpline which uses the spline at the first
#      observed time point as the baseline.
#   2. Negative lambda is preserved (no max(0, lag) floor).
#   3. The amplitude-band μ mask (0.05–0.95 of A) is removed because
#      it can suppress the true maximum slope on short data series
#      with a weak lag phase, causing systematic underestimation of μ.
#      R's gcFitSpline uses the global maximum of the first derivative.
# ============================================================
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .types import FitResult

# ── Smoothing constants ───────────────────────────────────────────────────────
SPAR_LAM_LOG_MIN = -6.0
SPAR_LAM_LOG_MAX =  2.0
GC_MIN_DF = 4.0
DR_MIN_DF = 3.5


def spar_to_lam(spar: float, *, log_min: float = SPAR_LAM_LOG_MIN, log_max: float = SPAR_LAM_LOG_MAX) -> float:
    s = float(np.clip(spar, 1e-6, 1.0))
    return float(10.0 ** (log_min + (log_max - log_min) * s))


def lam_to_spar(lam: float, *, log_min: float = SPAR_LAM_LOG_MIN, log_max: float = SPAR_LAM_LOG_MAX) -> float:
    if lam <= 0:
        return 0.0
    s = (np.log10(max(lam, 1e-12)) - log_min) / (log_max - log_min)
    return float(np.clip(s, 0.0, 1.0))


def effective_df(sp, x: np.ndarray) -> float:
    try:
        dy  = sp.derivative(1)(x)
        d2y = sp.derivative(2)(x)
        e1  = float(np.sum(dy ** 2))
        e2  = float(np.sum(d2y ** 2))
        if e1 < 1e-20:
            return 2.0
        return float(np.clip(2.0 + np.log1p(e2 / e1 * float(len(x))), 2.0, float(len(x))))
    except Exception:
        return 2.0


def _find_bounded_lambda(t: np.ndarray, y: np.ndarray, min_df: float, n_search: int = 30) -> float:
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
    cnt   = np.zeros_like(xu, dtype=float)
    np.add.at(y_sum, inv, y)
    np.add.at(cnt,   inv, 1.0)
    return xu, y_sum / np.maximum(cnt, 1.0)


def gc_fit_spline(
    t: np.ndarray,
    y: np.ndarray,
    lam: Optional[float] = None,
    auto_cv: bool = True,
    s_grid: Optional[np.ndarray] = None,
    *,
    smooth: Optional[float] = None,
    df: Optional[float] = None,
) -> FitResult:
    """
    Fit a smoothing spline to a growth curve.

    Changes vs original:
    - Baseline y0 = spline evaluated at t_min (y_grid[0]), not mean(y_grid[:3]).
      This matches R's gcFitSpline definition of the baseline.
    - Amplitude A = max(y_grid) - y_grid[0]  (same as R).
    - Lag computation does NOT floor to 0; negative values are allowed.
    - The amplitude-band mask on dy is removed; μ = global max of dy.
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

    resolved_lam: Optional[float] = lam
    if smooth is not None and resolved_lam is None:
        resolved_lam = spar_to_lam(smooth)
    if resolved_lam is not None:
        try:
            lam_num = float(resolved_lam)
        except Exception:
            lam_num = float("nan")
        resolved_lam = lam_num if np.isfinite(lam_num) else None

    order = np.argsort(t)
    t, y  = t[order], y[order]
    t, y  = _dedupe_sorted_xy(t, y)

    if len(t) < 4:
        return FitResult(
            method="spline", model="spline", success=False,
            message="Too few unique points for spline", n=len(t),
            fit_status="failed", fail_reason="insufficient_unique_points",
        )

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
        dy     = sp.derivative(1)(t_grid)

        # ── μ: global maximum of first derivative ─────────────────────
        # R's gcFitSpline uses max(smooth.spline.deriv(fit, x=t_grid))
        # without any amplitude-band masking.
        idx    = int(np.nanargmax(dy))
        mu     = float(dy[idx])
        t_star = float(t_grid[idx])
        y_star = float(np.interp(t_star, t_grid, y_grid))

        # ── Baseline y0 = spline at t_min (matches R's spline baseline) ──
        y0_baseline = float(y_grid[0])

        # ── Amplitude A = max(y_grid) - y0_baseline ───────────────────
        A = float(np.nanmax(y_grid)) - y0_baseline

        # ── Lag: tangent intercept, negative values allowed ───────────
        if mu <= 1e-12:
            lag = float("nan")
            lag_method_str = "tangent_spline_undefined"
        else:
            lag = float(t_star - (y_star - y0_baseline) / mu)
            lag_method_str = "tangent_spline"

        integral = float(np.trapz(y_grid, t_grid))
        rss      = float(np.sum((y - sp(t)) ** 2))

        df_eff   = effective_df(sp, t)
        smooth_out = lam_to_spar(lam_used) if np.isfinite(lam_used) else float("nan")
        lam_out    = float(lam_used) if np.isfinite(lam_used) else float("nan")

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
            smooth_used=smooth_out,
            df_effective=df_eff,
            lam_raw=lam_out,
            lag_method=lag_method_str,
            y0_baseline=y0_baseline,
            fit_status="ok",
            fail_reason=None,
            warnings=warn_list if warn_list else None,
            extra={
                "lam":        lam_out,
                "s":          lam_out,
                "lam_method": lam_method,
                "knots":      np.asarray(getattr(sp, "t", []), float),
                "mu_method":  "spline_derivative",
                "t_star":     float(t_star),
                "y_star":     float(y_star),
                "y0":         float(y0_baseline),
            },
        )
    except Exception as e:
        return FitResult(
            method="spline", model="spline", success=False,
            message=f"Spline fit failed: {e}", n=int(len(t)),
            fit_status="failed", fail_reason="fit_exception",
        )