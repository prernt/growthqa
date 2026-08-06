from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.interpolate import make_smoothing_spline
from growthqa.grofit.types import FitResult


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


def _estimate_lam_for_target_df(t: np.ndarray, y: np.ndarray, target_df: float, n_search: int = 30) -> float:
    lam_lo, lam_hi = 1e-12, 1e6
    try:
        df_lo = effective_df(make_smoothing_spline(t, y, lam=lam_lo), t)
        df_hi = effective_df(make_smoothing_spline(t, y, lam=lam_hi), t)
    except Exception:
        return float("nan")
    if not (df_hi <= target_df <= df_lo):
        
        return lam_lo if abs(df_lo - target_df) < abs(df_hi - target_df) else lam_hi
    for _ in range(n_search):
        lam_mid = np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15))))
        try:
            df_mid = effective_df(make_smoothing_spline(t, y, lam=lam_mid), t)
        except Exception:
            break
        if df_mid >= target_df:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    return float(np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15)))))


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
            achieved_df = effective_df(sp_gcv, t)
            if achieved_df >= min_df:
               
                lam_est = _estimate_lam_for_target_df(t, y, target_df=achieved_df)
                return sp_gcv, lam_est, "gcv_ok"
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
        idx    = int(np.nanargmax(dy))
        mu     = float(dy[idx])
        t_star = float(t_grid[idx])
        y_star = float(np.interp(t_star, t_grid, y_grid))
        y0_baseline = float(y_grid[0])
        A = float(np.nanmax(y_grid)) - y0_baseline
        if mu <= 1e-12:
            lag = float("nan")
            lag_method_str = "tangent_spline_undefined"
        else:
            lag = float(t_star - (y_star - y0_baseline) / mu)
            lag_method_str = "tangent_spline"
        integral = float(np.trapezoid(y_grid, t_grid)) if hasattr(np, "trapezoid") else float(np.trapz(y_grid, t_grid))
        rss      = float(np.sum((y - sp(t)) ** 2))
        df_eff   = effective_df(sp, t)
        smooth_out = lam_to_spar(lam_used) if np.isfinite(lam_used) else float("nan")
        lam_out    = float(lam_used) if np.isfinite(lam_used) else float("nan")
        warn_list = []
        if lam_method == "gcv_bounded":
            warn_list.append("GCV over-smoothed; df floor enforced")
        if lam_method == "fallback":
            warn_list.append("GCV failed; variance-scaled fallback used")
        if lam_method == "gcv_ok":
            warn_list.append("smooth_used is an estimate matching GCV's effective_df, not GCV's internal lambda (scipy does not expose it)")

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