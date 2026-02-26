# src/growthqa/grofit/gc_fit_model.py
# ============================================================
# CORRECTED VERSION v3  – multi-start fitting + degeneracy guard
#
# Changes vs v2:
#   1. _multi_start_fit(): tries multiple (p0) combinations per model
#      and keeps the best non-degenerate solution (lowest RSS).
#   2. _is_degenerate(): rejects fits whose SE > 1000× |parameter|
#      — the hallmark of a near-singular Jacobian (e.g. concentration
#      3 µM Richards blowup in v2 where mu_se = 5935, mu = 0.078).
#   3. _inhibition_seeds(): generates negative-λ starting points for
#      heavily inhibited curves, recovering the R solution at
#      30, 100, 300 µM.
#   4. Richards multi-start varies ν ∈ {0.5, 1.0, 2.0} across all
#      seed vectors to escape the local-minimum trap seen at 3 µM.
# ============================================================
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import curve_fit

from .types import FitResult
from .parametric_models import (
    get_model_specs, start_values_lowess, aic_from_rss,
    extract_grofit_params_from_curve, _extract_analytical_mu_lag,
    extract_A_from_params,
)


# ─────────────────────────────────────────────────────────────────────────────
# Degeneracy guard
# ─────────────────────────────────────────────────────────────────────────────

def _is_degenerate(popt: np.ndarray, pcov: np.ndarray,
                   se_ratio_threshold: float = 1000.0) -> bool:
    """
    Return True if any parameter's standard error exceeds
    `se_ratio_threshold` × |parameter value|.

    This catches near-singular Jacobian situations where scipy's
    curve_fit nominally converges but the solution is meaningless.
    R's nls() reports a similar singular gradient warning and grofit
    discards the fit; we do the same.

    Example: concentration 3 µM Richards in v2 returned
        mu=0.078, mu_se=5935  →  ratio=76000  →  degenerate=True
    """
    try:
        perr = np.sqrt(np.diag(pcov))
        for se, p in zip(perr, popt):
            if not np.isfinite(se) or not np.isfinite(p):
                return True
            if abs(p) > 1e-12 and se / abs(p) > se_ratio_threshold:
                return True
        return False
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Inhibition-aware negative-λ seed generation
# ─────────────────────────────────────────────────────────────────────────────

def _inhibition_seeds(t: np.ndarray, y: np.ndarray,
                      base_start: np.ndarray,
                      t_max: float) -> List[np.ndarray]:
    """
    Detect heavily inhibited curves and generate additional starting
    points with negative λ values.

    Detection criteria (either triggers):
      A) Relative amplitude (max−min) / max(y) < 15 % — near-flat curve.
      B) The lowess-derived λ guess ≥ 80 % of t_max — inflection so late
         it may be outside the observation window.

    For such curves, seeds with λ ∈ {−t_max, −t_max/2, −t_max/4} are
    added so the optimiser can explore the negative-lag basin where R
    finds the solution for 30, 100, 300 µM.
    """
    seeds: List[np.ndarray] = []
    amp     = float(np.nanmax(y) - np.nanmin(y))
    y_max   = float(np.nanmax(y))
    lam_base = float(base_start[3]) if len(base_start) > 3 else 0.0

    is_inhibited = (
        (y_max > 1e-10 and amp / y_max < 0.15)   # criterion A: low amplitude
        or lam_base >= 0.8 * t_max                 # criterion B: late inflection
    )

    if is_inhibited:
        for neg_lam in (-t_max, -t_max / 2.0, -t_max / 4.0):
            s = base_start.copy()
            s[3] = neg_lam
            seeds.append(s)

    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Multi-start fit for a single model
# ─────────────────────────────────────────────────────────────────────────────

def _multi_start_fit(
    name: str,
    spec,
    t: np.ndarray,
    y: np.ndarray,
    base_start: np.ndarray,
    t_max: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Try multiple starting-value sets for `spec`.  Return the (popt, pcov)
    pair with the lowest RSS that passes the degeneracy guard.

    Starting sets tried:
      1. base_start  (lowess-derived, same as v2)
      2. Inhibition seeds with negative λ (new in v3)
      3. For Richards only: ν ∈ {0.5, 2.0} variants of every seed above
         (ν=1.0 is the base; varying ν escapes the local-minimum trap)

    Returns None when every attempt either raises an exception or is
    flagged as degenerate.
    """
    candidates: List[np.ndarray] = [base_start]
    candidates.extend(_inhibition_seeds(t, y, base_start, t_max))

    # Richards: also vary ν across all current candidates
    if name == "richards":
        nu_variants: List[np.ndarray] = []
        for seed in candidates:
            for nu_val in (0.5, 2.0):   # nu=1.0 already in base
                s = seed.copy()
                s[4] = nu_val
                nu_variants.append(s)
        candidates.extend(nu_variants)

    lb = np.asarray(spec.bounds[0], float)
    ub = np.asarray(spec.bounds[1], float)

    best_rss  = float("inf")
    best_popt: Optional[np.ndarray] = None
    best_pcov: Optional[np.ndarray] = None

    for p0 in candidates:
        # Clip to interior of bounds to avoid immediate rejection
        p0_clipped = np.clip(p0, lb + 1e-9, ub - 1e-9)
        try:
            popt, pcov = curve_fit(
                spec.func, t, y,
                p0=p0_clipped,
                bounds=spec.bounds,
                maxfev=20_000,
                method="trf",   # Trust Region Reflective — handles bounds robustly
            )
        except Exception:
            continue

        if _is_degenerate(popt, pcov):
            continue

        y_hat = spec.func(t, *popt)
        rss   = float(np.sum((y - y_hat) ** 2))
        if rss < best_rss:
            best_rss  = rss
            best_popt = popt
            best_pcov = pcov

    if best_popt is None:
        return None
    return best_popt, best_pcov


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def gc_fit_model(t: np.ndarray, y: np.ndarray) -> FitResult:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    if len(t) < 4:
        return FitResult(
            method="parametric", model=None, success=False,
            message="Too few points for parametric fit", n=len(t),
            fit_status="failed", fail_reason="insufficient_points",
        )

    order  = np.argsort(t)
    t, y   = t[order], y[order]
    t_min  = float(np.min(t))
    t_max  = float(np.max(t))

    specs  = get_model_specs(t, y)
    starts = start_values_lowess(t, y)
    best: Optional[FitResult] = None
    aic_table: List[Dict[str, Any]] = []

    for name, spec in specs.items():
        p0_base = starts.get(name)
        if p0_base is None:
            aic_table.append({"model": name, "aic": float("nan"), "status": "no_start"})
            continue

        # ── Multi-start fit ───────────────────────────────────────────────
        result = _multi_start_fit(name, spec, t, y, p0_base, t_max)

        if result is None:
            aic_table.append({"model": name, "aic": float("nan"), "status": "exception"})
            continue

        popt, pcov = result

        # ── Standard errors ───────────────────────────────────────────────
        try:
            perr       = np.sqrt(np.diag(pcov))
            mu_se_val  = float(perr[2]) if np.isfinite(perr[2]) else np.nan
            A_se_val   = float(perr[1]) if np.isfinite(perr[1]) else np.nan
            lag_se_val = float(perr[3]) if np.isfinite(perr[3]) else np.nan
        except Exception:
            perr = np.full_like(popt, np.nan)
            mu_se_val = A_se_val = lag_se_val = np.nan

        # ── Bounds check ─────────────────────────────────────────────────
        lb, ub = spec.bounds
        hit_bounds = bool(
            np.any(np.isclose(popt, lb, atol=1e-5, rtol=0.0))
            or np.any(np.isclose(popt, ub, atol=1e-5, rtol=0.0))
        )

        y_hat    = spec.func(t, *popt)
        rss      = float(np.sum((y - y_hat) ** 2))
        n        = int(len(t))
        k        = int(spec.n_params)
        aic      = float(aic_from_rss(rss, n, k))
        bic      = float(n * np.log(max(rss, 1e-12) / n) + k * np.log(n))

        aic_table.append({
            "model": name, "aic": aic, "bic": bic, "rss": rss,
            "hit_bounds": hit_bounds, "status": "bounds" if hit_bounds else "ok",
        })

        if hit_bounds:
            continue

        y0_param = float(popt[0])
        A_param  = float(popt[1])

        # ── Parameter extraction ─────────────────────────────────────────
        ana_mu, ana_lam = _extract_analytical_mu_lag(name, popt)

        if np.isfinite(ana_mu) and np.isfinite(ana_lam):
            A_out    = extract_A_from_params(name, popt)
            t_grid   = np.linspace(t_min, t_max, 400)
            y_grid   = spec.func(t_grid, *popt)
            integral = float(np.trapz(y_grid, t_grid))
            y0_geo   = float(np.nanmean(y_grid[:3]))
            derived  = {
                "mu":       ana_mu,
                "lag":      ana_lam,
                "A":        A_out,
                "integral": integral,
            }
            lag_method_str = "analytical"
        else:
            fitted_func = lambda tt, f=spec.func, p=popt: f(tt, *p)
            derived = extract_grofit_params_from_curve(
                model_name=name, t=t, y0=y0_param, A=A_param,
                fitted_func=fitted_func, t_min=t_min, t_max=t_max,
                params=np.asarray(popt, float),
            )
            lag_method_str = "geometric"
            t_grid  = np.linspace(t_min, t_max, 400)
            y_grid  = spec.func(t_grid, *popt)
            y0_geo  = float(np.nanmean(y_grid[:3]))

        res = FitResult(
            method="parametric",
            model=name,
            success=True,
            message="ok",
            lag=float(derived["lag"]),
            mu=float(derived["mu"]),
            A=float(derived["A"]),
            integral=float(derived["integral"]),
            aic=aic,
            bic=bic,
            rss=rss,
            n=n,
            k=k,
            params=np.asarray(popt, float),
            cov=np.asarray(pcov, float),
            lag_method=lag_method_str,
            y0_baseline=y0_geo,
            fit_status="ok",
            fail_reason=None,
            mu_se=mu_se_val,
            A_se=A_se_val,
            lag_se=lag_se_val,
            extra={
                "y0_param":     y0_param,
                "A_param":      A_param,
                "param_se":     perr.tolist(),
                "mu_se":        float(perr[2]) if len(perr) > 2 else float("nan"),
                "A_se":         float(perr[1]) if len(perr) > 1 else float("nan"),
                "start_values": p0_base.tolist(),
                "lowess_start": True,
                "aic_table":    None,   # filled after loop
            },
        )

        if best is None or (
            res.aic is not None and best.aic is not None and res.aic < best.aic
        ):
            best = res

    # ── No valid fit ─────────────────────────────────────────────────────────
    if best is None:
        return FitResult(
            method="parametric", model=None, success=False,
            message="All parametric fits failed", n=int(len(t)),
            fit_status="failed", fail_reason="all_models_failed",
            extra={"aic_table": aic_table},
        )

    # ── AIC table enrichment ──────────────────────────────────────────────────
    best_aic    = best.aic or float("nan")
    valid_aics  = [r["aic"] for r in aic_table if np.isfinite(r.get("aic", float("nan")))]
    second_best = float("nan")
    if len(valid_aics) >= 2:
        sv = sorted(valid_aics)
        second_best = sv[1] if sv[0] == best_aic else sv[0]

    for row in aic_table:
        row["delta_aic"] = float(row.get("aic", float("nan"))) - best_aic

    if best.extra is not None:
        best.extra["aic_table"]        = aic_table
        best.extra["aic_best"]         = best_aic
        best.extra["aic_second"]       = second_best
        best.extra["delta_aic_second"] = second_best - best_aic
        best.extra["start_values"]     = starts.get(best.model, np.array([])).tolist()
        best.extra["lowess_start"]     = True

    return best