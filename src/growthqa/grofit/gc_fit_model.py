# src/growthqa/grofit/gc_fit_model.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List
from scipy.optimize import curve_fit

from .types import FitResult
from .parametric_models import (
    get_model_specs, start_values_lowess, aic_from_rss,
    extract_grofit_params_from_curve, _extract_analytical_mu_lag,
)


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

    order = np.argsort(t)
    t, y = t[order], y[order]

    specs = get_model_specs(t, y)
    starts = start_values_lowess(t, y)         # LOWESS-derived starts (item 7)
    best: Optional[FitResult] = None
    t_min, t_max = float(np.min(t)), float(np.max(t))

    # ── Per-model diagnostics table (item 8) ─────────────────────────────
    aic_table: List[Dict[str, Any]] = []

    for name, spec in specs.items():
        p0 = starts.get(name)
        if p0 is None:
            aic_table.append({"model": name, "aic": float("nan"), "status": "no_start"})
            continue

        try:
            popt, pcov = curve_fit(
                spec.func, t, y, p0=p0,
                bounds=spec.bounds,
                maxfev=20000,
            )
            try:
                perr = np.sqrt(np.diag(pcov))
                # Extract SE for Grofit-form models (params are [y0, A, mu, lam, ...])
                mu_se_val = float(perr[2]) if perr.size > 2 and np.isfinite(perr[2]) else np.nan
                A_se_val = float(perr[1]) if perr.size > 1 and np.isfinite(perr[1]) else np.nan
                lag_se_val = float(perr[3]) if perr.size > 3 and np.isfinite(perr[3]) else np.nan
            except Exception:
                perr = np.full_like(popt, np.nan)

            lb, ub = spec.bounds
            hit_bounds = bool(
                np.any(np.isclose(popt, lb, atol=1e-5, rtol=0.0))
                or np.any(np.isclose(popt, ub, atol=1e-5, rtol=0.0))
            )

            y_hat = spec.func(t, *popt)
            rss = float(np.sum((y - y_hat) ** 2))
            n = int(len(t))
            k = int(spec.n_params)
            aic = float(aic_from_rss(rss, n, k))
            bic = float(n * np.log(max(rss, 1e-12) / n) + k * np.log(n))

            aic_table.append({
                "model": name, "aic": aic, "bic": bic, "rss": rss,
                "hit_bounds": hit_bounds, "status": "bounds" if hit_bounds else "ok",
            })

            if hit_bounds:
                continue   # reject pegged-bounds fits

            y0 = float(popt[0])
            A_param = float(popt[1])

            # ── Parameter derivation (item 6) ────────────────────────────
            # Analytical formulas (exact Grofit R parity) preferred;
            # geometric fallback only when analytical returns NaN.
            ana_mu, ana_lag = _extract_analytical_mu_lag(name, np.asarray(popt, float))

            if np.isfinite(ana_mu) and np.isfinite(ana_lag):
                fitted_func = lambda tt, f=spec.func, p=popt: f(tt, *p)
                t_grid = np.linspace(t_min, t_max, 400)
                y_grid = fitted_func(t_grid)
                # ── Lag baseline (item 3): always use mean of first 3 grid points ──
                # Consistent with spline path; NOT popt[0] which can differ from
                # the measured start of the curve.
                y0_geo = float(np.nanmean(y_grid[:3]))
                A_est = float(np.nanmax(y_grid)) - y0_geo
                integral = float(np.trapz(y_grid, t_grid))
                derived = {"mu": ana_mu, "lag": ana_lag, "A": A_est, "integral": integral}
                lag_method_str = "analytical"
            else:
                fitted_func = lambda tt, f=spec.func, p=popt: f(tt, *p)
                derived = extract_grofit_params_from_curve(
                    model_name=name, t=t, y0=y0, A=A_param,
                    fitted_func=fitted_func, t_min=t_min, t_max=t_max,
                    params=np.asarray(popt, float),
                )
                lag_method_str = "geometric"

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
                # new fields
                lag_method=lag_method_str,
                y0_baseline=float(np.nanmean(np.linspace(t_min, t_max, 400)[:3]) if False else
                                  float(np.nanmean([spec.func(t_min + i*(t_max-t_min)/399, *popt)
                                                    for i in range(3)]))),
                fit_status="ok",
                fail_reason=None,
                mu_se=mu_se_val,
                A_se=A_se_val,
                lag_se=lag_se_val,
                extra={
                    "y0_param": y0,
                    "A_param": A_param,
                    "param_se": perr.tolist(),
                    "mu_se": float(perr[2]) if len(perr) > 2 else float("nan"),
                    "A_se": float(perr[1]) if len(perr) > 1 else float("nan"),
                    # item 7: store LOWESS-derived start values
                    "start_values": p0.tolist(),
                    "lowess_start": True,
                    # item 8: per-model AIC snapshot (filled after loop)
                    "aic_table": None,  # filled below after loop
                },
            )

            if best is None or (res.aic is not None and best.aic is not None and res.aic < best.aic):
                best = res

        except Exception:
            aic_table.append({"model": name, "aic": float("nan"), "status": "exception"})
            continue

    if best is None:
        return FitResult(
            method="parametric", model=None, success=False,
            message="All parametric fits failed", n=int(len(t)),
            fit_status="failed", fail_reason="all_models_failed",
            extra={"aic_table": aic_table},
        )

    # ── AIC table enrichment (item 8): add ΔAIC ──────────────────────────
    best_aic = best.aic or float("nan")
    valid_aics = [row["aic"] for row in aic_table if np.isfinite(row.get("aic", float("nan")))]
    second_best_aic = float("nan")
    if len(valid_aics) >= 2:
        sorted_aics = sorted(valid_aics)
        second_best_aic = sorted_aics[1] if sorted_aics[0] == best_aic else sorted_aics[0]

    for row in aic_table:
        row["delta_aic"] = float(row.get("aic", float("nan"))) - best_aic

    if best.extra is not None:
        best.extra["aic_table"] = aic_table
        best.extra["aic_best"] = best_aic
        best.extra["aic_second"] = second_best_aic
        best.extra["delta_aic_second"] = second_best_aic - best_aic
        best.extra["start_values"] = starts.get(best.model, np.array([])).tolist()
        best.extra["lowess_start"] = True

    return best