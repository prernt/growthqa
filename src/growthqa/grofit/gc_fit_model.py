# src/growthqa/grofit/gc_fit_model.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import curve_fit

from .types import FitResult
from .parametric_models import (
    get_model_specs, start_values_lowess, aic_from_rss, extract_grofit_params_from_curve
)

def gc_fit_model(t: np.ndarray, y: np.ndarray) -> FitResult:
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 4:
        return FitResult(method="parametric", model=None, success=False,
                         message="Too few points for parametric fit", n=len(t))

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    specs = get_model_specs(t, y)
    starts = start_values_lowess(t, y)

    best: Optional[FitResult] = None
    t_min, t_max = float(np.min(t)), float(np.max(t))

    for name, spec in specs.items():
        p0 = starts.get(name, None)
        if p0 is None:
            continue

        try:
            popt, pcov = curve_fit(
                spec.func, t, y, p0=p0,
                bounds=spec.bounds,
                maxfev=20000
            )
            y_hat = spec.func(t, *popt)
            rss = float(np.sum((y - y_hat) ** 2))
            n = int(len(t))
            k = int(spec.n_params)
            aic = float(aic_from_rss(rss, n, k))

            # baseline & amplitude from parameters:
            y0 = float(popt[0])
            # A parameter is popt[1] for all our models
            A_param = float(popt[1])

            fitted_func = lambda tt, f=spec.func, p=popt: f(tt, *p)
            derived = extract_grofit_params_from_curve(
                model_name=name,
                t=t,
                y0=y0,
                A=A_param,
                fitted_func=fitted_func,
                t_min=t_min,
                t_max=t_max,
            )

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
                rss=rss,
                n=n,
                k=k,
                params=np.asarray(popt, float),
                cov=np.asarray(pcov, float),
                extra={"y0": y0, "A_param": A_param},
            )

            if best is None or (res.aic is not None and best.aic is not None and res.aic < best.aic):
                best = res

        except Exception as e:
            continue

    if best is None:
        return FitResult(method="parametric", model=None, success=False,
                         message="All parametric fits failed", n=int(len(t)))

    return best
