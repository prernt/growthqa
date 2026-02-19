# src/growthqa/grofit/parametric_models.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
from .lowess import lowess_smooth

# --------- Model functions ---------
def logistic(t, y0, A, k, t0):
    # y(t) = y0 + A / (1 + exp(-k (t - t0)))
    return y0 + A / (1.0 + np.exp(-k * (t - t0)))

def gompertz(t, y0, A, k, t0):
    # y(t) = y0 + A * exp(-exp(-k (t - t0)))
    return y0 + A * np.exp(-np.exp(-k * (t - t0)))

def modified_gompertz(t, y0, A, mu, lam):
    # Common microbiology form:
    # y = y0 + A * exp( -exp( (mu*e/A)*(lam - t) + 1 ) )
    e = np.e
    return y0 + A * np.exp(-np.exp((mu * e / max(A, 1e-12)) * (lam - t) + 1.0))

def richards(t, y0, A, k, t0, nu):
    # Richards generalized logistic:
    # y = y0 + A / (1 + nu*exp(-k(t-t0)))^(1/nu)
    nu = np.clip(nu, 1e-6, 50.0)
    return y0 + A / np.power((1.0 + nu * np.exp(-k * (t - t0))), 1.0 / nu)

# --------- Utilities ---------
def aic_from_rss(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k

def _finite_diff_max_slope(t_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
    dy = np.gradient(y_grid, t_grid)
    idx = int(np.nanargmax(dy))
    mu = float(dy[idx])
    t_star = float(t_grid[idx])
    return mu, t_star

def _estimate_lag_from_tangent(t_star: float, y_star: float, mu: float, y0: float) -> float:
    # Lag time = x-intercept of tangent line at max slope:
    # y0 = y_star - mu*(t_star - lag)  => lag = t_star - (y_star - y0)/mu
    if mu <= 1e-12:
        return float("nan")
    return float(t_star - (y_star - y0) / mu)

@dataclass
class ModelSpec:
    name: str
    func: Callable
    n_params: int
    bounds: Tuple[np.ndarray, np.ndarray]

def get_model_specs(t: np.ndarray, y: np.ndarray) -> Dict[str, ModelSpec]:
    # bounds chosen to be broad but stable
    y0_min = float(np.nanmin(y)) - 2.0
    y0_max = float(np.nanmax(y)) + 2.0
    A_max = max(1e-6, float(np.nanmax(y) - np.nanmin(y)) * 10.0)
    t_min = float(np.nanmin(t))
    t_max = float(np.nanmax(t))

    specs = {
        "logistic": ModelSpec(
            "logistic", logistic, 4,
            (np.array([y0_min, 0.0, 1e-6, t_min]),
             np.array([y0_max, A_max, 50.0, t_max]))
        ),
        "gompertz": ModelSpec(
            "gompertz", gompertz, 4,
            (np.array([y0_min, 0.0, 1e-6, t_min]),
             np.array([y0_max, A_max, 50.0, t_max]))
        ),
        "modified_gompertz": ModelSpec(
            "modified_gompertz", modified_gompertz, 4,
            (np.array([y0_min, 0.0, 1e-6, t_min]),
             np.array([y0_max, A_max, 50.0, t_max]))
        ),
        "richards": ModelSpec(
            "richards", richards, 5,
            (np.array([y0_min, 0.0, 1e-6, t_min, 1e-3]),
             np.array([y0_max, A_max, 50.0, t_max, 20.0]))
        ),
    }
    return specs

def start_values_lowess(t: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Starting values inspired by grofit: smooth with LOWESS, infer baseline, amplitude, and a rough midpoint.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y_s = lowess_smooth(t, y, frac=0.25)

    y0 = float(np.nanpercentile(y_s, 5))
    y_end = float(np.nanpercentile(y_s, 95))
    A = max(1e-6, y_end - y0)

    # approximate midpoint time where y crosses y0 + 0.5*A
    target = y0 + 0.5 * A
    order = np.argsort(t)
    tt = t[order]
    yy = y_s[order]
    idx = np.argmin(np.abs(yy - target))
    t0 = float(tt[idx])

    # rough slope at midpoint
    if len(tt) >= 3:
        dy = np.gradient(yy, tt)
        k = float(np.nanmax(dy) / max(A, 1e-12))
        k = np.clip(k, 1e-3, 10.0)
    else:
        k = 0.5

    # modified gompertz needs mu, lam
    mu_guess = float(np.nanmax(np.gradient(yy, tt))) if len(tt) >= 3 else 0.5
    mu_guess = float(np.clip(mu_guess, 1e-6, 50.0))
    # lag guess: early time
    lam_guess = float(np.nanpercentile(tt, 20))

    starts = {
        "logistic": np.array([y0, A, k, t0], dtype=float),
        "gompertz": np.array([y0, A, k, t0], dtype=float),
        "modified_gompertz": np.array([y0, A, mu_guess, lam_guess], dtype=float),
        "richards": np.array([y0, A, k, t0, 1.0], dtype=float),
    }
    return starts

def extract_grofit_params_from_curve(
    model_name: str,
    t: np.ndarray,
    y0: float,
    A: float,
    fitted_func: Callable[[np.ndarray], np.ndarray],
    t_min: float,
    t_max: float,
) -> Dict[str, float]:
    """
    Derive mu, lag, integral from fitted curve numerically (stable across models).
    """
    t_grid = np.linspace(t_min, t_max, 400)
    y_grid = fitted_func(t_grid)
    mu, t_star = _finite_diff_max_slope(t_grid, y_grid)
    y_star = float(np.interp(t_star, t_grid, y_grid))
    lag = _estimate_lag_from_tangent(t_star, y_star, mu, y0)

    integral = float(np.trapz(y_grid, t_grid))

    # For A: we prefer amplitude in growth sense: max(y) - baseline
    A_est = float(np.nanmax(y_grid) - y0)
    A_est = max(A_est, 0.0)

    return {"mu": mu, "lag": lag, "A": A_est, "integral": integral}
