# src/growthqa/grofit/parametric_models.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
from .lowess import lowess_smooth

# --------- Model functions ---------
# def logistic(t, y0, A, k, t0):
#     # y(t) = y0 + A / (1 + exp(-k (t - t0)))
#     return y0 + A / (1.0 + np.exp(-k * (t - t0)))

# def gompertz(t, y0, A, k, t0):
#     # y(t) = y0 + A * exp(-exp(-k (t - t0)))
#     return y0 + A * np.exp(-np.exp(-k * (t - t0)))


def logistic(t, y0, A, mu, lam):
    """Grofit R Table 1 logistic form: params are directly A, mu, lambda.
    y = y0 + A / (1 + exp( 4*mu/A * (lam - t) + 2 ))
    Advantage over k/t0 form: mu and lam are the Grofit parameters by construction,
    no post-hoc analytical conversion needed (item 6).
    """
    A = max(float(A), 1e-12)
    return y0 + A / (1.0 + np.exp(4.0 * mu / A * (lam - t) + 2.0))


def gompertz(t, y0, A, mu, lam):
    """Grofit R Table 1 Gompertz form: params are directly A, mu, lambda.
    y = y0 + A * exp( -exp( mu*e/A * (lam - t) + 1 ) )
    Same as modified_gompertz with alpha=0 (item 6).
    """
    A = max(float(A), 1e-12)
    return y0 + A * np.exp(-np.exp(np.e * mu / A * (lam - t) + 1.0))

# def modified_gompertz(t, y0, A, mu, lam):
#     # Common microbiology form:
#     # y = y0 + A * exp( -exp( (mu*e/A)*(lam - t) + 1 ) )
#     e = np.e
#     return y0 + A * np.exp(-np.exp((mu * e / max(A, 1e-12)) * (lam - t) + 1.0))

# def modified_gompertz(t, y0, A, mu, lam, alpha=0.0, tshift=None):
#     """Full Grofit R modified Gompertz with optional second rise."""
#     e = np.e
#     if tshift is None:
#         tshift = float(np.max(t)) if hasattr(t, '__len__') else 0.0
#     part1 = y0 + A * np.exp(-np.exp((mu * e / max(A, 1e-12)) * (lam - t) + 1.0))
#     part2 = A * np.exp(np.clip(alpha * (t - tshift), -50, 50))
#     return part1 + part2

def modified_gompertz(t, y0, A, mu, lam):
    """Grofit R Table 1 modified Gompertz (4-parameter Grofit form).
    y = y0 + A * exp( -exp( mu*e/A * (lam - t) + 1 ) )
    Parameters mu and lam are the Grofit biological parameters directly.
    Identical to the Grofit R grofit() implementation.
    """
    e = np.e
    A_safe = max(float(A), 1e-12)
    inner = np.clip((mu * e / A_safe) * (lam - t) + 1.0, -50, 50)
    return y0 + A_safe * np.exp(-np.exp(inner))


def richards(t, y0, A, k, t0, nu):
    # Richards generalized logistic:
    # y = y0 + A / (1 + nu*exp(-k(t-t0)))^(1/nu)
    nu = np.clip(nu, 1e-6, 50.0)
    return y0 + A / np.power((1.0 + nu * np.exp(-k * (t - t0))), 1.0 / nu)

# --------- Utilities ---------
def aic_from_rss(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    # Apply small-sample correction (AICc) when n/k < 40
    if n > k + 1:
        aic += (2 * k * (k + 1)) / (n - k - 1)
    return aic


def _finite_diff_max_slope(t_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
    dy = np.gradient(y_grid, t_grid)
    idx = int(np.nanargmax(dy))
    mu = float(dy[idx])
    t_star = float(t_grid[idx])
    return mu, t_star

# def _estimate_lag_from_tangent(t_star: float, y_star: float, mu: float, y0: float) -> float:
#     # Lag time = x-intercept of tangent line at max slope:
#     # y0 = y_star - mu*(t_star - lag)  => lag = t_star - (y_star - y0)/mu
#     if mu <= 1e-12:
#         return float("nan")
#     return float(t_star - (y_star - y0) / mu)
def _estimate_lag_from_tangent(t_star: float, y_star: float, mu: float, y0: float) -> float:
    if mu <= 1e-12:
        return float("nan")
    lag = float(t_star - (y_star - y0) / mu)
    return float(max(0.0, lag))

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
    # mu_max = 50.0

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
            "modified_gompertz", modified_gompertz, 6,
            (np.array([y0_min, 0.0, 1e-6, t_min,]),
            np.array([y0_max, A_max, 50.0, t_max,]))
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
        "logistic": np.array([y0, A, mu_guess, lam_guess], dtype=float),
        "gompertz": np.array([y0, A, mu_guess, lam_guess], dtype=float),
        "modified_gompertz": np.array([y0, A, mu_guess, lam_guess], dtype=float),
        # "modified_gompertz": np.array([y0, A, mu_guess, lam_guess, 0.0, float(np.nanpercentile(tt, 80))], dtype=float),
        "richards": np.array([y0, A, k, t0, 1.0], dtype=float),
    }
    return starts

# def _extract_analytical_mu_lag(
#     model_name: str,
#     params: np.ndarray,
# ) -> tuple[float, float]:
#     name = str(model_name).lower()
#     p = np.asarray(params, dtype=float)

#     if name == "modified_gompertz" and p.size >= 4:
#         return float(p[2]), float(p[3])

#     if name in {"logistic", "gompertz"} and p.size >= 4:
#         A = float(p[1])
#         k = float(p[2])
#         t0 = float(p[3])
#         if k <= 1e-12 or A <= 0.0:
#             return float("nan"), float("nan")
#         if name == "logistic":
#             mu = (A * k) / 4.0
#             lag = t0 - (2.0 / k)
#             return float(mu), float(max(0.0, lag))
#         mu = (A * k) / np.e
#         lag = t0 - (1.0 / k)
#         return float(mu), float(max(0.0, lag))

#     if name == "richards" and p.size >= 5:
#         A = float(p[1])
#         k = float(p[2])
#         t0 = float(p[3])
#         nu = float(p[4])
#         if k <= 1e-12 or A <= 0.0 or nu <= 1e-12:
#             return float("nan"), float("nan")
#         mu = A * k * np.power(1.0 + nu, -(1.0 / nu) - 1.0)
#         lag = t0 - ((1.0 + nu) / k)
#         return float(mu), float(max(0.0, lag))

#     return float("nan"), float("nan")

def _extract_analytical_mu_lag(model_name: str, params: np.ndarray) -> tuple[float, float]:
    """Extract mu and lambda analytically from fitted parameters.

    For Grofit-parameterized models (logistic_grofit, gompertz_grofit,
    modified_gompertz), mu and lam ARE the parameters — trivial extraction.
    For richards (k/t0 form), compute via closed-form derivation.
    """
    name = str(model_name).lower()
    p = np.asarray(params, dtype=float)

    # Grofit-form models: params are [y0, A, mu, lam, ...] — direct extraction (item 6)
    if name in {"logistic", "gompertz", "modified_gompertz"} and p.size >= 4:
        mu_val, lam_val = float(p[2]), float(p[3])
        return float(mu_val), float(max(0.0, lam_val))

    # Richards still uses k/t0 form — analytical conversion
    if name == "richards" and p.size >= 5:
        A = float(p[1])
        k = float(p[2])
        t0 = float(p[3])
        nu = float(p[4])
        if k <= 1e-12 or A <= 0.0 or nu <= 1e-12:
            return float("nan"), float("nan")
        mu = A * k * np.power(1.0 + nu, -(1.0 / nu) - 1.0)
        lag = t0 - ((1.0 + nu) / k)
        return float(mu), float(max(0.0, lag))

    return float("nan"), float("nan")

def extract_grofit_params_from_curve(
    model_name: str,
    t: np.ndarray,
    y0: float,
    A: float,
    fitted_func: Callable[[np.ndarray], np.ndarray],
    t_min: float,
    t_max: float,
    params: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Derive grofit-aligned parameters geometrically from the fitted curve grid.
    This mirrors Grofit parity: mu from max slope, lag from tangent intercept.
    """
    t_grid = np.linspace(t_min, t_max, 400)
    y_grid = fitted_func(t_grid)
    # Grofit-parity baseline from the fitted curve start (robust mean of first 3 grid points).
    y0_geo = float(np.nanmean(y_grid[:3])) if y_grid.size >= 3 else float(y_grid[0])
    mu_num, t_star = _finite_diff_max_slope(t_grid, y_grid)
    y_star = float(np.interp(t_star, t_grid, y_grid))
    mu = float(mu_num)
    lag = _estimate_lag_from_tangent(t_star, y_star, mu, y0_geo)

    integral = float(np.trapz(y_grid, t_grid))

    # Grofit-parity amplitude from fitted max minus geometric baseline.
    A_est = float(np.nanmax(y_grid) - y0_geo)

    return {"mu": mu, "lag": lag, "A": A_est, "integral": integral}
