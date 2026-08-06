from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
from growthqa.grofit.lowess import lowess_smooth


def logistic(t, y0, A, mu, lam):
    A = max(float(A), 1e-12)
    return y0 + A / (1.0 + np.exp(4.0 * mu / A * (lam - t) + 2.0))


def gompertz(t, y0, A, mu, lam):
    A = max(float(A), 1e-12)
    return y0 + A * np.exp(-np.exp(np.e * mu / A * (lam - t) + 1.0))


def modified_gompertz(t, y0, A, mu, lam, alpha, t_shift):
    e = np.e
    A_safe = max(float(A), 1e-12)
    inner = np.clip((mu * e / A_safe) * (lam - t) + 1.0, -50, 50)
    second = np.clip(alpha * (t - t_shift), -50, 50)
    return y0 + A_safe * np.exp(-np.exp(inner)) + A_safe * np.exp(second)

def richards(t, y0, A, mu, lam, nu):
    nu = np.clip(float(nu), 1e-4, 50.0)
    A_safe = max(float(A), 1e-12)
    mu_safe = max(float(mu), 1e-12)
    exponent = (1.0 + nu
                + (mu_safe / A_safe) * (1.0 + nu) ** (1.0 + 1.0 / nu) * (lam - t))
    exponent = np.clip(exponent, -50, 50)
    return y0 + A_safe * (1.0 + nu * np.exp(exponent)) ** (-1.0 / nu)


def aic_from_rss(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-12)
    return float(n * np.log(rss / n) + 2 * k)


def _finite_diff_max_slope(t_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
    dy = np.gradient(y_grid, t_grid)
    idx = int(np.nanargmax(dy))
    return float(dy[idx]), float(t_grid[idx])


def _estimate_lag_from_tangent(t_star: float, y_star: float, mu: float, y0: float) -> float:
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
    y0_min = float(np.nanmin(y)) - 2.0
    y0_max = float(np.nanmax(y)) + 2.0
    A_max = max(1e-6, float(np.nanmax(y) - np.nanmin(y)) * 10.0)
    t_max = float(np.nanmax(t))
    lam_min = -t_max * 3.0
    lam_max = t_max * 1.5

    specs = {
        "logistic": ModelSpec(
            "logistic", logistic, 4,
            (np.array([y0_min, 0.0, 1e-6, lam_min]),
             np.array([y0_max, A_max, 50.0, lam_max]))
        ),
        "gompertz": ModelSpec(
            "gompertz", gompertz, 4,
            (np.array([y0_min, 0.0, 1e-6, lam_min]),
             np.array([y0_max, A_max, 50.0, lam_max]))
        ),
        "modified_gompertz": ModelSpec(
            "modified_gompertz", modified_gompertz, 6,
            (np.array([y0_min, 0.0, 1e-6, lam_min, 1e-6, 0.0]),
             np.array([y0_max, A_max, 50.0, lam_max, 10.0, t_max * 1.5]))
        ),
        "richards": ModelSpec(
            "richards", richards, 5,
            (np.array([y0_min, 0.0, 1e-6, lam_min, 1e-4]),
             np.array([y0_max, A_max, 50.0, lam_max, 20.0]))
        ),
    }
    return specs


def start_values_lowess(t: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y_s = lowess_smooth(t, y, frac=0.25)

    y0_guess = float(np.nanpercentile(y_s, 5))
    y_end = float(np.nanpercentile(y_s, 95))
    A_guess = max(1e-6, y_end - y0_guess)

    order = np.argsort(t)
    tt = t[order] 
    yy = y_s[order]

    mu_guess = float(np.nanmax(np.gradient(yy, tt))) if len(tt) >= 3 else 0.5
    mu_guess = float(np.clip(mu_guess, 1e-6, 50.0))
    lam_guess = float(np.nanpercentile(tt, 20))
    nu_guess = 1.0
    alpha_guess = float(np.clip(mu_guess * 0.3, 1e-6, 10.0))
    t_shift_guess = float(np.nanpercentile(tt, 75))

    starts = {
        "logistic": np.array([y0_guess, A_guess, mu_guess, lam_guess], dtype=float),
        "gompertz": np.array([y0_guess, A_guess, mu_guess, lam_guess], dtype=float),
        "modified_gompertz": np.array([y0_guess, A_guess, mu_guess, lam_guess,
                                        alpha_guess, t_shift_guess], dtype=float),
        "richards": np.array([y0_guess, A_guess, mu_guess, lam_guess, nu_guess], dtype=float),
    }
    return starts


def _extract_analytical_mu_lag(model_name: str, params: np.ndarray) -> Tuple[float, float]:
    name = str(model_name).lower()
    p = np.asarray(params, dtype=float)

    if name in {"logistic", "gompertz", "modified_gompertz"} and p.size >= 4:
        return float(p[2]), float(p[3])   

    if name == "richards" and p.size >= 5:
        return float(p[2]), float(p[3])

    return float("nan"), float("nan")


def extract_A_from_params(model_name: str, params: np.ndarray) -> float:
    p = np.asarray(params, dtype=float)
    if p.size >= 2:
        return float(max(p[1], 0.0))   # A is always index 1
    return float("nan")


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
    t_grid = np.linspace(t_min, t_max, 400)
    y_grid = fitted_func(t_grid)
    y0_geo = float(np.nanmean(y_grid[:3])) if y_grid.size >= 3 else float(y_grid[0])
    mu_num, t_star = _finite_diff_max_slope(t_grid, y_grid)
    y_star = float(np.interp(t_star, t_grid, y_grid))
    mu     = float(mu_num)
    lag    = _estimate_lag_from_tangent(t_star, y_star, mu, y0_geo)
    integral = float(np.trapezoid(y_grid, t_grid)) if hasattr(np, "trapezoid") else float(np.trapz(y_grid, t_grid))
    A_est = extract_A_from_params(model_name, params) if params is not None else float(np.nanmax(y_grid) - y0_geo)
    return {"mu": mu, "lambda": lag, "A": A_est, "integral": integral}