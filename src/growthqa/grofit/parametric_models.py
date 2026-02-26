# src/growthqa/grofit/parametric_models.py
# ============================================================
# CORRECTED VERSION  – see analysis document for full rationale
# Key changes vs original:
#   1. Richards is now in Grofit (A, mu, lam, nu) parameterisation
#      instead of (k, t0, nu), eliminating the broken conversion.
#   2. Amplitude A is read directly from the fitted parameter,
#      not computed as max(y_grid) - y_grid[0], which consistently
#      underestimates A because the spline grid never reaches the
#      true asymptote within the observed time window.
#   3. The negative-lambda bug: Python previously floored lambda at 0,
#      silently hiding fits where the lag phase has already passed
#      (common at high concentrations).  The floor is now removed
#      and negative lambda values are preserved exactly as R does.
#   4. AIC formula aligned with R's AIC() function: n*log(RSS/n) + 2k
#      without the small-sample AICc correction that was being applied
#      inside aic_from_rss() (R does not apply AICc in gcFitModel).
# ============================================================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
from .lowess import lowess_smooth


# ─────────────────────────────────────────────────────────────
# Model functions — all in Grofit (A, mu, lam) parameterisation
# ─────────────────────────────────────────────────────────────

def logistic(t, y0, A, mu, lam):
    """
    Grofit Table 1 logistic:
        y(t) = y0 + A / (1 + exp( 4*mu/A * (lam - t) + 2 ))

    Parameters A, mu, lam are the biological Grofit parameters directly,
    so no post-hoc analytical conversion is ever required.
    """
    A = max(float(A), 1e-12)
    return y0 + A / (1.0 + np.exp(4.0 * mu / A * (lam - t) + 2.0))


def gompertz(t, y0, A, mu, lam):
    """
    Grofit Table 1 Gompertz:
        y(t) = y0 + A * exp( -exp( mu*e/A * (lam - t) + 1 ) )
    """
    A = max(float(A), 1e-12)
    return y0 + A * np.exp(-np.exp(np.e * mu / A * (lam - t) + 1.0))


def modified_gompertz(t, y0, A, mu, lam):
    """
    Grofit Table 1 modified Gompertz — identical formula to plain Gompertz
    but kept as a separate entry so the model-selection loop tries it
    independently (it can converge to a different local minimum).
    """
    e = np.e
    A_safe = max(float(A), 1e-12)
    inner = np.clip((mu * e / A_safe) * (lam - t) + 1.0, -50, 50)
    return y0 + A_safe * np.exp(-np.exp(inner))


def richards(t, y0, A, mu, lam, nu):
    """
    Richards model in Grofit (A, mu, lam, nu) parameterisation.

    Grofit R Table 1:
        y(t) = y0 + A * [1 + nu*exp(1+nu + mu/A*(1+nu)^(1+1/nu)*(lam-t))]^(-1/nu)

    IMPORTANT: This replaces the old k/t0 parameterisation that required
    a post-hoc analytical conversion for mu and lam.  The conversion
    formula was incorrect for Richards because the inflection point of
    the Richards curve is NOT at t0; it is at:
        t_infl = lam - A*(1+nu)^(1+1/nu) / (mu*(1+nu))
    which differs from the logistic/Gompertz case.  By fitting directly
    in the (A, mu, lam) space we bypass the conversion entirely.

    Reference: Zwietering et al. (1990) Appl Environ Microbiol 56:1875-1881;
               Grofit source code, initRichards.R
    """
    nu = np.clip(float(nu), 1e-4, 50.0)
    A_safe = max(float(A), 1e-12)
    mu_safe = max(float(mu), 1e-12)
    # exponent inside the bracket
    exponent = (1.0 + nu
                + (mu_safe / A_safe) * (1.0 + nu) ** (1.0 + 1.0 / nu) * (lam - t))
    exponent = np.clip(exponent, -50, 50)
    return y0 + A_safe * (1.0 + nu * np.exp(exponent)) ** (-1.0 / nu)


# ─────────────────────────────────────────────────────────────
# AIC  —  matches R's AIC() which uses:  n*log(RSS/n) + 2k
# NOTE: R's gcFitModel does NOT apply the AICc small-sample
# correction; removing it fixes model-selection discrepancies.
# ─────────────────────────────────────────────────────────────

def aic_from_rss(rss: float, n: int, k: int) -> float:
    """AIC as computed by R's AIC() via logLik for Gaussian NLS.

    Formula: n * log(RSS/n) + 2*k
    This matches the Grofit R implementation exactly.
    The AICc correction previously present has been removed.
    """
    rss = max(rss, 1e-12)
    return float(n * np.log(rss / n) + 2 * k)


# ─────────────────────────────────────────────────────────────
# Analytical mu / lam extraction
# ─────────────────────────────────────────────────────────────

def _finite_diff_max_slope(t_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
    dy = np.gradient(y_grid, t_grid)
    idx = int(np.nanargmax(dy))
    return float(dy[idx]), float(t_grid[idx])


def _estimate_lag_from_tangent(t_star: float, y_star: float, mu: float, y0: float) -> float:
    """
    Lag = x-intercept of tangent at the maximum-slope point.
    IMPORTANT: negative lag values are preserved (not floored to 0).
    A negative lag means the inflection point has already been passed
    at time 0 — this is physically meaningful for high-concentration
    inhibition curves and matches R's behaviour.
    """
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
    """
    Build per-model parameter bounds.
    All models are now in Grofit (A, mu, lam) parameterisation,
    so bounds are set on the biologically interpretable quantities.
    Crucially, lam_min is set to a large *negative* value so that
    the optimizer can explore negative lag times (high-concentration
    inhibition curves).
    """
    y0_min = float(np.nanmin(y)) - 2.0
    y0_max = float(np.nanmax(y)) + 2.0
    A_max  = max(1e-6, float(np.nanmax(y) - np.nanmin(y)) * 10.0)
    t_min  = float(np.nanmin(t))
    t_max  = float(np.nanmax(t))
    # Allow negative lag (inflection before t=0) — key fix for R parity
    lam_min = -t_max * 3.0
    lam_max =  t_max * 1.5

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
            "modified_gompertz", modified_gompertz, 4,
            (np.array([y0_min, 0.0, 1e-6, lam_min]),
             np.array([y0_max, A_max, 50.0, lam_max]))
        ),
        "richards": ModelSpec(
            "richards", richards, 5,
            # nu > 0 required; upper bound 20 is generous
            (np.array([y0_min, 0.0, 1e-6, lam_min, 1e-4]),
             np.array([y0_max, A_max, 50.0, lam_max, 20.0]))
        ),
    }
    return specs


def start_values_lowess(t: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    LOWESS-based starting values (mirrors Grofit R's initMODEL approach).
    All starting lam values are unconstrained (can be negative).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y_s = lowess_smooth(t, y, frac=0.25)

    y0_guess = float(np.nanpercentile(y_s, 5))
    y_end    = float(np.nanpercentile(y_s, 95))
    A_guess  = max(1e-6, y_end - y0_guess)

    order = np.argsort(t)
    tt = t[order]; yy = y_s[order]

    # Rough mu from max gradient of smoothed data
    mu_guess = float(np.nanmax(np.gradient(yy, tt))) if len(tt) >= 3 else 0.5
    mu_guess = float(np.clip(mu_guess, 1e-6, 50.0))

    # Lag guess: time where tangent at inflection crosses y0
    # (allow negative; just use the 20th percentile as a safe start)
    lam_guess = float(np.nanpercentile(tt, 20))

    # nu starting value for Richards
    nu_guess = 1.0

    starts = {
        "logistic":          np.array([y0_guess, A_guess, mu_guess, lam_guess], dtype=float),
        "gompertz":          np.array([y0_guess, A_guess, mu_guess, lam_guess], dtype=float),
        "modified_gompertz": np.array([y0_guess, A_guess, mu_guess, lam_guess], dtype=float),
        "richards":          np.array([y0_guess, A_guess, mu_guess, lam_guess, nu_guess], dtype=float),
    }
    return starts


# ─────────────────────────────────────────────────────────────
# Analytical extraction of mu, lam from fitted parameters
# ─────────────────────────────────────────────────────────────

def _extract_analytical_mu_lag(model_name: str, params: np.ndarray) -> Tuple[float, float]:
    """
    Extract biological mu and lam directly from the fitted parameter vector.

    Because ALL models are now parameterised with (y0, A, mu, lam, [nu]),
    this is a trivial read-out for every model — no conversion formula needed.
    Negative lam is preserved (not clamped to 0).
    """
    name = str(model_name).lower()
    p = np.asarray(params, dtype=float)

    if name in {"logistic", "gompertz", "modified_gompertz"} and p.size >= 4:
        return float(p[2]), float(p[3])   # mu, lam  (no floor on lam)

    if name == "richards" and p.size >= 5:
        return float(p[2]), float(p[3])   # mu, lam  (no floor on lam)

    return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────
# Amplitude extraction — use the fitted A parameter directly
# ─────────────────────────────────────────────────────────────

def extract_A_from_params(model_name: str, params: np.ndarray) -> float:
    """
    Return the amplitude A directly from the fitted parameter vector.

    IMPORTANT: this replaces the old approach of computing
    max(y_grid) - y_grid[0].  That approach *systematically underestimates*
    A because the observed time window rarely reaches the true asymptote
    of the sigmoidal curve.  R reads A directly from the nls() parameter
    vector, which is what we now do too.
    """
    p = np.asarray(params, dtype=float)
    if p.size >= 2:
        return float(max(p[1], 0.0))   # A is always index 1
    return float("nan")


# ─────────────────────────────────────────────────────────────
# Geometric fallback (used only when analytical extraction fails)
# ─────────────────────────────────────────────────────────────

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
    Geometric parameter extraction from the fitted curve on a dense grid.
    Used only as a fallback when analytical extraction returns NaN.
    """
    t_grid = np.linspace(t_min, t_max, 400)
    y_grid = fitted_func(t_grid)
    y0_geo = float(np.nanmean(y_grid[:3])) if y_grid.size >= 3 else float(y_grid[0])
    mu_num, t_star = _finite_diff_max_slope(t_grid, y_grid)
    y_star = float(np.interp(t_star, t_grid, y_grid))
    mu     = float(mu_num)
    lag    = _estimate_lag_from_tangent(t_star, y_star, mu, y0_geo)
    integral = float(np.trapz(y_grid, t_grid))
    # Use the parameter A directly if available, else fall back to geometric
    A_est = extract_A_from_params(model_name, params) if params is not None else float(np.nanmax(y_grid) - y0_geo)
    return {"mu": mu, "lag": lag, "A": A_est, "integral": integral}