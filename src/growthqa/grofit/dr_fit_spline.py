# src/growthqa/grofit/dr_fit_spline.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.linalg import svd

from .dr_fit_model import dr_fit_model
from .parametric_models import aic_from_rss
from .gc_fit_spline import effective_df, _find_bounded_lambda, DR_MIN_DF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _is_monotonic(deriv: np.ndarray, eps: float) -> bool:
    d = np.asarray(deriv, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return False
    return bool(np.all(d >= -eps) or np.all(d <= eps))


def _pick_ec50_crossing(
    x_grid: np.ndarray,
    y_hat: np.ndarray,
    dy: np.ndarray,
    target: float,
) -> tuple[float, str]:
    """Find EC50 crossing on the fitted DR curve.
    
    When multiple crossings exist, selects the one at steepest slope (most
    biologically meaningful). Returns (ec50_in_transformed_x, status_code).
    Status codes: OK | MULTI_STEEPEST | NO_CROSS_NEAREST | AMBIGUOUS
    """
    diff = np.asarray(y_hat, float) - float(target)
    xg = np.asarray(x_grid, float)
    dyg = np.asarray(dy, float)

    roots: list[tuple[float, float]] = []
    n = len(xg)
    for i in range(max(0, n - 1)):
        d0, d1 = float(diff[i]), float(diff[i + 1])
        x0, x1 = float(xg[i]), float(xg[i + 1])
        if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(x0) and np.isfinite(x1)):
            continue
        if d0 == 0.0:
            slope = float(np.interp(x0, xg, np.abs(dyg)))
            roots.append((x0, slope))
        if d0 * d1 < 0.0:
            if abs(d1 - d0) < 1e-12:
                xr = x0
            else:
                xr = x0 + (0.0 - d0) * (x1 - x0) / (d1 - d0)
            slope = float(np.interp(xr, xg, np.abs(dyg)))
            roots.append((float(xr), slope))
    if n > 0 and float(diff[-1]) == 0.0:
        xr = float(xg[-1])
        roots.append((xr, float(np.interp(xr, xg, np.abs(dyg)))))

    if not roots:
        idx = int(np.argmin(np.abs(diff)))
        return float(xg[idx]), "NO_CROSS_NEAREST"

    if len(roots) == 1:
        return float(roots[0][0]), "OK"

    slopes = np.array([r[1] for r in roots], dtype=float)
    best = float(np.nanmax(slopes))
    tie_mask = np.isclose(slopes, best, rtol=0.0, atol=max(1e-6, 0.01 * max(1.0, abs(best))))
    if int(np.sum(tie_mask)) > 1:
        return float("nan"), "AMBIGUOUS"
    best_idx = int(np.nanargmax(slopes))
    return float(roots[best_idx][0]), "MULTI_STEEPEST"


# def _effective_df(sp, x: np.ndarray) -> float:
#     """
#     Estimate effective degrees of freedom of the fitted spline by computing
#     trace(H) where H is the hat/influence matrix evaluated at the knot points.

#     This mirrors R's smooth.spline df computation and lets us detect when
#     GCV has over-smoothed to near-linear (df ≈ 2 means straight line).

#     For a cubic smoothing spline, df = trace(S) where S maps y → fitted values.
#     We approximate this via the sum of squared basis evaluations (fast proxy).
#     """
#     try:
#         # Evaluate the spline and its derivative on a fine grid at x locations.
#         # A flat spline has nearly constant derivative → very low effective df.
#         # We use the ratio of curvature variance to derivative variance as proxy.
#         y_fit = sp(x)
#         dy = sp.derivative(1)(x)
#         d2y = sp.derivative(2)(x)

#         # Ratio of second derivative energy to first derivative energy.
#         # For a straight line: d2y ≈ 0 → ratio ≈ 0 → df_proxy ≈ 2.
#         # For a sigmoid: d2y has two peaks → ratio is large → df_proxy > 3.
#         e1 = float(np.sum(dy ** 2))
#         e2 = float(np.sum(d2y ** 2))
#         if e1 < 1e-20:
#             return 2.0
#         # Heuristic that maps curvature ratio to df on [2, n]:
#         # df ≈ 2 + log1p(e2/e1 * n)
#         n = float(len(x))
#         df_proxy = 2.0 + np.log1p(e2 / e1 * n)
#         return float(np.clip(df_proxy, 2.0, n))
#     except Exception:
#         return 2.0  # safe fallback: assume straight line if computation fails


# def _find_bounded_lambda(
#     xt: np.ndarray,
#     y: np.ndarray,
#     min_df: float = 3.5,
#     n_search: int = 30,
# ) -> float:
#     """
#     Find the largest lambda (smoothest spline) that still achieves at least
#     min_df effective degrees of freedom. This replicates R smooth.spline's
#     implicit df floor that prevents GCV from collapsing to a straight line
#     on sparse dose-response data.

#     Strategy: binary search on log(lambda) between a very tight lambda
#     (rough, df ≈ n) and a very loose lambda (smooth, df ≈ 2). Return the
#     largest lambda where df >= min_df.
#     """
#     n = len(xt)

#     # Bracket: lam_lo → df near n (wiggly), lam_hi → df near 2 (flat)
#     lam_lo = 1e-12
#     lam_hi = 1e6

#     # Quick check: does even the tightest lambda give enough df?
#     try:
#         sp_test = make_smoothing_spline(xt, y, lam=lam_lo)
#         df_lo = _effective_df(sp_test, xt)
#     except Exception:
#         df_lo = float(n)

#     if df_lo < min_df:
#         # Data is too sparse/flat to achieve min_df even when very wiggly.
#         # Return tightest lambda and accept the result.
#         return lam_lo

#     # Binary search for the crossover point where df drops below min_df.
#     for _ in range(n_search):
#         lam_mid = np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15))))
#         try:
#             sp_mid = make_smoothing_spline(xt, y, lam=lam_mid)
#             df_mid = _effective_df(sp_mid, xt)
#         except Exception:
#             df_mid = 2.0

#         if df_mid >= min_df:
#             lam_lo = lam_mid   # can go smoother (larger lambda) and still meet df floor
#         else:
#             lam_hi = lam_mid   # too smooth, need tighter lambda

#     # Return the smoothest lambda that still meets the df floor.
#     return float(np.exp(0.5 * (np.log(max(lam_lo, 1e-15)) + np.log(max(lam_hi, 1e-15)))))


# def _select_lambda(
#     xt: np.ndarray,
#     y: np.ndarray,
#     lam: Optional[float],
#     auto_cv: bool,
#     min_df: float = 3.5,
# ) -> tuple[float, str]:
#     """
#     Central lambda selection logic — mirrors R smooth.spline behaviour:

#     1. If user provides explicit lam → use it directly (no CV).
#     2. If auto_cv=True and lam=None → attempt GCV via SciPy (lam=None).
#        Then check effective df. If df < min_df (GCV over-smoothed to line),
#        fall back to bounded binary search that enforces df >= min_df.
#     3. If auto_cv=False and lam=None → use variance-scaled deterministic
#        fallback (avoids GCV instability on very sparse data).

#     Returns (lam_fit, selection_method_label).
#     """
#     n = len(xt)

#     # --- Case 1: explicit user lambda ---
#     if lam is not None:
#         return float(max(float(lam), 0.0)), "user"

#     # --- Case 2: GCV with df floor (Grofit R-like behaviour) ---
#     if auto_cv:
#         try:
#             sp_gcv = make_smoothing_spline(xt, y, lam=None)
#             df_gcv = _effective_df(sp_gcv, xt)

#             if df_gcv >= min_df:
#                 # GCV gave a biologically sensible smoothness → use it.
#                 # Recover the lambda SciPy chose by back-solving (it doesn't
#                 # expose lam directly, so we approximate from the spline).
#                 # We just reuse the spline object and return lam=None sentinel.
#                 return float("nan"), "gcv_ok"   # sentinel: caller uses sp_gcv
#             else:
#                 # GCV over-smoothed (sparse data problem). Fall back to
#                 # bounded search that enforces df >= min_df.
#                 lam_bounded = _find_bounded_lambda(xt, y, min_df=min_df)
#                 return lam_bounded, "gcv_bounded"

#         except Exception:
#             # GCV itself failed → fall through to deterministic fallback.
#             pass

#     # --- Case 3: deterministic fallback ---
#     # Data-variance-scaled lambda. Robust for very sparse/noisy datasets.
#     lam_fallback = float(max(np.nanvar(y) * max(n, 1), 1e-10))
#     return lam_fallback, "fallback"

def _select_lambda(xt, y, lam, auto_cv, min_df=DR_MIN_DF):
    """Thin wrapper using the shared GC/DR lambda selection logic (item 2)."""
    from .gc_fit_spline import _select_lam_and_fit
    sp, lam_used, method = _select_lam_and_fit(xt, y, lam=lam, auto_cv=auto_cv, min_df=min_df)
    return sp, lam_used, method

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dr_fit_spline(
    conc: np.ndarray,
    resp: np.ndarray,
    x_transform: Optional[str] = "log1p",
    lam: Optional[float] = None,
    auto_cv: bool = True,
    *,
    enforce_monotonic: bool = True,
    fallback_to_4pl: bool = True,
    min_df: float = 3.5,
) -> Dict[str, Any]:
    """
    Fit a dose-response spline in the style of Grofit R's drFitSpline.

    Key behaviours matching Grofit R:
    - x transformed by log1p (or log10/none) before fitting, EC50 inverted back.
    - Spline fit directly on biological (non-normalised) y values.
    - GCV smoothing selection with a df floor (min_df=3.5) to prevent
      over-smoothing on sparse DR data (6-10 points), matching R's behaviour.
    - EC50 = concentration at midpoint of fitted curve endpoints (Grofit def).
    - EC50_x_transformed = EC50 in transformed x space (reported as 'EC50' in R).
    - EC50 = back-transformed to original concentration units (R's 'EC50.orig').
    - Monotonicity check with 4PL fallback if spline is non-monotonic.
    - fail_reason reported for all NA outputs.

    Parameters
    ----------
    conc : array of concentrations (original scale, not transformed).
    resp : array of response values (e.g., mu from growth curve fit).
    x_transform : 'log1p' | 'log10' | 'log' | 'none' | None.
    lam : explicit SciPy smoothing penalty. If None, auto-selected via GCV.
    auto_cv : if True, use GCV with df floor. If False, use deterministic fallback.
    enforce_monotonic : if True and spline is non-monotonic, fallback to 4PL.
    fallback_to_4pl : allow 4PL fallback (only active when enforce_monotonic=True).
    min_df : minimum effective degrees of freedom floor for GCV (default 3.5).
              Prevents GCV from producing a straight line on sparse data.
              Equivalent to R smooth.spline's implicit df constraint.
    """
    x = np.asarray(conc, float)
    y = np.asarray(resp, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return {
            "success": False,
            "message": "Need >=4 points for dose-response",
            "fail_reason": "insufficient_points",
            "n": len(x),
        }

    # Defensive: treat non-finite lambda inputs as "not provided".
    if lam is not None:
        try:
            lam_num = float(lam)
        except Exception:
            lam_num = np.nan
        lam = lam_num if np.isfinite(lam_num) else None

    # --- X transform (on original x, before deduplication) ---
    x_transform_norm = (x_transform or "none").strip().lower()
    if x_transform_norm in {"log10", "log"}:
        pos = x[x > 0]
        if pos.size == 0:
            return {
                "success": False,
                "message": "Need at least one positive concentration for log10 transform",
                "fail_reason": "no_positive_concentration",
                "n": len(x),
            }
        pseudo = max(float(np.nanmin(pos)) / 10.0, 1e-12)
        x_for_log = np.where(x > 0, x, pseudo)
        xt = np.log10(x_for_log)
    elif x_transform_norm == "log1p":
        xt = np.log1p(x)
    else:
        xt = x.copy()

    # --- Sort and dedup on transformed x ---
    order = np.argsort(xt)
    xt = xt[order]
    x = x[order]
    y = y[order]

    xt_u, inv = np.unique(xt, return_inverse=True)
    if xt_u.size != xt.size:
        y_sum = np.zeros_like(xt_u, dtype=float)
        x_sum = np.zeros_like(xt_u, dtype=float)
        cnt = np.zeros_like(xt_u, dtype=float)
        np.add.at(y_sum, inv, y)
        np.add.at(x_sum, inv, x)
        np.add.at(cnt, inv, 1.0)
        xt = xt_u
        y = y_sum / np.maximum(cnt, 1.0)
        x = x_sum / np.maximum(cnt, 1.0)

    if len(xt) < 5:
        return {
            "success": False,
            "message": "Need >=4 unique points for dose-response",
            "fail_reason": "insufficient_unique_points",
            "n": len(xt),
        }

    # --- Lambda selection (GCV with df floor, matching R smooth.spline) ---
    sp, lam_chosen, lam_method = _select_lambda(xt, y, lam=lam, auto_cv=auto_cv, min_df=min_df)

    # --- Fit spline on raw (non-normalised) y values ---
    # NOTE: We deliberately do NOT min-max normalise y here.
    # Grofit R fits the spline on raw biological values (e.g., mu in OD/h).
    # Normalising changes the effective lambda scale and shifts EC50 midpoint
    # computation, producing EC50 values inconsistent with R's drFitSpline.
    try:
        if lam_method == "gcv_ok":
            # GCV passed df floor check — refit cleanly with lam=None.
            sp = make_smoothing_spline(xt, y, lam=None)
            lam_out = float("nan")  # GCV-chosen, not user-exposed
        else:
            sp = make_smoothing_spline(xt, y, lam=lam_chosen)
            lam_out = float(lam_chosen) if np.isfinite(lam_chosen) else float("nan")

        grid = np.linspace(float(np.min(xt)), float(np.max(xt)), 2000)
        y_hat = sp(grid)

        # Analytical derivative on original axes (no chain rule needed — no normalisation).
        dy = sp.derivative(1)(grid)

        dy_abs = np.abs(dy)
        is_linear = bool(np.nanmax(dy_abs) < 1.5 * max(float(np.nanmean(dy_abs)), 1e-8))

        eps = max(1e-8, 0.02 * float(np.nanstd(dy)))
        monotonic = _is_monotonic(dy, eps=eps)

    except Exception as e:
        return {
            "success": False,
            "message": f"DR spline fit failed: {e}",
            "fail_reason": "fit_failed",
            "n": int(len(x)),
        }

    # --- Monotonicity fallback to 4PL ---
    if enforce_monotonic and ((not monotonic) or is_linear) and fallback_to_4pl:
        model_fit = dr_fit_model(x, y)
        if model_fit.get("success"):
            ec50_4pl = model_fit.get("ec50", np.nan)
            return {
                "success": True,
                "message": "ok",
                "n": int(len(x)),
                "x_transform": x_transform,
                "x_transform_norm": x_transform_norm,
                "method": "4pl_fallback",
                "dr_monotonic": True,
                "lam_method": "4pl",
                # EC50 in transformed x (same scale as x-axis of fitted curve)
                "ec50_x_transformed": ec50_4pl,
                # EC50 in original concentration units (Grofit EC50.orig)
                "ec50": ec50_4pl,   # 4PL operates on original x, no inversion needed
                "ec50_status": "OK",
                "y_ec50": model_fit.get("y_ec50", np.nan),
                "endpoint_low": model_fit.get("bottom", np.nan),
                "endpoint_high": model_fit.get("top", np.nan),
                "aic": model_fit.get("aic", np.nan),
                "rss": model_fit.get("rss", np.nan),
                "lam": np.nan,
                "s": np.nan,
                "x_grid": model_fit.get("x_grid"),
                "y_hat": model_fit.get("y_hat"),
                "fail_reason": None,
            }
        return {
            "success": False,
            "message": "Spline non-monotonic and 4PL fallback also failed",
            "fail_reason": "non_monotonic_no_fallback",
            "n": int(len(x)),
        }

    # --- EC50: midpoint of fitted curve endpoints (Grofit R definition) ---
    # Grofit: target = (y_min_fitted + y_max_fitted) / 2, NOT global min/max of data.
    # This matches R drFitSpline behaviour.
    y0 = float(y_hat[0])    # response at lowest concentration
    y1 = float(y_hat[-1])   # response at highest concentration
    target = 0.5 * (y0 + y1)

    ec50_xt, ec50_status = _pick_ec50_crossing(grid, y_hat, dy, target)

    # --- Back-transform EC50 to original concentration units ---
    # ec50_x_transformed: EC50 in the transformed x-space (what R reports as 'xEC50')
    # ec50:               EC50 in original concentration units (what R reports as 'EC50.orig')
    if np.isfinite(ec50_xt):
        if x_transform_norm in {"log10", "log"}:
            ec50_orig = float(np.power(10.0, ec50_xt))
        elif x_transform_norm == "log1p":
            ec50_orig = float(np.expm1(ec50_xt))
        else:
            ec50_orig = float(ec50_xt)
    else:
        ec50_orig = float("nan")

    if not np.isfinite(ec50_orig):
        ec50_status = ec50_status if ec50_status != "OK" else "no_ec50_crossing"

    y_ec50 = float(np.interp(ec50_xt, grid, y_hat)) if np.isfinite(ec50_xt) else float("nan")

    # --- Residuals and fit quality on raw y ---
    y_fit_at_data = sp(xt)
    residual = y - y_fit_at_data
    rss = float(np.sum(residual ** 2))
    # k=4 matches Grofit's spline AIC convention (cubic spline ≈ 4 effective params)
    aic = float(aic_from_rss(rss, int(len(y)), 4))

    # --- Effective df of final fit (diagnostic) ---
    df_final = effective_df(sp, xt)

    fail_reason = None
    if not np.isfinite(ec50_orig):
        fail_reason = "no_ec50_crossing"
    elif ec50_status == "AMBIGUOUS":
        fail_reason = "ambiguous_ec50"

    return {
        "success": True,
        "message": "ok",
        "fail_reason": fail_reason,
        "n": int(len(x)),
        "x_transform": x_transform,
        "x_transform_norm": x_transform_norm,
        "method": "spline",
        "lam_method": lam_method,
        "lam": lam_out,
        "s": lam_out,           # backward-compat alias
        "effective_df": df_final,

        # Grofit R parity: ec50_x_transformed = xEC50 (in transformed space)
        #                  ec50 = EC50.orig (in original concentration units)
        "ec50_x_transformed": float(ec50_xt) if np.isfinite(ec50_xt) else float("nan"),
        "ec50": ec50_orig,
        "ec50_status": ec50_status,
        "y_ec50": y_ec50,

        "endpoint_low": y0,
        "endpoint_high": y1,
        "target_response": target,

        "dr_monotonic": bool(monotonic),
        "aic": aic,
        "rss": rss,
        "x_grid": grid,          # in transformed x-space
        "y_hat": y_hat,          # fitted response values
    }
