# src/growthqa/grofit/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import numpy as np

ParametricModelName = Literal["logistic", "gompertz", "modified_gompertz", "richards"]

@dataclass
class FitResult:
    method: Literal["parametric", "spline"]
    model: Optional[str]          # best model name; "spline" for spline fits
    success: bool
    message: str

    # Grofit-aligned growth parameters
    lag: Optional[float] = None        # λ (lag time)
    mu: Optional[float] = None         # μ (max specific growth rate)
    A: Optional[float] = None          # maximum growth amplitude
    integral: Optional[float] = None   # area under curve (AUC)

    # Fit quality
    aic: Optional[float] = None
    bic: Optional[float] = None
    rss: Optional[float] = None
    n: Optional[int] = None
    k: Optional[int] = None

    # Raw solver objects (for debugging / bootstrap)
    params: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None

    # Spline-specific diagnostics
    smooth_used: Optional[float] = None    # spar-equivalent [0,1] actually applied
    df_effective: Optional[float] = None   # effective degrees of freedom
    lam_raw: Optional[float] = None        # raw SciPy λ (for reproducibility)

    # Lag method metadata
    lag_method: Optional[str] = None       # "analytical" | "geometric" | "tangent"
    y0_baseline: Optional[float] = None    # baseline y value used in lag computation

    # Pipeline status
    fit_status: Optional[str] = None       # "ok" | "fallback" | "failed" | "invalid"
    fail_reason: Optional[str] = None      # machine-readable reason code
    warnings: Optional[List[str]] = field(default=None)

    # Parametric SE (from pcov diagonal)
    mu_se: Optional[float] = None
    A_se: Optional[float] = None
    lag_se: Optional[float] = None

    # Arbitrary extra diagnostics dict
    extra: Optional[Dict[str, Any]] = None