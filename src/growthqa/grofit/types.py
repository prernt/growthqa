# src/growthqa/grofit/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import numpy as np

ParametricModelName = Literal["logistic", "gompertz", "modified_gompertz", "richards"]

@dataclass
class FitResult:
    method: Literal["parametric", "spline"]
    model: Optional[str]  # for parametric best model; for spline use "spline"
    success: bool
    message: str

    # parameters (Grofit-like)
    lag: Optional[float] = None        # lambda
    mu: Optional[float] = None         # max slope / growth rate
    A: Optional[float] = None          # max growth (amplitude)
    integral: Optional[float] = None   # area under curve

    # fit quality
    aic: Optional[float] = None
    rss: Optional[float] = None
    n: Optional[int] = None
    k: Optional[int] = None

    # raw objects for debugging
    params: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    extra: Optional[Dict[str, Any]] = None
