from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import numpy as np

@dataclass
class FitResult:
    method: Literal["parametric", "spline"]
    model: Optional[str]      
    success: bool
    message: str
    lag: Optional[float] = None        
    mu: Optional[float] = None         
    A: Optional[float] = None         
    integral: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    rss: Optional[float] = None
    n: Optional[int] = None
    k: Optional[int] = None
    params: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    smooth_used: Optional[float] = None    
    df_effective: Optional[float] = None   
    lam_raw: Optional[float] = None        
    lag_method: Optional[str] = None       
    y0_baseline: Optional[float] = None    
    fit_status: Optional[str] = None  
    fail_reason: Optional[str] = None
    warnings: Optional[List[str]] = field(default=None)
    mu_se: Optional[float] = None
    A_se: Optional[float] = None
    lag_se: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None