# src/growthqa/grofit/lowess.py
from __future__ import annotations
import numpy as np

def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """
    LOWESS smoothing used to compute robust start values.
    Tries statsmodels if available; otherwise falls back to a simple robust smoother.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
        y_hat = lowess(y_s, x_s, frac=frac, return_sorted=False)
        out = np.empty_like(y_hat)
        out[order] = y_hat
        return out
    except Exception:
        # fallback: moving average with window proportional to frac
        n = len(x_s)
        w = max(3, int(np.ceil(frac * n)))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        y_pad = np.pad(y_s, (pad, pad), mode="edge")
        kernel = np.ones(w) / w
        y_hat = np.convolve(y_pad, kernel, mode="valid")
        out = np.empty_like(y_hat)
        out[order] = y_hat
        return out
