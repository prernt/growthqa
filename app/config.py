"""
Global constants and settings dataclasses.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _cand in {ROOT, ROOT / "src", Path.cwd(), Path.cwd() / "src"}:
    if _cand.exists():
        _sp = str(_cand)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

from growthqa.config import (
    MODEL_DIR,
    TRAIN_META_CSV as TRAIN_META,
    STEP_HOURS as _STEP_HOURS,
    MIN_POINTS as _MIN_POINTS,
    TMAX_HOURS as _TMAX_HOURS,
    SMOOTH_METHOD as _SMOOTH_METHOD,
    SMOOTH_WINDOW as _SMOOTH_WINDOW,
    RANDOM_STATE as _RANDOM_STATE,
    NORMALIZE as _NORMALIZE,
)

@dataclass
class InferenceSettings:
    step: float = _STEP_HOURS
    min_points: int = _MIN_POINTS
    auto_tmax: bool = False
    auto_tmax_coverage: float = 0.8
    tmax_hours: float | None = _TMAX_HOURS

    # Locked values
    clip_negatives: bool = False
    smooth_method: str = _SMOOTH_METHOD
    smooth_window: int = _SMOOTH_WINDOW
    normalize: str = _NORMALIZE


@dataclass
class GrofitOptions:
    response_var: str = "mu"
    have_atleast: int = 6
    fit_opt: str = "b"
    gc_boot_B: int = 200
    dr_boot_B: int = 300
    spline_auto_cv: bool = True
    spline_s: float | None = None 
    dr_s: float | None = None
    smooth_gc: float | None = None
    smooth_dr: float | None = None
    dr_x_transform: str | None = None
    dr_y_transform: str | None = None
    bootstrap_method: str = None
    random_state: int = _RANDOM_STATE


