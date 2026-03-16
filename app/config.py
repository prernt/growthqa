# app/config.py
"""
Global constants and settings dataclasses.
No Streamlit or heavy scientific dependencies — safe to import first.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root & sys.path bootstrap
# ---------------------------------------------------------------------------
# config.py lives in <repo>/app/, so parents[1] is the repo root.
ROOT = Path(__file__).resolve().parents[1]

for _cand in {ROOT, ROOT / "src", Path.cwd(), Path.cwd() / "src"}:
    if _cand.exists():
        _sp = str(_cand)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

# ---------------------------------------------------------------------------
# Well-known paths
# ---------------------------------------------------------------------------
MODEL_DIR  = ROOT / "classifier_output" / "saved_models_selected"
TRAIN_META = ROOT / "data" / "train_data" / "meta.csv"


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InferenceSettings:
    """Pre-processing options, partially exposed in the UI."""
    # Blank handling
    input_is_raw: bool         = False
    global_blank: float | None = None

    # Fixed pipeline defaults (not shown in UI)
    step: float                = 0.5
    min_points: int            = 3
    low_res_threshold: int     = 7
    auto_tmax: bool            = False
    auto_tmax_coverage: float  = 0.8
    tmax_hours: float | None   = 16.0

    # Locked values
    clip_negatives: bool = False
    smooth_method: str   = "SGF"    # Savitzky-Golay
    smooth_window: int   = 5
    normalize: str       = "MINMAX"


@dataclass
class GrofitOptions:
    """Options forwarded to the Grofit fitting pipeline."""
    response_var: str          = "mu"
    have_atleast: int          = 6
    fit_opt: str               = "b"
    gc_boot_B: int             = 200
    dr_boot_B: int             = 300
    spline_auto_cv: bool       = True
    spline_s: float | None     = None   # legacy manual spline smoothing
    dr_s: float | None         = None   # legacy manual DR smoothing
    smooth_gc: float | None    = None   # Grofit-R smooth.gc spar ∈ (0,1]
    smooth_dr: float | None    = None   # Grofit-R smooth.dr spar ∈ (0,1]
    dr_x_transform: str | None = None
    dr_y_transform: str | None = None
    bootstrap_method: str      = None
