# app/utils.py
"""
Pure utility functions used by the Streamlit layer: numeric helpers, label
normalisation, concentration parsing, and sample-data generators.
No Streamlit dependency.

Model loading, runtime/version checking and the legacy-pickle alias shim live
once in the pipeline layer (growthqa.pipelines.infer_labels); they are not
duplicated here.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        return default if s == "" else float(s)
    except Exception:
        return default


def to_numeric_scalar(x) -> float:
    return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])


def normalize_bootstrap_method(v: object) -> str:
    s = str(v).strip().lower()
    return s if s in {"pairs", "residual"} else "pairs"


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def normalize_label(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    if s in {"valid", "true", "1"}:
        return "Valid"
    if s in {"invalid", "false", "0"}:
        return "Invalid"
    if s in {"unsure", "unknown"}:
        return "Unsure"
    return str(v).strip()

def resolve_display_label(row, *, fallback: str = "") -> str:
    """
    Single source of truth for 'what label applies to this curve right now'.
    Priority: human review (true_label) > combined Stage-1+Stage-2 decision
    (final_label) > Stage-1-only Pred Label. Used wherever a label needs to
    be read for display or for deciding what feeds downstream, so all call
    sites agree on the same order instead of each re-deriving it.
    """
    getter = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    for key in ("true_label", "True Label", "final_label", "Final Label (S1+S2)", "Pred Label", "pred_label"):
        val = getter(key, None)
        if val is not None and str(val).strip():
            return normalize_label(val)
    return fallback


def label_is_valid(label: object) -> bool:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return False
    return str(label).strip().lower() in {"valid", "true", "1"}


# ---------------------------------------------------------------------------
# Concentration / curve-ID helpers
# ---------------------------------------------------------------------------

def extract_conc_from_curve_id(curve_id: str) -> float | None:
    """Parse concentration from well headers like ``A01[Conc=0.1]`` or ``A01[0.1]``."""
    if curve_id is None:
        return None
    m = re.search(
        r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]",
        str(curve_id), flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def make_curve_key(test_id: str, concentration: object) -> str:
    conc = "" if concentration is None or pd.isna(concentration) else str(concentration)
    return f"{test_id}|{conc}"


# ---------------------------------------------------------------------------
# Sample data generators
# ---------------------------------------------------------------------------

def make_sample_wide_csv_bytes() -> bytes:
    """Grofit-ready wide-format sample (one row per curve, concentration column)."""
    data = {
        "Test Id":       ["A01", "A02", "A03", "A04", "A05", "A06"],
        "concentration": [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        "T0.0 (h)": [0.05] * 6,
        "T1.0 (h)": [0.070, 0.070, 0.068, 0.065, 0.060, 0.055],
        "T2.0 (h)": [0.120, 0.115, 0.105, 0.095, 0.075, 0.060],
        "T3.0 (h)": [0.220, 0.205, 0.185, 0.155, 0.095, 0.070],
        "T4.0 (h)": [0.350, 0.320, 0.280, 0.220, 0.115, 0.080],
        "T5.0 (h)": [0.480, 0.430, 0.360, 0.260, 0.130, 0.090],
        "T6.0 (h)": [0.600, 0.520, 0.410, 0.290, 0.140, 0.095],
        "T7.0 (h)": [0.700, 0.580, 0.440, 0.310, 0.145, 0.098],
        "T8.0 (h)": [0.780, 0.620, 0.460, 0.320, 0.148, 0.100],
    }
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


def make_sample_long_csv_bytes() -> bytes:
    """Long-format sample: Time column + wells as columns with encoded concentrations."""
    times = [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
    cols = [
        ("A01[Conc=0.0]", 0.0), ("A02[0.1]",     0.1),
        ("A03[Conc=0.3]", 0.3), ("A04[1.0]",     1.0),
        ("A05[Conc=3.0]", 3.0), ("A06[10.0]",   10.0),
    ]
    base = np.array([0.05, 0.055, 0.065, 0.080, 0.110, 0.200,
                     0.270, 0.340, 0.410, 0.480, 0.600, 0.700, 0.780])
    strength_map = {0.0: 0.00, 0.1: 0.05, 0.3: 0.15, 1.0: 0.35, 3.0: 0.60, 10.0: 0.80}

    def _inhibited(strength: float) -> list:
        damp = 1.0 - strength
        y = base.copy()
        y[4:] = y[4:] * damp + 0.05 * (1.0 - damp)
        return y.tolist()

    data = {"Time (h)": times}
    for col_name, conc in cols:
        data[col_name] = _inhibited(strength_map[conc])
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")