from __future__ import annotations
import re
from typing import Optional
import numpy as np
import pandas as pd

_TIME_TOKEN_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?\s*$", re.I)

def _to_float(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_time_any_to_hours(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) 

    s = str(x).strip().lower()
    if s in {"t0", "to", "t 0", "0"}:
        return 0.0

    numeric = _to_float(s)
    if np.isfinite(numeric):
        return float(numeric)

    m = _TIME_TOKEN_RE.match(s.replace(",", "."))
    if not m:
        return np.nan

    hh = float(m.group(1)) if m.group(1) else 0.0
    mm = float(m.group(2)) if m.group(2) else 0.0
    ss = float(m.group(3)) if m.group(3) else 0.0
    return hh + (mm / 60.0) + (ss / 3600.0)

WELL_RE = re.compile(r"\(([A-Z]\d{2})\)\s*$", re.I)

def split_condition_and_well(colname: str) -> tuple[str, str]:
    s = str(colname).strip()
    m = WELL_RE.search(s)
    if m:
        well = m.group(1).upper()
        cond = s[:m.start()].strip()
        return cond if cond else s, well
    return s, "UNK"


def infer_and_convert_numeric_time_to_hours(t: pd.Series, hinted_unit: Optional[str] = None) -> pd.Series:
    tt = pd.to_numeric(t, errors="coerce")
    tt = tt.astype(float)

    if hinted_unit:
        u = hinted_unit.lower()
        if u in {"s", "sec", "secs", "second", "seconds"}:
            return tt / 3600.0
        if u in {"m", "min", "mins", "minute", "minutes"}:
            return tt / 60.0
        if u in {"h", "hr", "hrs", "hour", "hours"}:
            return tt

    finite = tt[np.isfinite(tt)]
    if finite.empty:
        return tt

    finite_sorted = np.sort(finite.unique())
    if len(finite_sorted) >= 2:
        diffs = np.diff(finite_sorted)
        med_step = np.nanmedian(diffs) if diffs.size else np.nan
    else:
        med_step = np.nan

    mx = float(np.nanmax(finite_sorted))

    if np.isfinite(med_step) and med_step > 100:
        return tt / 3600.0

    if mx > 50 and mx <= 5000:
        return tt / 60.0

    return tt


def make_time_header(hours: float) -> str:
    return f"T{hours:.2f} (h)"