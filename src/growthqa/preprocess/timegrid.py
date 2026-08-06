from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

_TIME_COL_RE = re.compile(r"^\s*T\s*([0-9]+(?:\.[0-9]+)?)\s*\(h\)\s*$", re.I)


def parse_time_from_header(col: str) -> Optional[float]:
    if not isinstance(col, str):
        return None
    m = _TIME_COL_RE.match(col.strip())
    if not m:
        return None
    return float(m.group(1))


def make_header_from_times(t_grid: np.ndarray) -> List[str]:
    headers: List[str] = []
    for t in t_grid:
        ts = f"{float(t):.2f}".rstrip("0").rstrip(".")
        if "." not in ts:
            ts = f"{ts}.0"
        headers.append(f"T{ts} (h)")
    return headers


def get_time_columns(df) -> List[str]:
    return [str(c) for c in df.columns if parse_time_from_header(str(c)) is not None]


def get_sorted_time_columns(df) -> List[str]:
    """Same as get_time_columns, but sorted by parsed time value."""
    cols = get_time_columns(df)
    return sorted(cols, key=lambda c: float(parse_time_from_header(str(c)) or 0.0))


def build_common_grid(step_hours: float, tmax_hours: float) -> np.ndarray:
    """Canonical 0..tmax grid at a fixed step. tmax is always supplied."""
    tmax = max(0.0, float(tmax_hours))
    n = max(int(np.floor(tmax / step_hours + 1e-9)) + 1, 1)
    return step_hours * np.arange(n, dtype=float)