from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _norm_blank_flag(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if s in {"ALREADY", "YES", "TRUE", "1"}:
        return "ALREADY"
    if s in {"RAW", "NO", "FALSE", "0"}:
        return "RAW"
    return s


def load_blank_status_map(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    """
    Returns dict:
      { FileName: {"status": "ALREADY"|"RAW", "blank_value": float|None} }
    """
    if not path:
        return {}
    df = pd.read_csv(path)
    if "FileName" not in df.columns or "already_blank_subtracted" not in df.columns:
        raise ValueError("blank-status-csv must contain columns: FileName, already_blank_subtracted (and optional blank_value)")
    out: Dict[str, Dict[str, object]] = {}
    for _, r in df.iterrows():
        fname = str(r["FileName"]).strip()
        status = _norm_blank_flag(r["already_blank_subtracted"])
        bv = r["blank_value"] if "blank_value" in df.columns else np.nan
        blank_value = float(bv) if pd.notna(bv) else None
        if status not in {"ALREADY", "RAW"}:
            raise ValueError(f"Invalid already_blank_subtracted for FileName={fname}: {status} (use RAW or ALREADY)")
        out[fname] = {"status": status, "blank_value": blank_value}
    return out

