from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

def standardize_long(df_long: pd.DataFrame, file_stem: str) -> pd.DataFrame:
    """
    Required columns after standardization:
      FileName, orig_TestId, Model Name, Is_Valid, time_h, OD
    """
    needed = {"time_h", "OD"}
    if not needed.issubset(df_long.columns):
        raise ValueError(f"Long DF missing required columns {needed}. Got: {df_long.columns.tolist()}")

    out = df_long.copy()

    if "FileName" not in out.columns:
        out["FileName"] = file_stem
    if "orig_TestId" not in out.columns:
        out["orig_TestId"] = "curve"
    if "Model Name" not in out.columns:
        out["Model Name"] = out["orig_TestId"].astype(str)
    if "Is_Valid" not in out.columns:
        out["Is_Valid"] = True

    out["OD"] = pd.to_numeric(out["OD"], errors="coerce")
    out["time_h"] = pd.to_numeric(out["time_h"], errors="coerce")
    if "Concentration" in out.columns:
        out["Concentration"] = pd.to_numeric(out["Concentration"], errors="coerce")
    else:
        conc = out["orig_TestId"].astype(str).str.extract(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]")[0]
        conc = pd.to_numeric(conc, errors="coerce")
        if conc.notna().any():
            out["Concentration"] = conc
    out = out.dropna(subset=["time_h", "OD"]).copy()

    # Ensure non-negative time
    out = out[out["time_h"] >= 0].copy()
    cols = ["FileName", "orig_TestId", "Model Name", "Is_Valid"]
    if "Concentration" in out.columns:
        cols.append("Concentration")
    cols += ["time_h", "OD"]
    return out[cols]
