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
from growthqa.preprocess.timegrid import get_time_columns

REQUIRED_META_COLS = ["FileName", "Test Id", "Model Name", "Is_Valid"]


def coerce_is_valid_bool(series: pd.Series) -> pd.Series:
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        s = str(x).strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f"}:
            return False
        return False
    return series.map(to_bool)


def load_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: Missing required columns: {missing}")

    tcols = get_time_columns(df)
    if not tcols:
        raise ValueError(f"{path}: No time columns like 'T0.00 (h)' found.")

    # strict boolean Is_Valid
    df["Is_Valid"] = coerce_is_valid_bool(df["Is_Valid"])

    # keep only known columns
    keep = REQUIRED_META_COLS + tcols
    df = df[keep].copy()

    # numeric timeseries
    for c in tcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_and_concat_wides(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            # logging.warning(f"Skipping missing input: {p}")
            continue
        dfs.append(load_wide_csv(p))
    if not dfs:
        raise SystemExit("No valid inputs loaded.")
    return pd.concat(dfs, ignore_index=True)
