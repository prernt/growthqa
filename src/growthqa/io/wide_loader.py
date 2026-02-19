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

def _find_concentration_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"concentration", "conc", "dose", "drug_conc"}:
            return c
    return None


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
    logging.info(f"load_wide_csv: {path} columns={list(df.columns)}")

    missing = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: Missing required columns: {missing}")

    conc_col = _find_concentration_col(df)
    if conc_col is not None and conc_col != "Concentration":
        df = df.rename(columns={conc_col: "Concentration"})

    tcols = get_time_columns(df)
    if not tcols:
        raise ValueError(f"{path}: No time columns like 'T0.00 (h)' found.")

    # strict boolean Is_Valid
    df["Is_Valid"] = coerce_is_valid_bool(df["Is_Valid"])

    # Domain flags (kept explicit in meta to mitigate synthetic/lab shift).
    stem = Path(path).stem.lower()
    inferred_source = "synthetic" if ("syn" in stem or "synthetic" in stem) else "lab"
    if "source_type" not in df.columns:
        df["source_type"] = inferred_source
    df["source_type"] = (
        df["source_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"syn": "synthetic", "synth": "synthetic"})
    )
    df.loc[~df["source_type"].isin(["synthetic", "lab"]), "source_type"] = inferred_source
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = (df["source_type"] == "synthetic").astype(int)
    else:
        df["is_synthetic"] = pd.to_numeric(df["is_synthetic"], errors="coerce").fillna(
            (df["source_type"] == "synthetic").astype(int)
        ).astype(int)

    # keep only known columns (plus optional Concentration)
    keep = REQUIRED_META_COLS + ["source_type", "is_synthetic"] + tcols
    if "Concentration" in df.columns:
        keep = REQUIRED_META_COLS + ["Concentration", "source_type", "is_synthetic"] + tcols
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
