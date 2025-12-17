from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .long_format import standardize_long
from .parsers import (
    convert_grofit_v_wide_to_long,
    convert_simple_wide_to_long,
    detect_wide_time_columns,
    parse_excel_any,
    parse_time_table_any,
)
from .time_parse import make_time_header


def long_to_wide_preserve_times(df_long: pd.DataFrame, file_tag: str, add_prefix: bool = True) -> pd.DataFrame:
    """
    Convert standardized long -> wide, preserving exact unique time_h values.
    """
    df = df_long.copy()

    # stable per-file curve id
    if add_prefix:
        df["Test Id"] = df["orig_TestId"].astype(str).map(lambda s: f"{file_tag}_{s}")
    else:
        df["Test Id"] = df["orig_TestId"].astype(str)

    # average duplicates at same time
    df = df.groupby(["FileName", "Test Id", "Model Name", "Is_Valid", "time_h"], as_index=False)["OD"].mean()

    # pivot
    wide = df.pivot_table(
        index=["FileName", "Test Id", "Model Name", "Is_Valid"],
        columns="time_h",
        values="OD",
        aggfunc="mean"
    )

    # rename time columns to Txx.xx (h)
    times = np.array(wide.columns.tolist(), dtype=float)
    times_sorted = np.sort(times)
    col_map = {t: make_time_header(float(t)) for t in times_sorted}
    wide = wide.reindex(columns=times_sorted)
    wide.columns = [col_map[t] for t in wide.columns]

    wide = wide.reset_index()
    return wide


def parse_any_file_to_long(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".xlsx", ".xls"}:
        return parse_excel_any(path)

    if ext in {".csv"}:
        # Try as simple time table first
        out = parse_time_table_any(path)
        if out is not None:
            return out

        # Try already-wide synthetic
        df = pd.read_csv(path)
        long = convert_simple_wide_to_long(df, p.stem)
        if long is not None:
            return long
        
        # 2) grofit-wide V-columns
        if any(isinstance(c, str) and re.match(r"^V\d+$", c.strip()) for c in df.columns):
            return convert_grofit_v_wide_to_long(df, p.stem)


        raise ValueError(f"CSV format not recognized (not a time-table and not synthetic-wide): {path}")

    raise ValueError(f"Unsupported file extension: {ext} ({path})")
